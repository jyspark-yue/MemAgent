#############################################################################
# File: binary_router.py
#
# Description:
#   Provides the production context-only binary router used to choose between
#   HVM and episodic memory before the evaluation question is revealed.
#
#   - Reconstructs evaluator-split MemoryEntry objects into complete sessions.
#   - Examines the full LongMemEval-M history instead of eight arbitrary samples.
#   - Uses session-pooled TF-IDF features so one informative session is not diluted
#     by hundreds of unrelated filler sessions.
#   - Adds cross-session repetition and recent-history views to expose update cues.
#   - Adds lightweight conversation-structure and preference/update cue features.
#   - Loads and validates one sklearn joblib artifact and caches it per process.
#   - Carries the exact held-out question IDs and test-file hash used for evaluation.
#   - Returns the predicted question type, selected memory, signed linear margin,
#     and lightweight routing diagnostics.
#############################################################################

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

from asdrp.eval_schemas import MemoryEntry


ARTIFACT_VERSION = 4
DEFAULT_LABELS = ("single-session-preference", "knowledge-update")
DEFAULT_LABEL_TO_MEMORY = {
    "single-session-preference": "hvm",
    "knowledge-update": "episodic",
}
SUPPORTED_MEMORIES = frozenset({"hvm", "episodic"})
SESSION_SEPARATOR = "\n\n<<<ROUTER_SESSION_BOUNDARY>>>\n\n"
TRUNCATION_MARKER = "\n\n<<<ROUTER_SESSION_TRUNCATED>>>\n\n"

_ROLE_RE = re.compile(r"(?m)^(?:USER|ASSISTANT|SYSTEM|TOOL):")
_PREFERENCE_RE = re.compile(
    r"\b(?:prefer(?:s|red|ring)?|preference|favorite|favourite|"
    r"like(?:s|d)?|love(?:s|d)?|enjoy(?:s|ed)?|dislike(?:s|d)?|"
    r"hate(?:s|d)?|rather|fan\s+of|into|avoid(?:s|ed)?|"
    r"can(?:not|'t)\s+stand)\b",
    re.IGNORECASE,
)
_UPDATE_RE = re.compile(
    r"\b(?:now|currently|recently|actually|instead|anymore|"
    r"no\s+longer|used\s+to|change(?:d|s|ing)?|update(?:d|s|ing)?|"
    r"switch(?:ed|es|ing)?|move(?:d|s|ing)?|replace(?:d|s|ing)?|"
    r"stop(?:ped|s|ping)?|start(?:ed|s|ing)?|became|become|"
    r"new|latest|previously|formerly)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class RouterFeatureConfig:
    # LongMemEval-M has roughly 500 sessions. 512 therefore keeps the normal
    # benchmark history intact while still bounding pathological inputs.
    max_sessions: int = 512
    max_chars_per_session: int = 2_400
    recent_fraction: float = 0.25

    def validate(self) -> None:
        if self.max_sessions < 1:
            raise ValueError("max_sessions must be at least 1")
        if self.max_chars_per_session < 256:
            raise ValueError("max_chars_per_session must be at least 256")
        if not 0.05 <= self.recent_fraction <= 0.5:
            raise ValueError("recent_fraction must be between 0.05 and 0.5")


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    predicted_label: str
    selected_memory: str
    decision_score: float
    positive_label: str
    sampled_session_count: int
    sampled_character_count: int


@dataclass(slots=True)
class BinaryTfidfRouter:
    # The historical class name is retained so evaluator imports do not change.
    # The fitted representation is now session-pooled rather than one flat sample.
    vectorizer: Any
    classifier: Any
    labels: tuple[str, str]
    label_to_memory: dict[str, str]
    feature_config: RouterFeatureConfig
    classifier_name: str = "linear"
    evaluation_question_ids: tuple[str, ...] = ()
    evaluation_data_sha256: str = ""

    def predict_entries(self, entries: Sequence[MemoryEntry]) -> RoutingDecision:
        sessions = prepare_router_sessions(entries, config=self.feature_config)
        if not sessions:
            raise ValueError("Binary router received no usable memory-entry text")

        features = transform_conversation_features(
            sessions,
            vectorizer=self.vectorizer,
            config=self.feature_config,
        )
        predicted_index = int(self.classifier.predict(features)[0])
        score = float(
            np.asarray(self.classifier.decision_function(features)).reshape(-1)[0]
        )
        predicted_label = self.labels[predicted_index]

        try:
            selected_memory = self.label_to_memory[predicted_label]
        except KeyError as error:
            raise ValueError(
                f"No memory mapping is stored for router label {predicted_label!r}"
            ) from error

        return RoutingDecision(
            predicted_label=predicted_label,
            selected_memory=selected_memory,
            decision_score=score,
            positive_label=self.labels[1],
            sampled_session_count=len(sessions),
            sampled_character_count=sum(len(session) for session in sessions),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "artifact_version": ARTIFACT_VERSION,
                "model_type": "session_pool_tfidf_linear_memory_router",
                "labels": self.labels,
                "label_to_memory": dict(self.label_to_memory),
                "feature_config": asdict(self.feature_config),
                "vectorizer": self.vectorizer,
                "classifier": self.classifier,
                "classifier_name": self.classifier_name,
                "evaluation_question_ids": self.evaluation_question_ids,
                "evaluation_data_sha256": self.evaluation_data_sha256,
                "context_only": True,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "BinaryTfidfRouter":
        payload = joblib.load(path)
        if not isinstance(payload, Mapping):
            raise ValueError("Binary router artifact is not a mapping")
        if payload.get("artifact_version") != ARTIFACT_VERSION:
            raise ValueError(
                "Unsupported binary router artifact version. Retrain the router with "
                "asdrp.classification_algorithms.train_binary_router; version 4 stores "
                "the exact held-out evaluation IDs and source test-file hash."
            )

        labels_raw = payload.get("labels")
        if not isinstance(labels_raw, (list, tuple)) or len(labels_raw) != 2:
            raise ValueError("Binary router artifact must contain exactly two labels")
        labels = (str(labels_raw[0]), str(labels_raw[1]))
        if len(set(labels)) != 2:
            raise ValueError("Binary router artifact labels must be unique")

        mapping_raw = payload.get("label_to_memory")
        if not isinstance(mapping_raw, Mapping):
            raise ValueError("Binary router artifact is missing label_to_memory")
        label_to_memory = {str(key): str(value) for key, value in mapping_raw.items()}
        if set(label_to_memory) != set(labels):
            raise ValueError("Binary router memory mapping does not match its labels")
        if set(label_to_memory.values()) != SUPPORTED_MEMORIES:
            raise ValueError(
                "Binary router must map exactly one label to HVM and one to episodic"
            )

        config_raw = payload.get("feature_config")
        if not isinstance(config_raw, Mapping):
            raise ValueError("Binary router artifact is missing feature_config")
        config = RouterFeatureConfig(
            max_sessions=int(config_raw["max_sessions"]),
            max_chars_per_session=int(config_raw["max_chars_per_session"]),
            recent_fraction=float(config_raw["recent_fraction"]),
        )
        config.validate()

        vectorizer = payload.get("vectorizer")
        classifier = payload.get("classifier")
        if vectorizer is None or classifier is None:
            raise ValueError("Binary router artifact is missing fitted sklearn objects")

        evaluation_ids_raw = payload.get("evaluation_question_ids")
        if not isinstance(evaluation_ids_raw, (list, tuple)) or not evaluation_ids_raw:
            raise ValueError(
                "Binary router artifact is missing held-out evaluation question IDs"
            )
        evaluation_question_ids = tuple(
            str(value).strip() for value in evaluation_ids_raw
        )
        if any(not value for value in evaluation_question_ids):
            raise ValueError("Binary router evaluation question IDs cannot be empty")
        if len(set(evaluation_question_ids)) != len(evaluation_question_ids):
            raise ValueError("Binary router evaluation question IDs must be unique")

        evaluation_data_sha256 = str(
            payload.get("evaluation_data_sha256") or ""
        ).lower()
        if re.fullmatch(r"[0-9a-f]{64}", evaluation_data_sha256) is None:
            raise ValueError(
                "Binary router artifact is missing a valid evaluation data SHA-256"
            )

        return cls(
            vectorizer=vectorizer,
            classifier=classifier,
            labels=labels,
            label_to_memory=label_to_memory,
            feature_config=config,
            classifier_name=str(payload.get("classifier_name") or "linear"),
            evaluation_question_ids=evaluation_question_ids,
            evaluation_data_sha256=evaluation_data_sha256,
        )


def _strip_session_id(text: str) -> str:
    # Session IDs are arbitrary benchmark identifiers and are not classifier features.
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("Session ID:"):
        lines = lines[1:]
    return "\n".join(lines).strip()


def _strip_repeated_session_header(text: str) -> str:
    lines = text.strip().splitlines()
    index = 0
    if index < len(lines) and lines[index].startswith("Session ID:"):
        index += 1
    if index < len(lines) and lines[index].startswith("Session Date:"):
        index += 1
    while index < len(lines) and not lines[index].strip():
        index += 1
    return "\n".join(lines[index:]).strip()


def _session_key(entry: MemoryEntry) -> tuple[int, str]:
    session_index = entry.metadata.get("session_index")
    if isinstance(session_index, int):
        return session_index, ""
    return entry.ordinal, str(entry.metadata.get("source_id") or entry.entry_id)


def _session_part_order(entry: MemoryEntry) -> tuple[int, int]:
    value = entry.metadata.get("session_part")
    return (value if isinstance(value, int) else 0), entry.ordinal


def reconstruct_sessions(entries: Sequence[MemoryEntry]) -> list[str]:
    grouped: dict[tuple[int, str], list[MemoryEntry]] = {}
    for entry in sorted(entries, key=lambda item: item.ordinal):
        if entry.text.strip():
            grouped.setdefault(_session_key(entry), []).append(entry)

    sessions: list[str] = []
    for key in sorted(grouped, key=lambda item: (item[0], item[1])):
        parts = sorted(grouped[key], key=_session_part_order)
        first = _strip_session_id(parts[0].text)
        if not first:
            continue
        text_parts = [first]
        for part in parts[1:]:
            body = _strip_repeated_session_header(part.text)
            if body:
                text_parts.append(body)
        sessions.append("\n\n".join(text_parts))
    return sessions


def _distributed_indices(length: int, limit: int) -> list[int]:
    if length <= 0:
        return []
    if length <= limit:
        return list(range(length))
    if limit == 1:
        return [length // 2]
    return [round(index * (length - 1) / (limit - 1)) for index in range(limit)]


def _clip_session(text: str, max_chars: int) -> str:
    clean = text.strip()
    if len(clean) <= max_chars:
        return clean
    remaining = max_chars - len(TRUNCATION_MARKER)
    if remaining < 2:
        return clean[:max_chars]
    head = remaining // 2
    return clean[:head] + TRUNCATION_MARKER + clean[-(remaining - head) :]


def prepare_router_sessions(
    entries: Sequence[MemoryEntry],
    *,
    config: RouterFeatureConfig,
) -> list[str]:
    config.validate()
    sessions = reconstruct_sessions(entries)
    indices = _distributed_indices(len(sessions), config.max_sessions)
    return [
        _clip_session(sessions[index], config.max_chars_per_session)
        for index in indices
        if sessions[index].strip()
    ]


def build_router_document(
    entries: Sequence[MemoryEntry],
    *,
    config: RouterFeatureConfig,
) -> tuple[str, int]:
    # Kept for compatibility with older callers/tests. The production classifier
    # no longer feeds this flat document directly to the vectorizer.
    sessions = prepare_router_sessions(entries, config=config)
    return SESSION_SEPARATOR.join(sessions), len(sessions)


def _max_pool(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    if matrix.shape[0] == 0:
        return sparse.csr_matrix((1, matrix.shape[1]), dtype=np.float32)
    pooled = sparse.csr_matrix(matrix.max(axis=0), dtype=np.float32)
    return normalize(pooled, norm="l2", copy=False)


def _repeat_pool(
    matrix: sparse.csr_matrix,
    *,
    idf: np.ndarray | None,
) -> sparse.csr_matrix:
    if matrix.shape[0] <= 1:
        return sparse.csr_matrix((1, matrix.shape[1]), dtype=np.float32)

    indicator = matrix.copy()
    indicator.data = np.ones_like(indicator.data, dtype=np.float32)
    session_counts = np.asarray(indicator.sum(axis=0)).reshape(-1)
    repeated = session_counts > 1
    if not np.any(repeated):
        return sparse.csr_matrix((1, matrix.shape[1]), dtype=np.float32)

    indices = np.flatnonzero(repeated)
    values = np.log1p(session_counts[indices] - 1.0) / math.log1p(matrix.shape[0])

    if idf is not None and len(idf) == matrix.shape[1]:
        safe_mean = float(np.mean(idf)) or 1.0
        values = values * (idf[indices] / safe_mean)

    pooled = sparse.csr_matrix(
        (
            values.astype(np.float32, copy=False),
            (np.zeros(len(indices), dtype=np.int32), indices),
        ),
        shape=(1, matrix.shape[1]),
        dtype=np.float32,
    )
    return normalize(pooled, norm="l2", copy=False)


def _cue_count(pattern: re.Pattern[str], text: str) -> int:
    return sum(1 for _ in pattern.finditer(text))


def structural_features(
    sessions: Sequence[str],
    *,
    config: RouterFeatureConfig,
) -> np.ndarray:
    if not sessions:
        return np.zeros(15, dtype=np.float32)

    lengths = np.asarray([len(session) for session in sessions], dtype=np.float32)
    turn_counts = np.asarray(
        [len(_ROLE_RE.findall(session)) for session in sessions],
        dtype=np.float32,
    )
    preference_counts = np.asarray(
        [_cue_count(_PREFERENCE_RE, session) for session in sessions],
        dtype=np.float32,
    )
    update_counts = np.asarray(
        [_cue_count(_UPDATE_RE, session) for session in sessions],
        dtype=np.float32,
    )

    recent_count = max(1, int(math.ceil(len(sessions) * config.recent_fraction)))
    recent_preference = preference_counts[-recent_count:]
    recent_update = update_counts[-recent_count:]

    total_preference = float(preference_counts.sum())
    total_update = float(update_counts.sum())
    total_cues = total_preference + total_update

    features = np.asarray(
        [
            min(len(sessions) / config.max_sessions, 1.0),
            min(math.log1p(float(lengths.sum())) / math.log1p(1_500_000.0), 1.25),
            min(float(lengths.mean()) / config.max_chars_per_session, 1.25),
            min(float(lengths.std()) / config.max_chars_per_session, 1.25),
            min(float(lengths.max()) / config.max_chars_per_session, 1.25),
            min(float(turn_counts.mean()) / 16.0, 1.25),
            min(float(turn_counts.max()) / 40.0, 1.25),
            float(np.count_nonzero(preference_counts)) / len(sessions),
            float(np.count_nonzero(update_counts)) / len(sessions),
            min(total_preference / max(len(sessions), 1) / 2.0, 1.25),
            min(total_update / max(len(sessions), 1) / 2.0, 1.25),
            float(np.count_nonzero(recent_preference)) / recent_count,
            float(np.count_nonzero(recent_update)) / recent_count,
            (total_preference - total_update) / (total_cues + 1.0),
            float(np.count_nonzero((preference_counts + update_counts) > 0))
            / len(sessions),
        ],
        dtype=np.float32,
    )
    return features


def transform_conversation_features(
    sessions: Sequence[str],
    *,
    vectorizer: Any,
    config: RouterFeatureConfig,
) -> sparse.csr_matrix:
    if not sessions:
        raise ValueError("Cannot transform an empty conversation")

    matrix = vectorizer.transform(list(sessions)).tocsr().astype(np.float32, copy=False)

    global_pool = _max_pool(matrix)
    repeat_pool = _repeat_pool(
        matrix,
        idf=(
            np.asarray(getattr(vectorizer, "idf_", None))
            if getattr(vectorizer, "idf_", None) is not None
            else None
        ),
    )

    recent_count = max(1, int(math.ceil(matrix.shape[0] * config.recent_fraction)))
    recent_pool = _max_pool(matrix[-recent_count:])

    numeric = sparse.csr_matrix(
        structural_features(sessions, config=config).reshape(1, -1),
        dtype=np.float32,
    )
    numeric = normalize(numeric, norm="l2", copy=False)

    # Four separately normalized views prevent the 500-session history length
    # from swamping the small but potentially decisive evidence-session signal.
    return sparse.hstack(
        [global_pool, repeat_pool, recent_pool, numeric],
        format="csr",
        dtype=np.float32,
    )


@lru_cache(maxsize=4)
def load_binary_router(path: str) -> BinaryTfidfRouter:
    return BinaryTfidfRouter.load(Path(path))
