#############################################################################
# File: train_binary_router.py
#
# Description:
#   Trains and evaluates the context-only full-history binary router used to
#   choose between HVM and episodic memory on two LongMemEval question types.
#
#   - Reads the supplied LongMemEval train/test files without creating a new split.
#   - Selects the same deterministic train count and test count for both labels.
#   - Caps the combined train/test total at 30 examples per label by default.
#   - Persists the exact held-out question IDs and test-file hash in the model artifact.
#   - Reconstructs and analyzes the complete LongMemEval-M history per example.
#   - Fits a word 1-3 gram TF-IDF vocabulary on training histories only.
#   - Pools session features with global, repetition, and recent-history views.
#   - Adds compact structure and preference/update cue features.
#   - Selects the linear classifier and regularization using repeated stratified
#     cross-validation on the training split only; the test split is untouched.
#   - Rejects complete-context overlap across train and test.
#   - Saves the production model plus held-out metrics, CV results, predictions,
#     data hashes, timing, and feature settings.
#############################################################################

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import sparse
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.svm import LinearSVC

from asdrp.classification_algorithms.binary_router import (
    ARTIFACT_VERSION,
    DEFAULT_LABELS,
    DEFAULT_LABEL_TO_MEMORY,
    BinaryTfidfRouter,
    RouterFeatureConfig,
    prepare_router_sessions,
    reconstruct_sessions,
    structural_features,
    transform_conversation_features,
)
from asdrp.eval_schemas import MemoryEntry


IMPLEMENTATION_NAME = "full-history-session-pool-v4"


@dataclass(frozen=True, slots=True)
class _Example:
    example_id: str
    label: str
    context_hash: str
    sessions: tuple[str, ...]


def _canonical_label(value: Any, question_id: Any = None) -> str | None:
    if str(question_id or "").strip().casefold().endswith("_abs"):
        return "abstention"
    label = str(value or "").strip().casefold().replace("_", "-").replace(" ", "-")
    if not label:
        return None
    aliases = {
        "multi-session-reasoning": "multi-session",
        "temporal": "temporal-reasoning",
        "knowledge-update-reasoning": "knowledge-update",
        "abstentation": "abstention",
    }
    return aliases.get(label, label)


def _iter_json_or_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        first = ""
        while character := handle.read(1):
            if not character.isspace():
                first = character
                break
        handle.seek(0)
        if first == "[":
            payload = json.load(handle)
            if not isinstance(payload, list):
                raise ValueError(f"Expected a JSON array in {path}")
            for index, row in enumerate(payload):
                if not isinstance(row, dict):
                    raise ValueError(f"Row {index} in {path} is not an object")
                yield index, row
            return

        for line_number, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {line_number} in {path} is not an object")
            yield line_number, row


def _entries_from_row(row: dict[str, Any], row_index: int) -> list[MemoryEntry]:
    raw_sessions = row.get("haystack_sessions")
    if not isinstance(raw_sessions, list):
        raise ValueError(f"LongMemEval row {row_index} is missing haystack_sessions")
    dates = (
        row.get("haystack_dates") if isinstance(row.get("haystack_dates"), list) else []
    )
    session_ids = (
        row.get("haystack_session_ids")
        if isinstance(row.get("haystack_session_ids"), list)
        else []
    )
    question_key = str(row.get("question_id") or row_index)
    entries: list[MemoryEntry] = []

    for session_index, session in enumerate(raw_sessions):
        if not isinstance(session, list):
            continue
        date = str(dates[session_index]) if session_index < len(dates) else ""
        session_id = (
            str(session_ids[session_index])
            if session_index < len(session_ids)
            else f"session_{session_index}"
        )
        messages: list[str] = []
        for turn in session:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role") or "unknown").upper()
            content = (
                str(turn.get("content") or "").replace("<|endoftext|>", "").strip()
            )
            if content:
                messages.append(f"{role}: {content}")
        if not messages:
            continue
        header = f"Session ID: {session_id}\nSession Date: {date}".strip()
        entries.append(
            MemoryEntry(
                entry_id=f"{question_key}_session_{session_index}_part_0",
                text=f"{header}\n\n" + "\n\n".join(messages),
                ordinal=len(entries),
                metadata={
                    "entry_kind": "session",
                    "source_id": session_id,
                    "timestamp": date,
                    "session_index": session_index,
                    "session_part": 0,
                },
            )
        )
    if not entries:
        raise ValueError(f"LongMemEval row {row_index} has no usable conversation text")
    return entries


def _context_hash(entries: Sequence[MemoryEntry]) -> str:
    raw = "\n\n<<<FULL_SESSION_BOUNDARY>>>\n\n".join(
        reconstruct_sessions(entries)
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _count_selected_labels(path: Path, labels: tuple[str, str]) -> Counter[str]:
    allowed = set(labels)
    counts: Counter[str] = Counter()
    for row_index, row in _iter_json_or_jsonl(path):
        question_id = str(row.get("question_id") or row.get("id") or f"row_{row_index}")
        label = _canonical_label(row.get("question_type"), question_id)
        if label in allowed:
            counts[label] += 1
    return counts


def _allocate_label_limits(
    train_counts: Counter[str],
    test_counts: Counter[str],
    *,
    labels: tuple[str, str],
    max_examples_per_label: int,
) -> tuple[dict[str, int], dict[str, int]]:
    if max_examples_per_label < 1:
        raise ValueError("max_examples_per_label must be positive")

    for label in labels:
        if train_counts[label] == 0:
            raise ValueError(f"No training examples found for selected label {label!r}")
        if test_counts[label] == 0:
            raise ValueError(f"No test examples found for selected label {label!r}")

    # Both classes receive exactly the same number of training examples and exactly
    # the same number of test examples. The combined per-class total is capped while
    # the supplied split ratio is preserved as closely as availability allows.
    common_train = min(train_counts[label] for label in labels)
    common_test = min(test_counts[label] for label in labels)
    target_total = min(max_examples_per_label, common_train + common_test)

    source_train = sum(train_counts[label] for label in labels)
    source_test = sum(test_counts[label] for label in labels)
    source_total = source_train + source_test
    desired_train = int(target_total * source_train / source_total + 0.5)

    # Clamp the rounded split so the exact same train/test counts are feasible for
    # both labels. _load_examples then takes the earliest rows in file order, making
    # the selected question IDs deterministic across runs on unchanged files.
    min_train = max(0, target_total - common_test)
    max_train = min(common_train, target_total)
    train_limit = min(max(desired_train, min_train), max_train)
    test_limit = target_total - train_limit

    if train_limit < 2:
        raise ValueError(
            "Balanced selection leaves fewer than two training examples per label; "
            "increase --max-examples-per-label or provide larger training splits"
        )
    if test_limit < 1:
        raise ValueError(
            "Balanced selection leaves no held-out examples per label; increase "
            "--max-examples-per-label or provide larger test splits"
        )

    return (
        {label: train_limit for label in labels},
        {label: test_limit for label in labels},
    )


def _load_examples(
    path: Path,
    *,
    labels: tuple[str, str],
    feature_config: RouterFeatureConfig,
    per_label_limits: dict[str, int],
) -> list[_Example]:
    examples: list[_Example] = []
    allowed = set(labels)
    selected: Counter[str] = Counter()

    for row_index, row in _iter_json_or_jsonl(path):
        question_id = str(row.get("question_id") or row.get("id") or f"row_{row_index}")
        label = _canonical_label(row.get("question_type"), question_id)
        if label not in allowed or selected[label] >= per_label_limits[label]:
            continue

        entries = _entries_from_row(row, row_index)
        sessions = tuple(prepare_router_sessions(entries, config=feature_config))
        if not sessions:
            continue

        examples.append(
            _Example(
                example_id=question_id,
                label=label,
                context_hash=_context_hash(entries),
                sessions=sessions,
            )
        )
        selected[label] += 1

        if all(selected[item] >= per_label_limits[item] for item in labels):
            break

    if not examples:
        raise ValueError(
            f"No selected binary LongMemEval examples were found in {path}"
        )

    missing = [label for label in labels if selected[label] != per_label_limits[label]]
    if missing:
        raise ValueError(
            f"{path} did not contain the expected selected counts for label(s): {missing}"
        )
    return examples


def _assert_unique_example_ids(examples: Sequence[_Example], split_name: str) -> None:
    counts = Counter(example.example_id for example in examples)
    duplicates = sorted(example_id for example_id, count in counts.items() if count > 1)
    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise ValueError(
            f"{split_name} contains duplicate selected question IDs: {preview}"
        )


def _assert_no_leakage(train: Sequence[_Example], test: Sequence[_Example]) -> None:
    train_contexts = {example.context_hash for example in train}
    test_contexts = {example.context_hash for example in test}
    overlap = train_contexts & test_contexts
    if overlap:
        raise ValueError(
            f"Train/test leakage: {len(overlap)} complete contexts appear in both splits"
        )


def _fit_vectorizer(
    train: Sequence[_Example],
    *,
    max_features: int,
) -> TfidfVectorizer:
    if max_features < 1:
        raise ValueError("max_features must be positive")

    # Vocabulary/IDF are learned only from training histories. Joining sessions
    # here is used only to fit the vocabulary; classification later transforms
    # every session separately and pools those vectors.
    documents = [
        "\n\n<<<SESSION_BOUNDARY>>>\n\n".join(example.sessions) for example in train
    ]
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
        dtype=np.float32,
        max_features=max_features,
    )
    vectorizer.fit(documents)
    return vectorizer


def _transform_examples(
    examples: Sequence[_Example],
    *,
    vectorizer: TfidfVectorizer,
    feature_config: RouterFeatureConfig,
) -> sparse.csr_matrix:
    rows = [
        transform_conversation_features(
            example.sessions,
            vectorizer=vectorizer,
            config=feature_config,
        )
        for example in examples
    ]
    return sparse.vstack(rows, format="csr", dtype=np.float32)


def _candidate_models(seed: int):
    for c in (0.03, 0.1, 0.3, 1.0, 3.0):
        yield (
            f"LinearSVC(C={c:g})",
            LinearSVC(
                C=c,
                class_weight="balanced",
                dual="auto",
                random_state=seed,
            ),
        )
    for c in (0.03, 0.1, 0.3, 1.0, 3.0):
        yield (
            f"LogisticRegression(C={c:g})",
            LogisticRegression(
                C=c,
                class_weight="balanced",
                solver="liblinear",
                max_iter=3_000,
                random_state=seed,
            ),
        )


def _select_classifier(
    features: sparse.csr_matrix,
    numeric_labels: np.ndarray,
    *,
    seed: int,
) -> tuple[str, Any, list[dict[str, Any]]]:
    counts = np.bincount(numeric_labels, minlength=2)
    if np.min(counts) < 2:
        raise ValueError("Cross-validation requires at least two examples per class")

    n_splits = min(5, int(np.min(counts)))
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=4,
        random_state=seed,
    )

    results: list[dict[str, Any]] = []
    best_name = ""
    best_estimator = None
    best_score = -1.0
    best_std = float("inf")

    for name, estimator in _candidate_models(seed):
        scores = cross_val_score(
            estimator,
            features,
            numeric_labels,
            scoring="f1_macro",
            cv=cv,
            n_jobs=1,
        )
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        results.append(
            {
                "model": name,
                "mean_macro_f1": mean_score,
                "std_macro_f1": std_score,
                "folds": int(len(scores)),
            }
        )

        # Prefer higher mean macro F1; for practically equal means, prefer the
        # lower-variance candidate rather than a more aggressive hyperparameter.
        if mean_score > best_score + 1e-6 or (
            abs(mean_score - best_score) <= 1e-6 and std_score < best_std
        ):
            best_name = name
            best_estimator = clone(estimator)
            best_score = mean_score
            best_std = std_score

    if best_estimator is None:
        raise RuntimeError("No classifier candidate was evaluated")

    best_estimator.fit(features, numeric_labels)
    results.sort(key=lambda item: (-item["mean_macro_f1"], item["std_macro_f1"]))
    return best_name, best_estimator, results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train the LongMemEval HVM/episodic full-history context-only router."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("train_file", type=Path)
    parser.add_argument("test_file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-a", default=DEFAULT_LABELS[0])
    parser.add_argument("--label-b", default=DEFAULT_LABELS[1])
    parser.add_argument(
        "--memory-a", default=DEFAULT_LABEL_TO_MEMORY[DEFAULT_LABELS[0]]
    )
    parser.add_argument(
        "--memory-b", default=DEFAULT_LABEL_TO_MEMORY[DEFAULT_LABELS[1]]
    )
    parser.add_argument("--max-sessions", type=int, default=512)
    parser.add_argument("--max-chars-per-session", type=int, default=2_400)
    parser.add_argument("--recent-fraction", type=float, default=0.25)
    parser.add_argument("--max-features", type=int, default=75_000)
    parser.add_argument(
        "--max-examples-per-label",
        type=int,
        default=30,
        help=(
            "Maximum combined train+test examples per selected label; both labels "
            "receive identical train counts and identical test counts"
        ),
    )
    parser.add_argument("--seed", type=int, default=224)
    args = parser.parse_args()

    if not args.train_file.is_file():
        parser.error(f"train file does not exist: {args.train_file}")
    if not args.test_file.is_file():
        parser.error(f"test file does not exist: {args.test_file}")

    labels = (
        _canonical_label(args.label_a),
        _canonical_label(args.label_b),
    )
    if labels[0] is None or labels[1] is None or labels[0] == labels[1]:
        parser.error("--label-a and --label-b must resolve to two different labels")
    labels = (labels[0], labels[1])
    mapping = {labels[0]: args.memory_a, labels[1]: args.memory_b}
    if set(mapping.values()) != {"hvm", "episodic"}:
        parser.error("--memory-a and --memory-b must map exactly to hvm and episodic")

    feature_config = RouterFeatureConfig(
        max_sessions=args.max_sessions,
        max_chars_per_session=args.max_chars_per_session,
        recent_fraction=args.recent_fraction,
    )
    feature_config.validate()

    print(f"router_implementation={IMPLEMENTATION_NAME}")
    print(f"trainer_file={Path(__file__).resolve()}")

    wall_start = time.perf_counter()
    prep_start = time.perf_counter()

    train_source_counts = _count_selected_labels(args.train_file, labels)
    test_source_counts = _count_selected_labels(args.test_file, labels)
    train_limits, test_limits = _allocate_label_limits(
        train_source_counts,
        test_source_counts,
        labels=labels,
        max_examples_per_label=args.max_examples_per_label,
    )

    train = _load_examples(
        args.train_file,
        labels=labels,
        feature_config=feature_config,
        per_label_limits=train_limits,
    )
    test = _load_examples(
        args.test_file,
        labels=labels,
        feature_config=feature_config,
        per_label_limits=test_limits,
    )
    _assert_unique_example_ids(train, "training split")
    _assert_unique_example_ids(test, "test split")
    _assert_no_leakage(train, test)
    test_question_ids = tuple(item.example_id for item in test)
    test_sha256 = _file_sha256(args.test_file)
    preparation_seconds = time.perf_counter() - prep_start

    print(
        f"source_train={sum(train_source_counts.values())} {dict(train_source_counts)}"
    )
    print(f"source_test={sum(test_source_counts.values())} {dict(test_source_counts)}")
    print(f"train={len(train)} {dict(Counter(item.label for item in train))}")
    print(f"test={len(test)} {dict(Counter(item.label for item in test))}")
    print(
        "train_sessions="
        f"{sum(len(item.sessions) for item in train)} "
        "test_sessions="
        f"{sum(len(item.sessions) for item in test)}"
    )

    vectorizer_start = time.perf_counter()
    vectorizer = _fit_vectorizer(train, max_features=args.max_features)
    train_features = _transform_examples(
        train,
        vectorizer=vectorizer,
        feature_config=feature_config,
    )
    vectorization_seconds = time.perf_counter() - vectorizer_start

    label_to_index = {label: index for index, label in enumerate(labels)}
    train_numeric_labels = np.asarray(
        [label_to_index[item.label] for item in train],
        dtype=np.int64,
    )

    training_start = time.perf_counter()
    classifier_name, classifier, cv_results = _select_classifier(
        train_features,
        train_numeric_labels,
        seed=args.seed,
    )
    training_seconds = time.perf_counter() - training_start

    router = BinaryTfidfRouter(
        vectorizer=vectorizer,
        classifier=classifier,
        labels=labels,
        label_to_memory=mapping,
        feature_config=feature_config,
        classifier_name=classifier_name,
        evaluation_question_ids=test_question_ids,
        evaluation_data_sha256=test_sha256,
    )

    model_path = args.output_dir / "model" / "tfidf_svc.joblib"
    router.save(model_path)

    evaluation_start = time.perf_counter()
    test_features = _transform_examples(
        test,
        vectorizer=vectorizer,
        feature_config=feature_config,
    )
    predicted_indices = classifier.predict(test_features)
    scores = np.asarray(classifier.decision_function(test_features)).reshape(-1)
    predicted_labels = [labels[int(index)] for index in predicted_indices]
    expected = [item.label for item in test]
    evaluation_seconds = time.perf_counter() - evaluation_start

    metrics = {
        "count": len(test),
        "labels": list(labels),
        "label_to_memory": mapping,
        "accuracy": float(accuracy_score(expected, predicted_labels)),
        "balanced_accuracy": float(balanced_accuracy_score(expected, predicted_labels)),
        "macro_f1": float(f1_score(expected, predicted_labels, average="macro")),
        "confusion_matrix": confusion_matrix(
            expected,
            predicted_labels,
            labels=list(labels),
        ).tolist(),
        "classification_report": classification_report(
            expected,
            predicted_labels,
            labels=list(labels),
            output_dict=True,
            zero_division=0,
        ),
    }

    predictions = [
        {
            "example_id": item.example_id,
            "label": item.label,
            "predicted_label": predicted,
            "selected_memory": mapping[predicted],
            "decision_score": float(score),
            "decision_score_positive_label": labels[1],
            "correct": predicted == item.label,
            "session_count": len(item.sessions),
        }
        for item, predicted, score in zip(
            test,
            predicted_labels,
            scores,
            strict=True,
        )
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "metrics.json", metrics)
    _write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    _write_json(
        args.output_dir / "cv_results.json",
        {
            "selection_metric": "macro_f1",
            "test_split_used_for_selection": False,
            "selected_model": classifier_name,
            "results": cv_results,
        },
    )
    _write_json(
        args.output_dir / "router_metadata.json",
        {
            "artifact_version": ARTIFACT_VERSION,
            "model_type": "session_pool_tfidf_linear_memory_router",
            "dataset": "longmemeval",
            "context_only": True,
            "question_text_in_features": False,
            "question_type_in_features": False,
            "labels": list(labels),
            "label_to_memory": mapping,
            "evaluation_contract": {
                "test_file_sha256": test_sha256,
                "question_ids": list(test_question_ids),
                "question_count": len(test_question_ids),
            },
            "feature_config": asdict(feature_config),
            "feature_extractor": {
                "word_ngrams": [1, 3],
                "max_features": args.max_features,
                "sublinear_tf": True,
                "pooling": [
                    "global_session_max",
                    "cross_session_repetition",
                    "recent_session_max",
                    "structural_and_cue_features",
                ],
                "uses_all_normal_longmemeval_m_sessions": args.max_sessions >= 500,
            },
            "classifier": {
                "selected_by": "4x repeated stratified 5-fold CV on training only",
                "selected_model": classifier_name,
                "seed": args.seed,
            },
            "model_artifact": str(model_path.relative_to(args.output_dir)),
        },
    )
    _write_json(
        args.output_dir / "data_manifest.json",
        {
            "train_file": str(args.train_file),
            "train_sha256": _file_sha256(args.train_file),
            "train_source_distribution": dict(train_source_counts),
            "train_examples": len(train),
            "train_distribution": dict(Counter(item.label for item in train)),
            "test_file": str(args.test_file),
            "test_sha256": test_sha256,
            "test_source_distribution": dict(test_source_counts),
            "test_examples": len(test),
            "test_distribution": dict(Counter(item.label for item in test)),
            "selected_train_question_ids": [item.example_id for item in train],
            "selected_test_question_ids": list(test_question_ids),
            "max_examples_per_label_total": args.max_examples_per_label,
            "selection_rule": (
                "Choose one common train count and one common test count for both "
                "labels, cap their combined per-label total by max_examples_per_label, "
                "preserve the supplied split ratio as closely as availability permits, "
                "and take the earliest required rows in each file deterministically."
            ),
            "uses_supplied_train_test_directly": True,
            "creates_or_resplits_data": False,
            "test_used_for_model_selection": False,
        },
    )
    _write_json(
        args.output_dir / "training_summary.json",
        {
            "preparation_seconds": preparation_seconds,
            "vectorization_seconds": vectorization_seconds,
            "classifier_selection_and_fit_seconds": training_seconds,
            "evaluation_seconds": evaluation_seconds,
            "mean_test_prediction_seconds": evaluation_seconds / len(test),
            "wall_runtime_seconds": time.perf_counter() - wall_start,
            "vocabulary_size": len(vectorizer.vocabulary_),
            "feature_count": int(train_features.shape[1]),
            "train_session_count": sum(len(item.sessions) for item in train),
            "test_session_count": sum(len(item.sessions) for item in test),
        },
    )

    print(f"selected_model={classifier_name}")
    print(
        f"cv_macro_f1={cv_results[0]['mean_macro_f1']:.4f}"
        f"+/-{cv_results[0]['std_macro_f1']:.4f}"
    )
    print(
        f"test_accuracy={metrics['accuracy']:.4f} "
        f"test_macro_f1={metrics['macro_f1']:.4f} "
        f"mean_prediction_seconds={evaluation_seconds / len(test):.6f}"
    )
    print(f"model={model_path}")


if __name__ == "__main__":
    main()
