#############################################################################
# File: test_binary_router.py
#
# Description:
#   Checks deterministic feature preparation and model persistence for the binary router.
#
#   - Confirms session IDs are excluded from classifier text features.
#   - Confirms long histories are sampled across the full conversation.
#   - Confirms split session fragments are reconstructed before feature extraction.
#   - Confirms a saved router round-trips with the same prediction and metadata.
#############################################################################

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from asdrp.classification_algorithms.binary_router import (
    BinaryTfidfRouter,
    RouterFeatureConfig,
    build_router_document,
    reconstruct_sessions,
)
from asdrp.eval_schemas import MemoryEntry


def _entry(index: int, text: str, *, part: int = 0) -> MemoryEntry:
    return MemoryEntry(
        entry_id=f"e{index}_{part}",
        text=text,
        ordinal=index + part,
        metadata={
            "session_index": index,
            "session_part": part,
            "source_id": f"session_{index}",
        },
    )


def test_router_features_exclude_session_ids_and_sample_across_history() -> None:
    entries = [
        _entry(
            index,
            f"Session ID: secret_{index}\nSession Date: 2026-01-{index + 1:02d}\n\n"
            f"USER: content {index}",
        )
        for index in range(10)
    ]
    document, count = build_router_document(
        entries,
        config=RouterFeatureConfig(max_sessions=4, max_chars_per_session=2_000),
    )

    assert count == 4
    assert "secret_0" not in document
    assert "content 0" in document
    assert "content 9" in document


def test_split_session_parts_are_reconstructed_once() -> None:
    entries = [
        _entry(
            0,
            "Session ID: s0\nSession Date: 2026-01-01\n\nUSER: first half",
            part=0,
        ),
        MemoryEntry(
            entry_id="e0_1",
            text="Session ID: s0\nSession Date: 2026-01-01\n\nASSISTANT: second half",
            ordinal=1,
            metadata={"session_index": 0, "session_part": 1, "source_id": "s0"},
        ),
    ]

    sessions = reconstruct_sessions(entries)
    assert len(sessions) == 1
    assert sessions[0].count("Session Date: 2026-01-01") == 1
    assert "USER: first half" in sessions[0]
    assert "ASSISTANT: second half" in sessions[0]


def test_saved_router_round_trip(tmp_path: Path) -> None:
    labels = ("single-session-preference", "knowledge-update")
    documents = [
        "USER: I prefer tea and quiet rooms",
        "USER: My favorite color is blue and I prefer jazz",
        "USER: I changed my address and the unnecessary one is obsolete",
        "USER: Correction, I now use a new phone number",
    ]
    y = np.asarray([0, 0, 1, 1])
    vectorizer = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2))),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))),
        ]
    )
    features = vectorizer.fit_transform(documents)
    classifier = LinearSVC(class_weight="balanced", random_state=224).fit(features, y)
    evaluation_question_ids = ("heldout_q1", "heldout_q2")
    evaluation_data_sha256 = "a" * 64
    router = BinaryTfidfRouter(
        vectorizer=vectorizer,
        classifier=classifier,
        labels=labels,
        label_to_memory={labels[0]: "hvm", labels[1]: "episodic"},
        feature_config=RouterFeatureConfig(),
        evaluation_question_ids=evaluation_question_ids,
        evaluation_data_sha256=evaluation_data_sha256,
    )

    path = tmp_path / "router.joblib"
    router.save(path)
    loaded = BinaryTfidfRouter.load(path)

    assert loaded.labels == labels
    assert loaded.label_to_memory[labels[0]] == "hvm"
    assert loaded.label_to_memory[labels[1]] == "episodic"
    assert loaded.evaluation_question_ids == evaluation_question_ids
    assert loaded.evaluation_data_sha256 == evaluation_data_sha256
