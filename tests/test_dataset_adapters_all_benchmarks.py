#############################################################################
# File: test_dataset_adapters_all_benchmarks.py
#
# Description:
#   Checks dataset parsing and token-safe splitting across every supported
#   benchmark family. The tests use small local samples that follow each
#   source's real field shape.
#
#   - Checks strict streaming JSON, tiny read chunks, Unicode, malformed
#     input, and trailing commas.
#   - Checks LongMemEval sessions, dates, updates, markers, and
#     oversized messages.
#   - Checks ecommerce sessions, preambles, unlabeled context, and long
#     turns.
#   - Checks RULER documents, EventQA books, chapter boundaries,
#     conflict facts, ReDial, and demonstrations.
#   - Checks both supported MemoryAgentBench LongMemEval literal
#     formats.
#   - Applies shared order, ID, content, and token-limit rules to
#     adapter output.
#############################################################################

from __future__ import annotations

import json

import pytest

from asdrp.dataset_adapters import (
    DatasetFormatError,
    _book_entries,
    _literal_longmemeval_sessions,
    _mab_accurate_retrieval_entries,
    _mab_ttl_entries,
    _numbered_fact_entries,
    _split_ecommerce_sessions,
    _split_paragraph_text,
    _split_units_safely,
    iter_evaluation_tasks,
    iter_json_array,
)

pytestmark = pytest.mark.unit  # All adapters are checked with local sample data.


def assert_entry_invariants(entries, counter, max_tokens):
    # Every adapter must return ordered, unique, non-empty, safe-sized entries.
    assert entries
    assert [entry.ordinal for entry in entries] == sorted(
        entry.ordinal for entry in entries
    )
    assert len({entry.entry_id for entry in entries}) == len(entries)
    assert all(entry.text.strip() for entry in entries)
    assert all(counter.count(entry.text) <= max_tokens for entry in entries)


def test_streaming_json_array_handles_tiny_file_chunks(tmp_path):
    rows = [{"i": index, "text": "ü" * index} for index in range(20)]
    path = tmp_path / "rows.json"  # Small Unicode JSON array.
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    assert list(iter_json_array(path, chunk_size=7)) == rows


@pytest.mark.parametrize(
    "bad",
    ["{}", "[1]", '[{"x": 1}', '[{"x":1}] trailing', '[{"x":1},]'],
)
def test_streaming_json_array_rejects_invalid_top_level_or_trailing_data(tmp_path, bad):
    path = tmp_path / "bad.json"  # Each parameter supplies invalid JSON input.
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(DatasetFormatError):
        list(iter_json_array(path, chunk_size=3))


def test_longmemeval_schema_and_oversize_session_chunking(
    tmp_path, counter, longmemeval_row
):
    # Make the first message too large for one memory entry.
    longmemeval_row["haystack_sessions"][0][0]["content"] = " ".join(
        f"token{i}" for i in range(80)
    )
    path = tmp_path / "lme.json"  # One real-shaped LongMemEval row.
    path.write_text(json.dumps([longmemeval_row]), encoding="utf-8")
    task = next(  # Adapter should split the oversized first session.
        iter_evaluation_tasks(
            data_file=path,
            dataset="longmemeval",
            embedding_model="x",
            max_entry_tokens=25,
            embedding_counter=counter,
        )
    )
    assert_entry_invariants(task.entries, counter, 25)
    assert any(entry.metadata["session_part"] > 0 for entry in task.entries)
    assert all(
        "Session ID:" in entry.text and "Session Date:" in entry.text
        for entry in task.entries
    )
    assert "<|endoftext|>" not in "\n".join(entry.text for entry in task.entries)
    assert task.retrieval.prefer_latest is True
    assert task.questions[0].metadata["answer_session_ids"] == ["s-new"]


def test_longmemeval_missing_sessions_fails(counter):
    # A row without its main session field should fail with a useful message.
    from asdrp.dataset_adapters import _longmemeval_task

    with pytest.raises(DatasetFormatError, match="haystack_sessions"):
        _longmemeval_task({}, 0, counter, 50)


def test_ecommerce_preserves_sessions_preamble_and_long_turns(counter):
    context = (  # Preamble followed by one large and one short session.
        "Customer ID: c1\nSource: synthetic\n\n[Session c1_s1]\nUSER: "
        + "word " * 50
        + "\nASSISTANT: yes\n\n[Session c1_s2]\nUSER: newest preference"
    )
    entries = _split_ecommerce_sessions(context, "c1", counter, 20)  # Split result.
    assert_entry_invariants(entries, counter, 20)
    assert {entry.metadata["source_id"] for entry in entries} == {"c1_s1", "c1_s2"}
    assert all("Customer ID: c1" in entry.text for entry in entries)
    assert any(entry.metadata["session_part"] > 0 for entry in entries)


def test_ecommerce_without_session_markers_still_chunks(counter):
    # Unlabeled context should still become safe-sized entries.
    entries = _split_ecommerce_sessions("paragraph " * 60, "c", counter, 15)
    assert_entry_invariants(entries, counter, 15)


def test_mab_ruler_documents_split_independently(counter):
    context = (  # Three labeled RULER documents with one target phrase.
        "Document 1:\n"
        + "a " * 40
        + "\nDocument 2:\nneedle fact\nDocument 3:\n"
        + "b " * 40
    )
    entries, plan = _mab_accurate_retrieval_entries(context, "ruler_qa_1", counter, 18)
    assert_entry_invariants(entries, counter, 18)
    assert {entry.metadata["section_index"] for entry in entries} >= {0, 1, 2}
    assert plan.top_k == 12
    assert any("needle fact" in entry.text for entry in entries)


def test_mab_eventqa_and_lru_book_chapter_boundaries(counter):
    context = "Preface text\nCHAPTER I\n" + "one " * 30 + "\nCHAPTER II\n" + "two " * 30
    entries = _book_entries(context, "eventqa", counter, 14)  # Chapter parts.
    assert_entry_invariants(entries, counter, 14)
    headings = {entry.metadata["section_heading"] for entry in entries}
    assert (
        "Opening" in headings and "CHAPTER I" in headings and "CHAPTER II" in headings
    )
    assert all(entry.metadata["entry_kind"] == "book_section" for entry in entries)


def test_mab_literal_longmemeval_both_serialization_shapes(counter):
    turns = [  # Shared chat turns used in both supported serializations.
        {"role": "user", "content": "alpha " * 30},
        {"role": "assistant", "content": "beta"},
    ]
    nested = repr([("2024-01-01", turns), ("2024-01-02", turns)])  # Pair form.
    flat = repr(["2024-01-01", turns, "2024-01-02", turns])  # Alternating form.
    for payload in (nested, flat):
        entries = _literal_longmemeval_sessions(payload, "longmemeval_s", counter, 15)
        assert_entry_invariants(entries, counter, 15)
        assert {entry.metadata["timestamp"] for entry in entries} == {
            "2024-01-01",
            "2024-01-02",
        }


def test_mab_literal_longmemeval_bad_literal_fails(counter):
    # Invalid Python-literal context should not be accepted as a session list.
    with pytest.raises(DatasetFormatError):
        _literal_longmemeval_sessions("not valid [", "longmemeval_s", counter, 20)


def test_conflict_resolution_keeps_every_fact_and_source_order():
    context = (
        "Here is a list of facts:\n0. Alice lives in Paris.\n"
        "1. Alice lives in Rome.\n10. Bob likes tea."
    )
    entries = _numbered_fact_entries(context, "conflict")  # Three fact records.
    assert [entry.ordinal for entry in entries] == [0, 1, 10]
    assert entries[1].text == "Alice lives in Rome."
    assert entries[1].metadata["fact_index"] == 1


def test_conflict_resolution_requires_numbered_facts():
    # Plain prose cannot supply the order needed for conflict resolution.
    with pytest.raises(DatasetFormatError):
        _numbered_fact_entries("Alice lives in Paris", "conflict")


def test_ttl_redial_splits_dialogues_and_nonredial_demonstrations(counter):
    redial, redial_plan = _mab_ttl_entries(  # Dialogue branch.
        "Dialogue 1:\nA\nDialogue 2:\nB", "recsys_redial_full", counter, 20
    )
    assert len(redial) == 2 and redial_plan.top_k == 32
    demos, demo_plan = _mab_ttl_entries(  # Demonstration branch.
        "Input: A\nOutput: 1\n\nInput: B\nOutput: 2", "icl", counter, 20
    )
    assert len(demos) == 2 and demo_plan.top_k == 80


def test_unit_splitter_never_exceeds_limit_even_single_huge_unit(counter):
    # The last-resort splitter must handle one unit larger than the limit.
    pieces = _split_units_safely(
        header="H", units=["x " * 100], counter=counter, max_tokens=12
    )
    assert len(pieces) > 1
    assert all(counter.count(piece) <= 12 for piece in pieces)


def test_paragraph_splitter_handles_unbroken_large_text(counter):
    # One large paragraph should split while repeating its header safely.
    pieces = _split_paragraph_text("x " * 100, counter, 11, header="HEADER")
    assert len(pieces) > 1
    assert all(counter.count(piece) <= 11 for piece in pieces)


@pytest.mark.parametrize(
    "dataset,source,context,expected_kind",
    [
        (
            "mab_long_range_understanding",
            "eventqa",
            "CHAPTER I\nfirst event\nCHAPTER II\nsecond event",
            "book_section",
        ),
        (
            "mab_test_time_learning",
            "recsys_redial_full",
            "Dialogue 1:\nA\nDialogue 2:\nB",
            "dialogue",
        ),
    ],
)
def test_canonical_mab_names_use_the_right_adapter(
    tmp_path,
    counter,
    dataset,
    source,
    context,
    expected_kind,
):
    row = {  # Small row accepted by both named MemoryAgentBench branches.
        "context": context,
        "questions": ["Question"],
        "answers": [["Answer"]],
        "metadata": {"source": source},
    }
    path = tmp_path / "mab.json"  # Adapter input file.
    path.write_text(json.dumps([row]), encoding="utf-8")

    task = next(  # First normalized task from this one-row file.
        iter_evaluation_tasks(
            data_file=path,
            dataset=dataset,
            embedding_model="unused",
            embedding_counter=counter,
            max_entry_tokens=20,
        )
    )

    assert task.dataset == dataset
    assert task.entries
    assert all(entry.metadata["entry_kind"] == expected_kind for entry in task.entries)
