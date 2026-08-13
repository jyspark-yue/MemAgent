#############################################################################
# File: dataset_adapters.py
#
# Description:
#   Converts every supported benchmark into the shared evaluation task
#   format. It preserves meaningful source units and splits them only when
#   an embedding limit requires it.
#
#   - Streams top-level JSON arrays or JSONL without loading full
#     benchmark files into memory.
#   - Builds LongMemEval and ecommerce tasks with stable questions,
#     sessions, dates, and retrieval defaults.
#   - Handles all supported MemoryAgentBench source families, including
#     RULER, EventQA, conflict facts, ReDial, and test-time learning.
#   - Parses Python-literal LongMemEval sessions and validates their
#     date/session pairs.
#   - Splits chats, chapters, documents, facts, dialogues,
#     demonstrations, and oversized paragraphs without overlap.
#   - Keeps entry IDs, source order, metadata, and token limits
#     consistent across adapters.
#############################################################################

from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from asdrp.eval_schemas import (
    EvaluationQuestion,
    EvaluationTask,
    MemoryEntry,
    RetrievalPlan,
)
from asdrp.runtime import TokenCounter


class DatasetFormatError(ValueError):
    # Raised when a dataset does not have the fields this code expects.
    pass


def iter_json_array(path: Path, chunk_size: int = 1 << 20) -> Iterator[dict[str, Any]]:
    # Stream a strict top-level JSON array without loading the whole file.

    decoder = json.JSONDecoder()  # Decodes one array item at a time.
    buffer = ""  # Holds the unread part of the file.
    position = 0  # Points to the next character in the buffer.
    started = False  # Becomes true after the opening bracket.
    closed = False  # Becomes true after the closing bracket.
    expect_value = True  # Tracks whether the next item or a comma is due.
    after_comma = False  # Distinguishes an empty array from a trailing comma.

    with path.open("r", encoding="utf-8") as handle:
        eof = False  # Marks the end of the input file.
        while True:
            if position:
                # Drop text that was parsed during the last pass.
                buffer = buffer[position:]
                position = 0

            if not eof:
                chunk = handle.read(chunk_size)  # Read a fixed-size piece.
                eof = not chunk  # An empty read means the file is complete.
                if chunk:
                    buffer += chunk

            while True:
                length = len(buffer)  # The current buffer may grow after each read.
                while position < length and buffer[position].isspace():
                    position += 1

                if closed:
                    if position < length:
                        raise DatasetFormatError(
                            "Trailing data after top-level JSON array"
                        )
                    break

                if not started:
                    if position >= length:
                        break
                    if buffer[position] != "[":
                        raise DatasetFormatError(
                            "Dataset must be a top-level JSON array"
                        )
                    started = True  # Opening array bracket was found.
                    expect_value = True  # First array item may follow.
                    position += 1
                    continue

                while position < length and buffer[position].isspace():
                    position += 1
                if position >= length:
                    break

                char = buffer[position]  # Next JSON separator or value marker.
                if expect_value:
                    if char == "]":
                        if after_comma:
                            raise DatasetFormatError(
                                "Trailing comma in top-level JSON array"
                            )
                        closed = True
                        position += 1
                        continue
                    try:
                        item, end = decoder.raw_decode(buffer, position)  # One item.
                    except json.JSONDecodeError:
                        break
                    if not isinstance(item, dict):
                        raise DatasetFormatError(
                            "Every dataset array item must be an object"
                        )
                    yield item
                    position = end  # Continue after the parsed item.
                    expect_value = False  # A comma or closing bracket must follow.
                    after_comma = False
                else:
                    if char == ",":
                        position += 1
                        expect_value = True  # A comma starts the next item.
                        after_comma = True
                    elif char == "]":
                        closed = True  # This was the final array bracket.
                        position += 1
                    else:
                        raise DatasetFormatError("Expected ',' or ']' in dataset array")

            if eof:
                remainder = buffer[position:].strip()  # Any unfinished JSON text.
                if remainder:
                    if closed:
                        raise DatasetFormatError(
                            "Trailing data after top-level JSON array"
                        )
                    raise DatasetFormatError(
                        f"Incomplete JSON near {remainder[:120]!r}"
                    )
                if not started:
                    raise DatasetFormatError("Dataset must be a top-level JSON array")
                if not closed:
                    raise DatasetFormatError(
                        "Incomplete JSON array: missing closing ']'"
                    )
                return


def iter_json_records(path: Path) -> Iterator[dict[str, Any]]:
    # Preserve the existing streaming JSON-array parser and add JSONL compatibility.

    with path.open("r", encoding="utf-8-sig") as handle:
        first_non_space = ""
        while character := handle.read(1):
            if not character.isspace():
                first_non_space = character
                break

    if first_non_space == "[":
        yield from iter_json_array(path)
        return

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise DatasetFormatError(
                    f"Invalid JSONL object on line {line_number}: {error.msg}"
                ) from error
            if not isinstance(item, dict):
                raise DatasetFormatError(
                    f"JSONL line {line_number} must contain an object"
                )
            yield item


def iter_evaluation_tasks(
    *,
    data_file: Path,
    dataset: str,
    embedding_model: str,
    max_entry_tokens: int = 7_000,
    source_filter: str | None = None,
    embedding_counter: Any | None = None,
) -> Iterator[EvaluationTask]:
    # Yield fully formatted tasks for the selected benchmark.

    # Reuse a supplied counter in tests, or make one for the embedding model.
    counter = embedding_counter or TokenCounter(embedding_model)
    for row_index, row in enumerate(iter_json_records(data_file)):
        # Send each row to the adapter for its selected benchmark.
        if dataset == "longmemeval":
            yield _longmemeval_task(row, row_index, counter, max_entry_tokens)
        elif dataset == "ecommerce":
            yield _ecommerce_task(row, row_index, counter, max_entry_tokens)
        elif dataset in {
            "mab_accurate_retrieval",
            "mab_conflict_resolution",
            "mab_long_range_understanding",
            "mab_test_time_learning",
        }:
            source = str((row.get("metadata") or {}).get("source") or "unknown")
            if source_filter and source_filter.casefold() not in source.casefold():
                continue
            yield _memory_agent_bench_task(
                row,
                row_index,
                dataset,
                counter,
                max_entry_tokens,
            )
        else:
            raise DatasetFormatError(f"Unsupported dataset: {dataset}")


def _longmemeval_task(
    row: dict[str, Any],
    row_index: int,
    counter: TokenCounter,
    max_tokens: int,
) -> EvaluationTask:
    # Store each LongMemEval conversation session as one memory entry.

    sessions = row.get("haystack_sessions")  # Full chat sessions in this row.
    if not isinstance(sessions, list):
        raise DatasetFormatError("LongMemEval row is missing haystack_sessions")
    dates = row.get("haystack_dates") or [""] * len(sessions)  # Session dates.
    session_ids = row.get("haystack_session_ids") or [  # Stable session labels.
        f"session_{i}" for i in range(len(sessions))
    ]
    if not isinstance(dates, list) or not isinstance(session_ids, list):
        raise DatasetFormatError(
            "LongMemEval session dates and IDs must be JSON arrays"
        )

    entries: list[MemoryEntry] = []  # Final memory entries for this task.
    ordinal = 0  # Keeps split sessions in source order.
    for session_index, session in enumerate(sessions):
        if not isinstance(session, list):
            continue
        date = str(dates[session_index]) if session_index < len(dates) else ""
        # Use the dataset ID when present and a stable fallback when absent.
        session_id = (
            str(session_ids[session_index])
            if session_index < len(session_ids)
            else f"session_{session_index}"
        )
        message_units = []  # Keeps each chat turn whole during splitting.
        for turn in session:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role") or "unknown").upper()  # Prompt speaker.
            # Remove the dataset's end marker before storing the message.
            content = (
                str(turn.get("content") or "").replace("<|endoftext|>", "").strip()
            )
            if content:
                message_units.append(f"{role}: {content}")
        header = f"Session ID: {session_id}\nSession Date: {date}".strip()
        text = f"{header}\n\n" + "\n\n".join(message_units)  # Unsplit form.
        # Split only when the complete session is too large to embed.
        pieces = _split_units_safely(
            header=header,
            units=message_units,
            counter=counter,
            max_tokens=max_tokens,
        )
        for part_index, piece in enumerate(pieces):
            # The part number keeps IDs unique when one session is split.
            question_key = row.get("question_id") or row_index
            entry_id = f"{question_key}_session_{session_index}_part_{part_index}"
            entries.append(
                MemoryEntry(
                    entry_id=entry_id,
                    text=piece if len(pieces) > 1 else text,
                    ordinal=ordinal,
                    metadata={
                        "entry_kind": "session",
                        "source_id": session_id,
                        "timestamp": date,
                        "session_index": session_index,
                        "session_part": part_index,
                    },
                )
            )
            ordinal += 1

    # Match the benchmark split/training rule exactly: abstention variants are
    # identified by the question ID even when the source row retains another
    # raw question_type value. This keeps runtime filtering aligned with training.
    question_id = str(row.get("question_id") or f"longmemeval_{row_index}")
    raw_question_type = str(row.get("question_type") or "unknown")
    question_type = (
        "abstention"
        if question_id.strip().casefold().endswith("_abs")
        else raw_question_type
    )

    # LongMemEval places one question beside its source sessions.
    question = EvaluationQuestion(
        question_id=question_id,
        text=str(row.get("question") or ""),
        answers=_normalize_answers(row.get("answer")),
        metadata={
            "question_type": question_type,
            "question_date": row.get("question_date"),
            "answer_session_ids": row.get("answer_session_ids"),
        },
    )
    return EvaluationTask(
        task_id=question.question_id,
        dataset="longmemeval",
        source="longmemeval",
        entries=entries,
        questions=[question],
        retrieval=RetrievalPlan(
            mode="top_k",
            top_k=12,
            candidate_multiplier=4,
            neighbor_window=1,
            max_context_tokens=96_000,
            prefer_latest=question_type in {"knowledge-update", "temporal-reasoning"},
        ),
        metadata={"row_index": row_index},
    )


def _ecommerce_task(
    row: dict[str, Any],
    row_index: int,
    counter: TokenCounter,
    max_tokens: int,
) -> EvaluationTask:
    # Split EcommerceMemEval context at explicit session boundaries.

    context = str(row.get("context") or "")  # All customer sessions in this row.
    metadata = row.get("metadata") or {}  # IDs, question types, and evidence.
    if not isinstance(metadata, dict):
        raise DatasetFormatError("Ecommerce metadata must be a JSON object")
    customer_id = str(metadata.get("customer_id") or f"customer_{row_index}")
    entries = _split_ecommerce_sessions(context, customer_id, counter, max_tokens)

    questions_raw = row.get("questions") or []  # Question text from the dataset.
    answers_raw = row.get("answers") or []  # Matching accepted answers.
    ids = metadata.get("qa_pair_ids") or []  # Optional stable question IDs.
    types = metadata.get("question_types") or []  # Optional question labels.
    evidence = metadata.get("evidence") or []  # Optional support for each answer.
    if not all(
        isinstance(value, list)
        for value in (questions_raw, answers_raw, ids, types, evidence)
    ):
        raise DatasetFormatError(
            "Ecommerce questions, answers, IDs, types, and evidence must be arrays"
        )
    questions = []  # Normalized questions for the evaluator.
    for index, question_text in enumerate(questions_raw):
        questions.append(
            EvaluationQuestion(
                question_id=(
                    str(ids[index]) if index < len(ids) else f"{customer_id}_q{index}"
                ),
                text=str(question_text),
                answers=_normalize_answers(
                    answers_raw[index] if index < len(answers_raw) else []
                ),
                metadata={
                    "question_type": types[index] if index < len(types) else None,
                    "evidence": evidence[index] if index < len(evidence) else None,
                },
            )
        )

    return EvaluationTask(
        task_id=customer_id,
        dataset="ecommerce",
        source=str(metadata.get("source") or "ecommerce"),
        entries=entries,
        questions=questions,
        retrieval=RetrievalPlan(
            mode="top_k",
            top_k=8,
            candidate_multiplier=4,
            neighbor_window=1,
            max_context_tokens=48_000,
            prefer_latest=True,
        ),
        metadata={"row_index": row_index, "customer_id": customer_id},
    )


def _memory_agent_bench_task(
    row: dict[str, Any],
    row_index: int,
    dataset: str,
    counter: TokenCounter,
    max_tokens: int,
) -> EvaluationTask:
    # Format each MemoryAgentBench competency according to its source family.

    metadata = row.get("metadata") or {}  # Source name and benchmark notes.
    if not isinstance(metadata, dict):
        raise DatasetFormatError("MemoryAgentBench metadata must be a JSON object")
    source = str(metadata.get("source") or "unknown")  # Dataset sub-family.
    context = str(row.get("context") or "")  # Raw memory text for this task.

    if dataset == "mab_accurate_retrieval":
        # Accurate retrieval has several source formats of its own.
        entries, plan = _mab_accurate_retrieval_entries(
            context, source, counter, max_tokens
        )
    elif dataset == "mab_conflict_resolution":
        entries = _numbered_fact_entries(context, source)  # One fact per entry.
        # Search a wide fact pool and favor the newest value.
        plan = RetrievalPlan(
            mode="top_k",
            top_k=40,
            candidate_multiplier=8,
            neighbor_window=0,
            graph_hops=3,
            max_context_tokens=64_000,
            prefer_latest=True,
        )
    elif dataset == "mab_long_range_understanding":
        entries = _book_entries(context, source, counter, max_tokens)
        if source == "detective_qa":
            # Detective questions need targeted chapters and their neighbors.
            plan = RetrievalPlan(
                mode="top_k",
                top_k=24,
                candidate_multiplier=5,
                neighbor_window=2,
                graph_hops=3,
                max_context_tokens=96_000,
            )
        else:
            # Other long-range sources need coverage of the whole book.
            plan = RetrievalPlan(
                mode="global",
                top_k=32,
                candidate_multiplier=4,
                neighbor_window=0,
                graph_hops=3,
                max_context_tokens=120_000,
            )
    else:
        # The only remaining accepted dataset is test-time learning.
        entries, plan = _mab_ttl_entries(context, source, counter, max_tokens)

    questions = _mab_questions(row, source, row_index)  # Normalized QA pairs.
    return EvaluationTask(
        task_id=f"{dataset}_{row_index}_{_safe_id(source)}",
        dataset=dataset,
        source=source,
        entries=entries,
        questions=questions,
        retrieval=plan,
        metadata={
            "row_index": row_index,
            "source": source,
            "keypoints": metadata.get("keypoints"),
            "previous_events": metadata.get("previous_events"),
        },
    )


def _mab_accurate_retrieval_entries(
    context: str,
    source: str,
    counter: TokenCounter,
    max_tokens: int,
) -> tuple[list[MemoryEntry], RetrievalPlan]:
    # Handle documents, event-order books, and LongMemEval-style sessions.

    if source.startswith("ruler_qa"):
        # RULER labels each source document on its own line.
        sections = _split_regex_sections(context, r"(?m)^Document\s+\d+:\s*")
        entries = _sections_to_entries(
            sections, source, "document", counter, max_tokens
        )
        plan = RetrievalPlan(  # Moderate document search with extra candidates.
            mode="top_k",
            top_k=12,
            candidate_multiplier=5,
            max_context_tokens=64_000,
        )
        return entries, plan

    if source.startswith("eventqa"):
        entries = _book_entries(context, source, counter, max_tokens)
        plan = RetrievalPlan(  # Add neighboring book sections around matches.
            mode="top_k",
            top_k=16,
            candidate_multiplier=5,
            neighbor_window=2,
            max_context_tokens=96_000,
        )
        return entries, plan

    if source.startswith("longmemeval"):
        entries = _literal_longmemeval_sessions(context, source, counter, max_tokens)
        plan = RetrievalPlan(  # Search sessions and favor newer updates.
            mode="top_k",
            top_k=12,
            candidate_multiplier=4,
            neighbor_window=1,
            max_context_tokens=96_000,
            prefer_latest=True,
        )
        return entries, plan

    # Unknown source families still get one safely split context block.
    entries = _sections_to_entries([context], source, "context", counter, max_tokens)
    return entries, RetrievalPlan(top_k=12, max_context_tokens=64_000)


def _mab_ttl_entries(
    context: str,
    source: str,
    counter: TokenCounter,
    max_tokens: int,
) -> tuple[list[MemoryEntry], RetrievalPlan]:
    # Split movie dialogues or labeled in-context-learning demonstrations.

    if source == "recsys_redial_full":
        # ReDial labels each movie conversation as a dialogue.
        sections = _split_regex_sections(context, r"(?m)^Dialogue\s+\d+:\s*")
        entries = _sections_to_entries(
            sections, source, "dialogue", counter, max_tokens
        )
        plan = RetrievalPlan(  # ReDial needs broad movie-association recall.
            mode="top_k",
            top_k=32,
            candidate_multiplier=5,
            max_context_tokens=80_000,
        )
        return entries, plan

    # Blank lines separate the examples used for test-time learning.
    demonstrations = [
        part.strip() for part in re.split(r"\n\s*\n", context) if part.strip()
    ]
    entries = _sections_to_entries(
        demonstrations,
        source,
        "demonstration",
        counter,
        max_tokens,
    )
    plan = RetrievalPlan(  # Many short demonstrations may be useful at once.
        mode="top_k",
        top_k=80,
        candidate_multiplier=3,
        max_context_tokens=96_000,
    )
    return entries, plan


def _mab_questions(
    row: dict[str, Any],
    source: str,
    row_index: int,
) -> list[EvaluationQuestion]:
    # Create stable question IDs and use decoded movie titles only for ReDial.

    questions_raw = row.get("questions") or []  # Question text in source order.
    answers_raw = row.get("answers") or []  # Accepted answers in matching order.
    metadata = row.get("metadata") or {}  # Optional IDs, dates, and type labels.
    ids = metadata.get("qa_pair_ids") or []  # Stable IDs when the source has them.
    if source == "recsys_redial_full" and row.get("decoded_answer_titles"):
        # Human-readable movie titles replace ReDial's encoded IDs.
        answers_raw = row["decoded_answer_titles"]

    question_types = metadata.get("question_types") or []  # Benchmark labels.
    question_dates = metadata.get("question_dates") or []  # Dates for time questions.
    previous_events = metadata.get("previous_events") or []  # Prior event hints.
    if not all(
        isinstance(value, list)
        for value in (
            questions_raw,
            answers_raw,
            ids,
            question_types,
            question_dates,
            previous_events,
        )
    ):
        raise DatasetFormatError(
            "MemoryAgentBench questions, answers, and question metadata must be arrays"
        )
    result = []  # Final normalized question list.
    for index, question_text in enumerate(questions_raw):
        result.append(
            EvaluationQuestion(
                question_id=(
                    str(ids[index]) if index < len(ids) else f"mab_{row_index}_q{index}"
                ),
                text=str(question_text),
                answers=_normalize_answers(
                    answers_raw[index] if index < len(answers_raw) else []
                ),
                metadata={
                    "question_type": (
                        question_types[index] if index < len(question_types) else None
                    ),
                    "question_date": (
                        question_dates[index] if index < len(question_dates) else None
                    ),
                    "previous_event": (
                        previous_events[index] if index < len(previous_events) else None
                    ),
                },
            )
        )
    return result


def _split_ecommerce_sessions(
    context: str,
    customer_id: str,
    counter: TokenCounter,
    max_tokens: int,
) -> list[MemoryEntry]:
    # Preserve each explicit ecommerce conversation session.

    # Find the session labels before cutting the source into blocks.
    matches = list(re.finditer(r"(?m)^\[Session\s+([^\]]+)\]\s*$", context))
    if not matches:
        return _sections_to_entries(
            [context], customer_id, "session", counter, max_tokens
        )
    preamble = context[: matches[0].start()].strip()  # Shared customer details.
    entries = []  # All session parts in source order.
    ordinal = 0  # Order across sessions and split parts.
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(context)
        session_id = match.group(1).strip()  # Label inside the brackets.
        body = context[match.end() : end].strip()  # Text until the next session.
        header = f"Customer: {customer_id}\nSession: {session_id}"  # Local label.
        if preamble:
            header = f"{preamble}\n{header}"
        pieces = _split_paragraph_text(body, counter, max_tokens, header=header)
        for part_index, piece in enumerate(pieces):
            entries.append(
                MemoryEntry(
                    entry_id=f"{customer_id}_{session_id}_{part_index}",
                    text=piece,
                    ordinal=ordinal,
                    metadata={
                        "entry_kind": "session",
                        "source_id": session_id,
                        "session_index": index,
                        "session_part": part_index,
                    },
                )
            )
            ordinal += 1
    return entries


def _literal_longmemeval_sessions(
    context: str,
    source: str,
    counter: TokenCounter,
    max_tokens: int,
) -> list[MemoryEntry]:
    # Parse MemoryAgentBench's Python-literal LongMemEval session serialization.

    try:
        payload = ast.literal_eval(context)  # Safe parser for the saved Python list.
    except (SyntaxError, ValueError) as error:
        raise DatasetFormatError(
            "Unable to parse MAB longmemeval_s* context"
        ) from error

    pairs: list[tuple[str, list[dict[str, Any]]]] = []  # Date and chat pairs.
    if isinstance(payload, list) and payload and isinstance(payload[0], (list, tuple)):
        for item in payload:
            if (
                isinstance(item, (list, tuple))
                and len(item) == 2
                and isinstance(item[1], list)
            ):
                pairs.append((str(item[0]), item[1]))
    elif isinstance(payload, list):
        for index in range(0, len(payload) - 1, 2):
            if isinstance(payload[index + 1], list):
                pairs.append((str(payload[index]), payload[index + 1]))

    if not pairs:
        raise DatasetFormatError(
            "MAB LongMemEval context contained no valid date/session pairs"
        )

    entries = []  # Parsed chat sessions ready for memory.
    ordinal = 0  # Order across every session part.
    for session_index, (date, turns) in enumerate(pairs):
        units = []  # Complete speaker turns from this session.
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role") or "unknown").upper()  # Speaker label.
            content = (  # Message text without the dataset end marker.
                str(turn.get("content") or "").replace("<|endoftext|>", "").strip()
            )
            if content:
                units.append(f"{role}: {content}")
        header = f"Chat Time: {date}"  # Repeated when a session is split.
        for part_index, piece in enumerate(
            _split_units_safely(
                header=header, units=units, counter=counter, max_tokens=max_tokens
            )
        ):
            entries.append(
                MemoryEntry(
                    entry_id=f"{_safe_id(source)}_session_{session_index}_{part_index}",
                    text=piece,
                    ordinal=ordinal,
                    metadata={
                        "entry_kind": "session",
                        "source_id": f"session_{session_index}",
                        "timestamp": date,
                    },
                )
            )
            ordinal += 1
    return entries


def _numbered_fact_entries(context: str, source: str) -> list[MemoryEntry]:
    # Store every conflict-resolution fact independently in source order.

    entries = []  # One memory entry for each numbered statement.
    for match in re.finditer(r"(?m)^\s*(\d+)\.\s*(.+?)\s*$", context):
        fact_index = int(match.group(1))  # Source number also gives the order.
        statement = match.group(2).strip()  # Fact text without its number.
        entries.append(
            MemoryEntry(
                entry_id=f"{_safe_id(source)}_fact_{fact_index}",
                text=statement,
                ordinal=fact_index,
                metadata={
                    "entry_kind": "fact",
                    "source_id": f"fact_{fact_index}",
                    "fact_index": fact_index,
                },
            )
        )
    if not entries:
        raise DatasetFormatError(
            "Conflict Resolution context contained no numbered facts"
        )
    return entries


def _book_entries(
    context: str,
    source: str,
    counter: TokenCounter,
    max_tokens: int,
) -> list[MemoryEntry]:
    # Create chapter-aware, non-overlapping book segments in chronological order.

    # Match common book headings while keeping the heading text in the entry.
    heading_pattern = re.compile(
        r"(?im)^(?:\s*<CHAPTER>.*|\s*(?:CHAPTER|BOOK|PART|VOLUME)\b.*|"
        r"\s*(?:Chapter|Book|Part|Volume)\s+[\wIVXLCDM.-]+.*|"
        r"\s*\[\d+\]\s*(?:Chapter|Part|Book|Prologue|Epilogue).*)$"
    )
    matches = list(heading_pattern.finditer(context))  # Every heading in order.
    sections: list[tuple[str, str]] = []  # Heading and body pairs.
    if matches:
        if context[: matches[0].start()].strip():
            # Keep text that appears before the first named chapter.
            sections.append(("Opening", context[: matches[0].start()].strip()))
        for index, match in enumerate(matches):
            end = (
                matches[index + 1].start() if index + 1 < len(matches) else len(context)
            )
            heading = match.group(0).strip()  # Chapter label.
            body = context[match.end() : end].strip()  # Chapter text.
            sections.append((heading, body))
    else:
        sections = [("Book segment", context)]  # Fallback when headings are absent.

    entries = []  # Finished book segments.
    ordinal = 0  # Preserves the reading order across chapters.
    for section_index, (heading, body) in enumerate(sections):
        pieces = _split_paragraph_text(body, counter, max_tokens, header=heading)
        for part_index, piece in enumerate(pieces):
            entries.append(
                MemoryEntry(
                    entry_id=f"{_safe_id(source)}_section_{section_index}_part_{part_index}",
                    text=piece,
                    ordinal=ordinal,
                    metadata={
                        "entry_kind": "book_section",
                        "source_id": f"section_{section_index}",
                        "section_heading": heading,
                        "section_index": section_index,
                        "section_part": part_index,
                    },
                )
            )
            ordinal += 1
    return entries


def _split_regex_sections(text: str, pattern: str) -> list[str]:
    # Split on labeled boundaries while retaining each boundary label.

    matches = list(re.finditer(pattern, text))  # Labeled section starts.
    if not matches:
        return [text.strip()] if text.strip() else []
    sections = []  # Full text from each label to the next.
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section = text[match.start() : end].strip()  # Keeps its leading label.
        if section:
            sections.append(section)
    return sections


def _sections_to_entries(
    sections: Sequence[str],
    source: str,
    entry_kind: str,
    counter: TokenCounter,
    max_tokens: int,
) -> list[MemoryEntry]:
    # Turn source sections into non-overlapping entries that fit the model.

    entries = []  # Converted and safely sized memory entries.
    ordinal = 0  # Order across all sections and parts.
    for section_index, section in enumerate(sections):
        pieces = _split_paragraph_text(section, counter, max_tokens)
        for part_index, piece in enumerate(pieces):
            entries.append(
                MemoryEntry(
                    entry_id=f"{_safe_id(source)}_{entry_kind}_{section_index}_{part_index}",
                    text=piece,
                    ordinal=ordinal,
                    metadata={
                        "entry_kind": entry_kind,
                        "source_id": f"{entry_kind}_{section_index}",
                        "section_index": section_index,
                        "section_part": part_index,
                    },
                )
            )
            ordinal += 1
    return entries


def _split_units_safely(
    *,
    header: str,
    units: Sequence[str],
    counter: TokenCounter,
    max_tokens: int,
) -> list[str]:
    # Pack complete units when possible and repeat the header on every chunk.

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    # Blank lines keep chats and other units readable in the stored entry.
    body = "\n\n".join(str(unit).strip() for unit in units if str(unit).strip())
    if not body:
        return (
            [header] if header.strip() and counter.count(header) <= max_tokens else []
        )
    return _split_paragraph_text(body, counter, max_tokens, header=header)


def _split_paragraph_text(
    text: str,
    counter: TokenCounter,
    max_tokens: int,
    *,
    header: str = "",
) -> list[str]:
    # Pack paragraphs without overlap and repeat a fitting header on each part.

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    # Paragraphs are the smallest parts we prefer to keep whole.
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs and text.strip():
        paragraphs = [text.strip()]  # Treat unbroken text as one paragraph.
    if not paragraphs:
        return []

    prefix = header.strip()  # Repeated on each piece to keep its source clear.
    separator = "\n\n" if prefix else ""  # Space between header and body.

    def render(body: str) -> str:
        # Add the optional header to one body piece.
        return f"{prefix}{separator}{body}" if prefix else body

    def split_oversized(body: str) -> list[str]:
        # Split one paragraph only when it cannot fit as a whole.
        if counter.count(render(body)) <= max_tokens:
            return [render(body)]

        # Determine the largest body budget that still fits with the repeated
        # prefix. Exact rendering checks avoid tokenizer-dependent constants.
        budget = max_tokens  # Starts with the full token limit.
        if prefix:
            # Reserve space for the header that appears on every piece.
            budget = max_tokens - counter.count(prefix)
        budget = max(1, budget)  # Always allow progress through the body.

        while budget > 0:
            chunks = counter.split(body, budget)  # Candidate body pieces.
            rendered = [render(chunk) for chunk in chunks if chunk.strip()]
            if rendered and all(
                counter.count(piece) <= max_tokens for piece in rendered
            ):
                return rendered
            budget -= 1

        # A header that consumes the entire limit cannot be repeated while also
        # retaining content. Fall back to a lossless hard split of the full text.
        return counter.split(render(body), max_tokens)

    pieces: list[str] = []  # Finished chunks within the token limit.
    current: list[str] = []  # Paragraphs being packed into the next chunk.

    for paragraph in paragraphs:
        candidate_body = "\n\n".join([*current, paragraph])  # Next trial chunk.
        if counter.count(render(candidate_body)) <= max_tokens:
            current.append(paragraph)
            continue

        if current:
            pieces.append(render("\n\n".join(current)))
            current = []  # The saved piece is no longer being filled.

        if counter.count(render(paragraph)) <= max_tokens:
            current = [paragraph]  # Start the next piece with this paragraph.
        else:
            pieces.extend(split_oversized(paragraph))

    if current:
        pieces.append(render("\n\n".join(current)))

    return pieces


def _normalize_answers(value: Any) -> list[str]:
    # Normalize scalar and nested benchmark answers into a flat string list.

    if value is None:
        return []
    if isinstance(value, list):
        result = []  # Flattened accepted answers.
        for item in value:
            if isinstance(item, list):
                result.extend(str(subitem) for subitem in item)
            else:
                result.append(str(item))
        return result
    return [str(value)]


def _safe_id(value: str) -> str:
    # Create stable IDs that are safe for file paths and Qdrant.

    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_") or "unknown"
