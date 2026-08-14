#############################################################################
# File: mab_percent_split_generator.py
#
# Description:
#   Create deterministic, mutually exclusive train/test files from the
#   MemoryAgentBench datasets.
#
#   Default split: 80% train / 20% test.
#   Default split unit: QA pair.
#
#   QA mode is designed for the actual MemoryAgentBench schema, where one context
#   row often contains 60-200 aligned questions/answers. It slices every aligned
#   list together, including metadata.qa_pair_ids, question_types, evidence-like
#   fields, decoded_answers, decoded_answer_titles, and qa_pairs_decoded. Because
#   MAB reuses some qa_pair_ids, split identity is based on exact question+answer
#   content; identical QA content is kept wholly in train or wholly in test.
#
#   Because several MAB task types occur in only one context row (especially TTL),
#   QA mode may repeat the shared context in train and test so that every task type
#   can be represented. Use --split-unit row for strict context-disjoint splitting;
#   that mode cannot preserve every singleton task type in both splits.

#   Outputs (exactly two dataset files):
#       <input_stem>_train.json or .jsonl
#       <input_stem>_test.json or .jsonl
#       Use --write-manifest to additionally write a validation manifest.
#
#   Command-line Arguments (Change as needed):
#       python mab_percent_split_generator.py \
#           separated_splits/Accurate_Retrieval.json \
#           separated_splits/Conflict_Resolution.json \
#           separated_splits/Long_Range_Understanding.json \
#           separated_splits/Test_Time_Learning_decoded.json \
#           --output-dir splits/mab_20pct \
#           --train-fraction 0.80 \
#           --split-unit qa \
#           --write-manifest
#############################################################################

"""
Create separate MemoryAgentBench train/test files for every input file.

Default split: 80% train / 20% test.
Default split unit: QA pair.

QA mode is designed for the actual MemoryAgentBench schema, where one context
row often contains 60-200 aligned questions/answers. It slices every aligned
list together, including metadata.qa_pair_ids, question_types, evidence-like
fields, decoded_answers, decoded_answer_titles, and qa_pairs_decoded. Because
MAB reuses some qa_pair_ids, split identity is based on exact question+answer
content; identical QA content is kept wholly in train or wholly in test.

Because several MAB task types occur in only one context row (especially TTL),
QA mode may repeat the shared context in train and test so that every task type
can be represented. Use --split-unit row for strict context-disjoint splitting;
that mode cannot preserve every singleton task type in both splits.

Outputs (exactly two dataset files):
    <input_stem>_train.json or .jsonl
    <input_stem>_test.json or .jsonl
    Use --write-manifest to additionally write a validation manifest.

Command-line Arguments (Change as needed):
    python mab_percent_split_generator.py \
        separated_splits/Accurate_Retrieval.json \
        separated_splits/Conflict_Resolution.json \
        separated_splits/Long_Range_Understanding.json \
        separated_splits/Test_Time_Learning_decoded.json \
        --output-dir splits/mab_20pct \
        --train-fraction 0.80 \
        --split-unit qa \
        --write-manifest
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input does not exist: {path}")

    if path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"{path}:{line_number} must contain a JSON object.")
                records.append(value)
        return records

    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)

    if isinstance(value, list):
        if not all(isinstance(item, dict) for item in value):
            raise TypeError(f"Every element in {path} must be a JSON object.")
        return value

    if isinstance(value, dict):
        for key in ("data", "records", "examples", "items"):
            wrapped = value.get(key)
            if isinstance(wrapped, list) and all(
                isinstance(item, dict) for item in wrapped
            ):
                return wrapped
        return [value]

    raise TypeError(f"Unsupported top-level JSON value: {type(value).__name__}")


def write_records(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".jsonl":
        with path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        with path.open("w", encoding="utf-8") as f:
            json.dump(list(records), f, ensure_ascii=False, indent=2)
            f.write("\n")


def stable_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_text(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_label(value: Any) -> str:
    if value is None:
        return "unknown"
    return str(value).strip().lower().replace(" ", "-").replace("_", "-")


def discover_inputs(inputs: Sequence[Path]) -> list[Path]:
    discovered: list[Path] = []
    for path in inputs:
        if path.is_dir():
            discovered.extend(
                p
                for p in sorted(path.rglob("*"))
                if p.is_file() and p.suffix.lower() in {".json", ".jsonl"}
            )
        else:
            discovered.append(path)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in discovered:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    if not unique:
        raise ValueError("No JSON or JSONL input files were found.")
    return unique


def validate_row(row: dict[str, Any], row_index: int) -> int:
    questions = row.get("questions")
    answers = row.get("answers")
    if not isinstance(questions, list) or not isinstance(answers, list):
        raise TypeError(
            f"Row {row_index} must contain list fields 'questions' and 'answers'."
        )
    if len(questions) != len(answers):
        raise ValueError(
            f"Row {row_index} has {len(questions)} questions but "
            f"{len(answers)} answers."
        )
    qa_ids = row.get("metadata", {}).get("qa_pair_ids")
    if qa_ids is not None and (
        not isinstance(qa_ids, list) or len(qa_ids) != len(questions)
    ):
        raise ValueError(
            f"Row {row_index}: metadata.qa_pair_ids is not aligned with questions."
        )
    return len(questions)


def qa_id(row: dict[str, Any], row_index: int, question_index: int) -> str:
    qa_ids = row.get("metadata", {}).get("qa_pair_ids")
    if isinstance(qa_ids, list):
        return str(qa_ids[question_index])
    return f"row:{row_index}:question:{question_index}"


def qa_stratum(row: dict[str, Any], question_index: int, input_stem: str) -> str:
    metadata = row.get("metadata", {})
    source = normalize_label(metadata.get("source") or input_stem)
    question_types = metadata.get("question_types")
    if isinstance(question_types, list) and question_index < len(question_types):
        question_type = normalize_label(question_types[question_index])
        return f"{source}::{question_type}"
    return source


def allocate_test_quotas(
    stratum_sizes: dict[str, int],
    target_test_items: int,
) -> dict[str, int]:
    """Allocate test items proportionally, retaining at least one train item."""
    sizes = {k: v for k, v in stratum_sizes.items() if v > 0}
    quotas = {k: 0 for k in sizes}
    if not sizes or target_test_items <= 0:
        return quotas

    max_test = {k: (v - 1 if v > 1 else 0) for k, v in sizes.items()}
    target = min(target_test_items, sum(max_test.values()))
    if target <= 0:
        return quotas

    eligible = [k for k in sizes if max_test[k] > 0]
    if target >= len(eligible):
        for key in eligible:
            quotas[key] = 1
        remaining = target - len(eligible)
    else:
        remaining = target

    capacities = {k: max_test[k] - quotas[k] for k in sizes}
    capacity_total = sum(capacities.values())
    if remaining <= 0 or capacity_total <= 0:
        return quotas

    raw = {k: remaining * capacities[k] / capacity_total for k in sizes}
    floors = {k: min(capacities[k], math.floor(raw[k])) for k in sizes}
    for key, amount in floors.items():
        quotas[key] += amount

    left = target - sum(quotas.values())
    order = sorted(
        sizes,
        key=lambda k: (raw[k] - floors[k], capacities[k], k),
        reverse=True,
    )
    while left > 0:
        progressed = False
        for key in order:
            if quotas[key] < max_test[key]:
                quotas[key] += 1
                left -= 1
                progressed = True
                if left == 0:
                    break
        if not progressed:
            break
    return quotas


def slice_aligned_row(
    row: dict[str, Any],
    indices: Sequence[int],
) -> dict[str, Any]:
    """Slice all top-level and metadata lists aligned to questions."""
    n = len(row["questions"])
    selected = list(indices)
    result = copy.deepcopy(row)

    for key, value in list(result.items()):
        if key == "metadata":
            continue
        if isinstance(value, list) and len(value) == n:
            result[key] = [value[i] for i in selected]

    metadata = result.get("metadata")
    if isinstance(metadata, dict):
        for key, value in list(metadata.items()):
            if isinstance(value, list) and len(value) == n:
                metadata[key] = [value[i] for i in selected]

    if len(result.get("questions", [])) != len(selected):
        raise AssertionError("Question slicing failed.")
    if len(result.get("answers", [])) != len(selected):
        raise AssertionError("Answer slicing failed.")
    return result


def multilabel_objective(
    selected_counts: Counter[str],
    targets: dict[str, float],
) -> float:
    score = 0.0
    for label, target in targets.items():
        denom = max(1.0, target)
        score += ((selected_counts[label] - target) / denom) ** 2
    return score


def select_multilabel_test_groups(
    group_labels: dict[str, Counter[str]],
    *,
    target_group_count: int,
    test_fraction: float,
    seed: int,
) -> set[str]:
    total_counts: Counter[str] = Counter()
    for counts in group_labels.values():
        total_counts.update(counts)
    targets = {label: count * test_fraction for label, count in total_counts.items()}

    rng = random.Random(seed)
    tie_break = {group_id: rng.random() for group_id in sorted(group_labels)}
    selected: set[str] = set()
    selected_counts: Counter[str] = Counter()

    # Seed rare strata first.
    for label in sorted(total_counts, key=lambda x: (total_counts[x], x)):
        if len(selected) >= target_group_count:
            break
        if selected_counts[label] > 0:
            continue
        candidates = [
            group_id
            for group_id, counts in group_labels.items()
            if group_id not in selected and counts[label] > 0
        ]
        if not candidates:
            continue
        best = min(
            candidates,
            key=lambda group_id: (
                multilabel_objective(selected_counts + group_labels[group_id], targets),
                -group_labels[group_id][label],
                tie_break[group_id],
                group_id,
            ),
        )
        selected.add(best)
        selected_counts.update(group_labels[best])

    while len(selected) < target_group_count:
        candidates = [g for g in group_labels if g not in selected]
        best = min(
            candidates,
            key=lambda group_id: (
                multilabel_objective(selected_counts + group_labels[group_id], targets),
                tie_break[group_id],
                group_id,
            ),
        )
        selected.add(best)
        selected_counts.update(group_labels[best])

    improved = True
    passes = 0
    while improved and passes < 8:
        improved = False
        passes += 1
        current_score = multilabel_objective(selected_counts, targets)
        unselected = set(group_labels) - selected
        for selected_id in sorted(selected):
            for unselected_id in sorted(unselected):
                candidate_counts = selected_counts.copy()
                candidate_counts.subtract(group_labels[selected_id])
                candidate_counts.update(group_labels[unselected_id])
                candidate_score = multilabel_objective(candidate_counts, targets)
                if candidate_score + 1e-12 < current_score:
                    selected.remove(selected_id)
                    selected.add(unselected_id)
                    selected_counts = candidate_counts
                    improved = True
                    break
            if improved:
                break
    return selected


def qa_content_identity(row: dict[str, Any], question_index: int) -> str:
    return stable_hash(
        {
            "question": row["questions"][question_index],
            "answer": row["answers"][question_index],
        }
    )


def split_by_qa(
    rows: Sequence[dict[str, Any]],
    *,
    train_fraction: float,
    seed: int,
    input_stem: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Split exact QA content groups while preserving aligned row fields."""
    occurrences_by_identity: dict[str, list[tuple[int, int]]] = defaultdict(list)
    labels_by_identity: dict[str, Counter[str]] = defaultdict(Counter)
    total_occurrences = 0

    for row_index, row in enumerate(rows):
        n = validate_row(row, row_index)
        total_occurrences += n
        for question_index in range(n):
            identity = qa_content_identity(row, question_index)
            occurrences_by_identity[identity].append((row_index, question_index))
            labels_by_identity[identity][
                qa_stratum(row, question_index, input_stem)
            ] += 1

    total_unique = len(occurrences_by_identity)
    if total_unique < 2:
        raise ValueError("At least two unique QA samples are required for a split.")

    test_fraction = 1.0 - train_fraction
    requested_test = round(total_unique * test_fraction)
    requested_test = max(1, min(total_unique - 1, requested_test))
    test_identities = select_multilabel_test_groups(
        labels_by_identity,
        target_group_count=requested_test,
        test_fraction=test_fraction,
        seed=seed,
    )
    train_identities = set(occurrences_by_identity) - test_identities

    if train_identities & test_identities:
        raise AssertionError("QA-content overlap detected between train and test.")
    if train_identities | test_identities != set(occurrences_by_identity):
        raise AssertionError("The split does not reconstruct all QA content groups.")

    test_locations = {
        location
        for identity in test_identities
        for location in occurrences_by_identity[identity]
    }

    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    shared_context_rows = 0
    train_occurrences = 0
    test_occurrences = 0

    for row_index, row in enumerate(rows):
        n = len(row["questions"])
        train_indices = [i for i in range(n) if (row_index, i) not in test_locations]
        test_indices = [i for i in range(n) if (row_index, i) in test_locations]
        train_occurrences += len(train_indices)
        test_occurrences += len(test_indices)
        if train_indices:
            train_rows.append(slice_aligned_row(row, train_indices))
        if test_indices:
            test_rows.append(slice_aligned_row(row, test_indices))
        if train_indices and test_indices:
            shared_context_rows += 1

    all_strata: Counter[str] = Counter()
    train_strata: Counter[str] = Counter()
    test_strata: Counter[str] = Counter()
    for identity, label_counts in labels_by_identity.items():
        all_strata.update(label_counts)
        (test_strata if identity in test_identities else train_strata).update(
            label_counts
        )

    # Recompute exact QA-content hashes from the outputs as an independent check.
    train_content_hashes = {
        qa_content_identity(row, question_index)
        for row in train_rows
        for question_index in range(len(row["questions"]))
    }
    test_content_hashes = {
        qa_content_identity(row, question_index)
        for row in test_rows
        for question_index in range(len(row["questions"]))
    }
    content_overlap = train_content_hashes & test_content_hashes
    if content_overlap:
        raise AssertionError(
            f"Detected {len(content_overlap)} identical QA samples in both splits."
        )

    report = {
        "split_unit": "qa_content",
        "qa_identity_rule": "sha256(question + answer)",
        "requested_train_fraction": train_fraction,
        "actual_train_unique_qa_fraction": len(train_identities) / total_unique,
        "actual_train_qa_occurrence_fraction": (
            train_occurrences / total_occurrences if total_occurrences else 0.0
        ),
        "total_input_rows": len(rows),
        "train_output_rows": len(train_rows),
        "test_output_rows": len(test_rows),
        "total_qa_occurrences": total_occurrences,
        "train_qa_occurrences": train_occurrences,
        "test_qa_occurrences": test_occurrences,
        "total_unique_qa_samples": total_unique,
        "train_unique_qa_samples": len(train_identities),
        "test_unique_qa_samples": len(test_identities),
        "qa_content_overlap_count": 0,
        "duplicate_qa_content_group_count": sum(
            1 for locations in occurrences_by_identity.values() if len(locations) > 1
        ),
        "shared_context_rows": shared_context_rows,
        "context_disjoint": shared_context_rows == 0,
        "selected_test_qa_content_sha256": sorted(test_identities),
        "all_strata": dict(sorted(all_strata.items())),
        "train_strata": dict(sorted(train_strata.items())),
        "test_strata": dict(sorted(test_strata.items())),
    }
    return train_rows, test_rows, report


def split_by_context_row(
    rows: Sequence[dict[str, Any]],
    *,
    train_fraction: float,
    seed: int,
    input_stem: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    for row_index, row in enumerate(rows):
        validate_row(row, row_index)

    groups: dict[str, list[int]] = defaultdict(list)
    group_stratum: dict[str, str] = {}
    for row_index, row in enumerate(rows):
        context_id = stable_hash(row.get("context", f"row:{row_index}"))
        groups[context_id].append(row_index)
        source = normalize_label(row.get("metadata", {}).get("source") or input_stem)
        group_stratum.setdefault(context_id, source)

    requested_test = round(len(groups) * (1.0 - train_fraction))
    requested_test = max(1, min(len(groups) - 1, requested_test))
    groups_by_stratum: dict[str, list[str]] = defaultdict(list)
    for group_id, stratum in group_stratum.items():
        groups_by_stratum[stratum].append(group_id)

    quotas = allocate_test_quotas(
        {k: len(v) for k, v in groups_by_stratum.items()},
        requested_test,
    )
    rng = random.Random(seed)
    test_groups: set[str] = set()
    for stratum in sorted(groups_by_stratum):
        candidates = sorted(groups_by_stratum[stratum])
        rng.shuffle(candidates)
        test_groups.update(candidates[: quotas.get(stratum, 0)])

    # Singleton strata can make perfect stratification impossible. Fill anyway.
    if len(test_groups) < requested_test:
        remaining = sorted(set(groups) - test_groups)
        rng.shuffle(remaining)
        test_groups.update(remaining[: requested_test - len(test_groups)])

    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for group_id, row_indices in groups.items():
        destination = test_rows if group_id in test_groups else train_rows
        destination.extend(rows[index] for index in row_indices)

    train_contexts = {stable_hash(row.get("context")) for row in train_rows}
    test_contexts = {stable_hash(row.get("context")) for row in test_rows}
    if train_contexts & test_contexts:
        raise AssertionError("Context overlap detected in row split mode.")

    # QA IDs are dataset-provided and remain valid even after row reindexing.
    provided_train_ids = {
        str(item)
        for row in train_rows
        for item in (row.get("metadata", {}).get("qa_pair_ids") or [])
    }
    provided_test_ids = {
        str(item)
        for row in test_rows
        for item in (row.get("metadata", {}).get("qa_pair_ids") or [])
    }
    if provided_train_ids & provided_test_ids:
        raise AssertionError("QA ID overlap detected in row split mode.")

    def source_counts(selected_rows: Sequence[dict[str, Any]]) -> Counter[str]:
        return Counter(
            normalize_label(row.get("metadata", {}).get("source") or input_stem)
            for row in selected_rows
        )

    report = {
        "split_unit": "row",
        "requested_train_fraction": train_fraction,
        "actual_train_row_fraction": len(train_rows) / len(rows),
        "total_input_rows": len(rows),
        "train_output_rows": len(train_rows),
        "test_output_rows": len(test_rows),
        "total_qa_pairs": sum(len(r["questions"]) for r in rows),
        "train_qa_pairs": sum(len(r["questions"]) for r in train_rows),
        "test_qa_pairs": sum(len(r["questions"]) for r in test_rows),
        "qa_id_overlap_count": 0,
        "shared_context_rows": 0,
        "context_disjoint": True,
        "selected_test_context_sha256": sorted(test_groups),
        "all_strata": dict(sorted(source_counts(rows).items())),
        "train_strata": dict(sorted(source_counts(train_rows).items())),
        "test_strata": dict(sorted(source_counts(test_rows).items())),
        "warning": (
            "Singleton source/task rows cannot appear in both train and test "
            "when strict context-disjoint splitting is used."
        ),
    }
    return train_rows, test_rows, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One or more MAB JSON/JSONL files, or directories containing them.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.resolve(),
    )
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument(
        "--split-unit",
        choices=("qa", "row"),
        default="qa",
        help=(
            "qa: preserve task representation with disjoint QA content; "
            "row: strict context-disjoint split."
        ),
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Also write one <stem>_split_manifest.json per input file.",
    )
    args = parser.parse_args()

    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be strictly between 0 and 1.")

    input_paths = discover_inputs(args.inputs)

    # Prevent two different input files from producing the same output names.
    stems = [path.stem for path in input_paths]
    duplicate_stems = [stem for stem, count in Counter(stems).items() if count > 1]
    if duplicate_stems:
        raise ValueError(
            "Output filename collisions for duplicate stems: "
            + ", ".join(sorted(duplicate_stems))
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_reports: list[dict[str, Any]] = []

    for input_path in input_paths:
        rows = read_records(input_path)

        if not rows:
            raise ValueError(f"Input dataset is empty: {input_path}")

        input_stem = input_path.stem

        if args.split_unit == "qa":
            train, test, report = split_by_qa(
                rows,
                train_fraction=args.train_fraction,
                seed=args.seed,
                input_stem=input_stem,
            )
        else:
            train, test, report = split_by_context_row(
                rows,
                train_fraction=args.train_fraction,
                seed=args.seed,
                input_stem=input_stem,
            )

        suffix = ".jsonl" if input_path.suffix.lower() == ".jsonl" else ".json"

        train_path = args.output_dir / f"{input_stem}_train{suffix}"
        test_path = args.output_dir / f"{input_stem}_test{suffix}"

        write_records(train_path, train)
        write_records(test_path, test)

        report.update(
            {
                "dataset": "MemoryAgentBench",
                "input": str(input_path),
                "input_stem": input_stem,
                "input_file_sha256": file_sha256(input_path),
                "input_records_sha256": stable_hash(rows),
                "seed": args.seed,
                "train_output": str(train_path),
                "test_output": str(test_path),
                "train_sha256": stable_hash(train),
                "test_sha256": stable_hash(test),
            }
        )

        if args.write_manifest:
            manifest_path = args.output_dir / f"{input_stem}_split_manifest.json"
            report["manifest_output"] = str(manifest_path)

            with manifest_path.open("w", encoding="utf-8") as f:
                json.dump(
                    report,
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
                f.write("\n")

        all_reports.append(report)

    summary = {
        "dataset": "MemoryAgentBench",
        "number_of_input_files": len(input_paths),
        "requested_train_fraction": args.train_fraction,
        "requested_test_fraction": 1.0 - args.train_fraction,
        "split_unit": args.split_unit,
        "seed": args.seed,
        "output_directory": str(args.output_dir),
        "files": all_reports,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
