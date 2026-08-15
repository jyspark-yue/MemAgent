#############################################################################
# File: ecommerce_percent_split_generator.py
#
# Description:
#   Create deterministic, mutually exclusive EcommerceMemEval train/test files.
#
#   Default split: 80% train / 20% test.
#
#   Complete customer groups stay together, so no customer, QA pair, question, or
#   record is shared between train and test. The test customers are selected with
#   multi-label stratification over metadata.question_types.
#
#   Outputs (exactly two dataset files):
#       <input_stem>_train.json or .jsonl
#       <input_stem>_test.json or .jsonl
#       Use --write-manifest to additionally write a validation manifest.
#
#   Command-line Arguments (Change as needed):
#       python ecommerce_percent_split_generator.py \
#           ecommerce_memeval_dataset_multi.json \
#           --output-dir splits/ecommerce_20pct \
#           --train-fraction 0.80 \
#           --write-manifest
#############################################################################

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


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


def get_path(record: dict[str, Any], dotted_path: str) -> Any:
    current: Any = record
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Missing field {dotted_path!r} at component {part!r}.")
        current = current[part]
    return current


def stable_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_text(value).encode("utf-8")).hexdigest()


def normalize_label(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "-").replace("_", "-")


def flatten(values: Iterable[Iterable[str]]) -> list[str]:
    return [item for group in values for item in group]


def objective(
    selected_counts: Counter[str],
    targets: dict[str, float],
) -> float:
    """Normalized squared distance from desired label totals."""
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
    if target_group_count <= 0:
        return set()

    total_counts: Counter[str] = Counter()
    for labels in group_labels.values():
        total_counts.update(labels)
    targets = {label: count * test_fraction for label, count in total_counts.items()}

    rng = random.Random(seed)
    tie_break = {group_id: rng.random() for group_id in sorted(group_labels)}
    selected: set[str] = set()
    selected_counts: Counter[str] = Counter()

    # Seed rare labels first so low-frequency question types are represented.
    labels_by_rarity = sorted(total_counts, key=lambda x: (total_counts[x], x))
    for label in labels_by_rarity:
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
                objective(selected_counts + group_labels[group_id], targets),
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
                objective(selected_counts + group_labels[group_id], targets),
                tie_break[group_id],
                group_id,
            ),
        )
        selected.add(best)
        selected_counts.update(group_labels[best])

    # Deterministic local swap optimization.
    improved = True
    passes = 0
    while improved and passes < 8:
        improved = False
        passes += 1
        current_score = objective(selected_counts, targets)
        for selected_id in sorted(selected):
            for unselected_id in sorted(set(group_labels) - selected):
                candidate_counts = selected_counts.copy()
                candidate_counts.subtract(group_labels[selected_id])
                candidate_counts.update(group_labels[unselected_id])
                candidate_score = objective(candidate_counts, targets)
                if candidate_score + 1e-12 < current_score:
                    selected.remove(selected_id)
                    selected.add(unselected_id)
                    selected_counts = candidate_counts
                    improved = True
                    break
            if improved:
                break

    return selected


def qa_ids(record: dict[str, Any], qa_id_field: str) -> list[str]:
    try:
        value = get_path(record, qa_id_field)
    except KeyError:
        return []
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"{qa_id_field!r} must be a list when present.")
    return [str(item) for item in value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent.resolve()
    )
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument(
        "--group-field",
        default="metadata.customer_id",
        help="Dotted field used to keep complete customer groups together.",
    )
    parser.add_argument(
        "--question-types-field",
        default="metadata.question_types",
        help="Dotted list field used for multi-label stratification.",
    )
    parser.add_argument(
        "--qa-id-field",
        default="metadata.qa_pair_ids",
        help="Dotted QA-ID list field used for overlap validation.",
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Also write <stem>_split_manifest.json.",
    )
    args = parser.parse_args()

    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be strictly between 0 and 1.")

    records = read_records(args.input)
    if len(records) < 2:
        raise ValueError("At least two records are required for a train/test split.")

    groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    group_label_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for index, record in enumerate(records):
        group_id = stable_text(get_path(record, args.group_field))
        labels = get_path(record, args.question_types_field)
        if not isinstance(labels, list):
            raise TypeError(
                f"Record {index}: {args.question_types_field!r} must be a list."
            )
        questions = record.get("questions", [])
        answers = record.get("answers", [])
        if len(labels) != len(questions) or len(questions) != len(answers):
            raise ValueError(
                f"Record {index} has misaligned questions/answers/question_types: "
                f"{len(questions)}/{len(answers)}/{len(labels)}."
            )
        groups[group_id].append((index, record))
        group_label_counts[group_id].update(normalize_label(x) for x in labels)

    test_fraction = 1.0 - args.train_fraction
    target_test_groups = round(len(groups) * test_fraction)
    target_test_groups = max(1, min(len(groups) - 1, target_test_groups))
    test_groups = select_multilabel_test_groups(
        group_label_counts,
        target_group_count=target_test_groups,
        test_fraction=test_fraction,
        seed=args.seed,
    )

    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        group_id = stable_text(get_path(record, args.group_field))
        (test if group_id in test_groups else train).append(record)

    train_groups = {stable_text(get_path(record, args.group_field)) for record in train}
    test_groups_recomputed = {
        stable_text(get_path(record, args.group_field)) for record in test
    }
    group_overlap = train_groups & test_groups_recomputed
    if group_overlap:
        raise AssertionError(
            f"Detected {len(group_overlap)} customer/group IDs in both splits."
        )

    train_qa_ids = set(flatten(qa_ids(r, args.qa_id_field) for r in train))
    test_qa_ids = set(flatten(qa_ids(r, args.qa_id_field) for r in test))
    qa_overlap = train_qa_ids & test_qa_ids
    if qa_overlap:
        raise AssertionError(
            f"Detected {len(qa_overlap)} QA IDs in both train and test."
        )

    train_hashes = {stable_hash(record) for record in train}
    test_hashes = {stable_hash(record) for record in test}
    record_overlap = train_hashes & test_hashes
    if record_overlap:
        raise AssertionError(
            f"Detected {len(record_overlap)} identical records in both splits."
        )

    def label_counts(rows: Sequence[dict[str, Any]]) -> Counter[str]:
        counts: Counter[str] = Counter()
        for row in rows:
            counts.update(
                normalize_label(x) for x in get_path(row, args.question_types_field)
            )
        return counts

    suffix = ".jsonl" if args.input.suffix.lower() == ".jsonl" else ".json"
    percentage = round(test_fraction * 100)
    train_path = args.output_dir / f"ecommerce_memeval_{percentage}pct_train{suffix}"
    test_path = args.output_dir / f"ecommerce_memeval_{percentage}pct_test{suffix}"
    write_records(train_path, train)
    write_records(test_path, test)

    report = {
        "dataset": "EcommerceMemEval",
        "input": str(args.input),
        "seed": args.seed,
        "requested_train_fraction": args.train_fraction,
        "actual_train_record_fraction": len(train) / len(records),
        "actual_train_group_fraction": len(train_groups) / len(groups),
        "total_records": len(records),
        "train_records": len(train),
        "test_records": len(test),
        "total_groups": len(groups),
        "train_groups": len(train_groups),
        "test_groups": len(test_groups_recomputed),
        "group_field": args.group_field,
        "question_types_field": args.question_types_field,
        "qa_id_field": args.qa_id_field,
        "group_overlap_count": 0,
        "qa_id_overlap_count": 0,
        "identical_record_overlap_count": 0,
        "all_question_type_counts": dict(sorted(label_counts(records).items())),
        "train_question_type_counts": dict(sorted(label_counts(train).items())),
        "test_question_type_counts": dict(sorted(label_counts(test).items())),
        "train_output": str(train_path),
        "test_output": str(test_path),
        "train_sha256": stable_hash(train),
        "test_sha256": stable_hash(test),
    }

    if args.write_manifest:
        manifest_path = (
            args.output_dir / f"ecommerce_memeval_{percentage}pct_split_manifest.json"
        )
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")
        report["manifest_output"] = str(manifest_path)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
