#############################################################################
# File: lme_percent_split_generator.py
#
# Description:
#   Create deterministic, mutually exclusive LongMemEval train/test files from the
#   longmemeval_m_cleaned dataset.
#
#   Default split: 80% train / 20% test.
#
#   The official LongMemEval format stores one evaluation instance per record and
#   uses `question_type`; abstention examples are identified by `_abs` at the end
#   of `question_id`. The script preserves those seven effective question types.
#
#   Outputs (exactly two dataset files):
#       <input_stem>_train.json or .jsonl
#       <input_stem>_test.json or .jsonl
#       Use --write-manifest to additionally write a validation manifest.
#
#   Command-line Arguments (Change as needed):
#       python lme_percent_split_generator.py \
#           longmemeval_m_cleaned.json \
#           --output-dir splits/longmemeval_20pct \
#           --train-fraction 0.80 \
#           --write-manifest
#############################################################################

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read a JSON array/object or JSONL file."""
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
        # Common wrappers used by exported datasets.
        for key in ("data", "records", "examples", "items"):
            wrapped = value.get(key)
            if isinstance(wrapped, list) and all(
                isinstance(item, dict) for item in wrapped
            ):
                return wrapped
        # A single record is also accepted.
        return [value]

    raise TypeError(f"Unsupported top-level JSON value in {path}: {type(value)}")


def write_records(path: Path, records: Sequence[dict[str, Any]]) -> None:
    """Write JSON or JSONL according to the output suffix."""
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


def first_present(record: dict[str, Any], candidates: Sequence[str]) -> Any | None:
    for candidate in candidates:
        if candidate in record and record[candidate] not in (None, ""):
            return record[candidate]
    return None


def normalize_label(value: Any) -> str:
    if value is None:
        return "unknown"
    return str(value).strip().lower().replace(" ", "-").replace("_", "-")


def desired_count(total: int, fraction: float) -> int:
    if total <= 0:
        return 0
    return max(1, min(total, round(total * fraction)))


def proportional_quotas(
    sizes: dict[str, int],
    target: int,
    *,
    ensure_each: bool,
) -> dict[str, int]:
    """
    Allocate exactly `target` samples with largest-remainder apportionment.

    When target >= number of non-empty strata and ensure_each=True, every
    stratum receives at least one sample.
    """
    sizes = {key: size for key, size in sizes.items() if size > 0}
    if not sizes or target <= 0:
        return {key: 0 for key in sizes}

    target = min(target, sum(sizes.values()))
    quotas = {key: 0 for key in sizes}

    if ensure_each and target >= len(sizes):
        for key in sizes:
            quotas[key] = 1
        remaining = target - len(sizes)
        capacities = {key: sizes[key] - 1 for key in sizes}
    else:
        remaining = target
        capacities = dict(sizes)

    if remaining <= 0:
        return quotas

    capacity_total = sum(capacities.values())
    if capacity_total <= 0:
        return quotas

    raw = {key: remaining * capacities[key] / capacity_total for key in sizes}
    floors = {key: min(capacities[key], math.floor(raw[key])) for key in sizes}
    for key, amount in floors.items():
        quotas[key] += amount

    left = target - sum(quotas.values())
    order = sorted(
        sizes,
        key=lambda key: (
            raw[key] - floors[key],
            capacities[key],
            key,
        ),
        reverse=True,
    )
    while left > 0:
        progressed = False
        for key in order:
            if quotas[key] < sizes[key]:
                quotas[key] += 1
                left -= 1
                progressed = True
                if left == 0:
                    break
        if not progressed:
            break

    return quotas


def grouped_stratified_sample(
    records: Sequence[dict[str, Any]],
    *,
    fraction: float,
    seed: int,
    stratum_fn: Callable[[dict[str, Any]], str],
    group_fn: Callable[[dict[str, Any], int], str],
    ensure_each_stratum: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """
    Sample complete groups while approximately preserving record-level strata.

    A group is assigned to the stratum most frequent among its records. This is
    appropriate when groups are normally label-homogeneous and still prevents
    context/user leakage if a malformed group contains multiple labels.
    """
    if not 0 < fraction < 1:
        raise ValueError("--fraction must be strictly between 0 and 1.")

    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[group_fn(record, index)].append((index, record))

    group_strata: dict[str, str] = {}
    for group_id, members in grouped.items():
        labels = [stratum_fn(record) for _, record in members]
        counts = Counter(labels)
        group_strata[group_id] = sorted(
            counts, key=lambda label: (-counts[label], label)
        )[0]

    groups_by_stratum: dict[str, list[str]] = defaultdict(list)
    for group_id, stratum in group_strata.items():
        groups_by_stratum[stratum].append(group_id)

    target_groups = desired_count(len(grouped), fraction)
    quotas = proportional_quotas(
        {key: len(value) for key, value in groups_by_stratum.items()},
        target_groups,
        ensure_each=ensure_each_stratum,
    )

    rng = random.Random(seed)
    selected_groups: set[str] = set()
    for stratum in sorted(groups_by_stratum):
        candidates = sorted(groups_by_stratum[stratum])
        rng.shuffle(candidates)
        selected_groups.update(candidates[: quotas.get(stratum, 0)])

    evaluation: list[dict[str, Any]] = []
    remainder: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        group_id = group_fn(record, index)
        (evaluation if group_id in selected_groups else remainder).append(record)

    eval_strata = Counter(stratum_fn(record) for record in evaluation)
    all_strata = Counter(stratum_fn(record) for record in records)
    selected_group_strata = Counter(group_strata[g] for g in selected_groups)

    report = {
        "seed": seed,
        "requested_fraction": fraction,
        "total_records": len(records),
        "evaluation_records": len(evaluation),
        "remainder_records": len(remainder),
        "actual_record_fraction": len(evaluation) / len(records) if records else 0,
        "total_groups": len(grouped),
        "evaluation_groups": len(selected_groups),
        "actual_group_fraction": len(selected_groups) / len(grouped) if grouped else 0,
        "all_record_strata": dict(sorted(all_strata.items())),
        "evaluation_record_strata": dict(sorted(eval_strata.items())),
        "evaluation_group_strata": dict(sorted(selected_group_strata.items())),
        "selected_group_ids": sorted(selected_groups),
        "evaluation_sha256": stable_hash(evaluation),
        "remainder_sha256": stable_hash(remainder),
    }
    return evaluation, remainder, report


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")


def effective_question_type(record: dict[str, Any]) -> str:
    question_id = str(record.get("question_id", ""))
    if question_id.endswith("_abs"):
        return "abstention"
    return normalize_label(record.get("question_type"))


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
        default=None,
        help=(
            "Optional field whose records must stay together. Normally each "
            "LongMemEval record is already one complete evaluation instance."
        ),
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Also write <stem>_split_manifest.json.",
    )
    args = parser.parse_args()

    records = read_records(args.input)
    if not records:
        raise ValueError("The input dataset is empty.")

    def group_id(record: dict[str, Any], index: int) -> str:
        if args.group_field:
            value = record.get(args.group_field)
            if value is None:
                raise KeyError(
                    f"Record {index} is missing --group-field " f"{args.group_field!r}."
                )
            return stable_text(value)
        question_id = record.get("question_id")
        return str(question_id) if question_id is not None else f"row:{index}"

    test_fraction = 1.0 - args.train_fraction
    test_records, train_records, report = grouped_stratified_sample(
        records,
        fraction=test_fraction,
        seed=args.seed,
        stratum_fn=effective_question_type,
        group_fn=group_id,
        ensure_each_stratum=True,
    )

    suffix = ".jsonl" if args.input.suffix.lower() == ".jsonl" else ".json"
    percentage = round(test_fraction * 100)
    train_path = args.output_dir / f"longmemeval_{percentage}pct_train{suffix}"
    test_path = args.output_dir / f"longmemeval_{percentage}pct_test{suffix}"

    write_records(train_path, train_records)
    write_records(test_path, test_records)

    report.update(
        {
            "dataset": "LongMemEval",
            "input": str(args.input),
            "input_file_sha256": file_sha256(args.input),
            "input_records_sha256": stable_hash(records),
            "train_fraction": args.train_fraction,
            "test_fraction": test_fraction,
            "train_records": len(train_records),
            "test_records": len(test_records),
            "train_output": str(train_path),
            "test_output": str(test_path),
            "effective_question_type_rule": (
                "question_id ending in '_abs' => abstention; otherwise question_type"
            ),
        }
    )

    if args.write_manifest:
        manifest_path = (
            args.output_dir / f"longmemeval_{percentage}pct_split_manifest.json"
        )
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")
        report["manifest_output"] = str(manifest_path)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
