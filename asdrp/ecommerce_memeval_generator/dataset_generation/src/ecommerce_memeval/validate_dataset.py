#############################################################################
# File: validate_dataset.py
#
# Description:
#   Validates generated EcommerceMemEval JSONL against the DatasetRow schema.
#
#   - Reports malformed/invalid rows without hiding line numbers.
#   - Counts valid rows, QA pairs, and question-type frequencies.
#   - Optionally prints a small set of validated examples and exits nonzero on failures.
#############################################################################

from __future__ import annotations

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

import argparse
from collections import Counter
from .io_utils import iter_jsonl
from .models import DatasetRow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl_path")
    parser.add_argument("--show-examples", type=int, default=2)
    args = parser.parse_args()

    n = 0
    bad = 0
    q_count = 0
    type_counts: Counter[str] = Counter()

    for line_num, obj in iter_jsonl(args.jsonl_path):
        try:
            row = DatasetRow.model_validate(obj)
            n += 1
            q_count += len(row.questions)
            type_counts.update([str(x) for x in row.metadata.question_types])
            if n <= args.show_examples:
                print(f"\nExample row {n}: {row.metadata.customer_id}")
                for qt, q, a in zip(
                    row.metadata.question_types, row.questions, row.answers
                ):
                    print(f"- [{qt}] {q} -> {a[0]}")
        except Exception as exc:
            bad += 1
            print(f"BAD line {line_num}: {exc}")

    print("\nValidation summary")
    print(f"Rows valid: {n}")
    print(f"Rows bad:   {bad}")
    print(f"QA pairs:   {q_count}")
    print("Question type counts:")
    for qt, count in sorted(type_counts.items()):
        print(f"  {qt}: {count}")

    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
