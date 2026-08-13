from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

CUSTOMER_RE = re.compile(r"cust_\d{5}")
CUSTOMER_ID_RE = re.compile(r"^cust_\d{5}$")
SESSION_RE = re.compile(r"cust_\d{5}_s(\d{3,})")
QA_RE = re.compile(r"cust_\d{5}_q(\d{3,})")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(
                    f"Line {line_num} is {type(obj).__name__}, expected object/dict"
                )
            rows.append(obj)
    return rows


def original_customer_number(row: dict[str, Any]) -> int | None:
    """Return numeric customer id when available, for optional stable sorting."""
    candidates: list[str] = []

    meta = row.get("metadata")
    if isinstance(meta, dict) and isinstance(meta.get("customer_id"), str):
        candidates.append(meta["customer_id"])

    context = row.get("context")
    if isinstance(context, str):
        candidates.extend(CUSTOMER_RE.findall(context[:500]))

    for candidate in candidates:
        m = re.search(r"cust_(\d{5})", candidate)
        if m:
            return int(m.group(1))
    return None


def replace_ids_in_string(value: str, new_customer_id: str) -> str:
    """Replace unnecessary customer id prefixes while preserving session/question suffixes."""
    value = SESSION_RE.sub(lambda m: f"{new_customer_id}_s{m.group(1)}", value)
    value = QA_RE.sub(lambda m: f"{new_customer_id}_q{m.group(1)}", value)
    value = CUSTOMER_RE.sub(new_customer_id, value)
    return value


def recursively_replace_ids(value: Any, new_customer_id: str) -> Any:
    if isinstance(value, str):
        return replace_ids_in_string(value, new_customer_id)
    if isinstance(value, list):
        return [recursively_replace_ids(item, new_customer_id) for item in value]
    if isinstance(value, dict):
        return {
            key: recursively_replace_ids(item, new_customer_id)
            for key, item in value.items()
        }
    return value


def renumber_row(row: dict[str, Any], new_index: int, prefix: str) -> dict[str, Any]:
    """
    Renumber one dataset row so IDs are compact/sequential.

    Example:
      row 1  -> cust_00001
      row 2  -> cust_00002
      row 99 -> cust_00099

    This updates:
      - metadata.customer_id
      - metadata.qa_pair_ids
      - metadata.evidence[*].qa_pair_id
      - metadata.evidence[*].session_ids
      - every customer/session/QA id string inside the row, including context text
    """
    new_customer_id = f"{prefix}_{new_index:05d}"

    # First do broad safe string replacement everywhere.
    fixed = recursively_replace_ids(row, new_customer_id)

    metadata = fixed.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        fixed["metadata"] = metadata

    metadata["customer_id"] = new_customer_id

    questions = fixed.get("questions")
    num_qas = len(questions) if isinstance(questions, list) else 0

    answers = fixed.get("answers")
    if isinstance(answers, list):
        num_qas = max(num_qas, len(answers))

    question_types = metadata.get("question_types")
    if isinstance(question_types, list):
        num_qas = max(num_qas, len(question_types))

    evidence = metadata.get("evidence")
    if isinstance(evidence, list):
        num_qas = max(num_qas, len(evidence))

    # Regenerate qa_pair_ids so gaps from failed rows disappear.
    if num_qas:
        metadata["qa_pair_ids"] = [
            f"{new_customer_id}_q{i:03d}" for i in range(1, num_qas + 1)
        ]

    # Align evidence qa_pair_id with the evidence index.
    if isinstance(metadata.get("evidence"), list):
        for i, ev in enumerate(metadata["evidence"], start=1):
            if isinstance(ev, dict):
                ev["qa_pair_id"] = f"{new_customer_id}_q{i:03d}"
                if isinstance(ev.get("session_ids"), list):
                    ev["session_ids"] = [
                        replace_ids_in_string(str(session_id), new_customer_id)
                        for session_id in ev["session_ids"]
                    ]

    return fixed


def check_sequential_customer_ids(rows: list[dict[str, Any]], prefix: str) -> None:
    expected = [f"{prefix}_{i:05d}" for i in range(1, len(rows) + 1)]
    actual = [row.get("metadata", {}).get("customer_id") for row in rows]
    bad = [
        (i + 1, exp, got)
        for i, (exp, got) in enumerate(zip(expected, actual))
        if exp != got
    ]
    if bad:
        shown = ", ".join(
            f"row {i}: expected {exp}, got {got}" for i, exp, got in bad[:5]
        )
        raise AssertionError(
            f"Customer IDs are not sequential after conversion: {shown}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ecommerce JSONL dataset to pretty JSON and compactly renumber customer/QA/session IDs."
    )
    parser.add_argument("input", type=Path, help="Input .jsonl file")
    parser.add_argument("output", type=Path, help="Output .json file")
    parser.add_argument(
        "--prefix",
        default="cust",
        help="Customer ID prefix before the numeric part; default: cust",
    )
    parser.add_argument(
        "--sort-by-original-id",
        action="store_true",
        help="Sort rows by their original customer id before renumbering. Useful if your JSONL was appended out of order.",
    )
    parser.add_argument(
        "--wrap-key",
        default="",
        help="Optional top-level key. Example: --wrap-key data writes {'data': [...]} instead of a bare array.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Pretty-print indentation; default: 2",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.input)

    if args.sort_by_original_id:
        rows = sorted(
            enumerate(rows),
            key=lambda pair: (
                original_customer_number(pair[1]) is None,
                original_customer_number(pair[1]) or 10**9,
                pair[0],
            ),
        )
        rows = [row for _, row in rows]

    fixed_rows = [
        renumber_row(row, i, args.prefix) for i, row in enumerate(rows, start=1)
    ]
    check_sequential_customer_ids(fixed_rows, args.prefix)

    payload: Any = {args.wrap_key: fixed_rows} if args.wrap_key else fixed_rows

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=args.indent)
        f.write("\n")

    print(f"Read {len(rows)} JSONL rows")
    print(f"Wrote clean JSON to {args.output}")
    print(
        f"Customer IDs now run from {args.prefix}_00001 to {args.prefix}_{len(rows):05d}"
    )


if __name__ == "__main__":
    main()
