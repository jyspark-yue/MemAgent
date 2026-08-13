#############################################################################
# File: compare_router_apples_to_apples.py
#
# Description:
#   Compares Binary Router, HVM, and Episodic results on one matched question set.
#
#   - Uses the evaluated router question IDs as the canonical comparison set.
#   - Reports end-to-end router, standalone baselines, fixed-output routing, and oracle references.
#   - Reuses existing judged labels and makes no new LLM calls.
#   - Verifies input coverage and records file hashes for reproducibility.
#   - Writes a human-readable report with overall, type-level, and per-question results.
#   - All correctness values come from the existing 'autoeval_label.label' fields,
#     this script does not call an LLM or re-judge any answer.
#############################################################################

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

DEFAULT_QUESTION_TYPES = ("single-session-preference", "knowledge-update")
DEFAULT_TYPE_ROUTES = {
    "single-session-preference": "hvm",
    "knowledge-update": "episodic",
}


class ComparisonError(RuntimeError):
    pass


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ComparisonError(f"File not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ComparisonError(
                    f"Invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ComparisonError(
                    f"Expected a JSON object in {path} at line {line_number}."
                )
            rows.append(row)
    return rows


def read_reference_map(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise ComparisonError(f"Reference file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, dict):
        # Support a few harmless wrappers while keeping the standard
        # LongMemEval list format as the primary case.
        for key in ("data", "records", "examples", "items"):
            if isinstance(data.get(key), list):
                data = data[key]
                break

    if not isinstance(data, list):
        raise ComparisonError(
            f"Expected {path} to contain a list of LongMemEval records."
        )

    result: dict[str, dict[str, Any]] = {}
    for row in data:
        if not isinstance(row, dict):
            continue
        qid = row.get("question_id") or row.get("task_id")
        if qid is not None:
            result[str(qid)] = row
    return result


def index_by_question_id(
    rows: Iterable[dict[str, Any]], *, source_name: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = row.get("question_id") or row.get("task_id")
        if not qid:
            raise ComparisonError(f"{source_name} contains a row without question_id.")
        qid = str(qid)
        if qid in indexed:
            raise ComparisonError(
                f"{source_name} contains duplicate question_id {qid!r}."
            )
        indexed[qid] = row
    return indexed


def judged_label(row: dict[str, Any], *, source_name: str, qid: str) -> bool:
    autoeval = row.get("autoeval_label")
    if not isinstance(autoeval, dict):
        raise ComparisonError(
            f"{source_name} question {qid} has no autoeval_label. "
            "Pass the evaluated *.eval-results-*.jsonl file, not the raw hypothesis JSONL."
        )

    label = autoeval.get("label")
    if not isinstance(label, bool):
        status = autoeval.get("status")
        raise ComparisonError(
            f"{source_name} question {qid} has no boolean judge label "
            f"(status={status!r}, label={label!r})."
        )
    return label


def question_type_for(
    qid: str,
    reference_map: dict[str, dict[str, Any]],
    candidate_rows: Iterable[dict[str, Any]],
) -> str:
    ref = reference_map.get(qid, {})
    qtype = ref.get("question_type")
    if isinstance(qtype, str) and qtype:
        return qtype

    # Fallback for evaluation pipelines that preserve the type directly.
    for row in candidate_rows:
        qtype = row.get("question_type")
        if isinstance(qtype, str) and qtype:
            return qtype
        metadata = row.get("metadata")
        if isinstance(metadata, dict):
            qtype = metadata.get("question_type")
            if isinstance(qtype, str) and qtype:
                return qtype

    raise ComparisonError(
        f"Could not determine question_type for {qid}. "
        "Check --reference-file or ensure question_type is preserved in the eval rows."
    )


def routing_for(
    qid: str,
    router_eval_row: dict[str, Any],
    router_hypothesis_map: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    routing = router_eval_row.get("routing")
    if isinstance(routing, dict):
        return routing

    if router_hypothesis_map is not None:
        row = router_hypothesis_map.get(qid)
        if row is not None and isinstance(row.get("routing"), dict):
            return row["routing"]

    raise ComparisonError(
        f"Router question {qid} has no routing metadata. "
        "Pass --router-hypotheses if the evaluated JSONL did not preserve it."
    )


def selected_memory_from_routing(routing: dict[str, Any], qid: str) -> str:
    selected = routing.get("selected_memory") or routing.get("selected_agent")
    if not isinstance(selected, str):
        raise ComparisonError(
            f"Router question {qid} has no selected_memory in routing metadata."
        )
    selected = selected.strip().lower()
    aliases = {
        "hvm": "hvm",
        "episodic": "episodic",
        "episode": "episodic",
    }
    if selected not in aliases:
        raise ComparisonError(
            f"Router question {qid} selected unsupported memory {selected!r}."
        )
    return aliases[selected]


def metric_from_labels(labels: list[bool]) -> dict[str, Any]:
    count = len(labels)
    correct = sum(labels)
    return {
        "accuracy": round(correct / count, 6) if count else None,
        "correct": correct,
        "count": count,
    }


def summarize_system(
    labels_by_qid: dict[str, bool],
    question_types_by_qid: dict[str, str],
    canonical_ids: list[str],
) -> dict[str, Any]:
    overall_labels = [labels_by_qid[qid] for qid in canonical_ids]

    grouped: dict[str, list[bool]] = defaultdict(list)
    for qid in canonical_ids:
        grouped[question_types_by_qid[qid]].append(labels_by_qid[qid])

    by_question_type = {
        qtype: metric_from_labels(grouped[qtype]) for qtype in sorted(grouped)
    }
    macro_values = [
        values["accuracy"]
        for values in by_question_type.values()
        if values["accuracy"] is not None
    ]

    return {
        "overall": metric_from_labels(overall_labels),
        "macro_accuracy_by_question_type": (
            round(sum(macro_values) / len(macro_values), 6) if macro_values else None
        ),
        "by_question_type": by_question_type,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_question_ids(question_ids: list[str]) -> str:
    payload = "\n".join(question_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def pp_delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return round((a - b) * 100.0, 4)


SYSTEM_LABELS = {
    "episodic": "Episodic",
    "hvm": "HVM",
    "binary_router_end_to_end": "Binary Router (end-to-end)",
    "router_fixed_output_selection": "Router (fixed standalone outputs)",
    "type_oracle": "Type Oracle",
    "best_of_two_oracle": "Best-of-Two Oracle",
}


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def pp(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2f} pp"


def yes_no(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return "Y" if value else "N"


def render_table(
    headers: list[str], rows: list[list[str]], aligns: list[str] | None = None
) -> list[str]:
    if aligns is None:
        aligns = ["left"] * len(headers)
    widths = [len(header) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt_row(row: list[str]) -> str:
        cells: list[str] = []
        for i, cell in enumerate(row):
            value = str(cell)
            if aligns[i] == "right":
                cells.append(value.rjust(widths[i]))
            elif aligns[i] == "center":
                cells.append(value.center(widths[i]))
            else:
                cells.append(value.ljust(widths[i]))
        return "  ".join(cells)

    lines = [fmt_row(headers)]
    lines.append("  ".join("-" * width for width in widths))
    lines.extend(fmt_row(row) for row in rows)
    return lines


def render_report(output: dict[str, Any]) -> str:
    scope = output["evaluation_scope"]
    systems = output["systems"]
    routing = output["routing"]
    comparisons = output["comparisons_percentage_points"]

    lines: list[str] = []
    title = "LONGMEMEVAL BINARY ROUTER V3 - APPLES-TO-APPLES COMPARISON"
    lines.extend(["=" * len(title), title, "=" * len(title), ""])

    lines.append("EVALUATION SCOPE")
    lines.append("----------------")
    lines.append(
        f"Canonical set: exactly the {scope['question_count']} question IDs in the Binary Router evaluation file"
    )
    qtype_counts = scope.get("question_type_counts", {})
    for qtype, count in sorted(qtype_counts.items()):
        lines.append(f"  - {qtype}: {count}")
    lines.append(f"Question-ID SHA256: {scope['question_ids_sha256']}")
    lines.append("")

    lines.append("OVERALL PERFORMANCE")
    lines.append("-------------------")
    overall_rows: list[list[str]] = []
    for name in (
        "episodic",
        "hvm",
        "binary_router_end_to_end",
        "router_fixed_output_selection",
        "type_oracle",
        "best_of_two_oracle",
    ):
        metric = systems[name]["overall"]
        overall_rows.append(
            [
                SYSTEM_LABELS[name],
                f"{metric['correct']}/{metric['count']}",
                pct(metric["accuracy"]),
                pct(systems[name]["macro_accuracy_by_question_type"]),
            ]
        )
    lines.extend(
        render_table(
            ["System", "Correct", "Accuracy", "Macro by Type"],
            overall_rows,
            ["left", "right", "right", "right"],
        )
    )
    lines.append("")

    lines.append("PERFORMANCE BY QUESTION TYPE")
    lines.append("----------------------------")
    for qtype in sorted(qtype_counts):
        lines.append("")
        lines.append(qtype)
        q_rows: list[list[str]] = []
        for name in (
            "episodic",
            "hvm",
            "binary_router_end_to_end",
            "router_fixed_output_selection",
            "type_oracle",
            "best_of_two_oracle",
        ):
            metric = systems[name]["by_question_type"][qtype]
            q_rows.append(
                [
                    SYSTEM_LABELS[name],
                    f"{metric['correct']}/{metric['count']}",
                    pct(metric["accuracy"]),
                ]
            )
        lines.extend(
            render_table(
                ["System", "Correct", "Accuracy"],
                q_rows,
                ["left", "right", "right"],
            )
        )
    lines.append("")

    lines.append("ROUTING PERFORMANCE")
    lines.append("-------------------")
    pred = routing["question_type_prediction"]
    if pred["evaluated_count"]:
        lines.append(
            f"Question-type classification: {pred['correct']}/{pred['evaluated_count']} "
            f"({pct(pred['accuracy'])})"
        )
    else:
        lines.append("Question-type classification: n/a")
    selected_counts = routing.get("selected_memory_counts", {})
    if selected_counts:
        selected_text = ", ".join(
            f"{name}={count}" for name, count in sorted(selected_counts.items())
        )
        lines.append(f"Selected memories: {selected_text}")
    confusion = pred.get("confusion", [])
    if confusion:
        lines.append("")
        confusion_rows = [
            [
                item["true_question_type"],
                item["predicted_question_type"],
                str(item["count"]),
            ]
            for item in confusion
        ]
        lines.extend(
            render_table(
                ["True Type", "Predicted Type", "Count"],
                confusion_rows,
                ["left", "left", "right"],
            )
        )
    lines.append("")

    lines.append("KEY DIFFERENCES")
    lines.append("---------------")
    comparison_labels = [
        ("Binary Router vs HVM", "binary_router_end_to_end_minus_hvm"),
        ("Binary Router vs Episodic", "binary_router_end_to_end_minus_episodic"),
        ("Fixed-output Router vs HVM", "router_fixed_output_selection_minus_hvm"),
        ("Type Oracle vs HVM", "type_oracle_minus_hvm"),
        ("Type Oracle vs Episodic", "type_oracle_minus_episodic"),
        ("Best-of-Two Oracle vs HVM", "best_of_two_oracle_minus_hvm"),
    ]
    diff_rows = [[label, pp(comparisons.get(key))] for label, key in comparison_labels]
    lines.extend(
        render_table(["Comparison", "Difference"], diff_rows, ["left", "right"])
    )
    lines.append("")

    lines.append("PER-QUESTION RESULTS")
    lines.append("--------------------")
    pq_rows: list[list[str]] = []
    for item in output["per_question"]:
        routing_info = item["routing"]
        correctness = item["correctness"]
        pq_rows.append(
            [
                item["question_id"],
                item["question_type"],
                routing_info.get("predicted_question_type") or "n/a",
                routing_info["selected_memory"],
                yes_no(correctness["episodic"]),
                yes_no(correctness["hvm"]),
                yes_no(correctness["binary_router_end_to_end"]),
                yes_no(correctness["router_fixed_output_selection"]),
                yes_no(correctness["type_oracle"]),
            ]
        )
    lines.extend(
        render_table(
            [
                "Question ID",
                "True Type",
                "Predicted",
                "Selected",
                "Epi",
                "HVM",
                "Router",
                "Fixed",
                "Oracle",
            ],
            pq_rows,
            [
                "left",
                "left",
                "left",
                "left",
                "center",
                "center",
                "center",
                "center",
                "center",
            ],
        )
    )
    lines.append("")

    lines.append("NOTES")
    lines.append("-----")
    lines.append(
        "- Binary Router (end-to-end): the actual judged answers produced by the router run."
    )
    lines.append(
        "- Router (fixed standalone outputs): uses the router's actual architecture decision, but scores the"
    )
    lines.append(
        "  already-judged standalone HVM/Episodic answer for that exact question. This isolates routing"
    )
    lines.append("  quality from fresh memory-build / generation variance.")
    lines.append(
        "- Type Oracle: HVM for single-session-preference and Episodic for knowledge-update, using true type."
    )
    lines.append(
        "- Best-of-Two Oracle: hindsight ceiling; correct when either HVM or Episodic was correct. Not realizable."
    )
    lines.append("- No answers are re-judged and no LLM calls are made by this script.")
    lines.append("")

    lines.append("INPUT FILES")
    lines.append("-----------")
    inputs = output["inputs"]
    lines.append(f"Router eval:       {inputs['router_eval']}")
    lines.append(f"HVM eval:          {inputs['hvm_eval']}")
    lines.append(f"Episodic eval:     {inputs['episodic_eval']}")
    lines.append(f"Reference file:    {inputs['reference_file']}")
    if inputs.get("router_hypotheses"):
        lines.append(f"Router hypotheses: {inputs['router_hypotheses']}")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Binary Router, HVM, Episodic, and oracle routing on exactly "
            "the question IDs present in the Binary Router evaluation JSONL."
        )
    )
    parser.add_argument(
        "--router-eval",
        type=Path,
        required=True,
        help="Binary Router v3 evaluated *.eval-results-*.jsonl file.",
    )
    parser.add_argument(
        "--hvm-eval",
        type=Path,
        required=True,
        help="Standalone HVM evaluated *.eval-results-*.jsonl file.",
    )
    parser.add_argument(
        "--episodic-eval",
        type=Path,
        required=True,
        help="Standalone Episodic evaluated *.eval-results-*.jsonl file.",
    )
    parser.add_argument(
        "--reference-file",
        type=Path,
        required=True,
        help="LongMemEval JSON split containing question_id and question_type.",
    )
    parser.add_argument(
        "--router-hypotheses",
        type=Path,
        default=None,
        help=(
            "Optional raw Binary Router JSONL. Only needed if routing metadata was "
            "not preserved in --router-eval."
        ),
    )
    parser.add_argument(
        "--question-types",
        nargs="+",
        default=list(DEFAULT_QUESTION_TYPES),
        help=(
            "Allowed question types in the canonical router set. Default: "
            "single-session-preference knowledge-update"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the human-readable apples-to-apples text report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    router_rows = read_jsonl(args.router_eval)
    hvm_rows = read_jsonl(args.hvm_eval)
    episodic_rows = read_jsonl(args.episodic_eval)
    reference_map = read_reference_map(args.reference_file)

    router_map = index_by_question_id(router_rows, source_name="Binary Router eval")
    hvm_map = index_by_question_id(hvm_rows, source_name="HVM eval")
    episodic_map = index_by_question_id(episodic_rows, source_name="Episodic eval")

    router_hypothesis_map: dict[str, dict[str, Any]] | None = None
    if args.router_hypotheses is not None:
        router_hypothesis_map = index_by_question_id(
            read_jsonl(args.router_hypotheses), source_name="Binary Router hypotheses"
        )

    # Preserve the router file's order. These IDs are the canonical apples-to-apples set.
    canonical_ids = [
        str(row.get("question_id") or row.get("task_id")) for row in router_rows
    ]
    if not canonical_ids:
        raise ComparisonError("Binary Router evaluation file contains no rows.")

    allowed_types = set(args.question_types)
    question_types_by_qid: dict[str, str] = {}

    missing_hvm = [qid for qid in canonical_ids if qid not in hvm_map]
    missing_episodic = [qid for qid in canonical_ids if qid not in episodic_map]
    if missing_hvm or missing_episodic:
        details = []
        if missing_hvm:
            details.append(f"HVM missing {missing_hvm}")
        if missing_episodic:
            details.append(f"Episodic missing {missing_episodic}")
        raise ComparisonError(
            "Cannot make an apples-to-apples comparison because baseline evaluated "
            "files are missing canonical router questions: " + "; ".join(details)
        )

    router_labels: dict[str, bool] = {}
    hvm_labels: dict[str, bool] = {}
    episodic_labels: dict[str, bool] = {}
    router_fixed_labels: dict[str, bool] = {}
    type_oracle_labels: dict[str, bool] = {}
    best_of_two_labels: dict[str, bool] = {}

    per_question: list[dict[str, Any]] = []
    route_confusion: Counter[tuple[str, str]] = Counter()
    selected_memory_counts: Counter[str] = Counter()
    routing_correct_count = 0

    for qid in canonical_ids:
        router_row = router_map[qid]
        hvm_row = hvm_map[qid]
        episodic_row = episodic_map[qid]

        qtype = question_type_for(
            qid, reference_map, (router_row, hvm_row, episodic_row)
        )
        question_types_by_qid[qid] = qtype
        if qtype not in allowed_types:
            raise ComparisonError(
                f"Canonical router question {qid} has type {qtype!r}, which is outside "
                f"--question-types {sorted(allowed_types)}."
            )

        router_label = judged_label(
            router_row, source_name="Binary Router eval", qid=qid
        )
        hvm_label = judged_label(hvm_row, source_name="HVM eval", qid=qid)
        episodic_label = judged_label(
            episodic_row, source_name="Episodic eval", qid=qid
        )

        routing = routing_for(qid, router_row, router_hypothesis_map)
        selected_memory = selected_memory_from_routing(routing, qid)
        predicted_type = routing.get("predicted_question_type")
        if not isinstance(predicted_type, str):
            predicted_type = None

        selected_memory_counts[selected_memory] += 1
        if predicted_type is not None:
            route_confusion[(qtype, predicted_type)] += 1
            if predicted_type == qtype:
                routing_correct_count += 1

        router_fixed_label = hvm_label if selected_memory == "hvm" else episodic_label

        oracle_memory = DEFAULT_TYPE_ROUTES.get(qtype)
        if oracle_memory is None:
            raise ComparisonError(
                f"No type-oracle route is defined for question type {qtype!r}."
            )
        type_oracle_label = hvm_label if oracle_memory == "hvm" else episodic_label
        best_of_two_label = hvm_label or episodic_label

        router_labels[qid] = router_label
        hvm_labels[qid] = hvm_label
        episodic_labels[qid] = episodic_label
        router_fixed_labels[qid] = router_fixed_label
        type_oracle_labels[qid] = type_oracle_label
        best_of_two_labels[qid] = best_of_two_label

        per_question.append(
            {
                "question_id": qid,
                "question_type": qtype,
                "question": router_row.get("question") or hvm_row.get("question"),
                "reference_answers": router_row.get("reference_answers")
                or hvm_row.get("reference_answers"),
                "routing": {
                    "predicted_question_type": predicted_type,
                    "selected_memory": selected_memory,
                    "type_prediction_correct": (
                        predicted_type == qtype if predicted_type is not None else None
                    ),
                    "decision_score": routing.get("decision_score"),
                },
                "correctness": {
                    "binary_router_end_to_end": router_label,
                    "hvm": hvm_label,
                    "episodic": episodic_label,
                    "router_fixed_output_selection": router_fixed_label,
                    "type_oracle": type_oracle_label,
                    "best_of_two_oracle": best_of_two_label,
                },
                "oracle": {
                    "type_oracle_selected_memory": oracle_memory,
                    "hvm_or_episodic_correct": best_of_two_label,
                },
            }
        )

    systems = {
        "binary_router_end_to_end": summarize_system(
            router_labels, question_types_by_qid, canonical_ids
        ),
        "hvm": summarize_system(hvm_labels, question_types_by_qid, canonical_ids),
        "episodic": summarize_system(
            episodic_labels, question_types_by_qid, canonical_ids
        ),
        "router_fixed_output_selection": summarize_system(
            router_fixed_labels, question_types_by_qid, canonical_ids
        ),
        "type_oracle": summarize_system(
            type_oracle_labels, question_types_by_qid, canonical_ids
        ),
        "best_of_two_oracle": summarize_system(
            best_of_two_labels, question_types_by_qid, canonical_ids
        ),
    }

    router_overall = systems["binary_router_end_to_end"]["overall"]["accuracy"]
    hvm_overall = systems["hvm"]["overall"]["accuracy"]
    episodic_overall = systems["episodic"]["overall"]["accuracy"]
    fixed_overall = systems["router_fixed_output_selection"]["overall"]["accuracy"]
    oracle_overall = systems["type_oracle"]["overall"]["accuracy"]
    best_overall = systems["best_of_two_oracle"]["overall"]["accuracy"]

    true_type_counts = Counter(question_types_by_qid.values())
    predicted_total = sum(route_confusion.values())

    output = {
        "comparison_name": "LongMemEval Binary Router v3 apples-to-apples comparison",
        "comparison_policy": {
            "canonical_question_set": (
                "Exactly the question IDs present in the Binary Router evaluated JSONL."
            ),
            "question_types": sorted(allowed_types),
            "type_routes": DEFAULT_TYPE_ROUTES,
            "router_fixed_output_selection": (
                "Uses the Binary Router's actual selected_memory for each question but "
                "takes correctness from the already-judged standalone HVM/Episodic "
                "output for that same question. This isolates routing from fresh-run variance."
            ),
            "type_oracle": (
                "Uses the true LongMemEval question type and the fixed route mapping: "
                "single-session-preference -> HVM; knowledge-update -> Episodic."
            ),
            "best_of_two_oracle": (
                "Hindsight empirical upper bound: a question is correct if either HVM "
                "or Episodic was correct. Not a realizable routing system."
            ),
            "rejudges_answers": False,
        },
        "inputs": {
            "router_eval": str(args.router_eval),
            "hvm_eval": str(args.hvm_eval),
            "episodic_eval": str(args.episodic_eval),
            "reference_file": str(args.reference_file),
            "router_hypotheses": (
                str(args.router_hypotheses) if args.router_hypotheses else None
            ),
            "sha256": {
                "router_eval": sha256_file(args.router_eval),
                "hvm_eval": sha256_file(args.hvm_eval),
                "episodic_eval": sha256_file(args.episodic_eval),
                "reference_file": sha256_file(args.reference_file),
            },
        },
        "evaluation_scope": {
            "question_count": len(canonical_ids),
            "question_ids": canonical_ids,
            "question_ids_sha256": sha256_question_ids(canonical_ids),
            "question_type_counts": dict(sorted(true_type_counts.items())),
            "all_baselines_have_every_question": True,
        },
        "routing": {
            "selected_memory_counts": dict(sorted(selected_memory_counts.items())),
            "question_type_prediction": {
                "evaluated_count": predicted_total,
                "correct": routing_correct_count if predicted_total else None,
                "accuracy": (
                    round(routing_correct_count / predicted_total, 6)
                    if predicted_total
                    else None
                ),
                "confusion": [
                    {
                        "true_question_type": true_type,
                        "predicted_question_type": predicted_type,
                        "count": count,
                    }
                    for (true_type, predicted_type), count in sorted(
                        route_confusion.items()
                    )
                ],
            },
        },
        "systems": systems,
        "comparisons_percentage_points": {
            "binary_router_end_to_end_minus_hvm": pp_delta(router_overall, hvm_overall),
            "binary_router_end_to_end_minus_episodic": pp_delta(
                router_overall, episodic_overall
            ),
            "router_fixed_output_selection_minus_hvm": pp_delta(
                fixed_overall, hvm_overall
            ),
            "type_oracle_minus_hvm": pp_delta(oracle_overall, hvm_overall),
            "type_oracle_minus_episodic": pp_delta(oracle_overall, episodic_overall),
            "best_of_two_oracle_minus_hvm": pp_delta(best_overall, hvm_overall),
        },
        "per_question": per_question,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(output), encoding="utf-8")

    print(f"Saved apples-to-apples text report: {args.output}")
    print(f"Canonical questions: {len(canonical_ids)}")
    print()
    print("System                           Correct/Total   Accuracy")
    print("--------------------------------------------------------")
    for name in (
        "episodic",
        "hvm",
        "binary_router_end_to_end",
        "router_fixed_output_selection",
        "type_oracle",
        "best_of_two_oracle",
    ):
        metric = systems[name]["overall"]
        accuracy = metric["accuracy"]
        accuracy_text = "n/a" if accuracy is None else f"{accuracy * 100:6.2f}%"
        print(
            f"{name:32s} {metric['correct']:>3}/{metric['count']:<3}      {accuracy_text}"
        )

    print()
    print("By question type:")
    for qtype in sorted(true_type_counts):
        print(f"  {qtype}:")
        for name in (
            "episodic",
            "hvm",
            "binary_router_end_to_end",
            "router_fixed_output_selection",
            "type_oracle",
        ):
            metric = systems[name]["by_question_type"][qtype]
            print(
                f"    {name:30s} {metric['correct']}/{metric['count']} "
                f"({metric['accuracy'] * 100:.2f}%)"
            )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ComparisonError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
