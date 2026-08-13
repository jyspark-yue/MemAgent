#############################################################################
# File: build_dataset.py
#
# Description:
#   Converts synthetic ecommerce conversations into schema-validated EcommerceMemEval rows.
#
#   - Formats conversation context and stable customer/question/session identifiers.
#   - Requests diverse QA types with an underrepresented-label balancing heuristic.
#   - Applies conservative grounding/label quality checks before accepting generated QA.
#   - Supports a deterministic no-API dry run for pipeline validation.
#   - Creates configured output directories and clears prior run files before generation.
#   - Writes accepted dataset rows and generation/validation failures incrementally.
#############################################################################

from __future__ import annotations

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .context_builder import build_context, customer_id_for_index
from .io_utils import append_jsonl, iter_jsonl, load_yaml
from .llm_client import LLMClient
from .models import (
    DatasetMetadata,
    DatasetRow,
    EvidenceItem,
    GenerationResult,
    QuestionType,
    RawConversation,
    model_json_schema_for_llm,
)
from .prompts import build_generation_prompt


def choose_desired_types(
    all_types: list[str], counts: Counter[str], k: int = 4
) -> list[str]:
    """Light balancing heuristic: keep asking the LLM for currently underrepresented labels."""
    return [
        qt
        for qt, _ in sorted(
            ((qt, counts[qt]) for qt in all_types), key=lambda x: (x[1], x[0])
        )[:k]
    ]


def make_qa_id(customer_id: str, qa_index_zero_based: int) -> str:
    return f"{customer_id}_q{qa_index_zero_based + 1:03d}"


def dry_run_generation(
    context: str, session_ids: list[str], min_qas: int
) -> dict[str, Any]:
    """
    Deterministic fake generation for pipeline testing only.
    It proves formatting/validation works; it is not a real benchmark output.
    """
    first_sid = session_ids[0] if session_ids else ""
    qas = [
        {
            "question": "What item did the customer discuss in this shopping conversation?",
            "answers": [
                "The answer should be derived from the provided ecommerce conversation."
            ],
            "question_type": "purchase_history_recall",
            "evidence": {
                "session_ids": [first_sid] if first_sid else [],
                "facts": ["Smoke-test QA generated without calling an LLM."],
            },
        },
        {
            "question": "What customer preference should the assistant remember from this conversation?",
            "answers": ["The preference should be read from the context."],
            "question_type": "preference_recall",
            "evidence": {
                "session_ids": [first_sid] if first_sid else [],
                "facts": ["Smoke-test QA generated without calling an LLM."],
            },
        },
        {
            "question": "What is the customer's ring size?",
            "answers": ["The context does not state the customer's ring size."],
            "question_type": "abstention",
            "evidence": {
                "session_ids": [],
                "facts": ["The context does not mention the customer's ring size."],
            },
        },
    ]
    return {"qas": qas[: max(1, min_qas)]}


def _contains_any(text: str, terms: set[str]) -> bool:
    text = text.lower()
    return any(term in text for term in terms)


PURCHASE_TERMS = {
    "purchased",
    "bought",
    "checked out",
    "completed checkout",
    "completed the checkout",
    "completed the purchase",
    "finalized the order",
    "order was finalized",
    "order has been finalized",
    "placed the order",
    "order confirmed",
    "purchase confirmed",
    "processed your order",
    "processed the order",
    "processed your purchase",
}

CART_WISHLIST_TERMS = {
    "cart",
    "wishlist",
    "wish list",
    "saved",
    "recently viewed",
    "viewed",
    "reserved",
    "added",
    "add it",
    "add them",
}

RETURN_SUPPORT_TERMS = {
    "return",
    "returned",
    "refund",
    "exchange",
    "defect",
    "defective",
    "support",
    "damaged",
    "flaw",
    "failed",
    "missing",
    "claim",
    "replacement",
}

TEMPORAL_TERMS = {
    "current",
    "currently",
    "now",
    "still",
    "latest",
    "new",
    "updated",
    "recent",
    "available",
    "in stock",
    "restock",
    "restocked",
    "promotion",
    "promo",
    "discount",
    "sale",
    "shipping",
    "delivery",
    "arrive",
    "arrival",
    "eta",
    "next week",
    "this week",
    "today",
    "tomorrow",
}

CONFLICT_TERMS = {
    "instead",
    "not",
    "avoid",
    "rather than",
    "last time",
    "before",
    "previous",
    "unnecessary",
    "new",
    "same",
    "duplicate",
    "duplicating",
    "mismatch",
    "wrong",
    "too small",
    "too tight",
    "too loose",
    "returned",
    "issue",
    "problem",
    "stick with",
    "switch",
    "changed",
    "updated",
    "correct",
}

RECOMMENDATION_TERMS = {
    "recommend",
    "suggest",
    "what kind",
    "which option",
    "best",
    "similar",
    "based on",
    "fit",
    "good option",
    "alternative",
}

ABSTENTION_ANSWER_TERMS = {
    "context does not",
    "does not provide",
    "not stated",
    "not mention",
    "not specified",
    "no information",
    "cannot determine",
}


def qa_quality_errors(qa: Any, context: str, valid_session_ids: set[str]) -> list[str]:
    """
    Conservative quality gate.
    It catches the most common label/answer failures without making the pipeline complex.
    """
    errors: list[str] = []

    q = qa.question.strip()
    q_lower = q.lower()
    a_text = " ".join(qa.answers).lower()
    facts_text = " ".join(qa.evidence.facts).lower()
    evidence_text = f"{q_lower} {a_text} {facts_text}"

    # Basic formatting.
    if len(q) < 12:
        errors.append("Question is too short")
    if len(q.split()) < 4:
        errors.append("Question is too vague")
    if "question_type" in q_lower or "preference_recall" in q_lower:
        errors.append("Question exposes benchmark label/style")
    if len(qa.answers) > 5:
        errors.append("Too many answer aliases")

    # Evidence validity.
    if qa.question_type != QuestionType.abstention:
        bad = [sid for sid in qa.evidence.session_ids if sid not in valid_session_ids]
        if bad:
            errors.append(f"Invalid evidence session IDs: {bad}")
        if not qa.evidence.session_ids:
            errors.append("Non-abstention QA has no evidence session_ids")
        if not qa.evidence.facts:
            errors.append("Non-abstention QA has no evidence facts")
    else:
        if qa.evidence.session_ids:
            errors.append("Abstention should not cite session_ids")
        if not _contains_any(a_text, ABSTENTION_ANSWER_TERMS):
            errors.append(
                "Abstention answer does not explicitly say the context lacks the detail"
            )

    # Label-specific checks.
    qt = qa.question_type

    if qt == QuestionType.purchase_history_recall:
        if not _contains_any(evidence_text, PURCHASE_TERMS):
            errors.append(
                "purchase_history_recall without clear purchase/order/finalization evidence"
            )

    elif qt == QuestionType.cart_or_wishlist_recall:
        if not _contains_any(evidence_text, CART_WISHLIST_TERMS):
            errors.append(
                "cart_or_wishlist_recall without cart/wishlist/saved/reserved evidence"
            )

    elif qt == QuestionType.return_support_recall:
        if not _contains_any(evidence_text, RETURN_SUPPORT_TERMS):
            errors.append(
                "return_support_recall without return/refund/warranty/defect/support evidence"
            )

    elif qt == QuestionType.temporal_update:
        if not _contains_any(evidence_text, TEMPORAL_TERMS):
            errors.append(
                "temporal_update without current/latest/stock/promo/shipping/restock evidence"
            )

    elif qt == QuestionType.constraint_recall:
        # Discount/shipping-only questions are usually temporal, not constraint.
        if _contains_any(
            q_lower,
            {
                "discount",
                "promo",
                "promotion",
                "sale",
                "shipping",
                "delivery",
                "arrive",
                "arrival",
            },
        ):
            errors.append(
                "constraint_recall appears to be discount/shipping/current-state question"
            )

    elif qt == QuestionType.conflict_resolution:
        if not _contains_any(evidence_text, CONFLICT_TERMS):
            errors.append(
                "conflict_resolution without clear conflict/correction/duplicate/stale-vs-current evidence"
            )

    elif qt == QuestionType.multi_session_synthesis:
        if len(set(qa.evidence.session_ids)) < 2:
            errors.append("multi_session_synthesis should cite at least two sessions")

    elif qt == QuestionType.recommendation_from_memory:
        if not _contains_any(q_lower, RECOMMENDATION_TERMS):
            errors.append(
                "recommendation_from_memory question does not ask for a recommendation"
            )
        if _contains_any(a_text, ABSTENTION_ANSWER_TERMS):
            errors.append("recommendation_from_memory answer is actually an abstention")
        if len(qa.answers) == 1 and len(qa.answers[0].split()) <= 3:
            errors.append(
                "recommendation_from_memory answer is too item-name-only; should include remembered basis"
            )

    elif qt == QuestionType.preference_recall:
        # If it only asks "which item" and not an attribute/taste, it is usually not preference.
        if q_lower.startswith(
            ("which item", "what item", "which brand", "which product")
        ):
            errors.append("preference_recall appears to be ordinary product recall")

    return errors


def validate_generated_qas(
    result: GenerationResult, context: str, session_ids: list[str]
) -> list[str]:
    valid_session_ids = set(session_ids)
    all_errors: list[str] = []

    seen_questions: set[str] = set()
    for i, qa in enumerate(result.qas):
        qa_key = qa.question.strip().lower()
        if qa_key in seen_questions:
            all_errors.append(f"q{i + 1}: duplicate question")
        seen_questions.add(qa_key)

        for err in qa_quality_errors(qa, context, valid_session_ids):
            all_errors.append(f"q{i + 1}: {err}")

    # Encourage at least one abstention if enough QAs were requested.
    if len(result.qas) >= 4 and not any(
        qa.question_type == QuestionType.abstention for qa in result.qas
    ):
        all_errors.append("missing abstention question")

    return all_errors


def build_dataset_row(
    *,
    raw_obj: dict[str, Any],
    row_index_zero_based: int,
    cfg: dict[str, Any],
    llm: LLMClient | None,
    type_counts: Counter[str],
) -> DatasetRow:
    raw = RawConversation.model_validate(raw_obj)
    customer_id = customer_id_for_index(
        row_index_zero_based, cfg.get("customer_id_prefix", "cust")
    )
    context, session_ids = build_context(raw, customer_id)

    max_generation_attempts = int(cfg.get("max_generation_attempts", 4))
    last_errors: list[str] = []

    for attempt in range(max_generation_attempts):
        desired_types = choose_desired_types(cfg["question_types"], type_counts)

        if cfg.get("dry_run") or llm is None:
            generated_obj = dry_run_generation(
                context,
                session_ids,
                int(cfg["min_qas_per_conversation"]),
            )
            result = GenerationResult.model_validate(generated_obj)
            break
        else:
            messages = build_generation_prompt(
                context=context,
                session_ids=session_ids,
                min_qas=int(cfg["min_qas_per_conversation"]),
                max_qas=int(cfg["max_qas_per_conversation"]),
                desired_types=desired_types,
            )

            if attempt > 0 and last_errors:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Regenerate the full QA JSON object from scratch. "
                            "Do not patch only the failing item.\n\n"
                            "The previous generation failed these quality checks:\n"
                            + "\n".join(f"- {e}" for e in last_errors[:20])
                            + "\n\nRequirements:\n"
                            "- Return only the corrected JSON object.\n"
                            "- Every non-abstention QA must cite at least one valid session_id.\n"
                            "- Every non-abstention QA must include concrete evidence facts from the cited sessions.\n"
                            "- Abstention answers must explicitly say the context does not provide the requested detail.\n"
                            "- Match question_type strictly to the label definitions."
                        ),
                    }
                )

            generated_obj = llm.generate_json(
                messages, schema=model_json_schema_for_llm()
            )

        result = GenerationResult.model_validate(generated_obj)

        max_qas = int(cfg["max_qas_per_conversation"])
        min_qas = int(cfg["min_qas_per_conversation"])
        candidate_qas = result.qas[:max_qas]

        if len(candidate_qas) < min_qas:
            last_errors = [
                f"Only generated {len(candidate_qas)} QA pairs; needed {min_qas}"
            ]
        else:
            trimmed_result = GenerationResult(qas=candidate_qas)
            last_errors = validate_generated_qas(trimmed_result, context, session_ids)

        if not last_errors:
            result = trimmed_result
            break
    else:
        raise ValueError(
            "Generated QA failed quality checks: " + "; ".join(last_errors[:12])
        )

    qas = result.qas

    if len(qas) < int(cfg["min_qas_per_conversation"]):
        raise ValueError(f"Only generated {len(qas)} QA pairs")

    questions: list[str] = []
    answers: list[list[str]] = []
    qa_pair_ids: list[str] = []
    question_types: list[QuestionType] = []
    evidence: list[EvidenceItem] = []

    valid_session_ids = set(session_ids)
    local_type_counts: Counter[str] = Counter()

    for i, qa in enumerate(qas):
        qa_id = make_qa_id(customer_id, i)
        ev = qa.evidence
        ev.qa_pair_id = qa_id

        if qa.question_type != QuestionType.abstention:
            bad = [sid for sid in ev.session_ids if sid not in valid_session_ids]
            if bad:
                raise ValueError(f"Invalid session IDs for {qa_id}: {bad}")
            if not ev.session_ids:
                raise ValueError(
                    f"Non-abstention QA {qa_id} has no evidence session_ids"
                )
            if not ev.facts:
                raise ValueError(f"Non-abstention QA {qa_id} has no evidence facts")
        else:
            # Abstention questions should not cite evidence sessions as support.
            ev.session_ids = []

        questions.append(qa.question)
        answers.append(qa.answers)
        qa_pair_ids.append(qa_id)
        question_types.append(qa.question_type)
        evidence.append(ev)
        local_type_counts[str(qa.question_type)] += 1

    metadata = DatasetMetadata(
        customer_id=customer_id,
        source=cfg.get("source", "synthetic_ecommerce"),
        qa_pair_ids=qa_pair_ids,
        question_types=question_types,
        evidence=evidence,
    )

    row = DatasetRow(
        context=context,
        questions=questions,
        answers=answers,
        metadata=metadata,
    )

    type_counts.update(local_type_counts)

    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="utils/generation.yaml")
    parser.add_argument("--input-jsonl", default=None)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--error-jsonl", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.input_jsonl:
        cfg["input_jsonl"] = args.input_jsonl
    if args.output_jsonl:
        cfg["output_jsonl"] = args.output_jsonl
    if args.error_jsonl:
        cfg["error_jsonl"] = args.error_jsonl
    if args.limit is not None:
        cfg["limit"] = args.limit
    if args.dry_run:
        cfg["dry_run"] = True

    random.seed(int(cfg.get("seed", 42)))

    input_path = Path(cfg["input_jsonl"])
    output_path = Path(cfg["output_jsonl"])
    error_path = Path(cfg.get("error_jsonl", "ecommerce_memeval_errors.jsonl"))
    output_path.write_text("", encoding="utf-8")
    error_path.write_text("", encoding="utf-8")

    llm = None
    if not cfg.get("dry_run"):
        llm = LLMClient(
            provider=cfg.get("provider", "openai"),
            model=cfg["model"],
            temperature=float(cfg.get("temperature", 0.2)),
        )

    type_counts: Counter[str] = Counter()
    total = 0
    failures = 0
    limit = cfg.get("limit")

    rows = list(iter_jsonl(input_path))
    if limit is not None:
        rows = rows[: int(limit)]

    for line_num, raw_obj in tqdm(rows, desc="Building ecommerce_memeval_generator"):
        try:
            row = build_dataset_row(
                raw_obj=raw_obj,
                row_index_zero_based=total,
                cfg=cfg,
                llm=llm,
                type_counts=type_counts,
            )
            append_jsonl(output_path, row.model_dump(mode="json"))
        except Exception as exc:
            failures += 1
            append_jsonl(
                error_path,
                {"line_num": line_num, "error": repr(exc), "raw_obj": raw_obj},
            )
        finally:
            total += 1

    print(f"Wrote: {output_path}")
    print(f"Errors: {failures} written to {error_path}")
    print("Question type counts:")
    for qt, count in sorted(type_counts.items()):
        print(f"  {qt}: {count}")


if __name__ == "__main__":
    main()
