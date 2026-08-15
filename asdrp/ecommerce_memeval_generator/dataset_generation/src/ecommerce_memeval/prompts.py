#############################################################################
# File: prompts.py
#
# Description:
#   Defines EcommerceMemEval QA-generation labels, grounding rules, and prompts.
#
#   - Gives precise definitions for recall, update, conflict, synthesis, recommendation,
#     and abstention question types.
#   - Builds system/user messages that require realistic, grounded, non-redundant QA pairs.
#   - Constrains evidence to supplied session IDs and forbids live-store assumptions.
#############################################################################

from __future__ import annotations

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from .models import QuestionType

QUESTION_TYPE_DEFINITIONS = {
    "preference_recall": (
        "Ask about a remembered stable preference, taste, favored attribute, usual choice, "
        "or avoidable style preference. Do NOT use this for ordinary item names unless the "
        "question is about why the user liked/preferred that attribute."
    ),
    "constraint_recall": (
        "Ask about a hard shopping constraint that affects product selection: size, fit, capacity, budget, "
        "material, scent sensitivity, compatibility, safety requirement, comfort requirement, noise limit, "
        "allergy/sensitivity, or avoid-list item. Only use for remembered user constraints/preferences "
        "that affect product choice: material, scent, size, fit, color, allergy/sensitivity, noise level, "
        "compatibility, budget limit, comfort requirement, non-scratchy, wool-free, etc. Do not use for "
        "shipping, stock, discounts, promos, delivery estimates, or current availability."
    ),
    "purchase_history_recall": (
        "Ask about an item, variant, size, brand, quantity, price, or order detail that was actually "
        "bought, reordered, checked out, finalized, or previously purchased. Only use when the "
        "context explicitly says the user purchased, ordered, checked out, completed checkout, "
        "finalized an order, or received order confirmation. Do not use for wishlist, cart, "
        "saved items, recommendations, or "
        "thinking of reordering."
        ""
    ),
    "cart_or_wishlist_recall": (
        "Ask about something saved, viewed, wishlisted, reserved, or added to cart. Use this when checkout "
        "or purchase is not explicitly completed. Use for saved items, wishlist items, or cart contents. "
        "Do not call these purchases unless checkout/finalization happened."
    ),
    "return_support_recall": (
        "Ask about a return, refund, exchange, defect, quality issue, warranty, support request, return window, "
        "or support status. Do NOT use this for discounts, shipping speed, or generic delivery estimates."
    ),
    "temporal_update": (
        "Ask for latest/current/updated state: current stock, current promotion, current discount, current cart, "
        "delivery estimate, shipping status, recently updated address/payment/default, restock, launch, or whether "
        "something is now available. Use only when the user asks about latest/current/updated information that is "
        "grounded in the conversation, such as a newer version, updated product option, restock, or changed preference. "
        "Do not force this label unless there is clear current/latest/update evidence."
    ),
    "conflict_resolution": (
        "Ask something that requires resolving an unnecessary-vs-new, stale-vs-current, rejected-vs-selected, duplicate-vs-distinct, "
        "or corrected fact. Only use this when the context contains an actual conflict, correction, uncertainty, duplicate risk, "
        "or prior failure that must be avoided."
    ),
    "multi_session_synthesis": (
        "Ask for a conclusion that combines facts from multiple sessions, not just a fact from one message. "
        "The answer should require at least two distinct pieces of evidence, such as selected item + reason, "
        "cart contents + discounts, unnecessary issue + new selected fix, or recipient profile + final recommendation. "
        "Must combine evidence from at least two different sessions. If only one session is available, do not generate this label."
    ),
    "recommendation_from_memory": (
        "Ask for a recommendation grounded in remembered preferences, constraints, prior failures, recipient needs, "
        "or shopping history. The answer should usually describe the kind of item to recommend and why, not merely repeat "
        "an item already added to cart. If the context does not contain enough evidence to recommend, use abstention instead."
    ),
    "abstention": (
        "Ask a plausible ecommerce memory question where the answer is not stated in the context. The answer must explicitly "
        "say the context does not provide the detail, and if useful, mention what related detail is known. The answer "
        "must still be non-empty and explain that the context does not provide the information."
    ),
}


def build_generation_prompt(
    *,
    context: str,
    session_ids: list[str],
    min_qas: int,
    max_qas: int,
    desired_types: list[str],
) -> list[dict[str, str]]:
    allowed = [qt.value for qt in QuestionType]
    type_block = "\n".join(f"- {k}: {v}" for k, v in QUESTION_TYPE_DEFINITIONS.items())
    desired = ", ".join(desired_types) if desired_types else "any suitable mix"
    sessions = ", ".join(session_ids)

    system = """
        You generate PhD-conference-quality ecommerce long-term-memory benchmark questions.
    
        Your job is NOT to make easy fact-extraction only. Your job is to create realistic future user questions that test whether an assistant can remember, update, distinguish, and abstain from ecommerce memory.
        
        Return only valid JSON matching the requested schema. Do not include markdown. Do not invent facts.
    """

    user = f"""
        Create as many high-quality QA pairs as the context genuinely supports, up to {max_qas} QA pairs.
        You must generate at least {min_qas} QA pairs.
        Do not stop at {min_qas} merely because the minimum is satisfied.
        Prefer generating closer to {max_qas} when the context supports distinct, grounded, non-redundant questions.
        Generate fewer than {max_qas} only when additional questions would be repetitive, weak, ungrounded, or label-ambiguous.
        
        Prefer these underrepresented question types only if the context genuinely supports them:
        {desired}
        
        Allowed question types:
        {allowed}
        
        Question type definitions:
        {type_block}
        
        Strict quality rules:
        - Questions must sound like realistic ecommerce follow-up questions a customer might ask later.
        - Do not ask generic benchmark-style questions like "What preference should be remembered?"
        - Do not ask multiple unrelated things in one question.
        - Avoid yes/no questions unless the task naturally requires checking a state, duplicate, omission, or abstention.
        - Do not label ordinary item recall as preference_recall.
        - Do not label discounts, stock, shipping, delivery, restocks, or current availability as constraint_recall; use temporal_update.
        - Use purchase_history_recall only if the item was actually purchased, reordered, checked out, finalized, or previously bought.
        - Use cart_or_wishlist_recall for saved, wishlisted, reserved, viewed, or added-to-cart items that were not clearly purchased.
        - Use return_support_recall only for returns, refunds, exchanges, warranties, defects, prior issues, or support outcomes.
        - Use conflict_resolution only if there is a real conflict/correction/duplicate/stale-vs-current issue.
        - Use multi_session_synthesis only when answering requires combining evidence from at least two sessions.
        - Use recommendation_from_memory only when the answer uses remembered preferences/constraints/history to recommend. Do not simply repeat an item already added to cart unless you explain the remembered basis.
        - Include at least one abstention question when possible.
        - Abstention questions should be tempting but unanswerable from the context, such as exact price, address, payment method, serial number, size, warranty, return status, or quantity when missing.
        - Abstention answers must say the context does not provide the detail. If related information is known, include it.
        
        Do not generate QA pairs whose answer would require live store lookup after the conversation, such as the store's current price, current stock, current shipping speed, current cart total, current return policy, or current warranty availability.
        It is allowed to ask about the latest state stated inside the provided context, such as the last-mentioned stock status, discount, shipping estimate, restock update, or availability, as long as the answer is grounded only in the conversation.
        
        Answer rules:
        - answers must be a list of acceptable answer strings.
        - Include concise aliases when useful, e.g. ["navy", "navy blue"] or ["size 8", "8"].
        - Do not include overly broad answers that mix background facts with the actual answer.
        - Keep answers short but complete.
        - For discounts that stack, do not add percentages unless the context explicitly states they combine additively. Say exactly what discounts apply.
        
        Evidence requirements:
        - For every non-abstention QA, cite one or more exact session IDs from this list:
        {sessions}
        - Evidence facts must be grounded in the context.
        - For abstention, use session_ids: [] and facts explaining that the requested detail is not stated.
        - Evidence facts should not be generic; they should justify the answer.
        
        Difficulty target:
        - Prefer a balanced mix of easy, medium, and hard memory questions.
        - At least one question should test current/latest state if the context contains stock, discount, shipping, restock, or updated availability.
        - At least one question should test cart/wishlist/purchase-state distinction if the context supports it.
        - At least one question should test avoidance of overclaiming or hallucination.
        
        Return this JSON object exactly:
        {{
          "qas": [
            {{
              "question": "...",
              "answers": ["...", "..."],
              "question_type": "preference_recall",
              "evidence": {{
                "session_ids": ["..."],
                "facts": ["..."]
              }}
            }}
          ]
        }}
        
        Context:
        {context}
    """.strip()

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
