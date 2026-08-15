#############################################################################
# File: context_builder.py
#
# Description:
#   Builds benchmark context from one raw ecommerce conversation.
#
#   - Assigns stable synthetic customer IDs.
#   - Groups user/assistant exchanges into ordered synthetic session IDs.
#   - Formats each session into the context representation used by QA generation.
#############################################################################

from __future__ import annotations

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from .models import RawConversation


def customer_id_for_index(index_zero_based: int, prefix: str = "cust") -> str:
    return f"{prefix}_{index_zero_based + 1:05d}"


def build_context(raw: RawConversation, customer_id: str) -> tuple[str, list[str]]:
    """
    Convert one input JSONL conversation into the final context string.

    The input has one long message list. For evidence, we create synthetic session IDs
    by grouping each user+assistant exchange as one session. If there is an odd trailing
    message, it gets its own final session.
    """
    sessions: list[tuple[str, list[str]]] = []
    current: list[str] = []
    session_index = 1

    for msg in raw.messages:
        current.append(f"{msg.role.upper()}: {msg.content.strip()}")
        if msg.role == "assistant":
            sid = f"{customer_id}_s{session_index:03d}"
            sessions.append((sid, current))
            current = []
            session_index += 1

    if current:
        sid = f"{customer_id}_s{session_index:03d}"
        sessions.append((sid, current))

    parts = [f"Customer ID: {customer_id}", "Source: synthetic_ecommerce", ""]
    for sid, lines in sessions:
        parts.append(f"[Session {sid}]")
        parts.extend(lines)
        parts.append("")

    return "\n".join(parts).strip(), [sid for sid, _ in sessions]
