#############################################################################
# File: qdrant_retry.py
#
# Description:
#   Keeps Qdrant retry behavior in one small module. It retries only
#   short-lived HTTP and network failures and leaves permanent errors
#   unchanged.
#
#   - Reads status codes from direct exceptions and nested responses.
#   - Recognizes common transient Qdrant and transport error messages.
#   - Validates retry counts and delay limits before starting work.
#   - Uses capped exponential backoff with small random jitter.
#   - Stops immediately when the evaluator cancels a task.
#   - Logs each retry with its operation, status, error type, and delay.
#############################################################################

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

# Keep the wrapped operation's return type when it passes through this helper.
T = TypeVar("T")

# These HTTP errors can clear up when the same request runs again.
RETRYABLE_QDRANT_STATUS_CODES = {
    408,
    409,
    429,
    500,
    502,
    503,
    504,
}

# Qdrant and its HTTP clients do not always expose a status code.
# These text fragments cover the same short-lived network failures.
RETRYABLE_QDRANT_MESSAGE_FRAGMENTS = (
    "request timeout",
    "timed out",
    "timeout",
    "connection reset",
    "connection refused",
    "connection aborted",
    "connection error",
    "all connection attempts failed",
    "temporarily unavailable",
    "service unavailable",
    "server disconnected",
    "broken pipe",
)


def _exception_chain(error: BaseException) -> tuple[BaseException, ...]:
    # Qdrant often wraps httpx/httpcore transport failures inside an outer
    # ResponseHandlingException whose own message is blank. Walk the common
    # wrapper attributes as well as Python's normal cause/context chain so the
    # retry decision sees the real network error.

    pending: list[BaseException] = [error]
    seen: set[int] = set()
    chain: list[BaseException] = []

    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        chain.append(current)

        for attribute in (
            "__cause__",
            "__context__",
            "source",
            "exception",
            "exc",
            "original_error",
        ):
            nested = getattr(current, attribute, None)
            if isinstance(nested, BaseException) and id(nested) not in seen:
                pending.append(nested)

    return tuple(chain)


def _qdrant_status_code(error: BaseException) -> int | None:
    # Read status codes from the outer exception and any wrapped transport/API
    # exceptions. This preserves the unnecessary behavior while handling Qdrant's
    # ResponseHandlingException wrappers correctly.

    for current in _exception_chain(error):
        for attribute in ("status_code", "status"):
            status = getattr(current, attribute, None)
            if isinstance(status, int):
                return status

        response = getattr(current, "response", None)
        if response is not None:
            for attribute in ("status_code", "status"):
                status = getattr(response, attribute, None)
                if isinstance(status, int):
                    return status

    return None


def _is_retryable_qdrant_error(error: BaseException) -> bool:
    # A known short-lived HTTP status is enough to allow another try.

    status = _qdrant_status_code(error)
    if status in RETRYABLE_QDRANT_STATUS_CODES:
        return True

    # Qdrant may wrap httpx.ReadError / ConnectError in an exception with an
    # empty string representation, so inspect every nested exception's message
    # and class name rather than only the outer wrapper.
    retryable_class_fragments = (
        "connecterror",
        "connecttimeout",
        "networkerror",
        "pooltimeout",
        "readerror",
        "readtimeout",
        "remoteprotocolerror",
        "writeerror",
        "writetimeout",
    )

    for current in _exception_chain(error):
        class_name = type(current).__name__.casefold()
        if any(fragment in class_name for fragment in retryable_class_fragments):
            return True

        message = str(current).casefold()
        if any(fragment in message for fragment in RETRYABLE_QDRANT_MESSAGE_FRAGMENTS):
            return True

    return False


async def retry_qdrant(
    operation: Callable[[], Awaitable[T]],
    *,
    operation_name: str,
    attempts: int,
    base_delay: float,
    max_delay: float,
) -> T:
    # Run the request once, then retry only short-lived failures.
    # The attempt count includes the first request.

    # Reject bad limits before starting a network request.
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    if base_delay < 0:
        raise ValueError("base_delay cannot be negative")
    if max_delay < 0:
        raise ValueError("max_delay cannot be negative")

    # Keep the final error for the unreachable-loop safety check below.
    last_error: Exception | None = None

    # Add more delay after each failed request.
    for attempt_index in range(attempts):
        try:
            return await operation()
        except asyncio.CancelledError:
            # Stop at once when the evaluator shuts down.
            raise
        except Exception as error:
            # Record the error and check if another attempt is allowed.
            last_error = error
            is_final_attempt = attempt_index + 1 >= attempts

            if not _is_retryable_qdrant_error(error) or is_final_attempt:
                raise

            # Cap exponential backoff, then add small random jitter.
            delay = min(
                base_delay * (2**attempt_index),
                max_delay,
            )
            delay *= random.uniform(0.8, 1.2)

            # Include a status in the log when the client supplied one.
            status = _qdrant_status_code(error)
            status_text = f" status={status}" if status is not None else ""

            retry_number = attempt_index + 1  # Number of the retry about to run.
            maximum_retries = attempts - 1  # Total retries after the first call.

            print(
                f"[qdrant] {operation_name} retry "
                f"{retry_number}/{maximum_retries} in {delay:.1f}s:"
                f"{status_text} {type(error).__name__}: {error}",
                flush=True,
            )

            # Wait outside Qdrant so other async work can keep running.
            await asyncio.sleep(delay)

    # Normal control flow always returns or raises inside the loop.
    if last_error is not None:
        raise last_error

    raise RuntimeError(
        f"Qdrant operation {operation_name!r} ended without a result or error"
    )
