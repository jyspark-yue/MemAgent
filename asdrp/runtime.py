#############################################################################
# File: runtime.py
#
# Description:
#   Connects native OpenAI Responses calls, local token counting, shared
#   embeddings, retries, and task-level usage accounting. Each task
#   receives its own runtime and usage ledger.
#
#   - Tracks memory, query, and per-question LLM and embedding usage
#     under concurrency.
#   - Separates short and long input, cache-read, cache-write, and
#     output tokens.
#   - Prices each request from its actual token category and context
#     length.
#   - Counts, splits, and truncates text locally with tiktoken.
#   - Calls the native async Responses API for plain and system/user
#     prompts.
#   - Retries only short-lived OpenAI and network failures.
#   - Reads native Responses usage fields and falls back to clearly
#     marked local estimates.
#   - Parses plain, fenced, or lightly wrapped JSON model output.
#############################################################################


from __future__ import annotations

import asyncio
import contextvars
import json
import random
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import tiktoken
from openai import AsyncOpenAI

from asdrp.eval_schemas import Pricing

if TYPE_CHECKING:
    # This import is only needed by type checkers, so normal imports stay light.
    from asdrp.openai_embedder import OpenAIEmbedder

# Keep the return type of any operation passed through the retry helper.
T = TypeVar("T")


@dataclass(slots=True)
class UsageBucket:
    # One bucket holds totals for either memory building or question answering.

    # LLM request counts.
    llm_requests: int = 0
    estimated_llm_requests: int = 0

    # Short-context LLM tokens.
    llm_input_tokens: int = 0
    llm_cached_input_tokens: int = 0
    llm_cache_write_tokens: int = 0
    llm_output_tokens: int = 0

    # Long-context LLM tokens.
    llm_long_input_tokens: int = 0
    llm_long_cached_input_tokens: int = 0
    llm_long_cache_write_tokens: int = 0
    llm_long_output_tokens: int = 0

    # LLM cost across both short- and long-context requests.
    llm_cost_usd: float = 0.0

    # Embedding workload and cost.
    embedding_requests: int = 0
    embedding_entry_count: int = 0
    embedding_input_tokens: int = 0
    embedding_cost_usd: float = 0.0

    # Keep this internally for task-level diagnostics.
    # It will no longer be included in the run summary.
    embedding_runtime_seconds: float = 0.0


class UsageLedger:
    # The ledger keeps shared task totals safe while questions run at once.

    def __init__(self, pricing: Pricing) -> None:
        self.memory = UsageBucket()  # Cost of loading the task context.
        self.query = UsageBucket()  # Cost of answering every task question.
        self._pricing = pricing  # Rates used when a call finishes.
        self._lock = asyncio.Lock()  # Guards every shared total.

        # Each async question gets its own current scope ID.
        self._scope: contextvars.ContextVar[str | None] = contextvars.ContextVar(
            f"usage_scope_{id(self)}",
            default=None,
        )
        # Store query totals under the question that caused them.
        self._query_by_scope: dict[str, UsageBucket] = {}

    @asynccontextmanager
    async def scope(self, scope_id: str) -> AsyncIterator[None]:
        # Mark every call in this block as work for one question.

        # Save the unnecessary scope so nested work can restore it on exit.
        token = self._scope.set(scope_id)
        try:
            yield
        finally:
            self._scope.reset(token)

    async def add_embedding(
        self,
        phase: str,
        token_count: int,
        *,
        request_count: int = 1,
        text_count: int = 0,
        runtime_seconds: float = 0.0,
    ) -> None:
        # Record one successful embedding batch and its API cost.

        async with self._lock:
            # Add the call to its memory or query phase.
            bucket = self.memory if phase == "memory" else self.query

            embedding_cost_usd = (
                token_count * self._pricing.embedding_input_per_million / 1_000_000
            )

            self._add_embedding_to_bucket(
                bucket,
                token_count,
                request_count,
                text_count,
                runtime_seconds,
                embedding_cost_usd,
            )
            # Query calls also belong to one question when a scope is active.
            scoped = self._scoped_bucket(phase)
            if scoped is not None:
                self._add_embedding_to_bucket(
                    scoped,
                    token_count,
                    request_count,
                    text_count,
                    runtime_seconds,
                    embedding_cost_usd,
                )

    async def add_llm(
        self,
        phase: str,
        input_tokens: int,
        cached_input_tokens: int,
        cache_write_tokens: int,
        output_tokens: int,
        estimated: bool,
    ) -> None:
        # Record one successful LLM request and its exact per-request cost.

        if (
            input_tokens < 0
            or cached_input_tokens < 0
            or cache_write_tokens < 0
            or output_tokens < 0
        ):
            raise ValueError("LLM token counts cannot be negative")
        if cached_input_tokens + cache_write_tokens > input_tokens:
            raise ValueError(
                "cached input and cache-write tokens cannot exceed total input tokens"
            )

        # Cache reads and writes are both part of the provider's input total.
        uncached_input_tokens = input_tokens - cached_input_tokens - cache_write_tokens
        input_rate, cached_rate, cache_write_rate, output_rate = (
            self._pricing.llm_rates(input_tokens)
        )
        cost_usd = (
            uncached_input_tokens * input_rate
            + cached_input_tokens * cached_rate
            + cache_write_tokens * cache_write_rate
            + output_tokens * output_rate
        ) / 1_000_000
        long_context = self._pricing.uses_long_context(input_tokens)

        async with self._lock:
            # Update the full phase total first.
            bucket = self.memory if phase == "memory" else self.query
            self._add_llm_to_bucket(
                bucket,
                input_tokens=input_tokens,
                cached_input_tokens=cached_input_tokens,
                cache_write_tokens=cache_write_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                estimated=estimated,
                long_context=long_context,
            )

            # Copy query use into the current question total as well.
            scoped = self._scoped_bucket(phase)
            if scoped is not None:
                self._add_llm_to_bucket(
                    scoped,
                    input_tokens=input_tokens,
                    cached_input_tokens=cached_input_tokens,
                    cache_write_tokens=cache_write_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                    estimated=estimated,
                    long_context=long_context,
                )

    def scope_snapshot(self, scope_id: str) -> dict[str, Any]:
        # Return a plain dictionary that can be saved as JSON.

        # An unanswered or failed question receives an empty bucket.
        bucket = asdict(self._query_by_scope.get(scope_id, UsageBucket()))
        return {**bucket, "estimated_cost_usd": self._cost(bucket)}

    def snapshot(self) -> dict[str, Any]:
        # Build memory, query, and combined totals for the task report.

        memory = asdict(self.memory)  # Make a copy of the memory totals.
        query = asdict(self.query)  # Make a copy of the query totals.

        # Both dictionaries share the same fields, so they can be summed by key.
        combined = {key: memory.get(key, 0) + query.get(key, 0) for key in memory}
        return {
            "memory": {**memory, "estimated_cost_usd": self._cost(memory)},
            "query": {**query, "estimated_cost_usd": self._cost(query)},
            "combined": {**combined, "estimated_cost_usd": self._cost(combined)},
        }

    def _scoped_bucket(self, phase: str) -> UsageBucket | None:
        # Memory building belongs to the task, not to any one question.

        if phase != "query":
            return None
        # Read the question ID set by the surrounding scope.
        scope_id = self._scope.get()
        if scope_id is None:
            return None
        # Create the question bucket on its first recorded call.
        return self._query_by_scope.setdefault(scope_id, UsageBucket())

    @staticmethod
    def _add_embedding_to_bucket(
        bucket: UsageBucket,
        token_count: int,
        request_count: int,
        text_count: int,
        runtime_seconds: float,
        cost_usd: float,
    ) -> None:
        bucket.embedding_input_tokens += token_count
        bucket.embedding_requests += request_count
        bucket.embedding_entry_count += text_count
        bucket.embedding_runtime_seconds += runtime_seconds
        bucket.embedding_cost_usd += cost_usd

    @staticmethod
    def _add_llm_to_bucket(
        bucket: UsageBucket,
        *,
        input_tokens: int,
        cached_input_tokens: int,
        cache_write_tokens: int,
        output_tokens: int,
        cost_usd: float,
        estimated: bool,
        long_context: bool,
    ) -> None:
        # Keep the actual request counts independent of token category.
        bucket.llm_requests += 1
        bucket.estimated_llm_requests += int(estimated)

        if long_context:
            bucket.llm_long_input_tokens += input_tokens
            bucket.llm_long_cached_input_tokens += cached_input_tokens
            bucket.llm_long_cache_write_tokens += cache_write_tokens
            bucket.llm_long_output_tokens += output_tokens
        else:
            bucket.llm_input_tokens += input_tokens
            bucket.llm_cached_input_tokens += cached_input_tokens
            bucket.llm_cache_write_tokens += cache_write_tokens
            bucket.llm_output_tokens += output_tokens

        bucket.llm_cost_usd += cost_usd

    @staticmethod
    def _cost(bucket: dict[str, int | float]) -> float:
        # Both costs are recorded when each successful API call finishes.
        return float(bucket.get("llm_cost_usd", 0.0)) + float(
            bucket.get("embedding_cost_usd", 0.0)
        )


class TokenCounter:
    # This local tokenizer keeps packing checks free of API calls.

    def __init__(self, model: str) -> None:
        try:
            # Use the exact tokenizer when tiktoken knows the model name.
            self._encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # New model aliases may arrive before the local tiktoken release.
            self._encoding = tiktoken.get_encoding("o200k_base")

    def count(self, text: str) -> int:
        # Count the encoded IDs without storing them.

        return len(self._encoding.encode(text, disallowed_special=()))

    def truncate(self, text: str, max_tokens: int) -> str:
        # Cut on token boundaries so the result fits the real model limit.

        if max_tokens <= 0:
            return ""
        # Encode once, then return the original text when no cut is needed.
        tokens = self._encoding.encode(text, disallowed_special=())
        if len(tokens) <= max_tokens:
            return text
        return self._encoding.decode(tokens[:max_tokens])

    def split(self, text: str, max_tokens: int) -> list[str]:
        # Split text into complete, non-overlapping token ranges.

        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        # Encode once so each slice uses the same token positions.
        tokens = self._encoding.encode(text, disallowed_special=())
        return [
            self._encoding.decode(tokens[start : start + max_tokens])
            for start in range(0, len(tokens), max_tokens)
        ] or [""]


class OpenAIRuntime:
    # This object joins one task's answer model with the shared embedding client.

    def __init__(
        self,
        *,
        llm_model: str,
        embedder: OpenAIEmbedder,
        api_semaphore: asyncio.Semaphore,
        usage: UsageLedger,
        request_timeout: float = 120.0,
        retry_attempts: int = 8,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 20.0,
    ) -> None:
        if request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        if retry_attempts < 1:
            raise ValueError("retry_attempts must be at least 1")
        if retry_base_delay < 0 or retry_max_delay < 0:
            raise ValueError("retry delays cannot be negative")
        if retry_max_delay < retry_base_delay:
            raise ValueError("retry_max_delay cannot be less than retry_base_delay")

        self.llm_model = llm_model  # Model name written to the run settings.
        self.usage = usage  # Task-local token and cost totals.
        self._api_semaphore = api_semaphore  # Shared API call limit.
        self._retry_attempts = retry_attempts  # First call plus later attempts.
        self._retry_base_delay = retry_base_delay  # First retry wait.
        self._retry_max_delay = retry_max_delay  # Longest retry wait.
        self.llm_counter = TokenCounter(llm_model)  # Packs answer prompts.
        self._embedder = embedder  # Shared batched embedding service.

        # Use the native async OpenAI client. This runtime owns retry behavior,
        # so SDK retries stay disabled to avoid nested retry loops.
        self._client = AsyncOpenAI(
            timeout=request_timeout,
            max_retries=0,
        )

    async def close(self) -> None:
        # Close the task-local OpenAI HTTP client and release its connections.
        await self._client.close()

    async def embed_many(
        self,
        texts: list[str],
        *,
        phase: str,
    ) -> list[list[float]]:
        # Embed all texts through the shared client and keep their order.

        if not texts:
            return []
        # Use the correct phase method so the call is clear in metrics.
        result = (
            await self._embedder.embed_documents(texts)
            if phase == "memory"
            else await self._embedder.embed_queries(texts)
        )
        # Record model batches, text count, tokens, and elapsed time together.
        await self.usage.add_embedding(
            phase,
            result.input_tokens,
            request_count=result.model_batches,
            text_count=result.text_count,
            runtime_seconds=result.elapsed_seconds,
        )
        return result.vectors

    async def embed_one(self, text: str, *, phase: str) -> list[float]:
        # Reuse the batch path so one-text calls get the same accounting.

        return (await self.embed_many([text], phase=phase))[0]

    async def complete(self, prompt: str, *, phase: str) -> str:
        # Send one plain prompt through OpenAI's native Responses API.

        async def call() -> Any:
            # Hold one shared API slot only while the request is active.
            async with self._api_semaphore:
                return await self._client.responses.create(
                    model=self.llm_model,
                    input=prompt,
                )

        response = await self._retry(call)
        text = self._response_text(response)
        (
            input_tokens,
            cached_input_tokens,
            cache_write_tokens,
            output_tokens,
            estimated,
        ) = self._extract_usage(response, prompt, text)
        await self.usage.add_llm(
            phase,
            input_tokens,
            cached_input_tokens,
            cache_write_tokens,
            output_tokens,
            estimated,
        )
        return text

    async def chat(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        phase: str,
    ) -> str:
        # Keep system instructions separate without a framework message class.

        async def call() -> Any:
            # The Responses API accepts instructions separately from user input.
            async with self._api_semaphore:
                return await self._client.responses.create(
                    model=self.llm_model,
                    instructions=system_prompt,
                    input=user_prompt,
                )

        response = await self._retry(call)
        text = self._response_text(response)
        # This plain form is used only if provider usage is unexpectedly absent.
        accounting_prompt = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}"
        (
            input_tokens,
            cached_input_tokens,
            cache_write_tokens,
            output_tokens,
            estimated,
        ) = self._extract_usage(response, accounting_prompt, text)
        await self.usage.add_llm(
            phase,
            input_tokens,
            cached_input_tokens,
            cache_write_tokens,
            output_tokens,
            estimated,
        )
        return text

    @staticmethod
    def _response_text(response: Any) -> str:
        # Native Responses objects provide output_text as the assembled text.
        output_text = getattr(response, "output_text", None)
        if output_text is not None:
            text = str(output_text).strip()
            if text:
                return text

        # Keep a small compatibility path for dictionary-shaped mocked responses.
        if isinstance(response, dict) and response.get("output_text") is not None:
            text = str(response["output_text"]).strip()
            if text:
                return text

        raise ValueError("OpenAI response did not include non-empty output_text")

    async def _retry(self, operation: Callable[[], Awaitable[T]]) -> T:
        # Retry only short-lived API and network failures.

        last_error: Exception | None = None
        for attempt in range(self._retry_attempts):
            try:
                return await operation()
            except Exception as error:
                last_error = error
                if not self._is_retryable(error) or attempt + 1 >= self._retry_attempts:
                    raise
                # Double the wait after each failure, up to the set cap.
                delay = min(
                    self._retry_base_delay * (2**attempt),
                    self._retry_max_delay,
                )
                # Jitter prevents all workers from retrying at the same moment.
                await asyncio.sleep(delay * random.uniform(0.8, 1.2))
        assert last_error is not None
        raise last_error

    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        # First check standard HTTP codes for short-lived failures.

        status = getattr(error, "status_code", None) or getattr(error, "status", None)
        if status in {408, 409, 429, 500, 502, 503, 504}:
            return True
        # Some transport errors only provide a useful text message.
        message = str(error).lower()
        return any(
            token in message
            for token in (
                "rate limit",
                "timeout",
                "timed out",
                "connection reset",
                "connection error",
                "temporarily unavailable",
                "server error",
            )
        )

    def _extract_usage(
        self,
        response: Any,
        prompt: str,
        output: str,
    ) -> tuple[int, int, int, int, bool]:
        # Read input, cache reads, cache writes, and output from Responses usage.

        usage = (
            response.get("usage")
            if isinstance(response, dict)
            else getattr(response, "usage", None)
        )

        def field(obj: Any, *names: str) -> Any:
            # Read the first present dictionary key or object attribute.
            if obj is None:
                return None
            for name in names:
                if isinstance(obj, dict):
                    if obj.get(name) is not None:
                        return obj[name]
                else:
                    value = getattr(obj, name, None)
                    if value is not None:
                        return value
            return None

        def integer(obj: Any, *names: str) -> int | None:
            value = field(obj, *names)
            return int(value) if value is not None else None

        if usage is not None:
            input_tokens = integer(usage, "input_tokens")
            output_tokens = integer(usage, "output_tokens")
            details = field(usage, "input_tokens_details")
            cached_input_tokens = integer(details, "cached_tokens")
            cache_write_tokens = integer(details, "cache_write_tokens")

            # Keep flattened fields for simple test doubles and serialized usage.
            if cached_input_tokens is None:
                cached_input_tokens = integer(
                    usage,
                    "cached_input_tokens",
                    "cached_tokens",
                )
            if cache_write_tokens is None:
                cache_write_tokens = integer(usage, "cache_write_tokens")

            if input_tokens is not None and output_tokens is not None:
                cached_input_tokens = cached_input_tokens or 0
                cache_write_tokens = cache_write_tokens or 0
                if (
                    input_tokens < 0
                    or cached_input_tokens < 0
                    or cache_write_tokens < 0
                    or output_tokens < 0
                ):
                    raise ValueError("API usage returned a negative token count")
                if cached_input_tokens + cache_write_tokens > input_tokens:
                    raise ValueError(
                        "API usage returned more cached/cache-write tokens than "
                        "total input tokens"
                    )
                return (
                    input_tokens,
                    cached_input_tokens,
                    cache_write_tokens,
                    output_tokens,
                    False,
                )

        # Local estimates cannot know how many prompt tokens were read or written.
        return (
            self.llm_counter.count(prompt),
            0,
            0,
            self.llm_counter.count(output),
            True,
        )


def parse_json_payload(text: str) -> Any:
    # Parse JSON even when the model adds a code fence or short lead-in.

    cleaned = text.strip()  # Remove outer whitespace before checking wrappers.
    if cleaned.startswith("```"):
        # Remove one outer Markdown fence while keeping the JSON itself.
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)  # Closing fence.
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Find the widest array or object when a model adds text around it.
        array_start, array_end = cleaned.find("["), cleaned.rfind("]")
        object_start, object_end = cleaned.find("{"), cleaned.rfind("}")
        candidates = []  # Possible JSON slices, checked from longest to shortest.
        if array_start >= 0 and array_end > array_start:
            candidates.append(cleaned[array_start : array_end + 1])
        if object_start >= 0 and object_end > object_start:
            candidates.append(cleaned[object_start : object_end + 1])
        # The larger slice is more likely to hold the full requested payload.
        for candidate in sorted(candidates, key=len, reverse=True):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        raise
