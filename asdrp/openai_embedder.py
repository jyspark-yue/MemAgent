#############################################################################
# File: openai_embedder.py
#
# Description:
#   Provides one shared asynchronous OpenAI embedding service for
#   evaluator workers. It batches text-embedding-3-small requests while
#   preserving order, limits, retries, and usage measurements.
#
#   - Validates request, retry, batch, concurrency, token, and pacing
#     settings.
#   - Rejects blank or oversized inputs before making an API request.
#   - Packs requests by both string count and aggregate token count.
#   - Limits active batches and reserves tokens in a rolling one-minute
#     window.
#   - Retries short-lived failures with server-directed waits or
#     jittered backoff.
#   - Checks embedding count and vector width before returning results.
#   - Provides local count, split, and truncate helpers for adapter
#     reuse.
#############################################################################

from __future__ import annotations

import asyncio
import random
import re
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import tiktoken
from openai import AsyncOpenAI


@dataclass(frozen=True, slots=True)
class EmbeddingBatchResult:
    # Vectors and the measurements returned with one embedding job.

    vectors: list[list[float]]  # One vector for each input text.
    elapsed_seconds: float  # Total time for all batches.
    text_count: int  # Number of input strings.
    input_tokens: int  # Tokens sent to the embedding API.
    model_batches: int  # Number of API requests made.


class OpenAIEmbedder:
    # Shared OpenAI client that sends embeddings in safe batches.
    #
    # One client is shared across evaluator workers. Experimental isolation still
    # comes from each task's independent memory block and Qdrant collection.

    MODEL_NAME = "text-embedding-3-small"  # Only supported embedding model.
    DIMENSION = 1536  # Vector width returned by the model.
    MAX_SEQUENCE_TOKENS = 8192  # Maximum tokens in one input string.

    def __init__(
        self,
        *,
        model_name: str = MODEL_NAME,
        batch_size: int = 256,
        batch_token_limit: int = 240_000,
        max_parallel_batches: int = 4,
        tokens_per_minute: int = 900_000,
        rate_limit_buffer_seconds: float = 0.75,
        api_semaphore: asyncio.Semaphore | None = None,
        request_timeout: float = 120.0,
        retry_attempts: int = 8,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 20.0,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if batch_token_limit < 1:
            raise ValueError("batch_token_limit must be at least 1")
        if max_parallel_batches < 1:
            raise ValueError("max_parallel_batches must be at least 1")
        if tokens_per_minute < 1:
            raise ValueError("tokens_per_minute must be at least 1")
        if rate_limit_buffer_seconds < 0:
            raise ValueError("rate_limit_buffer_seconds cannot be negative")
        if request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        if retry_attempts < 1:
            raise ValueError("retry_attempts must be at least 1")
        if retry_base_delay < 0 or retry_max_delay < 0:
            raise ValueError("retry delays cannot be negative")
        if retry_max_delay < retry_base_delay:
            raise ValueError("retry_max_delay cannot be less than retry_base_delay")

        self.model_name = model_name  # Model sent with each request.
        self.batch_size = batch_size  # Maximum strings in one request.
        self.batch_token_limit = batch_token_limit  # Maximum tokens per request.
        self.device = "openai-api"  # Saved in the run summary.
        self._encoding = tiktoken.get_encoding("cl100k_base")  # Token counter.
        self._client = AsyncOpenAI(timeout=request_timeout, max_retries=0)
        self._batch_semaphore = asyncio.Semaphore(max_parallel_batches)
        self._tokens_per_minute = tokens_per_minute  # Local pacing limit.
        self._rate_limit_buffer_seconds = rate_limit_buffer_seconds  # Safety wait.
        self._token_window: deque[tuple[float, int]] = deque()  # Recent requests.
        self._token_window_total = 0  # Tokens reserved in the last minute.
        self._token_window_lock = asyncio.Lock()  # Protects the shared window.
        self._completed_batches = 0  # Progress count for the current process.
        self._completed_tokens = 0  # Progress tokens for the current process.
        self._progress_started = perf_counter()  # Start of throughput tracking.
        self._last_progress_print = self._progress_started  # Last status time.
        self._api_semaphore = api_semaphore  # Shared limit across all API calls.
        self._retry_attempts = retry_attempts  # Tries for one failed request.
        self._retry_base_delay = retry_base_delay  # First retry wait.
        self._retry_max_delay = retry_max_delay  # Longest retry wait.

    async def close(self) -> None:
        # Close the shared HTTP client.

        await self._client.close()

    async def embed_documents(self, texts: Sequence[str]) -> EmbeddingBatchResult:
        # Documents and queries use the same OpenAI endpoint.
        return await self._encode(texts)

    async def embed_queries(self, texts: Sequence[str]) -> EmbeddingBatchResult:
        # Keep this named method clear at each call site.
        return await self._encode(texts)

    async def _encode(self, texts: Sequence[str]) -> EmbeddingBatchResult:
        cleaned = [str(text).strip() for text in texts]  # Normalize input values.
        if not cleaned:
            return EmbeddingBatchResult([], 0.0, 0, 0, 0)
        if any(not text for text in cleaned):
            raise ValueError("Embedding inputs must not contain blank text")

        token_counts = self.count_many(cleaned)  # Size of every input string.
        maximum = max(token_counts)  # Largest individual input.
        if maximum > self.MAX_SEQUENCE_TOKENS:
            raise ValueError(
                "Embedding input exceeds the OpenAI 8192-token limit: "
                f"maximum observed input was {maximum} tokens"
            )

        batches = self._pack_batches(cleaned, token_counts)  # Safe API groups.
        started = perf_counter()  # Start of this complete embedding job.
        results = await asyncio.gather(*(self._embed_batch(batch) for batch in batches))
        vectors = [vector for batch_vectors in results for vector in batch_vectors]
        return EmbeddingBatchResult(
            vectors=vectors,
            elapsed_seconds=perf_counter() - started,
            text_count=len(cleaned),
            input_tokens=sum(token_counts),
            model_batches=len(batches),
        )

    def _pack_batches(
        self,
        texts: Sequence[str],
        token_counts: Sequence[int],
    ) -> list[list[str]]:
        # Pack requests by entry count and aggregate API token load.

        batches: list[list[str]] = []  # Finished API batches.
        current: list[str] = []  # Strings in the batch being filled.
        current_tokens = 0  # Tokens in the current batch.
        for text, token_count in zip(texts, token_counts, strict=True):
            if current and (
                len(current) >= self.batch_size
                or current_tokens + token_count > self.batch_token_limit
            ):
                batches.append(current)
                current = []  # Start a new API batch.
                current_tokens = 0  # Reset its token count.
            current.append(text)
            current_tokens += token_count
        if current:
            batches.append(current)
        return batches

    async def _embed_batch(self, batch: Sequence[str]) -> list[list[float]]:
        batch_tokens = sum(self.count_many(batch))  # Used for pacing and totals.
        async with self._batch_semaphore:
            response = await self._retry_create(batch, batch_tokens)  # API result.
        ordered = sorted(response.data, key=lambda item: item.index)  # Input order.
        vectors = [item.embedding for item in ordered]  # Plain vector lists.
        if len(vectors) != len(batch) or any(
            len(vector) != self.DIMENSION for vector in vectors
        ):
            raise RuntimeError("OpenAI returned an invalid embedding response shape")
        self._completed_batches += 1
        self._completed_tokens += batch_tokens
        now = perf_counter()  # Current point in progress tracking.
        if now - self._last_progress_print >= 15.0:
            elapsed = max(now - self._progress_started, 1e-9)  # Safe divisor.
            print(
                f"[embeddings] {self._completed_batches} batches | "
                f"{self._completed_tokens:,} tokens | "
                f"{self._completed_tokens / elapsed:,.0f} tokens/s",
                flush=True,
            )
            self._last_progress_print = now  # Begin the next print interval.
        return vectors

    async def _reserve_tokens(self, token_count: int) -> None:
        # Reserve TPM capacity in one process-wide rolling 60-second window.

        if token_count > self._tokens_per_minute:
            raise ValueError(
                f"Embedding batch has {token_count:,} tokens, above the configured "
                f"{self._tokens_per_minute:,} TPM budget"
            )
        while True:
            async with self._token_window_lock:
                now = perf_counter()  # Time used for the rolling window.
                cutoff = now - 60.0  # Reservations older than this expire.
                while self._token_window and self._token_window[0][0] <= cutoff:
                    _, expired = self._token_window.popleft()  # Old token count.
                    self._token_window_total -= expired
                if self._token_window_total + token_count <= self._tokens_per_minute:
                    self._token_window.append((now, token_count))
                    self._token_window_total += token_count
                    return
                wait_seconds = max(  # Time until the oldest reservation expires.
                    0.0,
                    60.0
                    - (now - self._token_window[0][0])
                    + self._rate_limit_buffer_seconds,
                )
            print(
                f"[embeddings] TPM pacing: waiting {wait_seconds:.1f}s "
                f"before a {token_count:,}-token batch",
                flush=True,
            )
            await asyncio.sleep(wait_seconds)

    async def _retry_create(self, batch: Sequence[str], batch_tokens: int) -> Any:
        last_error: Exception | None = None  # Final ordinary request error.
        for attempt in range(self._retry_attempts):
            try:
                await self._reserve_tokens(batch_tokens)
                if self._api_semaphore is None:
                    return await self._client.embeddings.create(
                        model=self.model_name,
                        input=list(batch),
                        encoding_format="float",
                    )
                async with self._api_semaphore:
                    return await self._client.embeddings.create(
                        model=self.model_name,
                        input=list(batch),
                        encoding_format="float",
                    )
            except Exception as error:
                last_error = error  # Saved in case the loop exits unexpectedly.
                if not self._is_retryable(error) or attempt + 1 >= self._retry_attempts:
                    raise
                server_delay = self._retry_after_seconds(error)  # API requested wait.
                if server_delay is None:
                    delay = min(  # Exponential wait capped by configuration.
                        self._retry_base_delay * (2**attempt), self._retry_max_delay
                    )
                    delay *= random.uniform(0.8, 1.2)
                else:
                    delay = server_delay + self._rate_limit_buffer_seconds  # API wait.
                print(
                    f"[embeddings] retry {attempt + 1}/{self._retry_attempts - 1} "
                    f"in {delay:.1f}s: {type(error).__name__}",
                    flush=True,
                )
                await asyncio.sleep(delay)
        assert last_error is not None
        raise last_error

    @staticmethod
    def _retry_after_seconds(error: Exception) -> float | None:
        # Read OpenAI retry headers, falling back to the error-message delay.

        response = getattr(error, "response", None)  # Optional HTTP response.
        headers = getattr(response, "headers", None)  # Optional retry headers.
        if headers is not None:
            retry_after_ms = headers.get("retry-after-ms")  # Millisecond form.
            if retry_after_ms:
                try:
                    return max(0.0, float(retry_after_ms) / 1000.0)
                except (TypeError, ValueError):
                    pass
            retry_after = headers.get("retry-after")  # Standard second form.
            if retry_after:
                try:
                    return max(0.0, float(retry_after))
                except (TypeError, ValueError):
                    pass
        match = re.search(
            r"try again in ([0-9]+(?:\.[0-9]+)?)s",
            str(error),
            re.IGNORECASE,
        )
        return float(match.group(1)) if match else None

    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        status = getattr(error, "status_code", None) or getattr(error, "status", None)
        if status in {408, 409, 429, 500, 502, 503, 504}:
            return True
        message = str(error).lower()  # Fallback for transport errors.
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

    def count(self, text: str) -> int:
        # Return the model-style token count for one string.
        return len(self._encoding.encode(text, disallowed_special=()))

    def count_many(self, texts: Sequence[str]) -> list[int]:
        # Preserve input order so counts stay paired with texts.
        return [self.count(text) for text in texts]

    def truncate(self, text: str, max_tokens: int) -> str:
        # Keep only the first tokens without cutting through encoded pieces.
        if max_tokens <= 0:
            return ""
        token_ids = self._encoding.encode(text, disallowed_special=())  # Full text IDs.
        if len(token_ids) <= max_tokens:
            return text
        return self._encoding.decode(token_ids[:max_tokens])

    def split(self, text: str, max_tokens: int) -> list[str]:
        # Cut text into non-overlapping token-sized pieces.
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        token_ids = self._encoding.encode(text, disallowed_special=())  # Full text IDs.
        return [
            self._encoding.decode(token_ids[start : start + max_tokens])
            for start in range(0, len(token_ids), max_tokens)
        ] or [""]
