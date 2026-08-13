#############################################################################
# File: evaluate_agents_wcost.py
#
# Description:
#   Runs memory-architecture evaluations with isolated task state,
#   parallel workers, resumable output, and detailed usage measurements.
#   It uses native OpenAI clients and task-local Qdrant collections while
#   sharing one batched embedding service.
#
#   - Defines current standard short- and long-context rates for GPT-5.6
#     models, including cache reads and writes.
#   - Streams benchmark tasks into bounded workers and answers each
#     task's pending questions concurrently.
#   - Builds one independent runtime, memory block, agent, usage ledger,
#     and storage collection per task.
#   - Writes result and task JSONL records through one serialized queue
#     and resumes from successful task/question keys.
#   - Tracks LLM, embedding, cache, cost, latency, ingestion, task, and
#     question totals.
#   - Closes OpenAI and Qdrant clients on success, failure, and
#     cancellation.
#   - Validates paths, limits, retries, API settings, and optional
#     pricing overrides before network work begins.
#   - For binary routing, enforces the exact held-out IDs and test-file hash
#     stored by the router trainer so end-to-end evaluation cannot drift.
#   - Saves a rounded run summary with precise small costs and
#     environment details.
#############################################################################

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import sys
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from asdrp.agent import AGENT_REGISTRY
from asdrp.agent.binary_router_agent import BinaryRouterAgent
from asdrp.classification_algorithms.binary_router import (
    DEFAULT_LABELS,
    DEFAULT_LABEL_TO_MEMORY,
    load_binary_router,
)
from asdrp.dataset_adapters import iter_evaluation_tasks
from asdrp.eval_schemas import EvaluationQuestion, EvaluationTask, Pricing
from asdrp.memory.BaseMemBlock import BaseMemBlock
from asdrp.runtime import OpenAIRuntime, UsageLedger

# Keep the existing project registry untouched; register the proof-of-concept
# router only inside this evaluator module.
AGENT_REGISTRY = dict(AGENT_REGISTRY)
AGENT_REGISTRY["binary_router"] = BinaryRouterAgent

# Dataset names accepted by the command line.
DATASET_CHOICES = (
    "longmemeval",
    "ecommerce",
    "mab_accurate_retrieval",
    "mab_conflict_resolution",
    "mab_long_range_understanding",
    "mab_test_time_learning",
)

# OpenAI standard processing prices as of August 7, 2026.
# Each tuple stores short input/read/write/output, then the same long-context rates.
LLM_PRICING: dict[
    str,
    tuple[
        float,
        float,
        float,
        float,
        float | None,
        float | None,
        float | None,
        float | None,
    ],
] = {
    "gpt-5.6-sol": (
        5.00,  # short input
        0.50,  # short cached input
        6.25,  # short cache writes
        30.00,  # short output
        10.00,  # long input
        1.00,  # long cached input
        12.50,  # long cache writes
        45.00,  # long output
    ),
    "gpt-5.6-terra": (
        2.00,
        0.20,
        2.50,
        12.00,
        4.00,
        0.40,
        5.00,
        18.00,
    ),
    "gpt-5.6-luna": (
        0.20,
        0.02,
        0.25,
        1.20,
        0.40,
        0.04,
        0.50,
        1.80,
    ),
}

# Input price per million embedding tokens.
EMBEDDING_PRICING: dict[str, float] = {
    "text-embedding-3-small": 0.02,
}


SUMMARY_USAGE_FIELDS = (
    "llm_requests",
    "estimated_llm_requests",
    "llm_input_tokens",
    "llm_long_input_tokens",
    "llm_cached_input_tokens",
    "llm_long_cached_input_tokens",
    "llm_cache_write_tokens",
    "llm_long_cache_write_tokens",
    "llm_output_tokens",
    "llm_long_output_tokens",
    "llm_cost_usd",
    "embedding_requests",
    "embedding_entry_count",
    "embedding_input_tokens",
    "embedding_cost_usd",
)


@dataclass(slots=True)
class RunSummary:
    # Stores totals shared by all running task workers.

    tasks_started: int = 0  # Tasks picked up by workers.
    tasks_completed: int = 0  # Tasks with no failed questions.
    tasks_failed: int = 0  # Tasks with an ingestion or question failure.
    questions_completed: int = 0  # Successful answers.
    questions_failed: int = 0  # Failed answers.
    entries_ingested: int = 0  # Source entries in completed memory builds.
    task_runtime_seconds: float = 0.0  # Sum of task times across workers.
    usage: dict[str, float | int] = field(default_factory=dict)  # Combined usage.
    # This lock is internal and should not appear in saved dataclass output.
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    async def task_started(self) -> None:
        # Record that a task began without claiming ingestion succeeded.

        async with self._lock:
            self.tasks_started += 1

    async def entries_finished(self, entry_count: int) -> None:
        # Record entries only after the complete memory build succeeds.

        if entry_count < 0:
            raise ValueError("entry_count cannot be negative")

        async with self._lock:
            self.entries_ingested += entry_count

    async def task_finished(
        self,
        *,
        failed: bool,
        completed_questions: int,
        failed_questions: int,
        runtime_seconds: float,
        usage: dict[str, Any],
    ) -> None:
        # Update all task totals while holding the shared lock.
        async with self._lock:
            self.tasks_completed += int(not failed)
            self.tasks_failed += int(failed)
            self.questions_completed += completed_questions
            self.questions_failed += failed_questions
            self.task_runtime_seconds += runtime_seconds
            combined = usage["combined"]  # Memory and query usage together.
            for key in SUMMARY_USAGE_FIELDS:
                value = combined.get(key, 0)
                self.usage[key] = self.usage.get(key, 0) + value

    def snapshot(self) -> dict[str, Any]:
        # Keep only the usage fields intended for the final run summary.
        usage = {
            "llm_requests": int(self.usage.get("llm_requests", 0)),
            "estimated_llm_requests": int(self.usage.get("estimated_llm_requests", 0)),
            "llm_input_tokens": int(self.usage.get("llm_input_tokens", 0)),
            "llm_long_input_tokens": int(self.usage.get("llm_long_input_tokens", 0)),
            "llm_cached_input_tokens": int(
                self.usage.get("llm_cached_input_tokens", 0)
            ),
            "llm_long_cached_input_tokens": int(
                self.usage.get("llm_long_cached_input_tokens", 0)
            ),
            "llm_cache_write_tokens": int(self.usage.get("llm_cache_write_tokens", 0)),
            "llm_long_cache_write_tokens": int(
                self.usage.get("llm_long_cache_write_tokens", 0)
            ),
            "llm_output_tokens": int(self.usage.get("llm_output_tokens", 0)),
            "llm_long_output_tokens": int(self.usage.get("llm_long_output_tokens", 0)),
            "llm_cost_usd": float(self.usage.get("llm_cost_usd", 0.0)),
            "embedding_requests": int(self.usage.get("embedding_requests", 0)),
            "embedding_entry_count": int(self.usage.get("embedding_entry_count", 0)),
            "embedding_input_tokens": int(self.usage.get("embedding_input_tokens", 0)),
            "embedding_cost_usd": float(self.usage.get("embedding_cost_usd", 0.0)),
        }

        # Evaluation cost includes both generation and embeddings.
        total_cost = usage["llm_cost_usd"] + usage["embedding_cost_usd"]

        return {
            "tasks_started": self.tasks_started,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "questions_completed": self.questions_completed,
            "questions_failed": self.questions_failed,
            "entries_ingested": self.entries_ingested,
            "total_estimated_eval_cost_usd": total_cost,
            "sum_task_runtime_seconds": self.task_runtime_seconds,
            "usage": usage,
        }


class JsonlWriter:
    # Uses one queue so worker output cannot overlap in the JSONL files.

    def __init__(self, results_file: Path, tasks_file: Path) -> None:
        self._results_file = results_file  # Small answer records.
        self._tasks_file = tasks_file  # Detailed task records.
        self._queue: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue()

    async def write_result(self, payload: dict[str, Any]) -> None:
        # Queue one answer record for the result file.
        await self._queue.put(("result", payload))

    async def write_task(self, payload: dict[str, Any]) -> None:
        # Queue one full task record for the task sidecar file.
        await self._queue.put(("task", payload))

    async def close(self) -> None:
        # A blank item tells the writer that no more records will arrive.
        await self._queue.put(None)

    async def run(self) -> None:
        # Write and flush each completed record, preserving crash-resumability.

        self._results_file.parent.mkdir(parents=True, exist_ok=True)
        self._tasks_file.parent.mkdir(parents=True, exist_ok=True)
        with (
            self._results_file.open("a", encoding="utf-8") as result_handle,
            self._tasks_file.open("a", encoding="utf-8") as task_handle,
        ):
            while True:
                item = await self._queue.get()  # Waits without blocking workers.
                try:
                    if item is None:
                        return
                    kind, payload = item  # File label and JSON-ready record.
                    handle = result_handle if kind == "result" else task_handle
                    handle.write(
                        json.dumps(payload, ensure_ascii=False, default=str) + "\n"
                    )
                    handle.flush()
                finally:
                    self._queue.task_done()


class ProgressReporter:
    # Prints one clear progress line after each task.

    def __init__(self) -> None:
        self._lock = asyncio.Lock()  # Stops workers from mixing progress lines.
        self._finished_tasks = 0  # Tasks reported so far.
        self._finished_questions = 0  # Questions reported so far.
        self._start = time.perf_counter()  # Start of throughput timing.

    async def task_done(
        self, task: EvaluationTask, question_count: int, failed: bool
    ) -> None:
        async with self._lock:
            self._finished_tasks += 1
            self._finished_questions += question_count
            elapsed = time.perf_counter() - self._start
            average_seconds = (
                elapsed / self._finished_questions if self._finished_questions else 0.0
            )
            status = "failed" if failed else "done"
            print(
                f"[{self._finished_tasks} tasks | {self._finished_questions} questions | "
                f"{average_seconds:.2f} s/question] {status}: {task.task_id}",
                flush=True,
            )


def _sidecar_path(path: Path, label: str) -> Path:
    # Return ``name_<label>.jsonl/json`` without losing the original stem.

    if label == "summary":
        return path.with_name(f"{path.stem}_summary.json")
    return path.with_name(f"{path.stem}_{label}.jsonl")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_completed_question_keys(path: Path) -> set[tuple[str, str]]:
    # Read successful prior (task_id, question_id) pairs for resume support.
    # Some MemoryAgentBench sources reuse bare question IDs across tasks, so
    # question_id alone is not a safe global resume key.
    # Older LongMemEval result rows may omit task_id; for LongMemEval only,
    # task_id == question_id by construction, so this fallback is unambiguous.

    if not path.exists():
        return set()
    completed: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)  # Ignore a partial final line after a crash.
            except json.JSONDecodeError:
                continue
            if record.get("status") != "ok" or record.get("question_id") is None:
                continue

            question_id = str(record["question_id"])
            task_id = record.get("task_id")
            if task_id is None and record.get("dataset") == "longmemeval":
                task_id = question_id
            if task_id is not None:
                completed.add((str(task_id), question_id))
    return completed


def _error_payload(error: BaseException) -> dict[str, Any]:
    # Serialize useful diagnostics without exposing non-JSON exception objects.

    return {
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": "".join(traceback.format_exception(error))[-16_000:],
    }


async def _answer_question(
    *,
    task: EvaluationTask,
    question: EvaluationQuestion,
    agent: Any,
    usage: UsageLedger,
    writer: JsonlWriter,
    question_semaphore: asyncio.Semaphore,
) -> tuple[bool, dict[str, Any]]:
    # Answer one question and split clean output from task diagnostics.

    start = time.perf_counter()  # Includes search, prompt work, and generation.
    try:
        async with question_semaphore, usage.scope(question.question_id):
            answer = await agent.answer(question, task.retrieval)  # Agent result.
        question_runtime = time.perf_counter() - start  # Full question time.
        routing = getattr(agent, "routing_metadata", None)
        result_payload = {
            "status": "ok",
            "dataset": task.dataset,
            "source": task.source,
            "task_id": task.task_id,
            "question_id": question.question_id,
            "question": question.text,
            "reference_answers": question.answers,
            "hypothesis": answer.hypothesis,
        }
        if routing:
            result_payload["routing"] = routing
        await writer.write_result(result_payload)
        question_payload = {
            "question_id": question.question_id,
            "status": "ok",
            "retrieved_count": answer.retrieved_count,
            "retrieved_context_tokens": answer.retrieved_context_tokens,
            "retrieval_seconds": answer.retrieval_seconds,
            "context_preparation_seconds": answer.context_preparation_seconds,
            "generation_seconds": answer.generation_seconds,
            "question_runtime_seconds": question_runtime,
            "query_usage": usage.scope_snapshot(question.question_id),
            "retrieval_plan": asdict(task.retrieval),
            "effective_retrieval_plan": asdict(answer.effective_retrieval_plan),
            "retrieval_strategy": answer.retrieval_strategy,
            "question_metadata": question.metadata,
        }
        if routing:
            question_payload["routing"] = routing
        return True, question_payload
    except Exception as error:  # Errors are recorded per question.
        question_runtime = time.perf_counter() - start  # Time until failure.
        error_details = _error_payload(error)  # Safe fields for both output files.
        await writer.write_result(
            {
                "status": "error",
                "dataset": task.dataset,
                "source": task.source,
                "task_id": task.task_id,
                "question_id": question.question_id,
                "question": question.text,
                "reference_answers": question.answers,
                "hypothesis": None,
                "error_type": error_details["error_type"],
                "error": error_details["error"],
            }
        )
        return False, {
            "question_id": question.question_id,
            "status": "error",
            "question_runtime_seconds": question_runtime,
            "query_usage": usage.scope_snapshot(question.question_id),
            "question_metadata": question.metadata,
            **error_details,
        }


async def _process_task(
    *,
    task: EvaluationTask,
    pending_questions: list[EvaluationQuestion],
    args: argparse.Namespace,
    worker_id: int,
    run_id: str,
    api_semaphore: asyncio.Semaphore,
    embedder: Any,
    writer: JsonlWriter,
    totals: RunSummary,
) -> tuple[int, int, bool]:
    # Create isolated components, build memory once, and answer pending questions.

    task_start = time.perf_counter()  # Includes memory, questions, and cleanup.
    usage = UsageLedger(args.pricing)  # Usage for this task only.
    memory: BaseMemBlock | None = None  # Set after the memory class is made.
    memory_build_seconds = 0.0  # Time to transform and store the context.
    successful_questions = 0  # Answers written without error.
    failed_questions = 0  # Questions that raised an error.
    question_details: list[dict[str, Any]] = []  # Full per-question metrics.
    task_error: BaseException | None = None  # Setup, ingestion, or cancellation error.
    runtime: OpenAIRuntime | None = None  # Closed even when memory setup fails.

    await totals.task_started()
    try:
        # The runtime, usage ledger, agent, memory, and storage stay task-local.
        # Only the connection-pooled embedding client is shared across tasks.
        runtime = OpenAIRuntime(
            llm_model=args.llm_model,
            embedder=embedder,
            api_semaphore=api_semaphore,
            usage=usage,
            request_timeout=args.request_timeout,
            retry_attempts=args.retry_attempts,
            retry_base_delay=args.retry_base_delay,
            retry_max_delay=args.retry_max_delay,
        )
        agent_type = AGENT_REGISTRY[args.agent]  # Class chosen by the CLI name.
        memory_type = agent_type.memory_block_type  # Matching storage class.
        memory_kwargs: dict[str, Any] = {
            "task_id": task.task_id,
            "runtime": runtime,
            "qdrant_url": args.qdrant_url,
            "qdrant_timeout": args.qdrant_timeout,
            "qdrant_batch_size": args.qdrant_batch_size,
            "ingest_concurrency": args.ingest_concurrency,
            "retry_attempts": args.retry_attempts,
            "retry_base_delay": args.retry_base_delay,
            "retry_max_delay": args.retry_max_delay,
            "delete_collection_on_close": not args.keep_collections,
        }
        if args.agent == "binary_router":
            memory_kwargs["router_model_path"] = args.router_model
        created_memory: BaseMemBlock = memory_type(
            **memory_kwargs
        )  # Fresh task-local storage.
        memory = created_memory  # Retained separately for cleanup in finally.
        await created_memory.initialize()

        # Required control flow: runner -> memory.put(context), then questions -> agent.
        # The agent is not even constructed until direct memory ingestion finishes.
        memory_start = time.perf_counter()  # Starts after storage is ready.
        await created_memory.put(task.entries)
        memory_build_seconds = time.perf_counter() - memory_start  # Ingestion time.

        # Count entries only after the complete memory build succeeds.
        await totals.entries_finished(len(task.entries))

        agent = agent_type(  # Read-only question interface over the built memory.
            memory_block=created_memory,
            runtime=runtime,
            task=task,
        )

        question_semaphore = asyncio.Semaphore(args.question_concurrency)
        outcomes = await asyncio.gather(  # Answers stay in question source order.
            *(
                _answer_question(
                    task=task,
                    question=question,
                    agent=agent,
                    usage=usage,
                    writer=writer,
                    question_semaphore=question_semaphore,
                )
                for question in pending_questions
            )
        )
        successful_questions = sum(success for success, _ in outcomes)
        failed_questions = len(outcomes) - successful_questions
        question_details = [details for _, details in outcomes]  # Saved metrics.
    except asyncio.CancelledError as error:
        # Mark interrupted task records accurately, then let cancellation continue.
        task_error = error
        failed_questions = len(pending_questions)
        raise
    except Exception as error:  # Task construction/ingestion failure is persisted.
        task_error = error  # One setup failure affects every pending question.
        failed_questions = len(pending_questions)  # None could be answered.
        error_details = _error_payload(error)  # Reused for each pending question.
        for question in pending_questions:
            await writer.write_result(
                {
                    "status": "error",
                    "dataset": task.dataset,
                    "source": task.source,
                    "task_id": task.task_id,
                    "question_id": question.question_id,
                    "question": question.text,
                    "reference_answers": question.answers,
                    "hypothesis": None,
                    "error_type": error_details["error_type"],
                    "error": error_details["error"],
                }
            )
            question_details.append(
                {
                    "question_id": question.question_id,
                    "status": "error",
                    "question_runtime_seconds": 0.0,
                    "query_usage": usage.scope_snapshot(question.question_id),
                    "question_metadata": question.metadata,
                    **error_details,
                }
            )
    finally:
        close_errors: dict[str, dict[str, Any]] = {}  # Cleanup diagnostics by client.
        if memory is not None:
            try:
                # Qdrant clients must close even if collection creation failed.
                await memory.close()
            except Exception as error:
                # Keep cleanup errors without changing answer status.
                close_errors["memory"] = _error_payload(error)
        if runtime is not None:
            try:
                # Each task owns one Responses API client and its HTTP connections.
                await runtime.close()
            except Exception as error:
                close_errors["runtime"] = _error_payload(error)

        task_seconds = time.perf_counter() - task_start  # Full task time.
        usage_snapshot = usage.snapshot()  # Fixed totals for saved output.
        task_failed = task_error is not None or failed_questions > 0
        task_payload = {  # Detailed record written to the task sidecar.
            "run_id": run_id,
            "status": (
                "error"
                if task_error
                else ("partial_error" if failed_questions else "ok")
            ),
            "worker_id": worker_id,
            "dataset": task.dataset,
            "source": task.source,
            "task_id": task.task_id,
            "agent": args.agent,
            "entry_count": len(task.entries),
            "question_count": len(pending_questions),
            "successful_questions": successful_questions,
            "failed_questions": failed_questions,
            "memory_build_seconds": memory_build_seconds,
            "task_runtime_seconds": task_seconds,
            "usage": usage_snapshot,
            "questions": question_details,
            "task_metadata": task.metadata,
        }
        routing = (
            getattr(memory, "routing_metadata", None) if memory is not None else None
        )
        if routing:
            task_payload["routing"] = routing
        if task_error is not None:
            task_payload.update(_error_payload(task_error))
        if close_errors:
            task_payload["close_errors"] = close_errors  # Cleanup only.
        await writer.write_task(task_payload)
        await totals.task_finished(
            failed=task_failed,
            completed_questions=successful_questions,
            failed_questions=failed_questions,
            runtime_seconds=task_seconds,
            usage=usage_snapshot,
        )

    return successful_questions, failed_questions, task_failed


async def _worker(
    *,
    worker_id: int,
    queue: asyncio.Queue[tuple[EvaluationTask, list[EvaluationQuestion]] | None],
    args: argparse.Namespace,
    run_id: str,
    api_semaphore: asyncio.Semaphore,
    embedder: Any,
    writer: JsonlWriter,
    totals: RunSummary,
    progress: ProgressReporter,
) -> None:
    # Consume isolated tasks until the producer posts a sentinel.

    while True:
        item = await queue.get()  # A task or the end-of-work marker.
        try:
            if item is None:
                return
            task, pending_questions = item  # Questions not already in output.
            _, _, failed = await _process_task(  # Counts are already in totals.
                task=task,
                pending_questions=pending_questions,
                args=args,
                worker_id=worker_id,
                run_id=run_id,
                api_semaphore=api_semaphore,
                embedder=embedder,
                writer=writer,
                totals=totals,
            )
            await progress.task_done(task, len(pending_questions), failed)
        finally:
            queue.task_done()


async def _produce_tasks(
    *,
    queue: asyncio.Queue[tuple[EvaluationTask, list[EvaluationQuestion]] | None],
    args: argparse.Namespace,
    completed_keys: set[tuple[str, str]],
    embedder: Any,
) -> None:
    # Stream formatted tasks and keep only unanswered questions when resuming.

    selected_tasks = 0  # Needed only to enforce --max-tasks.
    visible_index = 0  # Index after applying any source filter.
    tasks = iter_evaluation_tasks(  # Streams rows instead of loading all data.
        data_file=args.data_file,
        dataset=args.dataset,
        embedding_model=args.embedding_model,
        max_entry_tokens=args.max_entry_tokens,
        source_filter=args.source_filter,
        embedding_counter=embedder,
    )
    question_type_filter = set(args.question_type_filter or ())
    router_question_ids = set(getattr(args, "router_evaluation_question_ids", ()))
    seen_router_ids: set[str] = set()

    for task in tasks:
        eligible_questions = []
        for question in task.questions:
            if router_question_ids and question.question_id not in router_question_ids:
                continue
            if router_question_ids:
                seen_router_ids.add(question.question_id)
            if (
                question_type_filter
                and str(question.metadata.get("question_type") or "")
                not in question_type_filter
            ):
                continue
            eligible_questions.append(question)

        if not eligible_questions:
            continue
        if visible_index < args.start_index:
            visible_index += 1
            continue
        if args.max_tasks is not None and selected_tasks >= args.max_tasks:
            break
        visible_index += 1
        pending = [  # Resume mode skips successful questions already on disk.
            question
            for question in eligible_questions
            if (task.task_id, question.question_id) not in completed_keys
        ]
        if not pending:
            continue
        await queue.put((task, pending))
        selected_tasks += 1

    if router_question_ids:
        missing = router_question_ids - seen_router_ids
        if missing:
            preview = ", ".join(sorted(missing)[:8])
            raise ValueError(
                "Binary-router evaluation file is missing question IDs stored in the "
                f"model artifact: {preview}"
            )

    # One marker lets each worker stop after the queue becomes empty.
    for _ in range(args.workers):
        await queue.put(None)


def _round_summary_values(value: Any, field_name: str = "") -> Any:
    # Keep integer counts as integers while limiting decimal values to two places.
    if isinstance(value, dict):
        return {key: _round_summary_values(item, key) for key, item in value.items()}

    if isinstance(value, list):
        return [_round_summary_values(item, field_name) for item in value]

    # bool is a subclass of int, so check it before numeric handling.
    if isinstance(value, bool):
        return value

    if isinstance(value, float):
        # Small costs and user-supplied rates need more than two decimal places.
        if field_name.endswith(("_cost_usd", "_per_million")):
            return round(value, 8)
        return round(value, 2)

    return value


async def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    # Run the full producer/worker/writer pipeline and persist a summary.

    run_id = uuid.uuid4().hex  # Joins files written during this run.
    wall_start = time.perf_counter()  # Measures real elapsed time.
    tasks_file = _sidecar_path(args.output_file, "tasks")  # Detailed JSONL.
    summary_file = _sidecar_path(args.output_file, "summary")  # Final JSON.

    if args.overwrite:
        for path in (args.output_file, tasks_file, summary_file):
            path.unlink(missing_ok=True)
    completed_keys = (  # Successful task/question pairs to skip in resume mode.
        set() if args.overwrite else _load_completed_question_keys(args.output_file)
    )

    from asdrp.openai_embedder import OpenAIEmbedder

    api_semaphore = asyncio.Semaphore(args.api_concurrency)  # All OpenAI calls.

    # Share one connection-pooled embedding client across all workers.
    embedder = OpenAIEmbedder(
        model_name=args.embedding_model,
        batch_size=args.embedding_batch_size,
        batch_token_limit=args.embedding_batch_token_limit,
        max_parallel_batches=args.embedding_parallel_batches,
        tokens_per_minute=args.embedding_tokens_per_minute,
        rate_limit_buffer_seconds=args.rate_limit_buffer_seconds,
        api_semaphore=api_semaphore,
        request_timeout=args.request_timeout,
        retry_attempts=args.retry_attempts,
        retry_base_delay=args.retry_base_delay,
        retry_max_delay=args.retry_max_delay,
    )
    writer = JsonlWriter(args.output_file, tasks_file)  # Shared output queue.
    writer_task = asyncio.create_task(writer.run(), name="jsonl-writer")
    # A bounded task queue keeps streamed dataset rows from growing in memory.
    queue: asyncio.Queue[tuple[EvaluationTask, list[EvaluationQuestion]] | None] = (
        asyncio.Queue(maxsize=max(2, args.workers * 2))
    )
    totals = RunSummary()  # Totals shared by all workers.
    progress = ProgressReporter()  # Shared progress line counts.
    workers = [  # Each worker handles complete tasks from the queue.
        asyncio.create_task(
            _worker(
                worker_id=index,
                queue=queue,
                args=args,
                run_id=run_id,
                api_semaphore=api_semaphore,
                embedder=embedder,
                writer=writer,
                totals=totals,
                progress=progress,
            ),
            name=f"evaluation-worker-{index}",
        )
        for index in range(args.workers)
    ]

    fatal_error: BaseException | None = None  # Includes cancellation for cleanup.
    try:
        await _produce_tasks(
            queue=queue,
            args=args,
            completed_keys=completed_keys,
            embedder=embedder,
        )
        await queue.join()
        await asyncio.gather(*workers)
    except BaseException as error:
        fatal_error = error  # Saved so cleanup can finish before it is raised.
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
    finally:
        await embedder.close()
        await writer.close()
        await writer_task

    wall_seconds = time.perf_counter() - wall_start  # Whole evaluation time.
    summary = {  # Final configuration, progress, usage, and environment record.
        "run_id": run_id,
        "configuration": {
            "dataset": args.dataset,
            "data_file": str(args.data_file),
            "agent": args.agent,
            "llm_model": args.llm_model,
            "embedding_model": args.embedding_model,
            "workers": args.workers,
            "question_concurrency": args.question_concurrency,
            "api_concurrency": args.api_concurrency,
            "embedding_batch_size": args.embedding_batch_size,
            "embedding_batch_token_limit": args.embedding_batch_token_limit,
            "embedding_parallel_batches": args.embedding_parallel_batches,
            "embedding_tokens_per_minute": args.embedding_tokens_per_minute,
            "rate_limit_buffer_seconds": args.rate_limit_buffer_seconds,
            "embedding_backend": embedder.device,
            "qdrant_batch_size": args.qdrant_batch_size,
            "ingest_concurrency": args.ingest_concurrency,
            "max_entry_tokens": args.max_entry_tokens,
            "request_timeout": args.request_timeout,
            "qdrant_timeout": args.qdrant_timeout,
            "retry_attempts": args.retry_attempts,
            "retry_base_delay": args.retry_base_delay,
            "retry_max_delay": args.retry_max_delay,
            "qdrant_url": args.qdrant_url,
            "keep_collections": args.keep_collections,
            "start_index": args.start_index,
            "max_tasks": args.max_tasks,
            "source_filter": args.source_filter,
            "question_type_filter": args.question_type_filter,
            "router_model": str(args.router_model) if args.router_model else None,
            "router_evaluation_question_ids": list(
                getattr(args, "router_evaluation_question_ids", ())
            ),
            "router_evaluation_data_sha256": getattr(
                args, "router_evaluation_data_sha256", None
            ),
            "pricing": asdict(args.pricing),
        },
        "completed_tasks": totals.tasks_completed,
        "completed_questions": totals.questions_completed,
        "previously_completed_questions": len(completed_keys),
        "wall_runtime_seconds": wall_seconds,
        "average_seconds_per_question": (
            wall_seconds / totals.questions_completed
            if totals.questions_completed
            else 0.0
        ),
        "totals": totals.snapshot(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
    }
    if fatal_error is not None:
        # Keep only the readable fatal error in the summary.
        summary["error"] = str(fatal_error)

    # Limit decimal values in the final summary to two places.
    summary = _round_summary_values(summary)

    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )

    if fatal_error is not None:
        raise fatal_error

    return summary


def build_parser() -> argparse.ArgumentParser:
    # Define the command-line options used by this evaluator.

    parser = argparse.ArgumentParser(  # Main evaluator command parser.
        description=(
            "Evaluate one isolated long-term-memory architecture with an OpenAI "
            "answer model and OpenAI text-embedding-3-small embeddings."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "data_file", type=Path, help="Existing benchmark JSON array file."
    )
    parser.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    parser.add_argument("--agent", required=True, choices=tuple(AGENT_REGISTRY))
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument(
        "--router-model",
        type=Path,
        help="TF-IDF + LinearSVC artifact required by --agent binary_router.",
    )
    parser.add_argument(
        "--question-type-filter",
        nargs="+",
        help=(
            "Evaluate only matching question types. For binary_router this is optional; "
            "the exact held-out question IDs are loaded from the model artifact."
        ),
    )
    parser.add_argument("--llm-model", default="gpt-5.6-luna")
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        choices=("text-embedding-3-small",),
        help="OpenAI 1536-dimensional embedding model.",
    )
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--question-concurrency", type=int, default=4)
    parser.add_argument("--api-concurrency", type=int, default=16)
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--embedding-batch-token-limit", type=int, default=240_000)
    parser.add_argument("--embedding-parallel-batches", type=int, default=4)
    parser.add_argument(
        "--embedding-tokens-per-minute",
        type=int,
        default=4_500_000,
        help="Shared embedding TPM target; defaults to 90%% of a 5M TPM limit.",
    )
    parser.add_argument(
        "--rate-limit-buffer-seconds",
        type=float,
        default=0.75,
        help="Small safety margin added to TPM and server retry waits.",
    )
    parser.add_argument("--qdrant-batch-size", type=int, default=256)
    parser.add_argument("--ingest-concurrency", type=int, default=4)
    parser.add_argument("--max-entry-tokens", type=int, default=7_000)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--qdrant-timeout", type=int, default=120)
    parser.add_argument("--retry-attempts", type=int, default=8)
    parser.add_argument("--retry-base-delay", type=float, default=1.0)
    parser.add_argument("--retry-max-delay", type=float, default=20.0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--source-filter")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--keep-collections",
        action="store_true",
        help="Retain task collections for debugging; off by default.",
    )
    parser.add_argument("--llm-input-per-million", type=float)
    parser.add_argument("--llm-cached-input-per-million", type=float)
    parser.add_argument("--llm-cache-write-per-million", type=float)
    parser.add_argument("--llm-output-per-million", type=float)

    parser.add_argument("--llm-long-input-per-million", type=float)
    parser.add_argument("--llm-long-cached-input-per-million", type=float)
    parser.add_argument("--llm-long-cache-write-per-million", type=float)
    parser.add_argument("--llm-long-output-per-million", type=float)
    parser.add_argument(
        "--llm-long-context-threshold",
        type=int,
        default=272_000,
    )

    parser.add_argument("--embedding-input-per-million", type=float)

    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    # Reject invalid concurrency and path settings before network calls begin.

    if not args.data_file.is_file():
        parser.error(f"data file does not exist: {args.data_file}")
    if args.agent == "binary_router":
        if args.dataset != "longmemeval":
            parser.error(
                "--agent binary_router currently supports only --dataset longmemeval"
            )
        if args.router_model is None:
            parser.error("--agent binary_router requires --router-model")
        if not args.router_model.is_file():
            parser.error(f"router model does not exist: {args.router_model}")

        router = load_binary_router(str(args.router_model.expanduser().resolve()))
        if set(router.labels) != set(DEFAULT_LABELS):
            parser.error(
                "--router-model must be the LongMemEval preference/update router with "
                f"labels {DEFAULT_LABELS}; loaded labels were {router.labels}"
            )
        if router.label_to_memory != DEFAULT_LABEL_TO_MEMORY:
            parser.error(
                "--router-model has the wrong memory mapping; expected "
                f"{DEFAULT_LABEL_TO_MEMORY}, loaded {router.label_to_memory}"
            )
        if not router.evaluation_question_ids:
            parser.error(
                "--router-model does not contain held-out evaluation question IDs; "
                "retrain it with the current train_binary_router.py"
            )

        data_sha256 = _file_sha256(args.data_file)
        if data_sha256 != router.evaluation_data_sha256:
            parser.error(
                "binary_router must be evaluated on the exact test file used by the "
                "trainer: the supplied data-file SHA-256 does not match the model artifact"
            )
        if args.start_index != 0 or args.max_tasks is not None:
            parser.error(
                "binary_router uses the exact held-out question set stored in its model; "
                "do not pass --start-index or --max-tasks"
            )
        if args.question_type_filter and set(args.question_type_filter) != set(
            DEFAULT_LABELS
        ):
            parser.error(
                "when supplied for binary_router, --question-type-filter must exactly "
                f"match {DEFAULT_LABELS}"
            )

        # _produce_tasks consumes this immutable contract instead of independently
        # choosing a subset from the test file. This prevents evaluation drift.
        args.router_evaluation_question_ids = tuple(router.evaluation_question_ids)
        args.router_evaluation_data_sha256 = router.evaluation_data_sha256
    output_paths = {
        args.output_file.resolve(),
        _sidecar_path(args.output_file, "tasks").resolve(),
        _sidecar_path(args.output_file, "summary").resolve(),
    }
    if args.data_file.resolve() in output_paths:
        parser.error("output and sidecar paths must not overwrite the data file")
    if args.output_file.exists() and args.output_file.is_dir():
        parser.error("--output-file must be a file path, not a directory")
    # These settings are counts or limits and must be above zero.
    for name in (
        "workers",
        "question_concurrency",
        "api_concurrency",
        "embedding_batch_size",
        "embedding_batch_token_limit",
        "embedding_parallel_batches",
        "embedding_tokens_per_minute",
        "qdrant_batch_size",
        "ingest_concurrency",
        "max_entry_tokens",
        "retry_attempts",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 512 <= args.max_entry_tokens <= 8_000:
        parser.error("--max-entry-tokens must be between 512 and 8000")
    if args.embedding_batch_token_limit < args.max_entry_tokens:
        parser.error(
            "--embedding-batch-token-limit must be at least --max-entry-tokens"
        )
    if args.rate_limit_buffer_seconds < 0:
        parser.error("--rate-limit-buffer-seconds cannot be negative")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be positive")

    if args.qdrant_timeout <= 0:
        parser.error("--qdrant-timeout must be positive")

    if args.retry_base_delay < 0:
        parser.error("--retry-base-delay cannot be negative")

    if args.retry_max_delay < 0:
        parser.error("--retry-max-delay cannot be negative")

    if args.retry_max_delay < args.retry_base_delay:
        parser.error(
            "--retry-max-delay must be greater than or equal to --retry-base-delay"
        )
    if args.embedding_batch_token_limit > args.embedding_tokens_per_minute:
        parser.error(
            "--embedding-batch-token-limit cannot exceed --embedding-tokens-per-minute"
        )
    if args.start_index < 0:
        parser.error("--start-index cannot be negative")
    if args.max_tasks is not None and args.max_tasks <= 0:
        parser.error("--max-tasks must be positive")
    if not os.getenv("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY is not set (a .env file is supported)")
    if args.llm_long_context_threshold <= 0:
        parser.error("--llm-long-context-threshold must be positive")

    # User-provided prices can be zero, but never negative.
    for name in (
        "llm_input_per_million",
        "llm_cached_input_per_million",
        "llm_cache_write_per_million",
        "llm_output_per_million",
        "llm_long_input_per_million",
        "llm_long_cached_input_per_million",
        "llm_long_cache_write_per_million",
        "llm_long_output_per_million",
        "embedding_input_per_million",
    ):
        value = getattr(args, name)  # Price supplied for this flag, if any.
        if value is not None and value < 0:
            parser.error(f"--{name.replace('_', '-')} cannot be negative")

    known_llm = LLM_PRICING.get(args.llm_model)  # Built-in rates, if known.

    if known_llm is None and any(
        value is None
        for value in (
            args.llm_input_per_million,
            args.llm_cached_input_per_million,
            args.llm_output_per_million,
        )
    ):
        print(
            f"Warning: no built-in price for {args.llm_model!r}; unspecified "
            "LLM rates will be recorded as $0. Pass the pricing flags for "
            "cost totals.",
            file=sys.stderr,
        )

    # Unknown models use zero for rates the user did not provide.
    (
        default_input,
        default_cached_input,
        default_cache_write,
        default_output,
        default_long_input,
        default_long_cached_input,
        default_long_cache_write,
        default_long_output,
    ) = known_llm or (0.0, 0.0, 0.0, 0.0, None, None, None, None)

    # User-supplied rates override built-ins one field at a time.
    input_rate = (
        default_input
        if args.llm_input_per_million is None
        else args.llm_input_per_million
    )
    cached_input_rate = (
        default_cached_input
        if args.llm_cached_input_per_million is None
        else args.llm_cached_input_per_million
    )
    cache_write_rate = (
        default_cache_write
        if args.llm_cache_write_per_million is None
        else args.llm_cache_write_per_million
    )
    long_input_rate = (
        default_long_input
        if args.llm_long_input_per_million is None
        else args.llm_long_input_per_million
    )
    long_cached_input_rate = (
        default_long_cached_input
        if args.llm_long_cached_input_per_million is None
        else args.llm_long_cached_input_per_million
    )
    long_cache_write_rate = (
        default_long_cache_write
        if args.llm_long_cache_write_per_million is None
        else args.llm_long_cache_write_per_million
    )

    # Attach one validated pricing object for every task runtime.
    args.pricing = Pricing(
        llm_input_per_million=input_rate,
        llm_cached_input_per_million=cached_input_rate,
        llm_cache_write_per_million=cache_write_rate,
        llm_output_per_million=(
            default_output
            if args.llm_output_per_million is None
            else args.llm_output_per_million
        ),
        embedding_input_per_million=(
            EMBEDDING_PRICING.get(args.embedding_model, 0.0)
            if args.embedding_input_per_million is None
            else args.embedding_input_per_million
        ),
        llm_long_input_per_million=long_input_rate,
        llm_long_cached_input_per_million=long_cached_input_rate,
        llm_long_cache_write_per_million=long_cache_write_rate,
        llm_long_output_per_million=(
            default_long_output
            if args.llm_long_output_per_million is None
            else args.llm_long_output_per_million
        ),
        llm_long_context_threshold=args.llm_long_context_threshold,
    )


def main() -> None:
    # CLI entry point.

    load_dotenv()
    parser = build_parser()  # Defines all supported command-line options.
    args = parser.parse_args()  # Values supplied for this run.
    _validate_args(parser, args)
    summary = asyncio.run(evaluate(args))  # Starts the async evaluation pipeline.
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
