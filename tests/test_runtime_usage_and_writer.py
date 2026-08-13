#############################################################################
# File: test_runtime_usage_and_writer.py
#
# Description:
#   Checks usage accounting, pricing, retries, output writing, resume
#   support, and evaluator validation without external calls. It covers
#   short and long prompt caching in both read and write directions.
#
#   - Checks per-question usage isolation while several async jobs
#     update one ledger.
#   - Checks current GPT-5.6 short and long input, cache-read,
#     cache-write, and output rates.
#   - Checks exact LLM and embedding cost formulas and invalid cache
#     totals.
#   - Checks native Responses usage objects, dictionary mocks, flattened
#     fields, and local estimates.
#   - Checks retry cancellation, serialized JSONL writes, resume
#     parsing, and public summary fields.
#   - Checks destructive output-path rejection and precise rounding of
#     small costs.
#############################################################################

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from asdrp.eval_schemas import Pricing
from asdrp.evaluate_agents_wcost import (
    JsonlWriter,
    RunSummary,
    _load_completed_question_keys,
    _round_summary_values,
    _validate_args,
    build_parser,
)
from asdrp.runtime import OpenAIRuntime, TokenCounter, UsageLedger

pytestmark = pytest.mark.unit  # No test in this file makes a network call.


@pytest.mark.asyncio
async def test_usage_scopes_remain_isolated_under_concurrency():
    pricing = Pricing(  # Simple rates make expected totals easy to inspect.
        llm_input_per_million=1,
        llm_cached_input_per_million=0.1,
        llm_cache_write_per_million=1.25,
        llm_output_per_million=2,
        embedding_input_per_million=0.02,
    )
    ledger = UsageLedger(pricing)  # Shared by both fake question jobs.

    async def work(question_id, token_count):
        # Add embedding and LLM use inside one question's scope.
        async with ledger.scope(question_id):
            await asyncio.gather(
                ledger.add_embedding("query", token_count, text_count=1),
                ledger.add_llm(
                    "query",
                    token_count * 3,
                    token_count,
                    token_count,
                    token_count,
                    False,
                ),
            )

    await asyncio.gather(work("q1", 10), work("q2", 20))
    question_one = ledger.scope_snapshot("q1")  # First job's isolated usage.
    question_two = ledger.scope_snapshot("q2")  # Second job's isolated usage.
    assert question_one["embedding_input_tokens"] == 10
    assert question_two["embedding_input_tokens"] == 20
    assert question_one["llm_input_tokens"] == 30
    assert question_two["llm_input_tokens"] == 60
    assert question_one["llm_cached_input_tokens"] == 10
    assert question_two["llm_cached_input_tokens"] == 20
    assert question_one["llm_cache_write_tokens"] == 10
    assert question_two["llm_cache_write_tokens"] == 20
    assert ledger.snapshot()["combined"]["llm_requests"] == 2


@pytest.mark.asyncio
async def test_runtime_retry_does_not_delay_task_cancellation():
    # Cancellation must leave the retry helper at once so shutdown can continue.
    runtime = OpenAIRuntime.__new__(OpenAIRuntime)
    runtime._retry_attempts = 3  # Would retry an ordinary temporary failure.
    runtime._retry_base_delay = 1.0  # A swallowed cancellation would slow the test.
    runtime._retry_max_delay = 1.0  # Keep every possible wait the same.

    async def cancelled_call():
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await runtime._retry(cancelled_call)


@pytest.mark.asyncio
async def test_jsonl_writer_serializes_concurrent_writes_and_flushes(tmp_path):
    results, tasks = tmp_path / "r.jsonl", tmp_path / "t.jsonl"  # Output files.
    writer = JsonlWriter(results, tasks)  # One queue serves both files.
    runner = asyncio.create_task(writer.run())  # Background writer task.
    await asyncio.gather(
        *(
            writer.write_result({"question_id": f"q{index}", "status": "ok"})
            for index in range(100)
        )
    )
    await asyncio.gather(
        *(writer.write_task({"task_id": f"t{index}"}) for index in range(20))
    )
    await writer.close()
    await writer._queue.join()
    await runner
    result_rows = [json.loads(line) for line in results.read_text().splitlines()]
    task_rows = [json.loads(line) for line in tasks.read_text().splitlines()]
    assert len(result_rows) == 100 and len(task_rows) == 20
    assert len({row["question_id"] for row in result_rows}) == 100


def test_resume_loader_ignores_errors_corrupt_lines_and_uses_task_keys(tmp_path):
    path = tmp_path / "r.jsonl"  # One success, one corrupt line, and one failure.
    path.write_text(
        '{"status":"ok","task_id":"task1","question_id":"q1"}\n'
        "not-json\n"
        '{"status":"error","task_id":"task2","question_id":"q2"}\n',
        encoding="utf-8",
    )
    assert _load_completed_question_keys(path) == {("task1", "q1")}


def test_resume_loader_supports_legacy_longmemeval_rows_without_task_id(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text(
        '{"status":"ok","dataset":"longmemeval","question_id":"q1"}\n',
        encoding="utf-8",
    )
    assert _load_completed_question_keys(path) == {("q1", "q1")}


def test_validation_prevents_output_from_overwriting_input(tmp_path, monkeypatch):
    data_file = tmp_path / "data.json"
    data_file.write_text("[]", encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(
        [
            str(data_file),
            "--dataset",
            "longmemeval",
            "--agent",
            "vector",
            "--output-file",
            str(data_file),
        ]
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    with pytest.raises(SystemExit):
        _validate_args(parser, args)


def test_summary_rounding_keeps_small_costs_and_rates():
    rounded = _round_summary_values(
        {
            "wall_runtime_seconds": 1.2345,
            "llm_cost_usd": 0.000012345,
            "llm_input_per_million": 0.123456789,
        }
    )
    assert rounded == {
        "wall_runtime_seconds": 1.23,
        "llm_cost_usd": 0.00001234,
        "llm_input_per_million": 0.12345679,
    }


@pytest.mark.parametrize(
    "model,expected",
    [
        ("gpt-5.6-sol", (5.00, 0.50, 6.25, 30.00, 10.00, 1.00, 12.50, 45.00)),
        ("gpt-5.6-terra", (2.00, 0.20, 2.50, 12.00, 4.00, 0.40, 5.00, 18.00)),
        ("gpt-5.6-luna", (0.20, 0.02, 0.25, 1.20, 0.40, 0.04, 0.50, 1.80)),
    ],
)
def test_default_pricing_includes_cached_and_long_context_rates(
    tmp_path,
    monkeypatch,
    model,
    expected,
):
    data_file = tmp_path / "data.json"  # Empty array is enough for CLI checks.
    data_file.write_text("[]", encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(
        [
            str(data_file),
            "--dataset",
            "longmemeval",
            "--agent",
            "vector",
            "--output-file",
            str(tmp_path / "results.jsonl"),
            "--llm-model",
            model,
        ]
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    _validate_args(parser, args)

    assert (
        args.pricing.llm_input_per_million,
        args.pricing.llm_cached_input_per_million,
        args.pricing.llm_cache_write_per_million,
        args.pricing.llm_output_per_million,
        args.pricing.llm_long_input_per_million,
        args.pricing.llm_long_cached_input_per_million,
        args.pricing.llm_long_cache_write_per_million,
        args.pricing.llm_long_output_per_million,
    ) == expected


@pytest.mark.asyncio
async def test_usage_ledger_prices_cached_short_and_long_requests_correctly():
    pricing = Pricing(
        llm_input_per_million=10.0,
        llm_cached_input_per_million=2.0,
        llm_cache_write_per_million=12.5,
        llm_output_per_million=20.0,
        embedding_input_per_million=0.5,
        llm_long_input_per_million=20.0,
        llm_long_cached_input_per_million=4.0,
        llm_long_cache_write_per_million=25.0,
        llm_long_output_per_million=30.0,
        llm_long_context_threshold=100,
    )
    ledger = UsageLedger(pricing)

    # Exactly at the threshold remains short; cached tokens are a subset of input.
    await ledger.add_llm("memory", 100, 30, 20, 10, False)
    # Above the threshold moves the complete request into the long-context bucket.
    await ledger.add_llm("query", 101, 40, 30, 3, False)
    await ledger.add_embedding("memory", 200, request_count=2, text_count=4)

    snapshot = ledger.snapshot()
    combined = snapshot["combined"]

    assert combined["llm_input_tokens"] == 100
    assert combined["llm_cached_input_tokens"] == 30
    assert combined["llm_cache_write_tokens"] == 20
    assert combined["llm_output_tokens"] == 10
    assert combined["llm_long_input_tokens"] == 101
    assert combined["llm_long_cached_input_tokens"] == 40
    assert combined["llm_long_cache_write_tokens"] == 30
    assert combined["llm_long_output_tokens"] == 3

    short_cost = ((100 - 30 - 20) * 10 + 30 * 2 + 20 * 12.5 + 10 * 20) / 1_000_000
    long_cost = ((101 - 40 - 30) * 20 + 40 * 4 + 30 * 25 + 3 * 30) / 1_000_000
    embedding_cost = 200 * 0.5 / 1_000_000
    assert combined["llm_cost_usd"] == pytest.approx(short_cost + long_cost)
    assert combined["embedding_cost_usd"] == pytest.approx(embedding_cost)
    assert snapshot["combined"]["estimated_cost_usd"] == pytest.approx(
        short_cost + long_cost + embedding_cost
    )


@pytest.mark.asyncio
async def test_usage_ledger_rejects_invalid_cached_token_totals():
    ledger = UsageLedger(
        Pricing(
            llm_input_per_million=1.0,
            llm_cached_input_per_million=0.1,
            llm_cache_write_per_million=1.25,
            llm_output_per_million=2.0,
            embedding_input_per_million=0.02,
        )
    )

    with pytest.raises(ValueError, match="cannot exceed"):
        await ledger.add_llm("query", 10, 6, 5, 1, False)


class _CounterStub(TokenCounter):
    def __init__(self) -> None:
        # These tests only need predictable counting, not a real tokenizer.
        pass

    def count(self, text: str) -> int:
        return len(text)


def _runtime_for_usage_extraction():
    runtime = OpenAIRuntime.__new__(OpenAIRuntime)
    runtime.llm_counter = _CounterStub()
    return runtime


def test_extract_usage_reads_responses_cache_tokens_from_dict():
    runtime = _runtime_for_usage_extraction()
    response = {
        "usage": {
            "input_tokens": 120,
            "output_tokens": 30,
            "input_tokens_details": {
                "cached_tokens": 64,
                "cache_write_tokens": 32,
            },
        }
    }

    assert runtime._extract_usage(response, "prompt", "answer") == (
        120,
        64,
        32,
        30,
        False,
    )


def test_extract_usage_reads_responses_cached_tokens_from_objects():
    runtime = _runtime_for_usage_extraction()
    usage = SimpleNamespace(
        input_tokens=300_001,
        output_tokens=45,
        input_tokens_details=SimpleNamespace(
            cached_tokens=250_000,
            cache_write_tokens=25_000,
        ),
    )
    response = SimpleNamespace(usage=usage)

    assert runtime._extract_usage(response, "prompt", "answer") == (
        300_001,
        250_000,
        25_000,
        45,
        False,
    )


def test_extract_usage_supports_flattened_cache_fields():
    runtime = _runtime_for_usage_extraction()
    response = {
        "usage": {
            "input_tokens": 90,
            "output_tokens": 12,
            "cached_input_tokens": 32,
            "cache_write_tokens": 16,
        }
    }

    assert runtime._extract_usage(response, "prompt", "answer") == (
        90,
        32,
        16,
        12,
        False,
    )


def test_extract_usage_estimate_never_invents_cached_tokens():
    runtime = _runtime_for_usage_extraction()
    response = SimpleNamespace()

    assert runtime._extract_usage(response, "abcd", "xy") == (4, 0, 0, 2, True)


def test_run_summary_keeps_only_public_usage_fields_in_requested_order():
    summary = RunSummary()
    summary.usage.update(
        {
            "llm_requests": 2,
            "estimated_llm_requests": 1,
            "llm_input_tokens": 100,
            "llm_long_input_tokens": 200,
            "llm_cached_input_tokens": 25,
            "llm_long_cached_input_tokens": 75,
            "llm_cache_write_tokens": 15,
            "llm_long_cache_write_tokens": 35,
            "llm_output_tokens": 10,
            "llm_long_output_tokens": 20,
            "llm_cost_usd": 1.25,
            "embedding_requests": 3,
            "embedding_entry_count": 4,
            "embedding_input_tokens": 500,
            "embedding_cost_usd": 0.01,
            "embedding_runtime_seconds": 99.0,
            "estimated_cost_usd": 999.0,
        }
    )

    usage = summary.snapshot()["usage"]
    assert list(usage) == [
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
    ]
    assert "embedding_runtime_seconds" not in usage
    assert "estimated_cost_usd" not in usage
