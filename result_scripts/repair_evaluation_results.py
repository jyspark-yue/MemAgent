#############################################################################
# File: repair_evaluation_results.py
#
# Description:
#   Audits the six core architecture outputs and rebuilds commands for unresolved questions only.
#
#   - Discovers result, task, and summary sidecars for Condensed, Episodic, Graph, HVM, Propositional, and Vector runs.
#   - Finds missing, duplicate, failed, or unresolved task/question pairs.
#   - Reconstructs resume-safe evaluator commands from saved run settings.
#   - Runs the repair commands only when --run is supplied.
#   - Re-audits the results after an executed repair pass.
#############################################################################

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

MEMORY_AGENTS = (
    "condensed",
    "episodic",
    "graph",
    "hvm",
    "propositional",
    "vector",
)

AGENT_CANONICAL = {
    "condensed": "condensed",
    "episodic": "episodic",
    "graph": "graph",
    "hvm": "hvm",
    "proposition": "propositional",
    "propositional": "propositional",
    "vector": "vector",
}

SCALAR_FLAGS = {
    "llm_model": "--llm-model",
    "embedding_model": "--embedding-model",
    "qdrant_url": "--qdrant-url",
    "workers": "--workers",
    "question_concurrency": "--question-concurrency",
    "api_concurrency": "--api-concurrency",
    "embedding_batch_size": "--embedding-batch-size",
    "embedding_batch_token_limit": "--embedding-batch-token-limit",
    "embedding_parallel_batches": "--embedding-parallel-batches",
    "embedding_tokens_per_minute": "--embedding-tokens-per-minute",
    "rate_limit_buffer_seconds": "--rate-limit-buffer-seconds",
    "qdrant_batch_size": "--qdrant-batch-size",
    "ingest_concurrency": "--ingest-concurrency",
    "max_entry_tokens": "--max-entry-tokens",
    "request_timeout": "--request-timeout",
    "qdrant_timeout": "--qdrant-timeout",
    "retry_attempts": "--retry-attempts",
    "retry_base_delay": "--retry-base-delay",
    "retry_max_delay": "--retry-max-delay",
    "start_index": "--start-index",
    "max_tasks": "--max-tasks",
    "source_filter": "--source-filter",
}

PRICING_FLAGS = {
    "llm_input_per_million": "--llm-input-per-million",
    "llm_cached_input_per_million": "--llm-cached-input-per-million",
    "llm_cache_write_per_million": "--llm-cache-write-per-million",
    "llm_output_per_million": "--llm-output-per-million",
    "llm_long_input_per_million": "--llm-long-input-per-million",
    "llm_long_cached_input_per_million": "--llm-long-cached-input-per-million",
    "llm_long_cache_write_per_million": "--llm-long-cache-write-per-million",
    "llm_long_output_per_million": "--llm-long-output-per-million",
    "llm_long_context_threshold": "--llm-long-context-threshold",
    "embedding_input_per_million": "--embedding-input-per-million",
}

QuestionKey = tuple[str, str]  # (task_id, question_id)


@dataclass(slots=True)
class RunFiles:
    summary_file: Path
    result_file: Path
    tasks_file: Path
    config: dict[str, Any]
    successful: set[QuestionKey]
    errors: dict[QuestionKey, list[dict[str, Any]]]
    task_keys: set[QuestionKey]
    result_keys: list[QuestionKey]
    malformed_result_lines: list[int]
    malformed_task_lines: list[int]

    @property
    def dataset(self) -> str:
        return str(self.config["dataset"])

    @property
    def agent(self) -> str:
        raw = str(self.config["agent"])
        return AGENT_CANONICAL.get(raw, raw)

    @property
    def group_key(self) -> tuple[str, str, str | None, tuple[str, ...]]:
        question_types = tuple(self.config.get("question_type_filter") or ())
        source_filter = self.config.get("source_filter")
        return (
            self.dataset,
            str(self.config.get("data_file") or ""),
            str(source_filter) if source_filter is not None else None,
            question_types,
        )


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[int]]:
    rows: list[dict[str, Any]] = []
    malformed: list[int] = []
    if not path.exists():
        return rows, malformed

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed.append(line_number)
                continue
            if isinstance(row, dict):
                rows.append(row)
            else:
                malformed.append(line_number)
    return rows, malformed


def _result_path_from_summary(summary_file: Path) -> Path:
    suffix = "_summary.json"
    if not summary_file.name.endswith(suffix):
        raise ValueError(f"Not an evaluator summary filename: {summary_file}")
    stem = summary_file.name[: -len(suffix)]
    return summary_file.with_name(f"{stem}.jsonl")


def _tasks_path_from_summary(summary_file: Path) -> Path:
    suffix = "_summary.json"
    stem = summary_file.name[: -len(suffix)]
    return summary_file.with_name(f"{stem}_tasks.jsonl")


def load_run(summary_file: Path) -> RunFiles | None:
    try:
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    config = summary.get("configuration")
    if not isinstance(config, dict):
        return None
    raw_agent = str(config.get("agent") or "")
    if raw_agent not in AGENT_CANONICAL or not config.get("dataset"):
        return None

    result_file = _result_path_from_summary(summary_file)
    tasks_file = _tasks_path_from_summary(summary_file)
    result_rows, malformed_result = _read_jsonl(result_file)
    task_rows, malformed_tasks = _read_jsonl(tasks_file)

    successful: set[QuestionKey] = set()
    errors: dict[QuestionKey, list[dict[str, Any]]] = defaultdict(list)
    result_keys: list[QuestionKey] = []

    dataset = str(config.get("dataset") or "")
    for row in result_rows:
        question_id = row.get("question_id")
        if question_id is None:
            continue

        # Backward compatibility for older LongMemEval result rows. Its adapter
        # defines one task per question, so task_id == question_id is unambiguous.
        task_id = row.get("task_id")
        if task_id is None and dataset == "longmemeval":
            task_id = question_id
        if task_id is None:
            continue

        key = (str(task_id), str(question_id))
        result_keys.append(key)
        if row.get("status") == "ok":
            successful.add(key)
        else:
            errors[key].append(row)

    task_keys: set[QuestionKey] = set()
    for task in task_rows:
        task_id = task.get("task_id")
        if task_id is None:
            continue
        for question in task.get("questions") or ():
            if not isinstance(question, dict) or question.get("question_id") is None:
                continue
            task_keys.add((str(task_id), str(question["question_id"])))

    return RunFiles(
        summary_file=summary_file,
        result_file=result_file,
        tasks_file=tasks_file,
        config=config,
        successful=successful,
        errors=dict(errors),
        task_keys=task_keys,
        result_keys=result_keys,
        malformed_result_lines=malformed_result,
        malformed_task_lines=malformed_tasks,
    )


def discover_runs(results_root: Path) -> list[RunFiles]:
    # Only audit the active canonical layout:
    #   results/<dataset>/<agent>/<run>_summary.json
    #
    # This deliberately ignores archival trees such as results/unnecessary/... so an
    # older partial run cannot overwrite the current run for the same
    # dataset/agent during grouping.
    runs: list[RunFiles] = []
    root = results_root.resolve()

    for summary_file in sorted(results_root.rglob("*_summary.json")):
        run = load_run(summary_file)
        if run is None:
            continue

        try:
            relative = summary_file.resolve().relative_to(root)
        except ValueError:
            continue

        parts = relative.parts
        if len(parts) < 3:
            continue

        expected_dataset_dir = run.dataset
        expected_agent_dir = run.agent
        if parts[0] != expected_dataset_dir or parts[1] != expected_agent_dir:
            continue

        runs.append(run)

    return runs


def group_runs(runs: Iterable[RunFiles]) -> dict[tuple[Any, ...], dict[str, RunFiles]]:
    groups: dict[tuple[Any, ...], dict[str, RunFiles]] = defaultdict(dict)
    for run in runs:
        groups[run.group_key][run.agent] = run
    return dict(groups)


def audit(
    results_root: Path,
) -> tuple[list[RunFiles], dict[Path, set[QuestionKey]], int]:
    runs = discover_runs(results_root)
    groups = group_runs(runs)
    unresolved_by_result: dict[Path, set[QuestionKey]] = {}
    total_unresolved = 0

    if not runs:
        print(f"No evaluator summary files found under {results_root}")
        return runs, unresolved_by_result, 0

    print("\nEvaluation completeness audit")
    print("=" * 90)

    for group_key, agent_runs in sorted(groups.items(), key=lambda item: item[0]):
        dataset, data_file, source_filter, question_types = group_key
        expected: set[QuestionKey] = set()
        for run in agent_runs.values():
            expected.update(run.task_keys)
            expected.update(run.result_keys)

        print(f"\n{dataset}")
        print(f"  data: {data_file}")
        if source_filter:
            print(f"  source filter: {source_filter}")
        if question_types:
            print(f"  question types: {', '.join(question_types)}")
        print(f"  expected task/question pairs: {len(expected)}")

        absent_agents = [agent for agent in MEMORY_AGENTS if agent not in agent_runs]
        if absent_agents:
            print(f"  WARNING: no run discovered for: {', '.join(absent_agents)}")

        for agent in MEMORY_AGENTS:
            run = agent_runs.get(agent)
            if run is None:
                continue

            unresolved = expected - run.successful
            unresolved_by_result[run.result_file] = unresolved
            total_unresolved += len(unresolved)
            unresolved_errors = {key for key in unresolved if key in run.errors}
            missing_without_error = unresolved - unresolved_errors
            duplicate_pairs = len(run.result_keys) - len(set(run.result_keys))

            print(
                f"  {agent:14} "
                f"ok={len(run.successful):4}/{len(expected):4}  "
                f"unresolved={len(unresolved):2}  "
                f"errors={len(unresolved_errors):2}  "
                f"missing={len(missing_without_error):2}  "
                f"duplicate-pairs={duplicate_pairs:2}"
            )

            if run.malformed_result_lines:
                print(f"    malformed result lines: {run.malformed_result_lines}")
            if run.malformed_task_lines:
                print(f"    malformed task lines: {run.malformed_task_lines}")

            for task_id, question_id in sorted(unresolved):
                error_rows = run.errors.get((task_id, question_id), [])
                if error_rows:
                    latest = error_rows[-1]
                    error_type = latest.get("error_type") or "unknown error"
                    error_text = str(latest.get("error") or "").replace("\n", " ")
                    if len(error_text) > 160:
                        error_text = error_text[:157] + "..."
                    print(
                        f"    ERROR   task={task_id} q={question_id}: "
                        f"{error_type}: {error_text}"
                    )
                else:
                    print(f"    MISSING task={task_id} q={question_id}")

    # Bare question IDs are not globally unique in every benchmark. Report that
    # explicitly because an evaluator that resumes on question_id alone is unsafe.
    for group_key, agent_runs in sorted(groups.items(), key=lambda item: item[0]):
        dataset = group_key[0]
        if dataset != "mab_accurate_retrieval":
            continue
        for run in agent_runs.values():
            bare_ids = [question_id for _, question_id in run.result_keys]
            if len(bare_ids) != len(set(bare_ids)):
                print(
                    "\nNOTE: mab_accurate_retrieval reuses some bare question_id values "
                    "across tasks. Resume must key on (task_id, question_id), not question_id alone."
                )
                break
        break

    print("\n" + "=" * 90)
    print(f"Total unresolved architecture/question pairs: {total_unresolved}")
    return runs, unresolved_by_result, total_unresolved


def _append_flag(command: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    command.extend((flag, str(value)))


def build_resume_command(
    run: RunFiles,
    *,
    python_executable: str,
    evaluator: Path,
) -> list[str]:
    config = run.config
    command = [
        python_executable,
        "-m",
        "asdrp.evaluate_agents_wcost",
        str(config["data_file"]),
        "--dataset",
        str(config["dataset"]),
        "--agent",
        str(config["agent"]),
        "--output-file",
        str(run.result_file),
    ]

    for key, flag in SCALAR_FLAGS.items():
        _append_flag(command, flag, config.get(key))

    question_types = config.get("question_type_filter") or ()
    if question_types:
        command.append("--question-type-filter")
        command.extend(str(value) for value in question_types)

    if config.get("keep_collections"):
        command.append("--keep-collections")

    pricing = config.get("pricing") or {}
    if isinstance(pricing, dict):
        for key, flag in PRICING_FLAGS.items():
            _append_flag(command, flag, pricing.get(key))

    # Deliberately never pass --overwrite: the evaluator's resume behavior is
    # what makes this rerun only unresolved questions and append new successes.
    return command


def shell_display(command: list[str]) -> str:
    # POSIX-safe enough for display without requiring a shell to execute it.
    import shlex

    return " \\\n  ".join(shlex.quote(part) for part in command)


def rerun_unresolved(
    runs: list[RunFiles],
    unresolved_by_result: dict[Path, set[QuestionKey]],
    *,
    project_root: Path,
    evaluator: Path,
    python_executable: str,
    execute: bool,
) -> None:
    pending_runs = [run for run in runs if unresolved_by_result.get(run.result_file)]

    if not pending_runs:
        print("\nNothing needs to be rerun.")
        return

    print("\nResume commands")
    print("=" * 90)
    for run in pending_runs:
        unresolved = unresolved_by_result[run.result_file]
        command = build_resume_command(
            run,
            python_executable=python_executable,
            evaluator=evaluator,
        )
        print(
            f"\n{run.dataset} / {run.agent}: " f"{len(unresolved)} unresolved pair(s)"
        )
        print(shell_display(command))

        if execute:
            subprocess.run(command, cwd=project_root, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit memory-evaluation JSONL files and resume only unresolved "
            "task/question pairs using each run's saved configuration."
        )
    )
    parser.add_argument(
        "results_root",
        type=Path,
        help="Root containing the evaluator result/summary/task files.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="MemAgent project root used to resolve saved relative data paths.",
    )
    parser.add_argument(
        "--evaluator",
        type=Path,
        default=Path("../asdrp/evaluate_agents_wcost.py"),
        help=(
            "Evaluator script to invoke. Use the resume-safe evaluator version so "
            "MAB Accurate Retrieval duplicate question IDs are handled correctly."
        ),
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used for reruns (normally the active .venv Python).",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually execute the resume commands. Without this flag, only audit/print.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    results_root = args.results_root.expanduser().resolve()
    project_root = args.project_root.expanduser().resolve()
    evaluator = args.evaluator
    if not evaluator.is_absolute():
        evaluator = project_root / evaluator

    runs, unresolved_by_result, unresolved_count = audit(results_root)
    rerun_unresolved(
        runs,
        unresolved_by_result,
        project_root=project_root,
        evaluator=evaluator,
        python_executable=args.python,
        execute=args.run,
    )

    if args.run:
        print("\nPost-rerun audit")
        audit(results_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
