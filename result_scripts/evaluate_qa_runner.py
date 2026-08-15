#############################################################################
# File: evaluate_qa_runner.py
#
# Description:
#   Runs resumable LLM-based QA judging across saved memory-evaluation runs.
#
#   - Discovers evaluator result files and their summary/task sidecars.
#   - Supports LongMemEval, EcommerceMemEval, and MemoryAgentBench result formats.
#   - Reuses saved judgments only when the current question, answer, and hypothesis still match.
#   - Computes overall, source, question-type, and abstention metrics.
#   - Writes per-run reports and one consolidated metrics report.
#   - Supports filtered runs, retries, parallel judges, and OpenAI-compatible endpoints.
#############################################################################

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import os
import random
import re
import sys
import threading
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from tqdm import tqdm

try:
    import openai
    from openai import OpenAI
except ImportError:  # Allows --list-runs/config inspection without the API package.
    openai = None
    OpenAI = None


MODEL_ZOO = {
    "llama-3.1-70b-instruct": (
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "local",
    ),
    "gpt-4o-mini": ("gpt-4o-mini-2024-07-18", "openai"),
    "gpt-4o": ("gpt-4o-2024-08-06", "openai"),
}

MEMORY_BLOCK_ORDER = (
    "episodic",
    "propositional",
    "condensed",
    "vector",
    "hvm",
    "graph",
)

ROUTING_SYSTEM_ORDER = ("binary_router",)

RUN_SYSTEM_ORDER = MEMORY_BLOCK_ORDER + ROUTING_SYSTEM_ORDER

BINARY_ROUTER_QUESTION_TYPES = (
    "single-session-preference",
    "knowledge-update",
)

BINARY_ROUTER_ROUTES = {
    "single-session-preference": "hvm",
    "knowledge-update": "episodic",
}

DATASET_ORDER = (
    "longmemeval",
    "ecommerce",
    "mab_accurate_retrieval",
    "mab_long_range_understanding",
    "mab_test_time_learning",
    "mab_conflict_resolution",
)

LONGMEMEVAL_QUESTION_TYPES = (
    "single-session-user",
    "single-session-preference",
    "single-session-assistant",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
)

RUN_IDENTITY_CONFIG_KEYS = {"agent", "dataset", "data_file"}

# Directories under the results root that are not active evaluation inputs.
IGNORED_RESULT_DIRS = {"unnecessary", "qa_evaluations"}

DATASET_ALIASES = {
    "lme": "longmemeval",
    "longmemeval": "longmemeval",
    "ecommerce": "ecommerce",
    "ecom": "ecommerce",
    "ar": "mab_accurate_retrieval",
    "mab_ar": "mab_accurate_retrieval",
    "mab_accurate_retrieval": "mab_accurate_retrieval",
    "accurate_retrieval": "mab_accurate_retrieval",
    "lru": "mab_long_range_understanding",
    "mab_lru": "mab_long_range_understanding",
    "mab_long_range_understanding": "mab_long_range_understanding",
    "long_range_understanding": "mab_long_range_understanding",
    "ttl": "mab_test_time_learning",
    "mab_ttl": "mab_test_time_learning",
    "mab_test_time_learning": "mab_test_time_learning",
    "test_time_learning": "mab_test_time_learning",
    "cr": "mab_conflict_resolution",
    "mab_cr": "mab_conflict_resolution",
    "mab_conflict_resolution": "mab_conflict_resolution",
    "conflict_resolution": "mab_conflict_resolution",
}

AGENT_ALIASES = {
    "episodic": "episodic",
    "proposition": "propositional",
    "propositional": "propositional",
    "condensed": "condensed",
    "vector": "vector",
    "hvm": "hvm",
    "graph": "graph",
    "binary_router": "binary_router",
}

_thread_local = threading.local()


@dataclass(frozen=True)
class ModelSpec:
    requested_name: str
    model: str
    source: str
    base_url: str | None


@dataclass(frozen=True)
class RunSpec:
    memory_block: str
    dataset: str
    hypothesis_file: Path
    task_file: Path | None
    summary_file: Path | None
    configuration: dict[str, Any]
    run_id: str | None

    @property
    def label(self) -> str:
        return f"{self.memory_block}/{self.dataset}"


class EvaluationError(RuntimeError):
    pass


class JudgeResponseError(EvaluationError):
    pass


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return slug or "model"


def normalize_memory_block(value: str) -> str:
    key = value.strip().lower().replace("-", "_")
    normalized = AGENT_ALIASES.get(key)
    if normalized is None:
        raise ValueError(f"Unknown memory block: {value}")
    return normalized


def normalize_dataset(value: str) -> str:
    key = value.strip().lower().replace("-", "_").replace("/", "_")
    normalized = DATASET_ALIASES.get(key)
    if normalized is None:
        raise ValueError(f"Unknown dataset: {value}")
    return normalized


def memory_block_sort_key(value: str) -> tuple[int, str]:
    try:
        return (RUN_SYSTEM_ORDER.index(value), value)
    except ValueError:
        return (len(RUN_SYSTEM_ORDER), value)


def dataset_sort_key(value: str) -> tuple[int, str]:
    try:
        return (DATASET_ORDER.index(value), value)
    except ValueError:
        return (len(DATASET_ORDER), value)


def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Could not read JSON file {path}: {exc}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise EvaluationError(
                        f"Invalid JSONL in {path} at line {line_number}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise EvaluationError(
                        f"Expected an object in {path} at line {line_number}."
                    )
                rows.append(row)
    except OSError as exc:
        raise EvaluationError(f"Could not read JSONL file {path}: {exc}") from exc
    return rows


def write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        handle.write(content.rstrip())
        handle.write("\n")
    temp_path.replace(path)


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False))
            handle.write("\n")
    temp_path.replace(path)


def append_jsonl(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(row, ensure_ascii=False, allow_nan=False)
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.write("\n")
            handle.flush()


def first_jsonl_row(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise EvaluationError(
                        f"Invalid JSONL in {path} at line {line_number}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise EvaluationError(f"Expected an object in {path}.")
                return row
    except OSError as exc:
        raise EvaluationError(f"Could not read {path}: {exc}") from exc
    raise EvaluationError(f"No JSON records found in {path}.")


def find_sidecar(primary: Path, suffix: str) -> Path | None:
    candidate = primary.with_name(f"{primary.stem}{suffix}")
    return candidate if candidate.is_file() else None


def infer_dataset_from_path(path: Path) -> str | None:
    parts = [part.lower() for part in path.parts]
    if "longmemeval" in parts:
        return "longmemeval"
    if "ecommerce" in parts:
        return "ecommerce"
    if "mab" in parts:
        mab_index = parts.index("mab")
        if mab_index + 1 < len(parts):
            category = parts[mab_index + 1]
            return {
                "ar": "mab_accurate_retrieval",
                "lru": "mab_long_range_understanding",
                "ttl": "mab_test_time_learning",
                "cr": "mab_conflict_resolution",
            }.get(category)
    return None


def discover_runs(results_dir: Path) -> list[RunSpec]:
    if not results_dir.is_dir():
        raise EvaluationError(f"Results directory does not exist: {results_dir}")

    primary_files = sorted(
        path
        for path in results_dir.rglob("*.jsonl")
        if path.is_file()
        and not path.name.endswith("_tasks.jsonl")
        and ".eval-results-" not in path.name
        and not any(
            part.lower() in IGNORED_RESULT_DIRS
            for part in path.relative_to(results_dir).parts[:-1]
        )
    )

    runs: list[RunSpec] = []
    for primary in primary_files:
        summary_file = find_sidecar(primary, "_summary.json")
        task_file = find_sidecar(primary, "_tasks.jsonl")

        configuration: dict[str, Any] = {}
        run_id: str | None = None
        if summary_file is not None:
            summary = read_json(summary_file)
            if not isinstance(summary, dict):
                raise EvaluationError(f"Summary must be a JSON object: {summary_file}")
            raw_configuration = summary.get("configuration") or {}
            if not isinstance(raw_configuration, dict):
                raise EvaluationError(
                    f"'configuration' must be an object in {summary_file}."
                )
            configuration = raw_configuration
            run_id_value = summary.get("run_id")
            run_id = str(run_id_value) if run_id_value is not None else None

        sample = first_jsonl_row(primary)

        raw_agent = configuration.get("../asdrp/agent") or primary.parent.name
        try:
            memory_block = normalize_memory_block(str(raw_agent))
        except ValueError:
            memory_block = normalize_memory_block(primary.parent.name)

        raw_dataset = configuration.get("dataset") or sample.get("dataset")
        dataset: str | None = None
        if raw_dataset is not None:
            try:
                dataset = normalize_dataset(str(raw_dataset))
            except ValueError:
                dataset = None
        if dataset is None:
            dataset = infer_dataset_from_path(primary)
        if dataset is None:
            raise EvaluationError(
                f"Could not identify dataset for hypothesis file: {primary}"
            )

        runs.append(
            RunSpec(
                memory_block=memory_block,
                dataset=dataset,
                hypothesis_file=primary,
                task_file=task_file,
                summary_file=summary_file,
                configuration=configuration,
                run_id=run_id,
            )
        )

    runs.sort(
        key=lambda run: (
            memory_block_sort_key(run.memory_block),
            dataset_sort_key(run.dataset),
            str(run.hypothesis_file),
        )
    )

    duplicates: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for run in runs:
        duplicates[(run.memory_block, run.dataset)].append(run.hypothesis_file)
    duplicate_groups = {
        key: paths for key, paths in duplicates.items() if len(paths) > 1
    }
    if duplicate_groups:
        details = "; ".join(
            f"{memory}/{dataset}: {', '.join(str(path) for path in paths)}"
            for (memory, dataset), paths in duplicate_groups.items()
        )
        raise EvaluationError(
            "Multiple primary result files map to the same memory-block/dataset pair. "
            f"Disambiguate the results tree before evaluating. {details}"
        )

    return runs


def select_runs(
    runs: list[RunSpec],
    memory_blocks: set[str] | None,
    datasets: set[str] | None,
    selected_file: Path | None,
) -> list[RunSpec]:
    selected = runs

    if selected_file is not None:
        resolved = selected_file.resolve()
        selected = [
            run for run in selected if run.hypothesis_file.resolve() == resolved
        ]
        if not selected:
            raise EvaluationError(
                f"Requested --file is not a discovered primary result JSONL: {selected_file}"
            )

    if memory_blocks:
        selected = [run for run in selected if run.memory_block in memory_blocks]
    if datasets:
        selected = [run for run in selected if run.dataset in datasets]

    if not selected:
        raise EvaluationError("No runs matched the requested filters.")
    return selected


def record_key(entry: dict[str, Any]) -> tuple[str, str]:
    task_id = entry.get("task_id")
    question_id = entry.get("question_id")
    return (
        "" if task_id is None else str(task_id),
        "" if question_id is None else str(question_id),
    )


def load_question_metadata(
    task_file: Path | None,
) -> dict[tuple[str, str], dict[str, Any]]:
    if task_file is None:
        return {}

    metadata: dict[tuple[str, str], dict[str, Any]] = {}
    for task in read_jsonl(task_file):
        questions = task.get("questions") or []
        if not isinstance(questions, list):
            continue
        for question in questions:
            if not isinstance(question, dict):
                continue
            question_id = question.get("question_id")
            if question_id is None:
                continue
            question_metadata = question.get("question_metadata") or {}
            if not isinstance(question_metadata, dict):
                question_metadata = {}
            metadata[(str(task.get("task_id") or ""), str(question_id))] = (
                question_metadata
            )
    return metadata


def load_canonical_longmemeval_types(run: RunSpec) -> dict[str, str]:
    """Load question_id -> canonical question_type from the dataset itself."""
    raw_path = run.configuration.get("data_file")
    if not raw_path:
        raise EvaluationError(
            f"{run.label} is missing configuration.data_file; "
            "cannot canonicalize LongMemEval question types."
        )

    data_path = Path(str(raw_path))
    data = read_json(data_path)
    if isinstance(data, dict):
        data = data.get("data") or data.get("records") or data.get("questions")
    if not isinstance(data, list):
        raise EvaluationError(
            f"Expected LongMemEval dataset list in {data_path}, got {type(data).__name__}."
        )

    mapping: dict[str, str] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        question_id = entry.get("question_id")
        question_type = entry.get("question_type")
        if question_id is None or question_type is None:
            continue
        question_id = str(question_id)
        question_type = str(question_type)
        previous = mapping.get(question_id)
        if previous is not None and previous != question_type:
            raise EvaluationError(
                f"Conflicting canonical LongMemEval question types for {question_id}: "
                f"{previous!r} vs {question_type!r}."
            )
        mapping[question_id] = question_type

    if not mapping:
        raise EvaluationError(f"No LongMemEval question metadata found in {data_path}.")
    return mapping


def normalize_reference_answers(entry: dict[str, Any]) -> list[str]:
    references = entry.get("reference_answers")
    if references is None and "answer" in entry:
        references = entry.get("answer")

    if references is None:
        return []
    if isinstance(references, list):
        return [str(value) for value in references if value is not None]
    return [str(references)]


def format_reference_answers(references: list[str]) -> str:
    if not references:
        return ""
    if len(references) == 1:
        return references[0]
    return "\n".join(f"- {answer}" for answer in references)


def is_abstention(question_id: str, question_type: str | None) -> bool:
    return "_abs" in question_id or question_type == "abstention"


def get_anscheck_prompt(
    dataset: str,
    question_type: str | None,
    question: str,
    references: list[str],
    response: str,
    abstention: bool,
) -> tuple[str, str]:
    answer = format_reference_answers(references)

    if abstention:
        template = (
            "I will give you an unanswerable question, an explanation, and a response "
            "from a model. Please answer yes if the model correctly identifies the "
            "question as unanswerable. The model could say that the information is "
            "incomplete, or some other information is given but the asked information "
            "is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: "
            "{}\n\nDoes the model correctly identify the question as unanswerable? "
            "Answer yes or no only."
        )
        return template.format(question, answer, response), "abstention"

    if question_type in {
        "single-session-user",
        "single-session-assistant",
        "multi-session",
    }:
        template = (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, "
            "answer no. If the response is equivalent to the correct answer or contains "
            "all the intermediate steps to get the correct answer, you should also answer "
            "yes. If the response only contains a subset of the information required by "
            "the answer, answer no.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel "
            "Response: {}\n\nIs the model response correct? Answer yes or no only."
        )
        return template.format(question, answer, response), "longmemeval_standard"

    if question_type == "temporal-reasoning":
        template = (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, "
            "answer no. If the response is equivalent to the correct answer or contains "
            "all the intermediate steps to get the correct answer, you should also answer "
            "yes. If the response only contains a subset of the information required by "
            "the answer, answer no. In addition, do not penalize off-by-one errors for the "
            "number of days. If the question asks for the number of days/weeks/months, "
            "etc., and the model makes off-by-one errors (e.g., predicting 19 days when "
            "the answer is 18), the model's response is still correct.\n\nQuestion: "
            "{}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response "
            "correct? Answer yes or no only."
        )
        return template.format(question, answer, response), "longmemeval_temporal"

    if question_type == "knowledge-update":
        template = (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, "
            "answer no. If the response contains some previous information along with an "
            "updated answer, the response should be considered as correct as long as the "
            "updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: "
            "{}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no "
            "only."
        )
        return (
            template.format(question, answer, response),
            "longmemeval_knowledge_update",
        )

    if question_type in {"temporal_update", "conflict_resolution"}:
        template = (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, "
            "answer no. If the response contains previous or conflicting information "
            "along with an updated answer, consider the response correct only when the "
            "required current answer is clearly identified as the answer to the question."
            "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the "
            "model response correct? Answer yes or no only."
        )
        return template.format(question, answer, response), "update_or_conflict"

    if question_type == "single-session-preference":
        template = (
            "I will give you a question, a rubric for a desired personalized response, "
            "and a response from a model. Please answer yes if the response satisfies the "
            "desired response. Otherwise, answer no. The model does not need to reflect "
            "all the points in the rubric. The response is correct as long as it recalls "
            "and utilizes the user's personal information correctly.\n\nQuestion: "
            "{}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? "
            "Answer yes or no only."
        )
        return template.format(question, answer, response), "longmemeval_preference"

    if dataset == "mab_long_range_understanding":
        template = (
            "I will give you a question, a reference answer, and a response from a model. "
            "Judge whether the model response satisfies the requested task to the same "
            "substantive standard as the reference answer. Do not require identical wording. "
            "For requested summaries, allow differences in phrasing and organization, but "
            "answer no if the response materially omits, invents, or contradicts important "
            "plot/character information or violates an explicit instruction in the question."
            "\n\nQuestion: {}\n\nReference Answer: {}\n\nModel Response: {}\n\nIs the "
            "model response correct? Answer yes or no only."
        )
        return template.format(question, answer, response), "mab_long_range"

    if len(references) > 1:
        reference_label = "Acceptable Reference Answers"
        correctness_clause = (
            "The listed reference answers are acceptable alternatives; matching any one "
            "of them semantically is sufficient when it fully answers the question."
        )
    else:
        reference_label = "Correct Answer"
        correctness_clause = (
            "Do not require exact wording; semantic equivalence is sufficient."
        )

    template = (
        "I will give you a question, one or more reference answers, and a response from "
        "a model. Answer yes if the model response is correct and fully satisfies the "
        "question; otherwise answer no. {} If the model response contradicts the reference, "
        "fails to provide required information, or provides only a materially incomplete "
        "subset of a required answer, answer no.\n\nQuestion: {}\n\n{}: {}\n\nModel "
        "Response: {}\n\nIs the model response correct? Answer yes or no only."
    )
    return (
        template.format(
            correctness_clause, question, reference_label, answer, response
        ),
        "generic_reference_answer",
    )


def resolve_model(metric_model: str, base_url: str | None) -> ModelSpec:
    if metric_model in MODEL_ZOO:
        model, source = MODEL_ZOO[metric_model]
    else:
        model = metric_model
        source = "local" if base_url else "openai"

    effective_base_url = base_url
    if source == "local" and effective_base_url is None:
        effective_base_url = "http://localhost:8001/v1"

    return ModelSpec(
        requested_name=metric_model,
        model=model,
        source=source,
        base_url=effective_base_url,
    )


def get_thread_client(model_spec: ModelSpec) -> Any:
    if OpenAI is None:
        raise EvaluationError(
            "The openai package is required for judging. Install the project dependencies "
            "before running evaluation."
        )

    cache_key = (model_spec.source, model_spec.base_url, model_spec.model)
    cached_key = getattr(_thread_local, "client_key", None)
    if cached_key == cache_key:
        client = getattr(_thread_local, "client", None)
        if client is not None:
            return client

    if model_spec.source == "local":
        api_key = os.getenv("OPENAI_API_KEY") or "EMPTY"
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EvaluationError(
                "OPENAI_API_KEY is not set. Add it to the environment/.env, or use "
                "--base-url for an OpenAI-compatible local server."
            )

    kwargs: dict[str, Any] = {"api_key": api_key}
    if model_spec.base_url:
        kwargs["base_url"] = model_spec.base_url
    organization = os.getenv("OPENAI_ORGANIZATION")
    if organization and model_spec.source == "openai":
        kwargs["organization"] = organization

    client = OpenAI(**kwargs)
    _thread_local.client = client
    _thread_local.client_key = cache_key
    return client


def retryable_exception(exc: Exception) -> bool:
    if openai is None:
        return False

    retryable_types = tuple(
        cls
        for cls in (
            getattr(openai, "RateLimitError", None),
            getattr(openai, "APIConnectionError", None),
            getattr(openai, "APITimeoutError", None),
            getattr(openai, "InternalServerError", None),
        )
        if isinstance(cls, type)
    )
    if retryable_types and isinstance(exc, retryable_types):
        return True

    status_code = getattr(exc, "status_code", None)
    return status_code in {408, 409, 429, 500, 502, 503, 504}


def parse_yes_no(response: str) -> bool:
    normalized = response.strip().lower()
    match = re.match(r"^[\s\W]*(yes|no)\b", normalized)
    if not match:
        raise JudgeResponseError(
            f"Judge returned neither an initial 'yes' nor 'no': {response!r}"
        )
    return match.group(1) == "yes"


def judge_prompt(
    model_spec: ModelSpec,
    prompt: str,
    max_retries: int,
    retry_base_delay: float,
    retry_max_delay: float,
) -> tuple[bool, str]:
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            client = get_thread_client(model_spec)

            if model_spec.source == "openai" and model_spec.model.startswith("gpt-5.6"):
                completion = client.responses.create(
                    model=model_spec.model,
                    input=prompt,
                    reasoning={"effort": "low"},
                    text={"verbosity": "low"},
                )
                content = completion.output_text
            else:
                completion = client.chat.completions.create(
                    model=model_spec.model,
                    messages=[{"role": "user", "content": prompt}],
                    n=1,
                    temperature=0,
                    max_tokens=10,
                )
                content = completion.choices[0].message.content

            if content is None:
                raise JudgeResponseError("Judge response content was null.")
            response = content.strip()
            return parse_yes_no(response), response
        except Exception as exc:  # Deliberately classified below before retrying.
            last_error = exc
            should_retry = retryable_exception(exc) or isinstance(
                exc, JudgeResponseError
            )
            if not should_retry or attempt >= max_retries:
                break
            delay = min(retry_max_delay, retry_base_delay * (2**attempt))
            delay *= random.uniform(0.8, 1.2)
            time.sleep(max(0.0, delay))

    assert last_error is not None
    raise last_error


def build_evaluated_entry(
    entry: dict[str, Any],
    dataset: str,
    question_metadata: dict[str, Any],
    model_spec: ModelSpec,
    max_retries: int,
    retry_base_delay: float,
    retry_max_delay: float,
) -> dict[str, Any]:
    output = copy.deepcopy(entry)
    question_id = str(entry.get("question_id", ""))
    question = entry.get("question")
    hypothesis = entry.get("hypothesis")
    references = normalize_reference_answers(entry)
    question_type_value = question_metadata.get("question_type")
    question_type = (
        str(question_type_value) if question_type_value is not None else None
    )

    if not question_id:
        raise EvaluationError("Result entry is missing question_id.")
    if not isinstance(question, str) or not question:
        raise EvaluationError(f"Question {question_id} is missing question text.")
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        raise EvaluationError(f"Question {question_id} has no hypothesis to evaluate.")
    if not references:
        raise EvaluationError(f"Question {question_id} has no reference answer.")

    abstention = is_abstention(question_id, question_type)
    prompt, prompt_type = get_anscheck_prompt(
        dataset=dataset,
        question_type=question_type,
        question=question,
        references=references,
        response=hypothesis,
        abstention=abstention,
    )
    label, raw_response = judge_prompt(
        model_spec=model_spec,
        prompt=prompt,
        max_retries=max_retries,
        retry_base_delay=retry_base_delay,
        retry_max_delay=retry_max_delay,
    )
    output["autoeval_label"] = {
        "model": model_spec.model,
        "requested_model": model_spec.requested_name,
        "label": label,
        "status": "ok",
        "prompt_type": prompt_type,
        "judge_response": raw_response,
    }
    return output


def build_skipped_generation_entry(
    entry: dict[str, Any], model_spec: ModelSpec
) -> dict[str, Any]:
    output = copy.deepcopy(entry)
    output["autoeval_label"] = {
        "model": model_spec.model,
        "requested_model": model_spec.requested_name,
        "label": None,
        "status": "skipped_generation_error",
    }
    return output


def build_judge_error_entry(
    entry: dict[str, Any], model_spec: ModelSpec, exc: Exception
) -> dict[str, Any]:
    output = copy.deepcopy(entry)
    output["autoeval_label"] = {
        "model": model_spec.model,
        "requested_model": model_spec.requested_name,
        "label": None,
        "status": "judge_error",
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    return output


def output_path_for_run(output_dir: Path, run: RunSpec, model_spec: ModelSpec) -> Path:
    model_slug = safe_slug(model_spec.requested_name)
    filename = f"{run.hypothesis_file.stem}.eval-results-{model_slug}.jsonl"
    return output_dir / run.memory_block / run.dataset / filename


def reusable_result_matches(
    cached: dict[str, Any],
    current: dict[str, Any],
    *,
    dataset: str,
    question_metadata: dict[str, Any],
    model_spec: ModelSpec,
) -> bool:
    """Return whether a saved judgment still represents the current JSONL row.

    Resume state is intentionally content-aware. A question ID identifies where a
    judgment belongs, but it is not enough to prove that the judgment is still valid:
    a regenerated run may keep the same ID while changing its hypothesis. Compare the
    actual judge inputs already present in the source/evaluation rows instead of
    persisting a separate fingerprint.
    """

    autoeval = cached.get("autoeval_label")
    if not isinstance(autoeval, dict) or not isinstance(autoeval.get("label"), bool):
        return False

    model_matches = (
        autoeval.get("model") == model_spec.model
        or autoeval.get("requested_model") == model_spec.requested_name
    )
    if not model_matches:
        return False

    # The hypothesis is the most important resume guard. Keep the comparison exact so
    # even a small regenerated-answer change forces a fresh judgment.
    current_hypothesis = current.get("hypothesis")
    if not isinstance(current_hypothesis, str):
        return False
    if cached.get("hypothesis") != current_hypothesis:
        return False

    # Question/reference changes also alter the judge prompt and must invalidate cache.
    if cached.get("question") != current.get("question"):
        return False
    if normalize_reference_answers(cached) != normalize_reference_answers(current):
        return False

    question_id = str(current.get("question_id") or "")
    question_type_value = question_metadata.get("question_type")
    question_type = (
        str(question_type_value) if question_type_value is not None else None
    )
    references = normalize_reference_answers(current)
    _, expected_prompt_type = get_anscheck_prompt(
        dataset=dataset,
        question_type=question_type,
        question=str(current.get("question") or ""),
        references=references,
        response=current_hypothesis,
        abstention=is_abstention(question_id, question_type),
    )
    cached_prompt_type = autoeval.get("prompt_type")
    if cached_prompt_type is not None and cached_prompt_type != expected_prompt_type:
        return False
    return True


def load_reusable_results(
    paths: Iterable[Path],
    model_spec: ModelSpec,
    *,
    current_entries: dict[tuple[str, str], dict[str, Any]],
    dataset: str,
    question_metadata: dict[tuple[str, str], dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Load only judgments whose saved judge inputs still match current rows.

    ``(task_id, question_id)`` remains the lookup key so datasets with multiple
    questions per context (for example Ecommerce and MemoryAgentBench) stay
    unambiguous. The key is only an index; reuse is authorized by direct content
    comparison, especially the current hypothesis, rather than by persisted identity
    state.
    """

    reusable: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            continue
        for row in read_jsonl(path):
            key = record_key(row)
            current = current_entries.get(key)
            if current is None:
                continue
            if reusable_result_matches(
                row,
                current,
                dataset=dataset,
                question_metadata=question_metadata.get(key, {}),
                model_spec=model_spec,
            ):
                reusable[key] = row
    return reusable


def legacy_source_result_path(run: RunSpec, model_spec: ModelSpec) -> Path:
    return Path(
        f"{run.hypothesis_file}.eval-results-{safe_slug(model_spec.requested_name)}"
    )


def evaluate_run(
    run: RunSpec,
    output_dir: Path,
    model_spec: ModelSpec,
    judge_workers: int,
    max_retries: int,
    retry_base_delay: float,
    retry_max_delay: float,
    resume: bool,
    reuse_legacy_source_results: bool,
    verbose_judgments: bool,
) -> tuple[list[dict[str, Any]], Path, dict[tuple[str, str], dict[str, Any]]]:
    hypotheses = read_jsonl(run.hypothesis_file)
    question_metadata = load_question_metadata(run.task_file)
    output_path = output_path_for_run(output_dir, run, model_spec)
    checkpoint_path = output_path.with_name(f"{output_path.name}.partial")

    reusable_paths: list[Path] = []
    if resume:
        reusable_paths.extend([output_path, checkpoint_path])
        if reuse_legacy_source_results:
            reusable_paths.append(legacy_source_result_path(run, model_spec))
    else:
        for path in (output_path, checkpoint_path):
            if path.exists():
                path.unlink()

    input_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    input_order: list[tuple[str, str]] = []
    generation_errors: dict[tuple[str, str], dict[str, Any]] = {}

    for entry in hypotheses:
        question_id_value = entry.get("question_id")
        if question_id_value is None:
            raise EvaluationError(
                f"Entry in {run.hypothesis_file} is missing question_id."
            )
        key = record_key(entry)
        if key in input_by_key:
            raise EvaluationError(
                "Duplicate (task_id, question_id) pair "
                f"{key!r} in {run.hypothesis_file}."
            )
        input_by_key[key] = entry
        input_order.append(key)

        if entry.get("status") != "ok" or not isinstance(entry.get("hypothesis"), str):
            generation_errors[key] = build_skipped_generation_entry(entry, model_spec)

    reusable = load_reusable_results(
        reusable_paths,
        model_spec,
        current_entries=input_by_key,
        dataset=run.dataset,
        question_metadata=question_metadata,
    )
    to_evaluate = [
        entry
        for key, entry in input_by_key.items()
        if key not in generation_errors and key not in reusable
    ]

    if run.dataset == "longmemeval":
        missing_types = [
            str(entry["question_id"])
            for entry in hypotheses
            if entry.get("status") == "ok"
            and record_key(entry) not in question_metadata
        ]
        if missing_types:
            preview = ", ".join(missing_types[:5])
            raise EvaluationError(
                "LongMemEval evaluation requires question_type metadata, but it was "
                f"missing for {len(missing_types)} questions in {run.task_file}. "
                f"Examples: {preview}"
            )

        if run.memory_block == "binary_router":
            observed_types = {
                str(question_metadata[record_key(entry)].get("question_type"))
                for entry in hypotheses
                if entry.get("status") == "ok"
                and record_key(entry) in question_metadata
            }
            unsupported_types = sorted(
                question_type
                for question_type in observed_types
                if question_type not in BINARY_ROUTER_QUESTION_TYPES
            )
            if unsupported_types:
                raise EvaluationError(
                    "binary_router LongMemEval results must be restricted to "
                    "single-session-preference and knowledge-update. Found unsupported "
                    f"question types: {', '.join(unsupported_types)}"
                )

    checkpoint_lock = threading.Lock()
    newly_evaluated: dict[tuple[str, str], dict[str, Any]] = {}

    if to_evaluate:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        description = f"{run.memory_block} | {run.dataset}"
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=judge_workers
        ) as executor:
            future_to_key = {
                executor.submit(
                    build_evaluated_entry,
                    entry,
                    run.dataset,
                    question_metadata.get(record_key(entry), {}),
                    model_spec,
                    max_retries,
                    retry_base_delay,
                    retry_max_delay,
                ): record_key(entry)
                for entry in to_evaluate
            }

            for future in tqdm(
                concurrent.futures.as_completed(future_to_key),
                total=len(future_to_key),
                desc=description,
                unit="q",
            ):
                key = future_to_key[future]
                source_entry = input_by_key[key]
                try:
                    evaluated = future.result()
                except Exception as exc:
                    evaluated = build_judge_error_entry(source_entry, model_spec, exc)
                newly_evaluated[key] = evaluated
                append_jsonl(checkpoint_path, evaluated, checkpoint_lock)

                if verbose_judgments:
                    label_info = evaluated.get("autoeval_label", {})
                    print(
                        json.dumps(
                            {
                                "memory_block": run.memory_block,
                                "dataset": run.dataset,
                                "task_id": key[0],
                                "question_id": key[1],
                                "label": label_info.get("label"),
                                "status": label_info.get("status"),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

    combined: list[dict[str, Any]] = []
    for key in input_order:
        if key in generation_errors:
            combined.append(generation_errors[key])
        elif key in newly_evaluated:
            combined.append(newly_evaluated[key])
        elif key in reusable:
            combined.append(reusable[key])
        else:
            combined.append(
                build_judge_error_entry(
                    input_by_key[key],
                    model_spec,
                    EvaluationError("No evaluation result was produced."),
                )
            )

    write_jsonl_atomic(output_path, combined)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    return combined, output_path, question_metadata


def metric_bucket(labels: list[bool]) -> dict[str, Any]:
    correct = sum(1 for label in labels if label)
    count = len(labels)
    return {
        "accuracy": round(correct / count, 6) if count else None,
        "correct": correct,
        "count": count,
    }


def calculate_metrics(
    run: RunSpec,
    evaluated_rows: list[dict[str, Any]],
    question_metadata: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    evaluated_labels: list[bool] = []
    type_labels: dict[str, list[bool]] = defaultdict(list)
    source_labels: dict[str, list[bool]] = defaultdict(list)
    abstention_labels: list[bool] = []
    canonical_lme_types = (
        load_canonical_longmemeval_types(run) if run.dataset == "longmemeval" else None
    )

    generation_errors = 0
    judge_errors = 0
    unevaluated = 0

    for row in evaluated_rows:
        question_id = str(row.get("question_id", ""))
        if canonical_lme_types is not None:
            qtype_value = canonical_lme_types.get(question_id)
            if qtype_value is None:
                raise EvaluationError(
                    f"LongMemEval question_id {question_id!r} is not present in the "
                    "canonical dataset configured for this run."
                )
            # Abstention is an exclusive seventh reporting category. LongMemEval
            # abstention examples retain a base question_type but are identified
            # by the _abs question-id suffix.
            question_type = (
                "abstention"
                if is_abstention(question_id, str(qtype_value))
                else str(qtype_value)
            )
        else:
            metadata = question_metadata.get(record_key(row), {})
            qtype_value = metadata.get("question_type")
            question_type = (
                str(qtype_value) if qtype_value is not None else "unspecified"
            )
        source = str(row.get("source") or "unspecified")

        autoeval = row.get("autoeval_label")
        if not isinstance(autoeval, dict):
            unevaluated += 1
            continue
        status = autoeval.get("status")
        label = autoeval.get("label")

        if status == "skipped_generation_error":
            generation_errors += 1
            continue
        if status == "judge_error":
            judge_errors += 1
            continue
        if not isinstance(label, bool):
            unevaluated += 1
            continue

        evaluated_labels.append(label)
        type_labels[question_type].append(label)
        source_labels[source].append(label)
        if is_abstention(
            question_id, None if question_type == "unspecified" else question_type
        ):
            abstention_labels.append(label)

    input_count = len(evaluated_rows)
    evaluated_count = len(evaluated_labels)
    correct = sum(1 for label in evaluated_labels if label)

    by_question_type = OrderedDict()
    question_types = list(type_labels.keys())
    if run.dataset == "longmemeval":
        question_types.sort(
            key=lambda value: (
                (
                    LONGMEMEVAL_QUESTION_TYPES.index(value)
                    if value in LONGMEMEVAL_QUESTION_TYPES
                    else len(LONGMEMEVAL_QUESTION_TYPES)
                ),
                value,
            )
        )
    else:
        question_types.sort()
    for question_type in question_types:
        by_question_type[question_type] = metric_bucket(type_labels[question_type])

    by_source = OrderedDict(
        (source, metric_bucket(source_labels[source]))
        for source in sorted(source_labels)
    )

    type_accuracies = [
        bucket["accuracy"]
        for bucket in by_question_type.values()
        if bucket["accuracy"] is not None
    ]
    macro_accuracy = (
        round(sum(type_accuracies) / len(type_accuracies), 6)
        if type_accuracies
        else None
    )

    metrics: dict[str, Any] = {
        "overall_accuracy": (
            round(correct / evaluated_count, 6) if evaluated_count else None
        ),
        "end_to_end_accuracy": round(correct / input_count, 6) if input_count else None,
        "correct": correct,
        "evaluated_questions": evaluated_count,
        "input_records": input_count,
        "generation_errors": generation_errors,
        "judge_errors": judge_errors,
        "unevaluated_records": unevaluated,
        "macro_accuracy_by_question_type": macro_accuracy,
        "by_question_type": by_question_type,
        "by_source": by_source,
    }

    if abstention_labels:
        metrics["abstention"] = metric_bucket(abstention_labels)
    if run.dataset == "longmemeval":
        metrics["task_averaged_accuracy"] = macro_accuracy
        if run.memory_block == "binary_router":
            metrics["evaluation_scope"] = {
                "question_types": list(BINARY_ROUTER_QUESTION_TYPES),
                "evaluates_every_result_record": True,
                "note": (
                    "Binary-router evaluation is intentionally limited to the two routed "
                    "LongMemEval types. The evaluator does not balance or subsample test "
                    "records; every result row in the selected JSONL is scored."
                ),
            }

    return metrics


def source_summary_fields(run: RunSpec) -> dict[str, Any]:
    if run.summary_file is None:
        return {
            "run_id": run.run_id,
            "summary_file": None,
        }

    summary = read_json(run.summary_file)
    totals = summary.get("totals") if isinstance(summary, dict) else None
    if not isinstance(totals, dict):
        totals = {}

    return {
        "run_id": summary.get("run_id"),
        "completed_tasks": summary.get("completed_tasks"),
        "completed_questions": summary.get("completed_questions"),
        "previously_completed_questions": summary.get("previously_completed_questions"),
        "wall_runtime_seconds": summary.get("wall_runtime_seconds"),
        "average_seconds_per_question": summary.get("average_seconds_per_question"),
        "tasks_failed": totals.get("tasks_failed"),
        "questions_failed": totals.get("questions_failed"),
        "total_estimated_eval_cost_usd": totals.get("total_estimated_eval_cost_usd"),
    }


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def consolidate_configuration(runs: list[RunSpec]) -> dict[str, Any]:
    configs = [(run, run.configuration) for run in runs if run.configuration]
    if not configs:
        return {
            "shared_across_runs": None,
            "note": "No run summary configuration was available.",
        }

    comparable_keys = sorted(
        set().union(*(config.keys() for _, config in configs))
        - RUN_IDENTITY_CONFIG_KEYS
    )
    shared: OrderedDict[str, Any] = OrderedDict()
    varying: OrderedDict[str, Any] = OrderedDict()

    for key in comparable_keys:
        value_groups: OrderedDict[str, dict[str, Any]] = OrderedDict()
        for run, config in configs:
            value = config.get(key, "<MISSING>")
            encoded = canonical_json(value)
            if encoded not in value_groups:
                value_groups[encoded] = {"value": value, "runs": []}
            value_groups[encoded]["runs"].append(run.label)

        if len(value_groups) == 1:
            shared[key] = next(iter(value_groups.values()))["value"]
        else:
            varying[key] = list(value_groups.values())

    dataset_inputs: OrderedDict[str, Any] = OrderedDict()
    for dataset in sorted({run.dataset for run, _ in configs}, key=dataset_sort_key):
        data_files = []
        for run, config in configs:
            if run.dataset != dataset:
                continue
            data_file = config.get("data_file")
            if data_file is not None and data_file not in data_files:
                data_files.append(data_file)
        dataset_inputs[dataset] = data_files[0] if len(data_files) == 1 else data_files

    agent_values: OrderedDict[str, Any] = OrderedDict()
    for memory in sorted(
        {run.memory_block for run, _ in configs}, key=memory_block_sort_key
    ):
        raw_values = []
        for run, config in configs:
            if run.memory_block != memory:
                continue
            raw = config.get("agent")
            if raw is not None and raw not in raw_values:
                raw_values.append(raw)
        agent_values[memory] = raw_values[0] if len(raw_values) == 1 else raw_values

    return {
        "shared_across_runs": not bool(varying),
        "comparison_excludes_run_identity_keys": sorted(RUN_IDENTITY_CONFIG_KEYS),
        "shared_configuration": shared,
        "varying_configuration": varying,
        "run_dimensions": {
            "dataset_data_files": dataset_inputs,
            "memory_block_agent_values": agent_values,
        },
    }


def evaluation_policy_description(dataset: str) -> str:
    if dataset == "longmemeval":
        return (
            "LongMemEval-compatible question-type prompts from evaluate_qa.py, using "
            "question_type metadata preserved in the task sidecar."
        )
    if dataset == "ecommerce":
        return (
            "Reference-answer LLM judge; abstention and update/conflict question types "
            "receive specialized prompts when metadata is available."
        )
    if dataset == "mab_long_range_understanding":
        return (
            "Reference-answer LLM judge with a summary-sensitive rubric for long-range "
            "understanding tasks. This is QA autoevaluation, not a claim of reproducing "
            "every official MemoryAgentBench task-specific metric."
        )
    if dataset.startswith("mab_"):
        return (
            "Reference-answer LLM judge, reusing LongMemEval-specific prompts when MAB "
            "question metadata exposes a LongMemEval question type. This is QA "
            "autoevaluation, not a claim of reproducing every official MemoryAgentBench "
            "task-specific metric."
        )
    return "Reference-answer LLM judge."


def build_consolidated_payload(
    selected_runs: list[RunSpec],
    model_spec: ModelSpec,
    results_dir: Path,
    output_dir: Path,
    per_run_results: dict[tuple[str, str], dict[str, Any]],
) -> OrderedDict[str, Any]:
    memory_blocks: OrderedDict[str, Any] = OrderedDict()
    routing_systems: OrderedDict[str, Any] = OrderedDict()

    selected_memory_blocks = {
        run.memory_block
        for run in selected_runs
        if run.memory_block in MEMORY_BLOCK_ORDER
    }
    for memory in sorted(selected_memory_blocks, key=memory_block_sort_key):
        datasets: OrderedDict[str, Any] = OrderedDict()
        for dataset in sorted(
            {run.dataset for run in selected_runs if run.memory_block == memory},
            key=dataset_sort_key,
        ):
            result = per_run_results.get((memory, dataset))
            if result is not None:
                datasets[dataset] = result
        memory_blocks[memory] = {"datasets": datasets}

    selected_routing_systems = {
        run.memory_block
        for run in selected_runs
        if run.memory_block in ROUTING_SYSTEM_ORDER
    }
    for routing_system in sorted(selected_routing_systems, key=memory_block_sort_key):
        datasets: OrderedDict[str, Any] = OrderedDict()
        for dataset in sorted(
            {
                run.dataset
                for run in selected_runs
                if run.memory_block == routing_system
            },
            key=dataset_sort_key,
        ):
            result = per_run_results.get((routing_system, dataset))
            if result is not None:
                datasets[dataset] = result

        routing_entry: OrderedDict[str, Any] = OrderedDict()
        if routing_system == "binary_router":
            routing_entry["router_type"] = "context-only binary architecture router"
            routing_entry["question_type_scope"] = list(BINARY_ROUTER_QUESTION_TYPES)
            routing_entry["routes"] = dict(BINARY_ROUTER_ROUTES)
            routing_entry["evaluation_note"] = (
                "The binary router is a routing wrapper over HVM and Episodic, not a "
                "seventh memory architecture. Its evaluator scores every result record "
                "present in the JSONL and does not rebalance the test set."
            )
        routing_entry["datasets"] = datasets
        routing_systems[routing_system] = routing_entry

    policies = OrderedDict()
    for dataset in sorted({run.dataset for run in selected_runs}, key=dataset_sort_key):
        policies[dataset] = evaluation_policy_description(dataset)

    payload: OrderedDict[str, Any] = OrderedDict()
    payload["generated_at"] = utc_now_iso()
    payload["metric_model"] = {
        "requested": model_spec.requested_name,
        "resolved": model_spec.model,
        "source": model_spec.source,
        "base_url": model_spec.base_url,
    }
    payload["source_results_directory"] = str(results_dir)
    payload["evaluation_output_directory"] = str(output_dir)
    payload["configuration"] = consolidate_configuration(selected_runs)
    payload["evaluation_policy"] = policies
    payload["memory_blocks"] = memory_blocks
    if routing_systems:
        payload["routing_systems"] = routing_systems
    return payload


DISPLAY_NAMES = {
    "episodic": "Episodic",
    "propositional": "Propositional",
    "condensed": "Condensed",
    "vector": "Vector",
    "hvm": "HVM",
    "graph": "Graph",
    "binary_router": "Binary Router",
    "longmemeval": "LongMemEval",
    "ecommerce": "Ecommerce",
    "mab_accurate_retrieval": "MAB - Accurate Retrieval",
    "mab_long_range_understanding": "MAB - Long-Range Understanding",
    "mab_test_time_learning": "MAB - Test-Time Learning",
    "mab_conflict_resolution": "MAB - Conflict Resolution",
}


def display_name(value: str) -> str:
    return DISPLAY_NAMES.get(value, value.replace("_", " ").replace("-", " ").title())


def format_percent(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    return f"{100.0 * float(value):.2f}%"


def format_number(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:,.6f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def render_table(headers: list[str], rows: list[list[Any]]) -> str:
    string_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in string_rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def render_row(row: list[str]) -> str:
        return "  ".join(
            cell.ljust(widths[index]) for index, cell in enumerate(row)
        ).rstrip()

    separator = "  ".join("-" * width for width in widths)
    lines = [render_row(headers), separator]
    lines.extend(render_row(row) for row in string_rows)
    return "\n".join(lines)


def render_mapping(mapping: dict[str, Any], indent: int = 0) -> list[str]:
    lines: list[str] = []
    prefix = " " * indent
    for key, value in mapping.items():
        label = key.replace("_", " ").replace("-", " ").title()
        if isinstance(value, dict):
            lines.append(f"{prefix}{label}:")
            lines.extend(render_mapping(value, indent + 2))
        elif isinstance(value, list):
            if not value:
                lines.append(f"{prefix}{label}: []")
            elif all(not isinstance(item, (dict, list)) for item in value):
                lines.append(
                    f"{prefix}{label}: {', '.join(str(item) for item in value)}"
                )
            else:
                lines.append(f"{prefix}{label}:")
                for item in value:
                    if isinstance(item, dict):
                        lines.append(f"{prefix}  -")
                        lines.extend(render_mapping(item, indent + 4))
                    else:
                        lines.append(f"{prefix}  - {item}")
        else:
            lines.append(f"{prefix}{label}: {format_number(value)}")
    return lines


def render_metric_details(metrics: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    lines.append("SUMMARY")
    lines.append("-" * 7)
    summary_rows = [
        ["Overall accuracy", format_percent(metrics.get("overall_accuracy"))],
        ["End-to-end accuracy", format_percent(metrics.get("end_to_end_accuracy"))],
        ["Correct", format_number(metrics.get("correct"))],
        ["Evaluated questions", format_number(metrics.get("evaluated_questions"))],
        ["Input records", format_number(metrics.get("input_records"))],
        [
            "Macro accuracy by question type",
            format_percent(metrics.get("macro_accuracy_by_question_type")),
        ],
        ["Generation errors", format_number(metrics.get("generation_errors"))],
        ["Judge errors", format_number(metrics.get("judge_errors"))],
        ["Unevaluated records", format_number(metrics.get("unevaluated_records"))],
    ]
    if metrics.get("task_averaged_accuracy") is not None:
        summary_rows.append(
            [
                "Task-averaged accuracy",
                format_percent(metrics["task_averaged_accuracy"]),
            ]
        )
    lines.append(render_table(["Metric", "Value"], summary_rows))

    by_question_type = metrics.get("by_question_type")
    if isinstance(by_question_type, dict) and by_question_type:
        lines.extend(["", "PERFORMANCE BY QUESTION TYPE", "-" * 28])
        rows = []
        for question_type, bucket in by_question_type.items():
            if not isinstance(bucket, dict):
                continue
            rows.append(
                [
                    display_name(question_type),
                    format_number(bucket.get("correct")),
                    format_number(bucket.get("count")),
                    format_percent(bucket.get("accuracy")),
                ]
            )
        lines.append(
            render_table(["Question Type", "Correct", "Count", "Accuracy"], rows)
        )

    by_source = metrics.get("by_source")
    if isinstance(by_source, dict) and by_source:
        lines.extend(["", "PERFORMANCE BY SOURCE", "-" * 21])
        rows = []
        for source, bucket in by_source.items():
            if not isinstance(bucket, dict):
                continue
            rows.append(
                [
                    source,
                    format_number(bucket.get("correct")),
                    format_number(bucket.get("count")),
                    format_percent(bucket.get("accuracy")),
                ]
            )
        lines.append(render_table(["Source", "Correct", "Count", "Accuracy"], rows))

    abstention = metrics.get("abstention")
    if isinstance(abstention, dict):
        lines.extend(["", "ABSTENTION", "-" * 10])
        lines.append(
            render_table(
                ["Correct", "Count", "Accuracy"],
                [
                    [
                        format_number(abstention.get("correct")),
                        format_number(abstention.get("count")),
                        format_percent(abstention.get("accuracy")),
                    ]
                ],
            )
        )

    evaluation_scope = metrics.get("evaluation_scope")
    if isinstance(evaluation_scope, dict):
        lines.extend(["", "EVALUATION SCOPE", "-" * 16])
        lines.extend(render_mapping(evaluation_scope))

    return lines


def render_run_report(
    memory_block: str,
    dataset: str,
    result: dict[str, Any],
    model_spec: ModelSpec,
    generated_at: str | None = None,
) -> str:
    title = f"{display_name(memory_block)} | {display_name(dataset)}"
    rule = "=" * max(72, len(title))
    lines = [
        rule,
        title.upper(),
        rule,
        "",
        f"Generated:       {generated_at or utc_now_iso()}",
        f"Status:          {str(result.get('status', 'unknown')).upper()}",
        f"Judge model:     {model_spec.model}",
        f"Hypothesis file: {result.get('hypothesis_file', '-')}",
    ]

    evaluation_file = result.get("evaluation_file")
    if evaluation_file:
        lines.append(f"Evaluation file: {evaluation_file}")

    if result.get("status") == "error":
        lines.extend(
            [
                "",
                "ERROR",
                "-" * 5,
                f"Type:    {result.get('error_type', '-')}",
                f"Message: {result.get('error', '-')}",
            ]
        )
    else:
        metrics = result.get("metrics")
        if isinstance(metrics, dict):
            lines.extend(["", *render_metric_details(metrics)])

    source_run = result.get("source_run")
    if isinstance(source_run, dict):
        lines.extend(["", "SOURCE RUN", "-" * 10])
        lines.extend(render_mapping(source_run))

    lines.extend(["", rule])
    return "\n".join(lines)


def per_run_report_path(
    output_dir: Path,
    memory_block: str,
    dataset: str,
) -> Path:
    return output_dir / memory_block / dataset / f"{memory_block}_{dataset}_metrics.txt"


def write_per_run_report(
    output_dir: Path,
    memory_block: str,
    dataset: str,
    result: dict[str, Any],
    model_spec: ModelSpec,
) -> Path:
    path = per_run_report_path(output_dir, memory_block, dataset)
    write_text_atomic(
        path, render_run_report(memory_block, dataset, result, model_spec)
    )
    return path


def iter_payload_results(
    payload: dict[str, Any],
) -> Iterable[tuple[str, str, dict[str, Any]]]:
    for top_level_key in ("memory_blocks", "routing_systems"):
        systems = payload.get(top_level_key)
        if not isinstance(systems, dict):
            continue
        for memory_block, memory_entry in systems.items():
            if not isinstance(memory_entry, dict):
                continue
            datasets = memory_entry.get("datasets")
            if not isinstance(datasets, dict):
                continue
            for dataset, result in datasets.items():
                if isinstance(result, dict):
                    yield memory_block, dataset, result


def render_consolidated_report(payload: dict[str, Any], model_spec: ModelSpec) -> str:
    title = "CONSOLIDATED QA EVALUATION REPORT"
    rule = "=" * 92
    lines = [
        rule,
        title.center(len(rule)),
        rule,
        "",
        f"Generated:             {payload.get('generated_at', '-')}",
        f"Judge model:           {model_spec.model}",
        f"Requested judge model: {model_spec.requested_name}",
        f"Judge source:          {model_spec.source}",
        f"Judge base URL:        {model_spec.base_url or '-'}",
        f"Source results:        {payload.get('source_results_directory', '-')}",
        f"Evaluation output:     {payload.get('evaluation_output_directory', '-')}",
        "",
        "RESULTS OVERVIEW",
        "-" * 16,
    ]

    overview_rows: list[list[Any]] = []
    results = list(iter_payload_results(payload))
    for memory_block, dataset, result in results:
        metrics = result.get("metrics") if isinstance(result, dict) else None
        metrics = metrics if isinstance(metrics, dict) else {}
        overview_rows.append(
            [
                display_name(memory_block),
                display_name(dataset),
                str(result.get("status", "unknown")).upper(),
                format_number(metrics.get("correct")),
                format_number(metrics.get("evaluated_questions")),
                format_percent(metrics.get("overall_accuracy")),
                format_percent(metrics.get("macro_accuracy_by_question_type")),
            ]
        )

    if overview_rows:
        lines.append(
            render_table(
                [
                    "System",
                    "Dataset",
                    "Status",
                    "Correct",
                    "Evaluated",
                    "Accuracy",
                    "Macro by Type",
                ],
                overview_rows,
            )
        )
    else:
        lines.append("No completed run results are available.")

    policies = payload.get("evaluation_policy")
    if isinstance(policies, dict) and policies:
        lines.extend(["", "EVALUATION POLICIES", "-" * 19])
        for dataset, description in policies.items():
            lines.append(f"{display_name(dataset)}:")
            lines.append(f"  {description}")

    configuration = payload.get("configuration")
    if isinstance(configuration, dict):
        lines.extend(["", "CONFIGURATION", "-" * 13])
        lines.extend(render_mapping(configuration))

    for memory_block, dataset, result in results:
        lines.extend(
            [
                "",
                "",
                "#" * 92,
                "",
                render_run_report(
                    memory_block,
                    dataset,
                    result,
                    model_spec,
                    generated_at=str(payload.get("generated_at") or ""),
                ),
            ]
        )

    routing_systems = payload.get("routing_systems")
    if isinstance(routing_systems, dict) and routing_systems:
        lines.extend(["", "", "ROUTING SYSTEM DEFINITIONS", "-" * 26])
        for name, entry in routing_systems.items():
            if not isinstance(entry, dict):
                continue
            lines.append(f"{display_name(name)}:")
            details = {key: value for key, value in entry.items() if key != "datasets"}
            lines.extend(render_mapping(details, indent=2))

    lines.extend(["", rule])
    return "\n".join(lines)


def print_discovered_runs(runs: list[RunSpec], results_dir: Path) -> None:
    print(f"Discovered {len(runs)} result runs under {results_dir}:\n")
    for run in runs:
        rel = run.hypothesis_file.relative_to(results_dir)
        print(f"  {run.memory_block:14} {run.dataset:29} {rel}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch QA autoevaluation runner for a results directory. By default it "
            "discovers and evaluates every primary result JSONL, writes one formatted metrics "
            "text report per memory-block/dataset pair, and writes one consolidated "
            "text report under qa_evaluations."
        )
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        type=Path,
        default=Path("../asdrp/results"),
        help="Root results directory (default: asdrp/results).",
    )
    parser.add_argument(
        "--metric-model",
        default="gpt-4o",
        help=(
            "Judge model alias or model ID. Preserved aliases: gpt-4o, gpt-4o-mini, "
            "llama-3.1-70b-instruct. Unknown values are treated as direct model IDs."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Optional OpenAI-compatible API base URL for a local/custom judge.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Generated-output directory (default: <results_dir>/qa_evaluations).",
    )
    parser.add_argument(
        "--consolidated-txt",
        type=Path,
        default=None,
        help=(
            "Consolidated text-report path (default: "
            "<output-dir>/consolidated_qa_metrics.txt)."
        ),
    )
    parser.add_argument(
        "--memory-block",
        "--agent",
        dest="memory_blocks",
        action="append",
        default=None,
        help=(
            "Evaluate only this architecture/routing system. Repeat for multiple values. "
            "Accepted: episodic, propositional/proposition, condensed, vector, hvm, graph, "
            "binary_router."
        ),
    )
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        default=None,
        help=(
            "Evaluate only this dataset. Repeat for multiple datasets. Aliases include "
            "longmemeval/lme, ecommerce, ar, lru, ttl, cr."
        ),
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Evaluate exactly one discovered primary result JSONL.",
    )
    parser.add_argument(
        "--judge-workers",
        type=int,
        default=8,
        help="Concurrent judge requests within each run (default: 8).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=8,
        help="Retries per judge request for transient/format errors (default: 8).",
    )
    parser.add_argument(
        "--retry-base-delay",
        type=float,
        default=1.0,
        help="Initial exponential-backoff delay in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--retry-max-delay",
        type=float,
        default=20.0,
        help="Maximum exponential-backoff delay in seconds (default: 20.0).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "Ignore even content-matching evaluator outputs/checkpoints and reevaluate "
            "all selected rows."
        ),
    )
    parser.add_argument(
        "--reuse-legacy-source-results",
        action="store_true",
        help=(
            "Also reuse matching evaluate_qa.py outputs located beside source hypothesis "
            "files (for example *.jsonl.eval-results-gpt-4o)."
        ),
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List discovered runs and exit without calling a judge model.",
    )
    parser.add_argument(
        "--verbose-judgments",
        action="store_true",
        help="Print one compact JSON status line per judged question.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately if one selected run fails instead of continuing the batch.",
    )
    return parser.parse_args()


def resolve_results_dir(path: Path) -> Path:
    """Resolve the results root, including compatibility with the unnecessary 'results' path."""
    expanded = path.expanduser()

    if expanded.is_absolute():
        candidates = [expanded]
    else:
        cwd = Path.cwd()
        script_dir = Path(__file__).resolve().parent
        candidates = [cwd / expanded, script_dir / expanded]

        if expanded == Path("../asdrp/results"):
            candidates.extend(
                [
                    cwd / "asdrp" / "results",
                    script_dir / "asdrp" / "results",
                ]
            )

    seen: set[Path] = set()
    tried: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        tried.append(resolved)
        if resolved.is_dir():
            return resolved

    raise EvaluationError(
        "Results directory does not exist. Tried: "
        + ", ".join(str(candidate) for candidate in tried)
    )


def main() -> int:
    args = parse_args()
    load_dotenv()

    if args.judge_workers < 1:
        raise EvaluationError("--judge-workers must be at least 1.")
    if args.max_retries < 0:
        raise EvaluationError("--max-retries cannot be negative.")
    if args.retry_base_delay < 0 or args.retry_max_delay < 0:
        raise EvaluationError("Retry delays cannot be negative.")

    results_dir = resolve_results_dir(args.results_dir)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else results_dir / "qa_evaluations"
    )
    consolidated_txt = (
        args.consolidated_txt.expanduser().resolve()
        if args.consolidated_txt is not None
        else output_dir / "consolidated_qa_metrics.txt"
    )

    runs = discover_runs(results_dir)
    if args.list_runs:
        print_discovered_runs(runs, results_dir)
        return 0

    memory_blocks = None
    if args.memory_blocks:
        memory_blocks = {normalize_memory_block(value) for value in args.memory_blocks}
    datasets = None
    if args.datasets:
        datasets = {normalize_dataset(value) for value in args.datasets}

    selected_file = args.file
    if selected_file is not None and not selected_file.is_absolute():
        candidate = (Path.cwd() / selected_file).resolve()
        if candidate.is_file():
            selected_file = candidate
        else:
            selected_file = (results_dir / selected_file).resolve()

    selected_runs = select_runs(runs, memory_blocks, datasets, selected_file)
    model_spec = resolve_model(args.metric_model, args.base_url)

    print(
        f"Selected {len(selected_runs)} run(s); judge={model_spec.model}; "
        f"judge_workers={args.judge_workers}; output={output_dir}",
        flush=True,
    )

    per_run_results: dict[tuple[str, str], dict[str, Any]] = {}
    failures = 0

    for index, run in enumerate(selected_runs, start=1):
        print(f"\n[{index}/{len(selected_runs)}] {run.label}", flush=True)
        try:
            evaluated_rows, eval_output_path, question_metadata = evaluate_run(
                run=run,
                output_dir=output_dir,
                model_spec=model_spec,
                judge_workers=args.judge_workers,
                max_retries=args.max_retries,
                retry_base_delay=args.retry_base_delay,
                retry_max_delay=args.retry_max_delay,
                resume=not args.no_resume,
                reuse_legacy_source_results=args.reuse_legacy_source_results,
                verbose_judgments=args.verbose_judgments,
            )
            metrics = calculate_metrics(run, evaluated_rows, question_metadata)
            per_run_results[(run.memory_block, run.dataset)] = {
                "status": (
                    "ok"
                    if (
                        metrics["generation_errors"] == 0
                        and metrics["judge_errors"] == 0
                        and metrics["unevaluated_records"] == 0
                    )
                    else "partial"
                ),
                "hypothesis_file": str(run.hypothesis_file),
                "evaluation_file": str(eval_output_path),
                "source_run": source_summary_fields(run),
                "metrics": metrics,
            }
            run_report_path = write_per_run_report(
                output_dir,
                run.memory_block,
                run.dataset,
                per_run_results[(run.memory_block, run.dataset)],
                model_spec,
            )
            print(f"  metrics report: {run_report_path}", flush=True)
            print(
                "  accuracy={} evaluated={} generation_errors={} judge_errors={}".format(
                    metrics["overall_accuracy"],
                    metrics["evaluated_questions"],
                    metrics["generation_errors"],
                    metrics["judge_errors"],
                ),
                flush=True,
            )
            if metrics["judge_errors"] or metrics["unevaluated_records"]:
                failures += 1
        except Exception as exc:
            failures += 1
            per_run_results[(run.memory_block, run.dataset)] = {
                "status": "error",
                "hypothesis_file": str(run.hypothesis_file),
                "source_run": source_summary_fields(run),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            run_report_path = write_per_run_report(
                output_dir,
                run.memory_block,
                run.dataset,
                per_run_results[(run.memory_block, run.dataset)],
                model_spec,
            )
            print(f"  metrics report: {run_report_path}", flush=True)
            print(f"  ERROR {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            if args.fail_fast:
                payload = build_consolidated_payload(
                    selected_runs,
                    model_spec,
                    results_dir,
                    output_dir,
                    per_run_results,
                )
                write_text_atomic(
                    consolidated_txt,
                    render_consolidated_report(payload, model_spec),
                )
                raise

        payload = build_consolidated_payload(
            selected_runs,
            model_spec,
            results_dir,
            output_dir,
            per_run_results,
        )
        write_text_atomic(
            consolidated_txt,
            render_consolidated_report(payload, model_spec),
        )

    print(f"\nConsolidated metrics saved to: {consolidated_txt}", flush=True)
    if failures:
        print(f"Completed with {failures} failed run(s).", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "\nInterrupted. Completed judge responses remain resumable.",
            file=sys.stderr,
        )
        raise SystemExit(130)
    except EvaluationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)
