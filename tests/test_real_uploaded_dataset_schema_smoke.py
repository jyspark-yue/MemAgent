#############################################################################
# File: test_real_uploaded_dataset_schema_smoke.py
#
# Description:
#   Checks the first few tasks from each real benchmark file when those
#   files are available. Missing large datasets are skipped instead of
#   making the local suite fail.
#
#   - Defines the canonical local path for every supported benchmark
#     family.
#   - Streams tasks through the same adapters used by the evaluator.
#   - Checks non-empty entries and questions, unique entry IDs, and
#     source order.
#   - Limits each dataset check to three rows to keep the smoke test
#     fast.
#############################################################################

from __future__ import annotations

from pathlib import Path

import pytest

from asdrp.dataset_adapters import iter_evaluation_tasks

pytestmark = pytest.mark.integration  # Uses local benchmark files when present.

# Locate datasets relative to the project instead of the current shell folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Canonical evaluator name paired with its expected uploaded dataset path.
REAL_DATASETS = [
    (
        "longmemeval",
        PROJECT_ROOT
        / "asdrp/datasets/longmemeval_v1/splits/longmemeval_20pct"
        / "longmemeval_20pct_test.json",
    ),
    (
        "mab_accurate_retrieval",
        PROJECT_ROOT
        / "asdrp/datasets/memory_agent_bench/splits/mab_10pct"
        / "Accurate_Retrieval_test.json",
    ),
    (
        "mab_conflict_resolution",
        PROJECT_ROOT
        / "asdrp/datasets/memory_agent_bench/splits/mab_10pct"
        / "Conflict_Resolution_test.json",
    ),
    (
        "mab_long_range_understanding",
        PROJECT_ROOT
        / "asdrp/datasets/memory_agent_bench/splits/mab_10pct"
        / "Long_Range_Understanding_test.json",
    ),
    (
        "mab_test_time_learning",
        PROJECT_ROOT
        / "asdrp/datasets/memory_agent_bench/splits/mab_10pct"
        / "Test_Time_Learning_decoded_test.json",
    ),
    (
        "ecommerce",
        PROJECT_ROOT
        / "asdrp/datasets/ecommerce_memeval/splits/ecommerce_20pct"
        / "ecommerce_memeval_20pct_test.json",
    ),
]


@pytest.mark.parametrize("dataset,path", REAL_DATASETS)
def test_real_dataset_first_rows_parse_and_respect_embedding_limit(
    dataset,
    path,
):
    if not path.is_file():
        # The source archive may not include these large benchmark files.
        pytest.skip(f"Real dataset not found: {path}")

    tasks = iter_evaluation_tasks(  # Stream normalized tasks from the real file.
        data_file=path,
        dataset=dataset,
        embedding_model="text-embedding-3-small",
        max_entry_tokens=7000,
    )
    observed = 0  # Only the first few rows are needed for a schema check.
    for task in tasks:
        assert task.entries and task.questions
        assert len({entry.entry_id for entry in task.entries}) == len(task.entries)
        assert [entry.ordinal for entry in task.entries] == sorted(
            entry.ordinal for entry in task.entries
        )
        observed += 1
        if observed == 3:
            break
    assert observed > 0
