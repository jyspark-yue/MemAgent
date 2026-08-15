# MemAgent Project

Research code for **_Predicting Future Memory Demand for Architecture Routing in LLM-based Agents_**.

This repository contains the memory systems, benchmark adapters, evaluation code, EcommerceMemEval dataset tools, and history-only architecture router used in the paper.

The study asks two questions:

1. Do long-term memory architectures perform differently across memory demands when the rest of the evaluation pipeline is held fixed?
2. Can conversation history predict which memory architecture will be useful before the future question is known?

The code compares six memory architectures on LongMemEval, MemoryAgentBench, and EcommerceMemEval. It then uses a small binary classifier to route LongMemEval histories between HVM and Episodic Memory before the evaluation question is revealed.

## Paper overview

The six evaluated memory systems use the same benchmark adapters, answer interface, generation model, embedding model where needed, and judging procedure. The main difference is how each system writes and retrieves memory. The paper therefore reports **system-level comparisons**, not isolated component ablations.

| Architecture | Stored representation | Retrieval |
| --- | --- | --- |
| **Condensed** | Rolling or segment summaries | Dense + BM25 rank fusion over summaries |
| **Propositional** | LLM-extracted atomic propositions | Dense + BM25 rank fusion over propositions |
| **Episodic** | Events with session and temporal metadata | Dense + BM25 rank fusion over events |
| **Graph** | Entities, relations, and linked factual nodes | Entity/relation traversal with graph-specific scoring |
| **Vector** | Minimally transformed conversation chunks | Dense + BM25 rank fusion over chunks |
| **HVM** | Source leaves and recursively clustered summary nodes | Tree traversal and similarity retrieval over leaves and summaries |

Two additional systems are included for controlled experiments:

- `binary_router` predicts either `single-session-preference` or `knowledge-update` from conversation history, then selects HVM or Episodic Memory.
- `qdrant_graph` stores graph-shaped records but retrieves them with flat Qdrant similarity search. It is an ablation and is not one of the six main paper architectures.

## Paper configuration

The main reported experiments use the following shared settings.

| Setting | Paper configuration |
| --- | --- |
| Answer model | `gpt-5.6-luna` |
| Judge | `gpt-5.6-sol` |
| Embeddings | `text-embedding-3-small` where required |
| LongMemEval | Fixed stratified 20% test split: 100 histories / 100 questions |
| MemoryAgentBench | Fixed 20% task splits: 400 Accurate Retrieval, 34 Long-Range Understanding, 140 Test-Time Learning, 183 Conflict Resolution questions |
| EcommerceMemEval | 351 synthetic customer histories, 1,747 questions, 10 memory operations |
| Router training | 23 complete-context-disjoint histories per class |
| Router test | 7 histories per class; 14 total |

LongMemEval and MemoryAgentBench use fixed 20% subsets in this study. Their scores are intended for comparison inside this paper and should not be presented as full-benchmark leaderboard results.

## Repository layout

```text
MemAgent/
├── asdrp/
│   ├── agent/                         # Query-time agents for each memory system
│   ├── memory/                        # Memory construction, storage, and retrieval
│   ├── classification_algorithms/    # Binary router training, inference, and figures
│   ├── ecommerce_memeval_generator/
│   │   ├── dataset_generation/        # EcommerceMemEval QA generation and validation
│   │   └── synthetic-conversation-generation/
│   │                                    # Separate source-conversation generator
│   ├── datasets/                     # Benchmark download, decoding, and split tools
│   ├── dataset_adapters.py           # Benchmark -> shared evaluation schema
│   ├── eval_schemas.py               # Shared evaluation records
│   ├── evaluate_agents_wcost.py      # Main architecture evaluator
│   ├── openai_embedder.py            # Batched embedding service
│   ├── qdrant_retry.py               # Qdrant retry helpers
│   └── runtime.py                    # Model calls, retries, usage, and cost tracking
├── result_scripts/                    # Judging, metrics, repair, comparisons, and figures
├── tests/                             # Unit, property, schema, and optional live tests
├── pytest.ini
└── README.md
```

### Paper-to-code map

| Paper component | Main code |
| --- | --- |
| Six memory architectures | `asdrp/memory/`, `asdrp/agent/` |
| Shared evaluation protocol | `asdrp/evaluate_agents_wcost.py` |
| Benchmark normalization | `asdrp/dataset_adapters.py` |
| LongMemEval preparation | `asdrp/datasets/longmemeval_v1/download_lme.py`, `asdrp/datasets/longmemeval_v1/lme_percent_split_generator.py` |
| MemoryAgentBench preparation | `asdrp/datasets/memory_agent_bench/download_mab.py`, `asdrp/datasets/memory_agent_bench/mab_ttl_id_decoder.py`, `asdrp/datasets/memory_agent_bench/mab_percent_split_generator.py` |
| EcommerceMemEval construction | `asdrp/ecommerce_memeval_generator/dataset_generation/` |
| Binary architecture router | `asdrp/classification_algorithms/train_binary_router.py`, `binary_router.py` |
| Router confusion matrix | `asdrp/classification_algorithms/create_confusion_matrix.py` |
| Answer judging and consolidated metrics | `result_scripts/evaluate_qa_runner.py` |
| Cross-benchmark heatmap | `result_scripts/generate_memory_heatmap.py` |
| Matched router comparison | `result_scripts/compare_router_apples_to_apples.py` |
| Interrupted-run audit/repair | `result_scripts/repair_evaluation_results.py` |

## Setup

Unless a section says otherwise, run commands from the repository root.

### 1. Create a Python environment

Python 3.12 is the target environment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the packages used by the evaluator, router, benchmark tools, figures, and tests:

```bash
python -m pip install -r requirements.txt
```

The separate LlamaIndex demo and source-conversation generator have extra dependencies and are not required for the paper evaluator.

### 2. Set the OpenAI API key

```bash
export OPENAI_API_KEY="..."
```

A root `.env` file is also supported.

### 3. Start Qdrant

Vector, Episodic, Condensed, Propositional, HVM, `qdrant_graph`, and the binary router require Qdrant. The default URL is `http://127.0.0.1:6333`.

```bash
docker compose up -d qdrant
```

Use `--qdrant-url` when Qdrant runs elsewhere.

### 4. Download and prepare external benchmarks

LongMemEval and MemoryAgentBench are not stored in this repository because the raw data and generated splits are large. The repository instead keeps the download, preprocessing, and split code needed to reconstruct them. The upstream benchmark data remains subject to the original projects' licenses and terms.

For paper reproduction, **do not rely on the random seed alone**. An exact split also depends on the exact upstream dataset revision, the split-script version, and the split arguments. The download scripts therefore record the resolved Hugging Face revision and source-file hashes, while the split scripts record the seed, source hashes, selected test identities, and output hashes. Commit these small manifests even when the large JSON files are ignored.

Both split generators use `20260720` as their default seed. The paper commands below pass it explicitly so the intended configuration is visible.

#### LongMemEval

The paper uses the cleaned `LongMemEval_M` source. The official LongMemEval repository distributes the cleaned benchmark through Hugging Face. Download only the `m` variant unless another LongMemEval file is needed for a separate experiment.

```bash
python asdrp/datasets/longmemeval_v1/download_lme.py --variant m
```

This writes:

```text
asdrp/datasets/longmemeval_v1/
├── longmemeval_m_cleaned.json          # large; do not commit
└── longmemeval_source_manifest.json    # small; commit
```

Create the paper's 80/20 split with the fixed seed:

```bash
python asdrp/datasets/longmemeval_v1/lme_percent_split_generator.py \
  asdrp/datasets/longmemeval_v1/longmemeval_m_cleaned.json \
  --output-dir asdrp/datasets/longmemeval_v1/splits/longmemeval_20pct \
  --train-fraction 0.80 \
  --seed 20260720 \
  --write-manifest
```

The test file should contain 100 of the 500 LongMemEval questions. Abstention samples are stratified as their own effective type when `question_id` ends in `_abs`. The split manifest records the exact selected question IDs and hashes, so it is the canonical record of which questions belong to the paper test set.

Keep the split manifest in Git, but ignore the generated train/test JSON files:

```text
asdrp/datasets/longmemeval_v1/splits/longmemeval_20pct/
├── longmemeval_20pct_train.json             # large; do not commit
├── longmemeval_20pct_test.json              # large; do not commit
└── longmemeval_20pct_split_manifest.json    # small; commit
```

#### MemoryAgentBench

Download the official Hugging Face dataset and export each benchmark split to JSON:

```bash
python asdrp/datasets/memory_agent_bench/download_mab.py
```

The downloader also saves `entity2id.json`, which MemoryAgentBench requires for the recommendation/Test-Time Learning entity mapping, and writes `memory_agent_bench_source_manifest.json` with the resolved upstream revision and hashes.

Decode the Test-Time Learning answers before creating the paper split:

```bash
python asdrp/datasets/memory_agent_bench/mab_ttl_id_decoder.py
```

Then create the fixed 80/20 task splits:

```bash
python asdrp/datasets/memory_agent_bench/mab_percent_split_generator.py \
  asdrp/datasets/memory_agent_bench/separated_splits/Accurate_Retrieval.json \
  asdrp/datasets/memory_agent_bench/separated_splits/Conflict_Resolution.json \
  asdrp/datasets/memory_agent_bench/separated_splits/Long_Range_Understanding.json \
  asdrp/datasets/memory_agent_bench/separated_splits/Test_Time_Learning_decoded.json \
  --output-dir asdrp/datasets/memory_agent_bench/splits/mab_20pct \
  --train-fraction 0.80 \
  --split-unit qa \
  --seed 20260720 \
  --write-manifest
```

`--split-unit qa` is intentional for the paper split. MemoryAgentBench often stores many aligned QA pairs in one shared context row, and some task types occur in only one row. QA mode keeps identical question/answer content wholly in one split while allowing the shared context row itself to appear in both train and test. This preserves task coverage but is **not context-disjoint**. Use `--split-unit row` only when a strict context-disjoint split is required; its composition will differ from the paper split.

For the paper configuration, the resulting test counts are 400 Accurate Retrieval, 34 Long-Range Understanding, 140 Test-Time Learning, and 183 Conflict Resolution questions. Treat a count mismatch as a sign that the upstream data, preprocessing, split arguments, or code revision differs from the paper setup.

#### What to commit

Commit the reproducibility metadata, not the large benchmark payloads. At minimum, keep:

- `longmemeval_source_manifest.json`;
- `longmemeval_20pct_split_manifest.json`;
- `memory_agent_bench_source_manifest.json`;
- each `mab_20pct/*_split_manifest.json`;
- the download, decoder, and split scripts themselves.

The source manifests identify the upstream revision and source hashes. The split manifests identify the exact split seed, input hashes, selected test identities, and generated-output hashes. Together with the Git commit, these are enough to detect whether a regenerated benchmark matches the paper data exactly.

If the current local paper splits already exist, **preserve their manifests before deleting the large files**. Check the `seed` field in those manifests and compare the recorded train/test hashes against a regeneration. If the manifest does not say `20260720`, use the seed recorded there instead. A matching seed by itself is not proof of an identical split.

## Recommended paper workflow

The shortest complete workflow is:

1. Download the external benchmarks and regenerate the fixed paper splits; verify their manifests.
2. Start Qdrant.
3. Run each memory architecture on the intended benchmark split.
4. Audit the generated outputs with `repair_evaluation_results.py`; rerun unresolved questions if needed.
5. Judge the generated answers with `evaluate_qa_runner.py` and write `asdrp/results/final_eval_metrics.txt`.
6. Generate the cross-benchmark heatmap from that consolidated metrics file.
7. Evaluate and judge the binary router.
8. Run the matched HVM/Episodic/router apples-to-apples comparison.
9. Generate the router confusion matrix from the classifier `metrics.json`.

The sections below give generic command forms and every supported command-line adjustment exposed by the attached evaluation utilities. Paper-specific values belong in result metadata and experiment records, not in the generic run syntax.

# Architecture evaluation

`asdrp/evaluate_agents_wcost.py` is the main generation evaluator. Benchmark inputs are normalized to the shared evaluation schema, each task receives isolated memory state, and successful result pairs can be resumed without recomputing them.

## Command form

For a normal architecture run, only the benchmark file and the three required flags need to be specified explicitly:

```bash
python -m asdrp.evaluate_agents_wcost DATA_FILE \
  --dataset DATASET \
  --agent AGENT \
  --output-file OUTPUT_FILE
```

Do not treat a long command containing concurrency, batching, retry, or pricing values as the canonical way to run the evaluator. Those are optional controls. Use them only when the run requires non-default behavior. The table below is the exhaustive command-line option reference for the attached evaluator revision.

Supported benchmark identifiers are:

```text
longmemeval
ecommerce
mab_accurate_retrieval
mab_long_range_understanding
mab_test_time_learning
mab_conflict_resolution
```

The six paper architectures are:

```text
condensed
episodic
graph
proposition
vector
hvm
```

`binary_router` is registered by this evaluator in addition to the normal project agent registry. Because `--agent` choices are built from that registry at runtime, `python -m asdrp.evaluate_agents_wcost --help` is the source of truth for any additional agents in the checked-out revision.

## Complete evaluator option reference

The defaults below are **implementation defaults**, not recommended experiment settings. For paper reproduction, use the experiment configuration recorded with the corresponding result set.

| Argument | Required / default | Purpose / constraint |
| --- | --- | --- |
| `DATA_FILE` | required | Existing benchmark JSON array file. |
| `--dataset DATASET` | required | Benchmark adapter to use. Accepted values: `longmemeval`, `ecommerce`, `mab_accurate_retrieval`, `mab_conflict_resolution`, `mab_long_range_understanding`, `mab_test_time_learning`. |
| `--agent AGENT` | required | Memory agent from the project agent registry; this evaluator additionally registers `binary_router`. |
| `--output-file OUTPUT_FILE` | required | Primary result JSONL. The task and summary sidecars are derived from this path. |
| `--router-model ROUTER_MODEL` | none | TF-IDF + LinearSVC router artifact. Required only when `--agent binary_router`. |
| `--question-type-filter TYPE [TYPE ...]` | none | Evaluate only matching question types. For `binary_router`, if supplied, it must exactly match the router's two configured labels. |
| `--llm-model MODEL` | `gpt-5.6-luna` | Answer-generation model ID. |
| `--embedding-model MODEL` | `text-embedding-3-small` | Embedding model. The current CLI accepts only `text-embedding-3-small`. |
| `--qdrant-url URL` | `http://127.0.0.1:6333` | Qdrant endpoint. |
| `--workers N` | `4` | Concurrent task workers; must be positive. |
| `--question-concurrency N` | `4` | Concurrent questions within a task; must be positive. |
| `--api-concurrency N` | `16` | Shared API concurrency limit; must be positive. |
| `--embedding-batch-size N` | `256` | Maximum entries per embedding batch; must be positive. |
| `--embedding-batch-token-limit N` | `240000` | Maximum embedding tokens per batch; must be positive, at least `--max-entry-tokens`, and no greater than `--embedding-tokens-per-minute`. |
| `--embedding-parallel-batches N` | `4` | Maximum embedding batches processed in parallel; must be positive. |
| `--embedding-tokens-per-minute N` | `4500000` | Shared embedding TPM target; must be positive. |
| `--rate-limit-buffer-seconds SECONDS` | `0.75` | Safety margin added to rate-limit waits; cannot be negative. |
| `--qdrant-batch-size N` | `256` | Qdrant write batch size; must be positive. |
| `--ingest-concurrency N` | `4` | Concurrent ingestion work; must be positive. |
| `--max-entry-tokens N` | `7000` | Per-entry token cap; must be between 512 and 8000. |
| `--request-timeout SECONDS` | `120.0` | Model request timeout; must be positive. |
| `--qdrant-timeout SECONDS` | `120` | Qdrant timeout; must be positive. |
| `--retry-attempts N` | `8` | Number of request attempts; must be positive. |
| `--retry-base-delay SECONDS` | `1.0` | Initial retry delay; cannot be negative. |
| `--retry-max-delay SECONDS` | `20.0` | Maximum retry delay; cannot be negative and must be at least the base delay. |
| `--start-index N` | `0` | Zero-based task index at which to begin; cannot be negative. Rejected for `binary_router`. |
| `--max-tasks N` | none | Maximum number of tasks to process; must be positive when supplied. Rejected for `binary_router`. |
| `--source-filter SOURCE` | none | Restrict evaluation to one source value when the benchmark adapter exposes source metadata. |
| `--overwrite` | off | Start from scratch instead of resuming successful existing `(task_id, question_id)` pairs. |
| `--keep-collections` | off | Retain per-task Qdrant collections after completion for debugging. |
| `--llm-input-per-million PRICE` | built-in / none | Override short-context uncached input price per million tokens; cannot be negative. |
| `--llm-cached-input-per-million PRICE` | built-in / none | Override short-context cached-input price per million tokens; cannot be negative. |
| `--llm-cache-write-per-million PRICE` | built-in / none | Override short-context cache-write price per million tokens; cannot be negative. |
| `--llm-output-per-million PRICE` | built-in / none | Override short-context output price per million tokens; cannot be negative. |
| `--llm-long-input-per-million PRICE` | built-in / none | Override long-context uncached input price per million tokens; cannot be negative. |
| `--llm-long-cached-input-per-million PRICE` | built-in / none | Override long-context cached-input price per million tokens; cannot be negative. |
| `--llm-long-cache-write-per-million PRICE` | built-in / none | Override long-context cache-write price per million tokens; cannot be negative. |
| `--llm-long-output-per-million PRICE` | built-in / none | Override long-context output price per million tokens; cannot be negative. |
| `--llm-long-context-threshold TOKENS` | `272000` | Token threshold separating short- and long-context pricing; must be positive. |
| `--embedding-input-per-million PRICE` | built-in / none | Override embedding input price per million tokens; cannot be negative. |

The evaluator contains built-in pricing for its known GPT-5.6 models and `text-embedding-3-small`. If a different LLM is used and accurate cost totals matter, provide the applicable pricing overrides rather than assuming the built-in rates apply.

For `binary_router`, see the router-specific restrictions below. They are additional validation rules on top of this same evaluator CLI.

Check the exact parser in the current checkout with:

```bash
python -m asdrp.evaluate_agents_wcost --help
```

# Result files and resume behavior

For an output such as:

```text
asdrp/results/longmemeval/hvm/lme_hvm.jsonl
```

the evaluator also writes:

```text
lme_hvm_tasks.jsonl     # task-level ingestion, timing, usage, and diagnostics
lme_hvm_summary.json    # run settings, counts, usage, costs, and environment summary
```

Existing successful `(task_id, question_id)` pairs are skipped when a run resumes. This is also why the repair workflow below does not use `--overwrite`.

Preserve the summary sidecar with the result JSONL. The repair utility reconstructs rerun commands from the saved configuration.

# Audit and repair interrupted evaluations

`result_scripts/repair_evaluation_results.py` audits the six core architecture outputs. It reports successful, unresolved, error, missing, duplicate, and malformed records and reconstructs resume-safe evaluator commands from each run's summary sidecar.

## Command form

Audit only:

```bash
python result_scripts/repair_evaluation_results.py RESULTS_ROOT
```

Execute the generated resume commands after reviewing the audit:

```bash
python result_scripts/repair_evaluation_results.py RESULTS_ROOT --run
```

The repair script deliberately does **not** add `--overwrite`; the evaluator's normal resume behavior is what limits reruns to unresolved `(task_id, question_id)` pairs. After `--run`, the script performs a second audit.

## Repair options

| Argument | Required / default | Purpose |
| --- | --- | --- |
| `RESULTS_ROOT` | required | Root containing evaluator result, task, and summary files. |
| `--project-root PROJECT_ROOT` | current working directory | Repository root used as the subprocess working directory and to resolve saved relative paths. |
| `--evaluator EVALUATOR` | `../asdrp/evaluate_agents_wcost.py` | Exposed by the CLI, but the current command builder still invokes `python -m asdrp.evaluate_agents_wcost` directly. Do not rely on this flag to select another evaluator until the script is changed. |
| `--python PYTHON` | current Python executable | Python interpreter used for reruns, normally the active virtual environment. |
| `--run` | off | Actually execute the generated resume commands. Without this flag, only audit and print them. |

If the command is launched outside the repository root, supply the actual repository path through `--project-root` rather than relying on the current working directory.

# Judge generated answers

`result_scripts/evaluate_qa_runner.py` is the repository-wide judge and metrics runner. It discovers architecture result JSONLs, performs resumable LLM judging, writes judged JSONLs, writes one text report per run, and writes a consolidated metrics report.

The judge output preserves the source result record and adds `autoeval_label`, so metadata such as binary-router routing decisions remains available to downstream comparison scripts.

## Command form

The results directory is optional because the script has a built-in default:

```bash
python result_scripts/evaluate_qa_runner.py [RESULTS_DIR] [OPTIONS]
```

For a normal run, choose only the options that need to differ from the defaults. For example, a custom judge model or custom consolidated report path can be supplied as:

```bash
python result_scripts/evaluate_qa_runner.py RESULTS_DIR \
  --metric-model JUDGE_MODEL \
  --consolidated-txt CONSOLIDATED_METRICS.txt
```

To inspect discovery without making judge calls:

```bash
python result_scripts/evaluate_qa_runner.py RESULTS_DIR --list-runs
```

The model and output paths used for a paper reproduction should come from the recorded experiment configuration, not from a hardcoded README command.

## Complete judge-runner option reference

| Argument | Required / default | Purpose / constraint |
| --- | --- | --- |
| `RESULTS_DIR` | `../asdrp/results` with compatibility resolution | Root containing generation results. |
| `--metric-model MODEL` | `gpt-4o` | Judge alias or direct model ID. Preserved aliases include `gpt-4o`, `gpt-4o-mini`, and `llama-3.1-70b-instruct`; unknown values are treated as direct model IDs. |
| `--base-url URL` | none | Optional OpenAI-compatible API base URL for a local or custom judge. |
| `--output-dir OUTPUT_DIR` | `<results_dir>/qa_evaluations` | Directory for judged JSONLs and per-run reports. |
| `--consolidated-txt FILE` | `<output-dir>/consolidated_qa_metrics.txt` | Consolidated text metrics report path. |
| `--memory-block BLOCK` / `--agent BLOCK` | all | Repeatable architecture/routing-system filter. Accepted aliases include `proposition`/`propositional`; `binary_router` is supported. |
| `--dataset DATASET` | all | Repeatable dataset filter. Aliases include `longmemeval`/`lme`, `ecommerce`, `ar`, `lru`, `ttl`, and `cr`. |
| `--file FILE` | none | Evaluate exactly one discovered primary result JSONL. Relative paths are resolved against the current working directory and then the results root. |
| `--judge-workers N` | `8` | Concurrent judge requests within each run; must be at least 1. |
| `--max-retries N` | `8` | Retries per judge request for transient or format errors; cannot be negative. |
| `--retry-base-delay SECONDS` | `1.0` | Initial exponential-backoff delay; cannot be negative. |
| `--retry-max-delay SECONDS` | `20.0` | Maximum exponential-backoff delay; cannot be negative. |
| `--no-resume` | off | Ignore content-matching evaluator outputs/checkpoints and re-judge all selected rows. |
| `--reuse-legacy-source-results` | off | Also reuse matching legacy `evaluate_qa.py` outputs located beside source hypothesis files. |
| `--list-runs` | off | List discovered runs and exit without calling a judge model. |
| `--verbose-judgments` | off | Print one compact JSON status line per judged question. |
| `--fail-fast` | off | Stop immediately if one selected run fails instead of continuing the batch. |

Repeat `--memory-block` or `--dataset` to select multiple values. Use `--file` when exactly one discovered result JSONL should be judged.

Check the exact parser in the current checkout with:

```bash
python result_scripts/evaluate_qa_runner.py --help
```

## LongMemEval evaluation-code attribution

`result_scripts/evaluate_qa.py` and `result_scripts/print_qa_metrics.py` are copied from the official LongMemEval evaluation code and kept as legacy compatibility utilities. Preserve LongMemEval's original copyright and MIT license notice when redistributing those files.

`result_scripts/evaluate_qa_runner.py` is the repository-wide runner built around the evaluation workflow. It reuses LongMemEval-specific judging behavior for LongMemEval while adding support for the other benchmark families and the current result format.

When using or redistributing the LongMemEval benchmark or its evaluation code, cite LongMemEval and preserve its upstream license notice.

# Evaluate the binary router

The router is a **pre-query, history-level** experiment. It does not see the evaluation question, reference answer, or question-type metadata when it selects an architecture.

The fixed mapping is:

```text
single-session-preference -> HVM
knowledge-update          -> Episodic Memory
```

The training script uses session-pooled word 1-3-gram TF-IDF features plus lightweight history features. It compares regularized LinearSVC and logistic-regression candidates by repeated stratified cross-validation on the training data only. The checked-in paper artifact selects LinearSVC.

## Routed-system command form

`binary_router` uses the same evaluator and therefore accepts the evaluator options listed above. Its required command shape is:

```bash
python -m asdrp.evaluate_agents_wcost LONGMEMEVAL_TEST_FILE \
  --dataset longmemeval \
  --agent binary_router \
  --router-model ROUTER_MODEL \
  --output-file ROUTER_OUTPUT_FILE
```

Add only the evaluator options that are needed for the run. Do not copy a fixed concurrency, batching, model, or rate-limit profile into the README command.

For `binary_router`, the evaluator additionally enforces the test contract stored in the router artifact:

- the dataset must be `longmemeval`;
- `--router-model` is required and must point to the expected preference/update router artifact;
- the supplied test file SHA-256 must match the test file recorded by the router artifact;
- the held-out evaluation question IDs come from the router artifact;
- `--start-index` and `--max-tasks` are rejected;
- if `--question-type-filter` is supplied, it must contain exactly `single-session-preference` and `knowledge-update`;
- the router artifact's label-to-memory mapping must match the evaluator's expected mapping.

All other evaluator options remain available subject to the normal evaluator validation rules.

## Judge the router

The router is judged by the same `evaluate_qa_runner.py` command. To restrict judging to router results, use the router filter and add any other judge options needed for the run:

```bash
python result_scripts/evaluate_qa_runner.py RESULTS_DIR \
  --memory-block binary_router
```

Use `--metric-model`, `--output-dir`, `--consolidated-txt`, retry controls, or other judge options from the complete judge-runner table when needed.

# Generate the router confusion matrix

`asdrp/classification_algorithms/create_confusion_matrix.py` reads the classifier `metrics.json`. The positional input may be either the metrics file itself or its containing directory.

## Command form

```bash
python -m asdrp.classification_algorithms.create_confusion_matrix RESULTS [OPTIONS]
```

No title, normalization mode, output path, DPI, font size, colormap, or label formatting is required in the command. Supply those flags only when a non-default figure is wanted.

By default, the current script row-normalizes each true-label row. `--no-normalize` switches the cells and colorbar to raw sample counts.

## Confusion-matrix options

| Argument | Required / default | Purpose |
| --- | --- | --- |
| `RESULTS` | required | Path to `metrics.json` or a directory containing `metrics.json`. |
| `--output OUTPUT.png` | `confusion_matrix.png` beside metrics | Output PNG path. |
| `--no-normalize` | off | Show raw counts instead of row-normalized percentages. |
| `--title TITLE` | none | Optional figure title. |
| `--dpi N` | `400` | Output resolution. |
| `--font-size SIZE` | `11.0` | Base font size. |
| `--cmap COLORMAP` | `Blues` | Matplotlib continuous colormap. |
| `--raw-labels` | off | Use labels exactly as stored in `metrics.json` instead of publication-formatted labels. |
| `--hide-counts` | off | In normalized mode, hide the raw count shown beneath each percentage. |
| `--x-rotation DEGREES` | `35.0` | Rotation angle for predicted-class labels. |

# Generate the cross-benchmark heatmap

`result_scripts/generate_memory_heatmap.py` has **no command-line arguments** in the current version. It uses these fixed paths relative to its working directory:

```text
input:  ../asdrp/results/final_eval_metrics.txt
PNG:    ../figures/memory_architecture_accuracy_heatmap.png
SVG:    ../figures/memory_architecture_accuracy_heatmap.svg
```

Because those paths are relative to the current working directory, run the script from `result_scripts/`. From the repository root, the safest one-line command is:

```bash
(
  cd result_scripts
  python generate_memory_heatmap.py
)
```

The script reads LongMemEval question-type accuracy plus overall EcommerceMemEval and MemoryAgentBench accuracy, uses one shared 0-100% scale, and annotates each cell with accuracy and the raw denominator.

Do not rename the expected section labels inside `final_eval_metrics.txt` before running the current heatmap parser. It looks up the benchmark and question-type labels by exact text.

If different input/output paths are needed, the current script must be edited; there are no `--input`, `--output`, or other CLI flags yet.

# Run the router apples-to-apples comparison

`result_scripts/compare_router_apples_to_apples.py` compares the router, HVM, and Episodic Memory on exactly the evaluated router question IDs. It reuses existing `autoeval_label.label` values and makes **no new LLM calls**.

The three evaluation inputs must be **judged** `*.eval-results-*.jsonl` files, not raw generation JSONLs.

## Command form

```bash
python result_scripts/compare_router_apples_to_apples.py \
  --router-eval ROUTER_EVAL.jsonl \
  --hvm-eval HVM_EVAL.jsonl \
  --episodic-eval EPISODIC_EVAL.jsonl \
  --reference-file REFERENCE_FILE.json \
  --output REPORT.txt
```

If routing metadata is absent from the judged router file, add `--router-hypotheses ROUTER_HYPOTHESES.jsonl` pointing to the raw router output.

## Apples-to-apples options

| Argument | Required / default | Purpose |
| --- | --- | --- |
| `--router-eval FILE` | required | Judged Binary Router JSONL. The question IDs in this file define the canonical comparison set. |
| `--hvm-eval FILE` | required | Judged standalone HVM JSONL covering every canonical router question. |
| `--episodic-eval FILE` | required | Judged standalone Episodic JSONL covering every canonical router question. |
| `--reference-file FILE` | required | LongMemEval split containing `question_id` and `question_type`. |
| `--router-hypotheses FILE` | none | Optional raw router JSONL if routing metadata is absent from the judged router file. |
| `--question-types TYPE [TYPE ...]` | `single-session-preference knowledge-update` | Allowed question types in the canonical router set. |
| `--output FILE` | required | Human-readable comparison report path. |

The report includes:

- Binary Router end-to-end accuracy;
- standalone HVM accuracy;
- standalone Episodic accuracy;
- fixed-output router selection using the router's actual architecture choice but the already-judged standalone answer for that architecture;
- a true-question-type routing reference;
- a best-of-two hindsight upper bound;
- type-level and per-question results;
- file and question-ID hashes for reproducibility.

# Workflow command map

The result-processing workflow can be run without embedding paper-specific model names, file names, or concurrency values in the README. Replace the placeholders with the paths and settings for the run being processed.

```bash
# 1. Audit generation completeness.
python result_scripts/repair_evaluation_results.py RESULTS_ROOT

# 2. If needed, execute resume-safe repair commands.
python result_scripts/repair_evaluation_results.py RESULTS_ROOT --run

# 3. Inspect the generation runs that the judge discovers.
python result_scripts/evaluate_qa_runner.py RESULTS_ROOT --list-runs

# 4. Judge selected generation outputs using the desired judge options.
python result_scripts/evaluate_qa_runner.py RESULTS_ROOT [OPTIONS]

# 5. Generate the heatmap. This script currently has fixed paths and must be run
#    from result_scripts unless its source constants are changed.
(
  cd result_scripts
  python generate_memory_heatmap.py
)

# 6. Run the matched router/HVM/Episodic comparison.
python result_scripts/compare_router_apples_to_apples.py \
  --router-eval ROUTER_EVAL.jsonl \
  --hvm-eval HVM_EVAL.jsonl \
  --episodic-eval EPISODIC_EVAL.jsonl \
  --reference-file REFERENCE_FILE.json \
  --output REPORT.txt

# 7. Generate the router confusion matrix.
python -m asdrp.classification_algorithms.create_confusion_matrix RESULTS [OPTIONS]
```

`[OPTIONS]` above is documentation notation, not literal shell text. Choose flags from the corresponding option table rather than copying a fixed experiment profile.

# Reproducibility

For each reported experiment, preserve:

- the upstream benchmark source manifest and resolved dataset revision;
- the exact benchmark split and split manifest;
- the code commit;
- the evaluator summary sidecar;
- generation, judge, and embedding model names;
- non-default concurrency, retry, Qdrant, and pricing settings;
- the raw architecture result JSONL;
- the judged result used for accuracy calculations;
- the consolidated metrics file used for tables and figures;
- the router model and data manifest for routed runs.

Use canonical benchmark files or split manifests as the source of truth for question identity and category counts. Do not recover category denominators from individual architecture outputs when a canonical manifest is available.

The LongMemEval and MemoryAgentBench split scripts default to seed `20260720`, but reproducibility requires more than the seed. Verify the source-file hash, resolved upstream revision, split arguments, selected test identities, and output hashes recorded in the committed manifests. If any of these differ, treat the regenerated data as a different split until the mismatch is explained.

The paper's router results should be read as a small proof of concept. The 14-history test set is too small for broad routing claims, and the category-to-architecture mapping was selected from the architecture study rather than an independent validation set. Fixed and routed answers were also generated in separate runs. These limits are part of the experimental interpretation, not implementation errors.

# Tests

Most tests use fake model responses, embeddings, and Qdrant clients, so the default test suite does not consume API credit:

```bash
pytest -q
```

The suite covers benchmark adapters, chunking, memory packing and retrieval behavior, graph variants, router features and serialization, runtime usage and retries, JSONL resume behavior, and local dataset-schema checks.

To run the full test suite, including the live embedding smoke test that is disabled by default, use:

```bash
RUN_LIVE_API_TESTS=1 pytest -v -ra 2>&1 | tee test_results.txt
```

Only enable live API testing when `OPENAI_API_KEY` is set and a live API call is intended.

# Third-party code and benchmark attribution

This repository combines original experiment code with third-party benchmarks and selected upstream utilities. Keep upstream licenses and attribution with any redistributed benchmark data or copied code.

## LongMemEval

LongMemEval provides one of the three benchmark settings used in the paper. The legacy `result_scripts/evaluate_qa.py` and `result_scripts/print_qa_metrics.py` files are copied from the official LongMemEval evaluation code. Preserve the original LongMemEval copyright and MIT license notice when redistributing those files. The repository-wide `evaluate_qa_runner.py` reuses LongMemEval-specific judging behavior where applicable but adds the multi-benchmark result handling used by this project.

Paper citation:

> D. Wu, H. Wang, W. Yu, Y. Zhang, K.-W. Chang, and D. Yu, "LongMemEval: Benchmarking chat assistants on long-term interactive memory," ICLR, 2025.

## MemoryAgentBench

MemoryAgentBench provides the Accurate Retrieval, Long-Range Understanding, Test-Time Learning, and Conflict Resolution tasks used in the paper.

Paper citation:

> Y. Hu, Y. Wang, and J. McAuley, "Evaluating memory in LLM agents via incremental multi-turn interactions," ICLR, 2026.

## Channel Labs Synthetic Conversation Generation

EcommerceMemEval uses synthetic customer conversations generated with the Channel Labs `synthetic-conversation-generation` project as the starting point for its conversation data. The dataset construction, task definitions, and evaluation setup used in this repository are part of this study.

EcommerceMemEval is introduced in this study and is built from synthetic customer conversations; no real customer data is used.

## Architecture references

HVM follows the hierarchical retrieval direction of RAPTOR, while Graph Memory is motivated by graph-based long-term memory work such as HippoRAG. The implementations in this repository are part of the shared MemAgent evaluation system rather than direct reproductions of those systems.

> P. Sarthi et al., "RAPTOR: Recursive abstractive processing for tree-organized retrieval," ICLR, 2024.

> B. Jimenez Gutierrez et al., "HippoRAG: Neurobiologically inspired long-term memory for large language models," NeurIPS, 2024.
