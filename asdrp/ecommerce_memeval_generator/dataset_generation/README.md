# EcommerceMemEval Dataset Generation

This directory builds **EcommerceMemEval**, the synthetic customer-state dataset introduced in _Predicting Future Memory Demand for Architecture Routing in LLM-based Agents_.

The paper uses EcommerceMemEval to test whether the memory architectures transfer to a domain with persistent customer state. The evaluated release contains **351 synthetic customer histories and 1,747 questions across ten memory operations**. Histories include preferences, constraints, cart and wishlist state, purchases, returns, budgets, deadlines, and later updates. No real customer data is used.

This README covers only **dataset construction**. For the six memory architectures, shared evaluator, answer judging, router, paper figures, and full experiment workflow, use the repository-level `README.md`.

## Scope of this directory

This stage starts from existing synthetic ecommerce conversations. It does **not** create those source conversations.

The source-conversation generator is kept separately at:

```text
../synthetic-conversation-generation/
```

For each source conversation, this pipeline:

1. validates the message structure;
2. assigns stable customer and session IDs;
3. formats the conversation as dataset context;
4. asks a model to create grounded memory questions and answers;
5. checks the question type, evidence IDs, grounding, and answer format;
6. retries invalid generations;
7. writes validated dataset rows;
8. records failed rows separately for audit.

A deterministic `--dry-run` path tests the pipeline without an API call. Dry-run questions are placeholders and must not be used as dataset data.

## Directory layout

```text
dataset_generation/
├── README.md
├── configs/
│   └── generation.yaml
└── src/
    └── ecommerce_memeval/
        ├── __init__.py
        ├── build_dataset.py       # Main generation pipeline
        ├── context_builder.py     # Customer/session IDs and context formatting
        ├── io_utils.py            # JSONL/YAML helpers and JSON cleanup
        ├── llm_client.py          # OpenAI JSON-generation client
        ├── models.py              # Pydantic schemas and question-type enum
        ├── prompts.py             # Generation prompt and label definitions
        ├── split_dataset.py       # Generic random split helper
        └── validate_dataset.py    # Dataset schema and label validation
```

The paper uses the customer-disjoint split utility in the repository root:

```text
../../../datasets/ecommerce_percent_split_generator.py
```

The local `split_dataset.py` helper is kept for general development, but it does not enforce customer-disjoint groups and should not be used for the paper split.

## Before running the generator

Complete the main environment setup in the root `README.md` first. Live generation needs `openai`, `python-dotenv`, Pydantic 2, PyYAML, and `tqdm`, plus `OPENAI_API_KEY`.

The generator package uses a nested `src/` layout. Its configured data paths are also resolved from the current working directory. For that reason, run the commands in this README from `asdrp/`:

```bash
cd asdrp
export PYTHONPATH="$PWD/ecommerce_memeval_generator/dataset_generation/src${PYTHONPATH:+:$PYTHONPATH}"
```

The remaining commands assume this working directory and `PYTHONPATH`.

## Input conversations

Each input JSONL line must contain an object with a `messages` array. Each message needs:

- `role`: `user`, `assistant`, or `system`;
- `content`: a non-empty string.

Minimal example:

```json
{
  "messages": [
    {"role": "user", "content": "I usually prefer unscented products."},
    {"role": "assistant", "content": "I can keep that preference in mind."}
  ]
}
```

The default configuration points to the synthetic conversation data under:

```text
ecommerce_memeval_generator/synthetic-conversation-generation/data/conversations/
```

Use the configuration file as the source of truth for the exact input file used by a generation run.

## Generated dataset row

Each accepted source conversation becomes one dataset row with one shared context and several aligned QA records:

```json
{
  "context": "<formatted multi-session conversation>",
  "questions": [
    "What type of products does the customer prefer?"
  ],
  "answers": [
    ["The customer prefers unscented products."]
  ],
  "metadata": {
    "customer_id": "cust_00001",
    "source": "ecommerce_conversations_final",
    "qa_pair_ids": ["cust_00001_q001"],
    "question_types": ["preference_recall"],
    "evidence": [
      {
        "qa_pair_id": "cust_00001_q001",
        "session_ids": ["cust_00001_s001"],
        "facts": ["The customer says they usually prefer unscented products."]
      }
    ]
  }
}
```

These arrays are position-aligned:

```text
questions[i]
answers[i]
metadata.qa_pair_ids[i]
metadata.question_types[i]
metadata.evidence[i]
```

`models.py` validates the array lengths and checks that each evidence record uses the matching QA ID.

## Question types

The generator supports ten memory operations.

| Question type | What it tests |
| --- | --- |
| `preference_recall` | Stable customer preferences or usual choices |
| `constraint_recall` | Hard constraints such as size, compatibility, budget, material, allergy, or capacity |
| `purchase_history_recall` | Confirmed purchases or finalized orders |
| `cart_or_wishlist_recall` | Viewed, saved, wishlisted, reserved, or carted items that were not confirmed purchases |
| `return_support_recall` | Returns, refunds, exchanges, defects, warranties, replacements, or support issues |
| `temporal_update` | The current state after a later change |
| `conflict_resolution` | Correcting stale, rejected, duplicated, or conflicting state |
| `multi_session_synthesis` | Combining evidence from at least two sessions |
| `recommendation_from_memory` | Making a recommendation from remembered preferences, constraints, or history |
| `abstention` | Recognizing that the requested information was never provided |

The generation prompt and exact label rules live in:

```text
src/ecommerce_memeval/prompts.py
```

Use those definitions when changing or auditing question-type behavior.

## Grounding and evidence rules

Every non-abstention QA must point to evidence from the current conversation. It needs:

- at least one valid `session_id`;
- concrete evidence facts that support the answer.

The validator rejects session IDs that do not exist in the formatted context.

Abstention examples use a different rule:

- `session_ids` must be empty;
- the answer must state that the requested detail is not available in the context;
- the evidence note must explain what information is missing rather than invent support.

The generator also applies label-specific checks. For example, a `purchase_history_recall` item needs evidence that a purchase was completed, while `cart_or_wishlist_recall` should not describe an item that was already purchased.

These checks reduce common generation errors, but they do not replace manual review of dataset quality.

## Generation configuration

The main configuration file is:

```text
configs/generation.yaml
```

It controls:

- input conversation path;
- dataset and error output paths;
- provider and model;
- temperature;
- minimum and maximum QA pairs per conversation;
- maximum generation attempts;
- customer ID prefix and source label;
- random seed;
- optional row limit;
- dry-run mode;
- allowed question types.

The current config uses the OpenAI provider and `gpt-4.1` for QA generation.

Keep the exact YAML used for any released dataset version. Command-line overrides are useful for smoke tests, but a research release should still preserve the final effective configuration.

## 1. Run a no-API smoke test

Start with a small dry run:

```bash
python -m ecommerce_memeval.build_dataset \
  --config ecommerce_memeval_generator/dataset_generation/configs/generation.yaml \
  --dry-run \
  --limit 3
```

The default configuration writes the dataset and error files under:

```text
datasets/ecommerce_memeval/jsonl_format/
datasets/ecommerce_memeval/generation_errors/
```

The builder clears the configured output and error files at the start of a run. Do not point a smoke test at a dataset file you need to keep.

Validate the dry-run output:

```bash
python -m ecommerce_memeval.validate_dataset \
  datasets/ecommerce_memeval/jsonl_format/ecommerce_memeval_dataset_latest.jsonl
```

A successful validation reports the row count, QA count, and question-type counts. Delete or overwrite the dry-run output before producing real dataset data.

## 2. Generate real QA data

Set `OPENAI_API_KEY`, then run:

```bash
python -m ecommerce_memeval.build_dataset \
  --config ecommerce_memeval_generator/dataset_generation/configs/generation.yaml
```

For a small live check first:

```bash
python -m ecommerce_memeval.build_dataset \
  --config ecommerce_memeval_generator/dataset_generation/configs/generation.yaml \
  --limit 5
```

The generator tracks question-type counts and asks for underrepresented types. It retries a conversation when a generated candidate fails schema or quality checks. A failed conversation is written to the configured error JSONL instead of stopping the full generation run.

## 3. Validate the final dataset

Run the schema and label validator before splitting:

```bash
python -m ecommerce_memeval.validate_dataset \
  datasets/ecommerce_memeval/jsonl_format/ecommerce_memeval_dataset_latest.jsonl \
  --show-examples 3
```

Review both the validation result and the question-type distribution. Schema validity does not guarantee that every generated question is useful or difficult, so manual QA review remains important for a research release.

## 4. Create the paper split

Return to the repository root:

```bash
cd ..
```

Create the customer-disjoint split:

```bash
python datasets/ecommerce_percent_split_generator.py \
  asdrp/datasets/ecommerce_memeval/jsonl_format/ecommerce_memeval_dataset_latest.jsonl \
  --output-dir asdrp/datasets/ecommerce_memeval/splits/ecommerce_20pct \
  --train-fraction 0.80 \
  --write-manifest
```

The splitter keeps all rows for one `metadata.customer_id` on the same side of the split. It also checks QA-ID overlap and identical-record overlap and records output hashes in the manifest.

This customer-disjoint split is the research path used by the repository. Do not substitute the local random `split_dataset.py` helper when reproducing the paper.

## What to preserve for a dataset release

Keep the following together for each regenerated EcommerceMemEval version:

- source conversation file or its hash;
- exact `generation.yaml`;
- code commit;
- generated dataset JSONL;
- generation error JSONL;
- validation counts;
- customer-disjoint split manifest;
- final train/test files and hashes.

Do not silently edit generated labels, answers, or evidence after the split. Record corrections in Git or a separate repair log, then regenerate the affected manifest and hashes.

## Dataset interpretation

EcommerceMemEval was designed to cover a broad set of persistent customer-state operations. In the paper, however, the strongest memory systems score close to the dataset ceiling. The dataset therefore shows domain transfer more clearly than it separates the strongest architectures.

This matters when extending the dataset. Harder future versions should place more pressure on current-state tracking, including denser obsolete-but-similar facts, repeated preference revisions, close cart-to-purchase transitions, longer evidence gaps, and stronger multi-session evidence requirements.

## Common problems

### `ModuleNotFoundError: No module named 'ecommerce_memeval'`

Run from `asdrp/` and add the nested source directory to `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/ecommerce_memeval_generator/dataset_generation/src${PYTHONPATH:+:$PYTHONPATH}"
```

### Input file cannot be found

Paths inside `generation.yaml` are resolved from the current working directory. Confirm that you are inside `asdrp/` and that the configured synthetic conversation file exists.

### Config file cannot be found

Pass the config path explicitly as shown above. For custom runs, confirm the path supplied to `--config`.

### `OPENAI_API_KEY` is missing

Set the key for live generation. Use `--dry-run` when you only want to test formatting and validation.

### Many rows are written to the error JSONL

Inspect each row's recorded error and raw model output. Common causes include malformed source messages, invalid model JSON, unsupported evidence IDs, weak label evidence, or repeated failure to produce enough valid QA pairs.

## Continue with paper evaluation

After generation, validation, and the customer-disjoint split are complete, return to the root `README.md` for:

- running all six memory architectures;
- judging answers with the paper judge;
- generating consolidated metrics;
- creating the cross-benchmark heatmap;
- reproducing the router experiments.
