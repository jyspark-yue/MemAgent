# ecommerce_memeval automation

This turns `ecommerce_conversations_final.jsonl` into your target ecommerce memory-eval format.
Each input JSONL line becomes one dataset row:

```json
{
  "context": "...",
  "questions": ["..."],
  "answers": [["..."]],
  "metadata": {
    "customer_id": "cust_00042",
    "source": "synthetic_ecommerce",
    "qa_pair_ids": ["cust_00042_q001"],
    "question_types": ["preference_recall"],
    "evidence": [{"qa_pair_id": "cust_00042_q001", "session_ids": ["cust_00042_s003"], "facts": ["..."]}]
  }
}
```

## Setup

```bash
cd ecommerce_memeval_code
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your_key_here"
cp /path/to/ecommerce_conversations_small_final.jsonl .
```

## Smoke test without API calls

```bash
python -m ecommerce_memeval_generator.build_dataset --config configs/generation.yaml --dry-run --limit 3
python -m ecommerce_memeval_generator.validate_dataset ecommerce_memeval_dataset.jsonl
```

## Real generation

Edit `configs/generation.yaml` and set `dry_run: false`, then:

```bash
python -m ecommerce_memeval_generator.build_dataset --config configs/generation.yaml
python -m ecommerce_memeval_generator.validate_dataset ecommerce_memeval_dataset.jsonl
```

## Useful outputs

- `ecommerce_memeval_dataset.jsonl`: final dataset.
- `ecommerce_memeval_errors.jsonl`: rows that failed generation/validation.
