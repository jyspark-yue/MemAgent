#############################################################################
# File: download_mab.py
#
# Description:
#   Downloads MemoryAgentBench from Hugging Face and saves local JSON copies.
#
#   - Fetches the ai-hyz/MemoryAgentBench dataset.
#   - Writes each source split under separated_splits/.
#   - Writes one combined memory_agent_bench_pooled.json file.
#############################################################################

import json
from pathlib import Path

from datasets import load_dataset

OUT_DIR = Path(".")
SPLIT_DIR = Path("separated_splits")
SPLIT_DIR.mkdir(exist_ok=True)

dataset = load_dataset("ai-hyz/MemoryAgentBench")

# Save each split as a readable JSON file
for split_name, split_data in dataset.items():
    rows = [dict(row) for row in split_data]

    out_path = SPLIT_DIR / f"{split_name}.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(rows)} rows to {out_path}")

# Save all splits together in one readable JSON file
combined = {}

for split_name, split_data in dataset.items():
    combined[split_name] = [dict(row) for row in split_data]

combined_path = OUT_DIR / "memory_agent_bench_pooled.json"

with combined_path.open("w", encoding="utf-8") as f:
    json.dump(combined, f, ensure_ascii=False, indent=2)

print(f"Saved combined file to {combined_path}")
