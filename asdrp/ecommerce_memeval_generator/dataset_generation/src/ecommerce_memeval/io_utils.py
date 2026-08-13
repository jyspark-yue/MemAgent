#############################################################################
# File: io_utils.py
#
# Description:
#   Provides small YAML/JSONL I/O helpers for EcommerceMemEval generation.
#
#   - Loads YAML configuration and streams JSONL with line-aware parse errors.
#   - Writes or appends UTF-8 JSONL records.
#   - Recovers a JSON object from model output, including accidental markdown fences.
#############################################################################

from __future__ import annotations

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

import json
from pathlib import Path
from typing import Any, Iterable
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def iter_jsonl(path: str | Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_num, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num}: {exc}") from exc


def write_jsonl(
    path: str | Path, rows: Iterable[dict[str, Any]], append: bool = False
) -> None:
    mode = "a" if append else "w"
    with Path(path).open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    write_jsonl(path, [row], append=True)


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract a JSON object from model text, allowing accidental markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])
