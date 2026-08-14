"""Download the official cleaned LongMemEval files used by MemAgent.

The downloader resolves the requested Hugging Face revision to an immutable
commit, copies only the requested benchmark variants into this directory, and
writes a small source manifest with file SHA-256 hashes. Commit that manifest
for paper reproducibility; the downloaded benchmark JSON files can remain
ignored by Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "xiaowu0162/longmemeval-cleaned"
VARIANT_FILES = {
    "m": "longmemeval_m_cleaned.json",
    "s": "longmemeval_s_cleaned.json",
    "oracle": "longmemeval_oracle.json",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_in_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=tuple(VARIANT_FILES),
        action="append",
        help=(
            "LongMemEval variant to download. Repeat to fetch multiple variants. "
            "Default: m, which is the source used by the paper split."
        ),
    )
    parser.add_argument(
        "--revision",
        default="main",
        help=(
            "Hugging Face revision, tag, or commit. The resolved immutable commit "
            "is written to the source manifest."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.resolve(),
        help="Directory for downloaded JSON files and the source manifest.",
    )
    args = parser.parse_args()

    variants = unique_in_order(args.variant or ["m"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    info = HfApi().dataset_info(REPO_ID, revision=args.revision)
    resolved_revision = info.sha
    if not resolved_revision:
        raise RuntimeError(f"Could not resolve Hugging Face revision {args.revision!r}.")

    files: list[dict[str, object]] = []
    for variant in variants:
        filename = VARIANT_FILES[variant]
        cached_path = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=filename,
                revision=resolved_revision,
            )
        )
        destination = args.output_dir / filename
        shutil.copy2(cached_path, destination)
        entry = {
            "variant": variant,
            "filename": filename,
            "size_bytes": destination.stat().st_size,
            "sha256": file_sha256(destination),
        }
        files.append(entry)
        print(f"Saved {filename} -> {destination}")

    manifest = {
        "dataset": "LongMemEval",
        "huggingface_repo": REPO_ID,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "files": files,
    }
    manifest_path = args.output_dir / "longmemeval_source_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Saved source manifest -> {manifest_path}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
