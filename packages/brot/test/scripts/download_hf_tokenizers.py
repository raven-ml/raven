#!/usr/bin/env python3
"""
Download selected HuggingFace tokenizer JSON files into brot/test/fixtures/hf.

Run this script whenever you need to refresh the fixtures:

    python3 brot/test/scripts/download_hf_tokenizers.py

The files are ignored by git, so each developer/machine maintains its own cache.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Iterable, Tuple


FIXTURES: Iterable[Tuple[str, str]] = (
    (
        "bert-base-uncased",
        "https://huggingface.co/bert-base-uncased/resolve/main/tokenizer.json?download=1",
    ),
    (
        "gpt2",
        "https://huggingface.co/gpt2/resolve/main/tokenizer.json?download=1",
    ),
    (
        "roberta-base",
        "https://huggingface.co/roberta-base/resolve/main/tokenizer.json?download=1",
    ),
)


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(".tmp")

    print(f"→ downloading {url}…")
    with urllib.request.urlopen(url) as response, open(tmp_path, "wb") as out:
        while True:
            chunk = response.read(1024 * 64)
            if not chunk:
                break
            out.write(chunk)
    tmp_path.replace(dest)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 64), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize(path: Path) -> None:
    try:
        with path.open("r", encoding="utf-8") as fh:
            metadata = json.load(fh)
        model_type = metadata.get("model", {}).get("type", "<unknown>")
        size = path.stat().st_size
        digest = sha256(path)[:12]
        print(f"  saved {path} ({size} bytes, model={model_type}, sha256={digest})")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"  warning: failed to inspect {path}: {exc}")


def main() -> int:
    test_root = Path(__file__).resolve().parents[1]
    fixtures_dir = test_root / "fixtures" / "hf"
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    for model, url in FIXTURES:
        target = fixtures_dir / model / "tokenizer.json"
        if target.exists():
            print(f"✓ {model} already present at {target}")
            continue
        print(f"Downloading {model} tokenizer…")
        try:
            download(url, target)
            summarize(target)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"  failed to download {model}: {exc}", file=sys.stderr)
            if target.exists():
                target.unlink()
            return 1

    print("All fixtures downloaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
