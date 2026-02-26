from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List

from tokenizers import Tokenizer

_ROOT = Path(__file__).resolve().parent
_DATA_DIR = _ROOT / "data"

import sys

vendor_dir = _ROOT.parents[1] / "vendor" / "ubench"
if str(vendor_dir) not in sys.path:
    sys.path.insert(0, str(vendor_dir))

import ubench  # type: ignore


SHORT_TEXT = (_DATA_DIR / "news_1k.txt").read_text(encoding="utf-8")
LONG_TEXT = (_DATA_DIR / "wiki_64k.txt").read_text(encoding="utf-8")
BATCH_32 = [SHORT_TEXT] * 32


def load_tokenizer(filename: str) -> Tokenizer:
    path = _DATA_DIR / filename
    return Tokenizer.from_file(str(path))


def make_suite(label: str, tokenizer: Tokenizer) -> Any:
    decode_ids = tokenizer.encode(LONG_TEXT).ids

    benches: List[Any] = [
        ubench.bench("Encode/single_short", lambda: tokenizer.encode(SHORT_TEXT)),
        ubench.bench("Encode/single_long", lambda: tokenizer.encode(LONG_TEXT)),
        ubench.bench("Encode/batch_32", lambda: tokenizer.encode_batch(BATCH_32)),
        ubench.bench("Decode/long", lambda: tokenizer.decode(decode_ids)),
    ]

    return ubench.group(label, benches)


def build_benchmarks() -> List[Any]:
    return [
        make_suite("GPT-2", load_tokenizer("gpt2.json")),
        make_suite("BERT-base", load_tokenizer("bert_base.json")),
        make_suite("LLaMA", load_tokenizer("llama.json")),
    ]


def default_config() -> ubench.Config:
    return ubench.Config.default().build()


def main() -> None:
    benchmarks = build_benchmarks()
    config = default_config()
    ubench.run(benchmarks, config=config, output_format="pretty", verbose=False)


if __name__ == "__main__":
    main()
