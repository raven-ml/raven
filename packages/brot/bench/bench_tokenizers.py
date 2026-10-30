# /// script
# requires-python = ">=3.12"
# dependencies = ["tokenizers==0.23.3"]
# ///
"""HuggingFace tokenizers' side of brot's bench, through its Python binding:
each row's time per call. download_data.sh fetches the tokenizers it reads.

usage: uv run packages/brot/bench/bench_tokenizers.py
"""

from __future__ import annotations

import gc
import timeit
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

from tokenizers import Tokenizer

_DATA_DIR = Path(__file__).resolve().parent / "data"
REPEATS = 5

SHORT_TEXT = (_DATA_DIR / "news_1k.txt").read_text(encoding="utf-8")
LONG_TEXT = (_DATA_DIR / "wiki_64k.txt").read_text(encoding="utf-8")
BATCH_32 = [SHORT_TEXT] * 32

Row = Tuple[str, Callable[[], object]]


# Timing


def time_per_call(fn: Callable[[], object]) -> float:
    """Seconds per call of fn: the fastest of REPEATS batches, each of as many
    calls as take 0.2 s. The collector stays on, as in a program."""
    timer = timeit.Timer(fn, setup=gc.enable)
    number, _ = timer.autorange()
    return min(timer.repeat(repeat=REPEATS, number=number)) / number


def format_time(seconds: float) -> str:
    for scale, unit in ((1.0, "s"), (1e-3, "ms"), (1e-6, "µs")):
        if seconds >= scale:
            return f"{seconds / scale:.2f}{unit}"
    return f"{seconds * 1e9:.2f}ns"


def run(rows: Sequence[Row]) -> None:
    width = max(len(name) for name, _ in rows)
    for name, fn in rows:
        print(f"{name:<{width}}  {format_time(time_per_call(fn))}", flush=True)


# Rows


def suite(label: str, filename: str) -> List[Row]:
    tokenizer = Tokenizer.from_file(str(_DATA_DIR / filename))
    decode_ids = tokenizer.encode(LONG_TEXT).ids
    return [
        (f"{label}/Encode/single_short", lambda: tokenizer.encode(SHORT_TEXT)),
        (f"{label}/Encode/single_long", lambda: tokenizer.encode(LONG_TEXT)),
        (f"{label}/Encode/batch_32", lambda: tokenizer.encode_batch(BATCH_32)),
        (f"{label}/Decode/long", lambda: tokenizer.decode(decode_ids)),
    ]


def build_rows() -> List[Row]:
    return [
        *suite("GPT-2", "gpt2.json"),
        *suite("BERT-base", "bert_base.json"),
        *suite("LLaMA", "llama.json"),
    ]


if __name__ == "__main__":
    run(build_rows())
