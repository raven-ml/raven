# /// script
# requires-python = ">=3.11"
# dependencies = ["tokenizers==0.23.1"]
# ///
"""Generate the truncation windows that HuggingFace tokenizers gives.

    uv run packages/brot/test/gen/gen_stride_expected.py [--check]

Writes `support/parity/stride_windows.txt`, which `test_encode_decode.ml`
reads. With `--check` it writes nothing and fails if the file differs from
what it would write.

For each case, the kept ids and the ids of every overflowing window that
tokenizers gives with `enable_truncation(max_length, stride, direction)` on
the trimmed tokenizer files of `support/tokenizers`. After a comment line,
the file holds the single text, the first and the second text of the pair,
one per line, then one line per case:

    <model> single|pair <max_length> <stride> left|right | <ids> | <ids> ...

the kept ids first, then each window's.
"""

import sys
from pathlib import Path

from tokenizers import Tokenizer

TEST_ROOT = Path(__file__).resolve().parents[1]
OUT = TEST_ROOT / "support" / "parity" / "stride_windows.txt"

# The trimmed copies the tests read, which trim_parity_tokenizers.py makes
# encode these texts as the full files do.
DATA = TEST_ROOT / "support" / "tokenizers"

SINGLE = "The quick brown fox jumps over the lazy dog while seventeen astronauts"
PAIR = (
    "The quick brown fox jumps over the lazy dog",
    "Seventeen astronauts orbit the small green planet tonight",
)

# (model, single|pair, max_length, stride, direction)
CASES = [
    ("gpt2", "single", 8, 0, "right"),
    ("gpt2", "single", 8, 2, "right"),
    ("gpt2", "single", 12, 7, "right"),
    ("gpt2", "pair", 8, 0, "right"),
    ("gpt2", "pair", 10, 2, "right"),
    ("gpt2", "pair", 20, 7, "right"),
    ("gpt2", "pair", 10, 2, "left"),
    ("bert_base", "single", 8, 0, "right"),
    ("bert_base", "single", 8, 2, "right"),
    ("bert_base", "single", 12, 7, "right"),
    ("bert_base", "single", 12, 7, "left"),
    ("bert_base", "pair", 8, 0, "right"),
    ("bert_base", "pair", 12, 2, "right"),
    ("bert_base", "pair", 20, 7, "right"),
]


def text() -> str:
    lines = ["# tokenizers==0.23.1", SINGLE, *PAIR]
    tokenizers = {}
    for model, mode, max_length, stride, direction in CASES:
        tok = tokenizers.setdefault(
            model, Tokenizer.from_file(str(DATA / f"{model}.json"))
        )
        tok.enable_truncation(
            max_length=max_length, stride=stride, direction=direction
        )
        enc = tok.encode(SINGLE) if mode == "single" else tok.encode(*PAIR)
        rows = [enc.ids] + [o.ids for o in enc.overflowing]
        windows = " | ".join(" ".join(map(str, row)) for row in rows)
        lines.append(
            f"{model} {mode} {max_length} {stride} {direction} | {windows}"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    expected = text()
    if "--check" in sys.argv[1:]:
        if not OUT.exists() or OUT.read_text() != expected:
            print(f"{OUT} differs from what tokenizers gives", file=sys.stderr)
            return 1
        return 0
    OUT.write_text(expected)
    return 0


if __name__ == "__main__":
    sys.exit(main())
