# /// script
# requires-python = ">=3.11"
# dependencies = ["tokenizers==0.23.1"]
# ///
"""Generate the expected rows of the stride differential in
test_tokenization.ml (`gpt2_stride_cases`, `bert_stride_cases`).

    uv run packages/brot/test/gen/gen_stride_expected.py [--check]

For each case, the primary ids and every overflowing window's ids that
HuggingFace tokenizers gives on the same tokenizer.json. Without `--check` it
prints the fragments to paste over the case lists; with `--check` it compares
them with the lists in test_tokenization.ml and fails, naming each case that
differs.
"""

import sys
from pathlib import Path

from tokenizers import Tokenizer

import ocaml_cases

TEST_ROOT = Path(__file__).resolve().parents[1]
TEST = TEST_ROOT / "test_tokenization.ml"

# The trimmed copies test_tokenization.ml reads, which trim_parity_tokenizers.py
# makes encode these texts as the full files do.
DATA = TEST_ROOT / "support" / "tokenizers"

SINGLE = "The quick brown fox jumps over the lazy dog while seventeen astronauts"
PAIR = (
    "The quick brown fox jumps over the lazy dog",
    "Seventeen astronauts orbit the small green planet tonight",
)

# The case list of each model in test_tokenization.ml.
GROUPS = {"gpt2": "gpt2_stride_cases", "bert_base": "bert_stride_cases"}

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


def ocaml_array(ids):
    return "[| " + "; ".join(map(str, ids)) + " |]"


def fragments() -> dict[str, list[str]]:
    """The lines of each case list, by its name in the test."""
    out = {group: [] for group in GROUPS.values()}
    tokenizers = {}
    for model, mode, max_length, stride, direction in CASES:
        tok = tokenizers.setdefault(
            model, Tokenizer.from_file(str(DATA / f"{model}.json"))
        )
        tok.enable_truncation(
            max_length=max_length, stride=stride, direction=direction
        )
        enc = tok.encode(SINGLE) if mode == "single" else tok.encode(*PAIR)
        rows = [list(enc.ids)] + [list(o.ids) for o in enc.overflowing]
        variant = "`Left" if direction == "left" else "`Right"
        pair = "true" if mode == "pair" else "false"
        lines = out[GROUPS[model]]
        lines.append(
            f"      case ~max_length:{max_length} ~stride:{stride} "
            f"~direction:{variant} ~pair:{pair}"
        )
        lines.append("        [")
        lines.extend(f"          {ocaml_array(row)};" for row in rows)
        lines.append("        ];")
    return out


def check(groups: dict[str, list[str]]) -> int:
    defined = ocaml_cases.definitions(ocaml_cases.read(TEST))
    problems = []
    for name, texts in (("stride_single", (SINGLE,)), ("stride_pair", PAIR)):
        found = [value for kind, value in defined[name] if kind == "str"]
        if found != [text.encode() for text in texts]:
            problems.append(f"{name}: the test's text is not the generator's")
    for group, lines in groups.items():
        generated = ocaml_cases.cases(
            ocaml_cases.tokens("\n".join(lines)), "case", 1
        )
        recorded = ocaml_cases.cases(defined[group], "case", 1)
        problems += ocaml_cases.differences(
            group, generated, recorded, partial=False
        )
    return ocaml_cases.report(problems)


def main() -> int:
    groups = fragments()
    if "--check" in sys.argv[1:]:
        return check(groups)
    for group, lines in groups.items():
        print(f"(* {group} *)")
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
