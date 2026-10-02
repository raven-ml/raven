# Generates the expected rows of the stride differential in
# test_tokenization.ml (`gpt2_stride_cases` / `bert_stride_cases`): for each
# case, the primary ids and every overflowing window's ids from HuggingFace
# tokenizers on the same tokenizer.json. Run with
#
#   uv run --with tokenizers==0.23.1 python3 gen_stride_expected.py
#
# from this directory and paste the printed fragments over the case lists.

from pathlib import Path

from tokenizers import Tokenizer

DATA = Path(__file__).resolve().parents[2] / "bench" / "data"

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


def ocaml_array(ids):
    return "[| " + "; ".join(map(str, ids)) + " |]"


def main():
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
        print(
            f"      case ~max_length:{max_length} ~stride:{stride} "
            f"~direction:{variant} ~pair:{pair}"
        )
        print("        [")
        for row in rows:
            print(f"          {ocaml_array(row)};")
        print("        ];")


if __name__ == "__main__":
    main()
