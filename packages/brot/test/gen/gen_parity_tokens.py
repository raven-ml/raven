# /// script
# requires-python = ">=3.11"
# dependencies = ["tokenizers==0.23.1"]
# ///
"""Generate the tokens HuggingFace tokenizers gives for the parity corpora.

    uv run packages/brot/test/gen/gen_parity_tokens.py [--check]

Writes `support/parity/<corpus>.<name>.tokens`, which `test_parity.ml` reads.
With `--check` it writes nothing and fails, naming each file, if a file
differs from what it would write.

It reads the trimmed tokenizer files of `support/tokenizers`, which
`trim_parity_tokenizers.py` makes encode the parity corpora as the full files
do, so it needs no download. The corpora are split into documents as
`gen_parity_expected.py` splits them. A file holds a comment line, then one
line per document: the JSON array of the tokens of the document encoded with
`add_special_tokens=True`, the special tokens included.
"""

import json
import sys
from pathlib import Path

import tokenizers
from tokenizers import Tokenizer

from gen_parity_expected import CORPORA, REGEX_SPLIT_CORPORA, documents

TEST_ROOT = Path(__file__).resolve().parents[1]
PARITY = TEST_ROOT / "support" / "parity"
DATA = TEST_ROOT / "support" / "tokenizers"

# The tokenizers whose tokens brot gives as tokenizers does. T5's Unigram model
# spells an unknown token as the text it stands for, where brot gives the
# vocabulary's entry.
TOKENIZERS = {
    "gpt2": CORPORA,
    "bert_base": CORPORA,
    "roberta_base": CORPORA,
    "mistral": CORPORA,
    "qwen2_5": REGEX_SPLIT_CORPORA,
    "deepseek_v3": REGEX_SPLIT_CORPORA,
    "gpt_oss": REGEX_SPLIT_CORPORA,
    "phi4": REGEX_SPLIT_CORPORA,
    "mistral_nemo": REGEX_SPLIT_CORPORA,
}


def text(tokenizer: Tokenizer, corpus: str, name: str) -> str:
    lines = [
        f"# tokenizers {tokenizers.__version__}, corpus {corpus}.txt, "
        f"tokenizer {name}"
    ]
    source = (PARITY / f"{corpus}.txt").read_bytes().decode("utf-8")
    for document in documents(source):
        tokens = tokenizer.encode(document, add_special_tokens=True).tokens
        lines.append(json.dumps(tokens, ensure_ascii=False))
    return "\n".join(lines) + "\n"


def main() -> int:
    check = "--check" in sys.argv[1:]
    stale = []
    for name, corpora in TOKENIZERS.items():
        tokenizer = Tokenizer.from_file(str(DATA / f"{name}.json"))
        for corpus in corpora:
            out = PARITY / f"{corpus}.{name}.tokens"
            data = text(tokenizer, corpus, name)
            if not check:
                out.write_text(data, encoding="utf-8")
            elif not out.exists() or out.read_text(encoding="utf-8") != data:
                print(f"{out} differs from what the generator writes",
                      file=sys.stderr)
                stale.append(out)
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
