#!/usr/bin/env python3
"""Generate the expected encodings that brot is checked against.

Run from anywhere, with the version the committed fixtures were generated with:

    uv run --with tokenizers==0.23.1 python3 packages/brot/test/scripts/gen_parity_expected.py

The reference is the HuggingFace `tokenizers` library. `test_parity.ml` reads
the files this writes and requires brot to produce exactly the same encodings,
so regenerate only when a fixture changes or when a divergence has been
confirmed to be a bug in the reference rather than in brot. Regenerating with
the pinned version rewrites every file byte for byte, so a `git diff` after a
run over an unchanged corpus is a reference that moved, not a fixture to
accept.

Tokenizers
----------

`bench/download_data.sh` writes the tokenizer files this reads to `bench/data/`,
and `test_parity.ml` reads the same files, so the reference and brot are always
given one tokenizer:

- `gpt2` — byte-level BPE, ByteLevel pre-tokenizer and decoder.
- `llama` — BPE with byte fallback, a Prepend/Replace normalizer, no
  pre-tokenizer.
- `bert_base` — WordPiece, BertNormalizer, BertPreTokenizer, TemplateProcessing.
- `roberta_base` — byte-level BPE with `RobertaProcessing`, which trims the
  offsets the ByteLevel pre-tokenizer produced and wraps the ids in
  `<s>`/`</s>`.
- `t5_base_nonorm` — Unigram with a Metaspace pre-tokenizer and decoder, and a
  TemplateProcessing that appends `</s>`.

`t5_base_nonorm` is not stock T5: T5's only normalizer is a `Precompiled`
SentencePiece charsmap, which brot does not implement and refuses to load, so
`download_data.sh` drops the `normalizer` field on the way in. What that costs
is the NFKC-like character folding the charsmap performs, which on these corpora
reaches little more than the edge cases; both sides read the derived file, so
the fixture stays self-consistent.

A Unigram model is rebuilt from the file's scores as Python reads them (see
`load_tokenizer`). `Tokenizer.from_file` reads them through a decimal
conversion that is off by one unit in the last place for about a quarter of
them (serde_json without its `float_roundtrip` feature: 8334 of T5's 32100
scores). A segmentation is a sum of scores compared for the maximum, and a run
of one repeated character can tie exactly between two orders of the same pieces
— `▁ - --- ---` against `▁ --- --- -` — where that last place decides. brot
reads the numbers as written, so the reference is made to score the same
vocabulary; the ids `from_file` gives differ from these only on such ties.

Corpora
-------

`fixtures/parity/sample.txt` is ordinary prose: the unique block of
`bench/data/wiki_64k.txt` (that file is a 2533-byte block repeated to fill
64 KB), `bench/data/news_1k.txt`, and a few of the project's own English
documentation pages, which add code identifiers, URLs and punctuation.
`fixtures/parity/edge_cases.txt` is hand-written and covers whitespace runs,
contractions, digits, scripts, combining marks, emoji, over-long words, special
tokens, and spans whose byte and character extents differ.

Fixture format
--------------

A corpus file holds several documents separated by a line containing exactly
`====`; documents may therefore contain newlines, and a document may be empty.
The last byte of the file is a newline that belongs to the format, not to the
final document. Splitting is line-based, so a separator at the start or end of
the file, and two adjacent separators, all denote empty documents; a line that
merely starts with `====` is ordinary text. See `documents` below.

Expected-encoding format
------------------------

For corpus `<corpus>.txt` and tokenizer `<name>` this writes
`fixtures/parity/<corpus>.<name>.ids`:

    # a header comment, recording the reference version
    IDS <ids of document 0, add_special_tokens=False>
    OFFSETS <byte spans of document 0, one `start,end` per IDS token>
    DECODE <the IDS row decoded back to text, as a JSON string>
    SPECIAL <ids of document 0, add_special_tokens=True>
    MASK <special-token mask of the SPECIAL row>
    IDS <ids of document 1>
    ...
    ALL <ids of the whole corpus file as a single document>

Lines starting with `#` are comments. Every other line is a row `<KIND>` or
`<KIND> <value> <value> ...`; a row with no values is a document that encodes
to no tokens at all. Rows of the five per-document kinds appear once per
document, in document order, so the n-th `IDS` row and the n-th `OFFSETS` row
describe the same document. The trailing `ALL` row holds the whole corpus file
(again minus its final newline) encoded as one document, which exercises the
long-input paths; only its ids are recorded, because its offsets would more
than double the size of these fixtures for no coverage the per-document rows
do not already give.

`OFFSETS` values are byte spans into the document. The reference reports
character spans, which `byte_spans` converts through a UTF-8 prefix sum. The
conversion is per span, so a character whose bytes are split across several
tokens keeps one span repeated once per token, as the reference reports it.

`DECODE` is the one row whose value is text rather than numbers, so it is
written as a JSON string: the quotes keep leading and trailing spaces visible,
and the escapes keep a newline or a tab in decoded text from breaking the
line-based format. Non-ASCII characters stay literal, which keeps the row
readable and the file small. It records `decode` of the `IDS` row with
`skip_special_tokens=False`, which is a round trip only up to what the
tokenizer's normalizer and decoder discard.

There is no row for `type_ids` or `attention_mask`: for a single sequence they
are all `0` and all `1`, so `test_parity.ml` asserts that directly rather than
carrying tens of thousands of constants here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import tokenizers
from tokenizers import Tokenizer, models

CORPORA = ("sample", "edge_cases")
TOKENIZERS = ("gpt2", "llama", "bert_base", "roberta_base", "t5_base_nonorm")


def documents(text: str) -> list[str]:
    docs: list[str] = []
    current: list[str] = []
    for line in text.removesuffix("\n").split("\n"):
        if line == "====":
            docs.append("\n".join(current))
            current = []
        else:
            current.append(line)
    docs.append("\n".join(current))
    return docs


def byte_spans(
    text: str, spans: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Map character spans into `text` to byte spans."""
    starts = [0]
    total = 0
    for char in text:
        total += len(char.encode("utf-8"))
        starts.append(total)
    return [(starts[start], starts[stop]) for start, stop in spans]


def row(kind: str, values: list[str]) -> str:
    return " ".join([kind, *values])


def ints(values: list[int]) -> list[str]:
    return [str(value) for value in values]


def load_tokenizer(path: Path) -> Tokenizer:
    """The tokenizer at `path`, a Unigram model rebuilt from the scores as
    Python reads them rather than as `from_file` reads them (see above)."""
    tokenizer = Tokenizer.from_file(str(path))
    with path.open(encoding="utf-8") as file:
        model = json.load(file)["model"]
    if model.get("type") == "Unigram" or "unk_id" in model:
        tokenizer.model = models.Unigram(
            [(token, score) for token, score in model["vocab"]],
            unk_id=model["unk_id"],
            byte_fallback=model.get("byte_fallback", False),
        )
    return tokenizer


def main() -> int:
    test_root = Path(__file__).resolve().parents[1]
    parity_dir = test_root / "fixtures" / "parity"
    models_dir = test_root.parent / "bench" / "data"

    for name in TOKENIZERS:
        model = models_dir / f"{name}.json"
        if not model.exists():
            print(
                f"missing {model}; run packages/brot/bench/download_data.sh",
                file=sys.stderr,
            )
            return 1
        tokenizer = load_tokenizer(model)

        for corpus in CORPORA:
            text = (parity_dir / f"{corpus}.txt").read_bytes().decode("utf-8")
            lines = [
                (
                    "# generated by gen_parity_expected.py with tokenizers "
                    f"{tokenizers.__version__}"
                ),
                f"# corpus {corpus}.txt, tokenizer {name}",
            ]
            for document in documents(text):
                plain = tokenizer.encode(document, add_special_tokens=False)
                special = tokenizer.encode(document, add_special_tokens=True)
                offsets = [
                    f"{start},{stop}"
                    for start, stop in byte_spans(document, plain.offsets)
                ]
                decoded = tokenizer.decode(
                    plain.ids, skip_special_tokens=False
                )
                lines.append(row("IDS", ints(plain.ids)))
                lines.append(row("OFFSETS", offsets))
                lines.append(
                    row("DECODE", [json.dumps(decoded, ensure_ascii=False)])
                )
                lines.append(row("SPECIAL", ints(special.ids)))
                lines.append(row("MASK", ints(special.special_tokens_mask)))
            whole = tokenizer.encode(
                text.removesuffix("\n"), add_special_tokens=False
            )
            lines.append(row("ALL", ints(whole.ids)))

            out = parity_dir / f"{corpus}.{name}.ids"
            out.write_text(
                "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
            )
            print(f"wrote {out} ({out.stat().st_size} bytes)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
