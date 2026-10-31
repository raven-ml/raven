#!/usr/bin/env python3
"""Build the small SentencePiece charsmap that the Precompiled tests read.

Run from anywhere:

    uv run --with sentencepiece==0.2.2 python3 packages/brot/test/gen/gen_precompiled.py

A `Precompiled` normalizer carries a SentencePiece `precompiled_charsmap`: a
darts-clone double-array trie over the UTF-8 bytes of the rules' sources,
followed by their replacements, each ended by a NUL. Real ones (T5, ALBERT,
XLNet) hold some 220 000 rules in 237 KB. This writes one compiled by
SentencePiece's own builder from the few rules below, all but the last copied
from T5's charsmap (Apache-2.0, google-t5/t5-base; SentencePiece's
`nmt_nfkc`), so the trie has the shape the real ones have.

The rules cover what the normalizer has to get right: removal (control
characters), one character to one, one to several (ligatures, `™`, `½`), and
several to one (combining sequences), including a source that is a prefix of
another (`e\u0302` and `e\u0302\u0300`, `\uff45` and `\uff45\u0301`, `\uff76` and `\uff76\uff9e`).

It writes `support/precompiled/charsmap.json`: the rules, which the tests'
reference implementation reads, and the normalizer object as a
`tokenizer.json` holds it. `hf_alignments.py` reads the same file for the
Precompiled cases of `test_unicode.ml`.

It also writes `support/precompiled/graphemes.json`, a charsmap with a rule
for each code point of Unicode's GraphemeBreakTest-17.0.0 (as gen_grapheme.py
commits it), NUL aside, which no source may hold: U+0308 becomes `<0308>`. A
cluster under six bytes then comes out as its first character's name and a
longer one as the names of all its characters, so the normalized text shows
where the clusters of a test line end.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import sentencepiece as spm

RULES = [
    # Control characters are removed.
    ("\x01", ""),
    ("\x1f", ""),
    ("\x7f", ""),
    # White space becomes a space, U+FFFD included.
    ("\t", " "),
    ("\n", " "),
    ("\r", " "),
    ("\u00a0", " "),
    ("\u200b", " "),
    ("\u3000", " "),
    ("\ufffd", " "),
    # Full-width forms.
    ("\uff21", "A"),
    ("\uff45", "e"),
    ("\uff11", "1"),
    ("\uff01", "!"),
    # Ligatures and other compatibility characters.
    ("\ufb01", "fi"),
    ("\ufb00", "ff"),
    ("\ufb03", "ffi"),
    ("\u01c4", "D\u017d"),
    ("\u2122", "TM"),
    ("\u2460", "1"),
    ("\u00bd", "1\u20442"),
    ("\u2026", "..."),
    ("\u00b2", "2"),
    ("\u210c", "H"),
    # Half-width katakana: a letter, the voicing mark, and the two composed.
    ("\uff76", "\u30ab"),
    ("\uff9e", "\u3099"),
    ("\uff76\uff9e", "\u30ac"),
    # Combining sequences composed.
    ("e\u0301", "\u00e9"),
    ("e\u0302", "\u00ea"),
    ("e\u0302\u0300", "\u1ec1"),
    ("\uff45\u0301", "\u00e9"),
    ("\uff45\u0300", "\u00e8"),
    ("C\u0327", "\u00c7"),
    ("\u1100\u1161", "\uac00"),
    # Not T5's: a source that is a Prepend character, so that a cluster it
    # opens (GB9b) is replaced whole.
    ("\u0600", "#"),
]


def charsmap(rules: list[tuple[str, str]]) -> bytes:
    """The `precompiled_charsmap` SentencePiece compiles from `rules`."""
    normalizer = spm.SentencePieceNormalizer(norm_map=rules)
    spec = normalizer.serialized_normalizer_spec()
    # NormalizerSpec is a protocol buffer; field 2 is precompiled_charsmap.
    pos = 0
    while pos < len(spec):
        key, pos = varint(spec, pos)
        field, wire = key >> 3, key & 7
        if wire == 0:
            _, pos = varint(spec, pos)
        elif wire == 2:
            length, pos = varint(spec, pos)
            if field == 2:
                return spec[pos : pos + length]
            pos += length
        else:
            raise ValueError(f"unexpected wire type {wire}")
    raise ValueError("no precompiled_charsmap in the normalizer spec")


def varint(data: bytes, pos: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        shift += 7
        if byte < 0x80:
            return value, pos


def normalizer(blob: bytes) -> dict:
    return {
        "type": "Precompiled",
        "precompiled_charsmap": base64.b64encode(blob).decode("ascii"),
    }


def grapheme_rules(test: str) -> list[tuple[str, str]]:
    """A rule from each non-NUL code point of `test` to its name."""
    points = set()
    for line in test.splitlines():
        data = line.split("#", 1)[0]
        for field in data.split():
            if field not in ("\u00f7", "\u00d7"):
                points.add(int(field, 16))
    points.discard(0)
    return [(chr(cp), f"<{cp:04X}>") for cp in sorted(points)]


def main() -> None:
    support = Path(__file__).resolve().parents[1] / "support"
    blob = charsmap(RULES)
    # One rule per line, so that a change to the rules reads as one in a diff.
    rules = ",\n  ".join(json.dumps(rule) for rule in RULES)
    document = (
        f'{{\n "rules": [\n  {rules}\n ],\n'
        f' "normalizer": {json.dumps(normalizer(blob))}\n}}\n'
    )
    out = support / "precompiled"
    out.mkdir(exist_ok=True)
    path = out / "charsmap.json"
    path.write_text(document, encoding="utf-8", newline="\n")
    print(f"wrote {path} ({len(blob)} bytes of charsmap)")

    test = (support / "unicode" / "GraphemeBreakTest-17.0.0.txt").read_text(
        encoding="utf-8"
    )
    rules = grapheme_rules(test)
    blob = charsmap(rules)
    path = out / "graphemes.json"
    path.write_text(
        json.dumps(normalizer(blob)) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {path} ({len(rules)} rules, {len(blob)} bytes of charsmap)")


if __name__ == "__main__":
    main()
