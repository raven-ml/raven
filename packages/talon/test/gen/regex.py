# /// script
# requires-python = ">=3.11"
# dependencies = ["polars==1.35.2"]
# ///
"""Writes golden/regex.txt, what Str.regex must agree with, from Rust's regex
crate through Polars.

    uv run packages/talon/test/gen/regex.py [--check]

The crate's semantics are what Str.regex follows: leftmost-first, Unicode's \\w,
\\d and \\s, simple case folding, no empty match abutting the previous match.
Patterns are drawn from a grammar over the language both accept, against texts
of a fixed alphabet of scalars present in every Unicode version the crate and
this repository read, so that the data does not depend on which one a version
assigned. Each line holds, for one pattern and one text, what the crate reports
with Polars' str.contains, str.extract, str.extract_groups, str.extract_all and
str.replace_all(`<$0>`). Every column is one hex string of UTF-8 bytes, `-` for
a null; a list joins its hex strings with `,`. test_regex.ml reads each line
through the public Str operations and compares every column.

Polars cannot show a split's boundaries, so split is checked through find_all
and the round-trip law in the suite.
"""

from __future__ import annotations

import filecmp
import random
import sys
import tempfile
from pathlib import Path

import polars as pl

# A scalar for each code-length bucket and case-fold orbit worth stressing.
ALPHABET = (
    list("abcxyzABCXYZ0129 _-")
    + ["é", "É", "ß", "ς", "σ", "Σ", "ſ", "K", "K", "k"]
    + ["日", "本", "😀", "𝟙", "ǅ", "ǆ", "Ǆ"]
    + ["́", "‍", " ", " ", "\n", "\t", "५"]
)
TEXT_ALPHABET = ALPHABET + ["ab", "xyz", "éa", "á", "😀😀", "日本語"]

ATOMS = (
    list("abxz")
    + ["é", "日", "😀", ".", r"\d", r"\w", r"\s", r"\D", r"\W", r"\S"]
    + [r"\b", r"\B", "^", "$", r"\A", r"\z"]
    + ["[ab]", "[a-c]", "[^a]", "[^\\s]", "[é-ü]", "[\\d_]", "[a-zé]", "[日本]"]
    + [r"\p{L}", r"\p{Lu}", r"\p{Script=Greek}", r"\P{L}", r"\p{Nd}"]
    + ["ς", "σ", "k", "K"]
)
QUANTIFIERS = ["*", "+", "?", "*?", "+?", "??", "{2}", "{1,2}", "{0,1}", "{1,}"]
WRAPPERS = ["(%s)", "(?:%s)", "(?i:%s)", "(?s:%s)", "(?m:%s)", "(%s)?"]


def pattern(rng: random.Random, depth: int) -> str:
    if depth == 0 or rng.random() < 0.4:
        return rng.choice(ATOMS)
    roll = rng.random()
    if roll < 0.35:
        return pattern(rng, depth - 1) + pattern(rng, depth - 1)
    if roll < 0.55:
        a = pattern(rng, depth - 1)
        b = pattern(rng, depth - 1) if rng.random() < 0.8 else ""
        return f"{a}|{b}"
    if roll < 0.8:
        inner = pattern(rng, depth - 1)
        # An atom needing a group first: repeat that group.
        return "(?:%s)%s" % (inner, rng.choice(QUANTIFIERS))
    return rng.choice(WRAPPERS) % pattern(rng, depth - 1)


def texts(rng: random.Random, n: int) -> list[str]:
    fixed = [
        "",
        "a",
        "aaa",
        "abxd",
        "ab",
        "é.x",
        "ÉéŚ",
        "Σσς",
        "日本語テキスト",
        "😀a😀",
        "hello world_foo 123",
        "x\ny\n",
        "́a",
        "áb",
        "1२३",
    ]
    return fixed + [
        "".join(rng.choices(TEXT_ALPHABET, k=rng.randint(1, 12))) for _ in range(n)
    ]


def sampled(rng: random.Random, base: list[str]) -> list[str]:
    """A few of the fixed texts and a few random ones, per random pattern."""
    random_ones = [
        "".join(rng.choices(TEXT_ALPHABET, k=rng.randint(1, 12))) for _ in range(2)
    ]
    return rng.sample(base[:15], 2) + random_ones


def hexed(s) -> str:
    """A text as hex; `-` for null, `=` for the empty text."""
    if s is None:
        return "-"
    return s.encode("utf-8").hex() or "="


def row(pat: str, text: str) -> str | None:
    frame = pl.DataFrame({"s": [text]})
    try:
        out = frame.with_columns(
            pl.col("s").str.contains(pat).alias("contains"),
            pl.col("s").str.extract(pat, group_index=0).alias("whole"),
            pl.col("s").str.extract_all(pat).alias("all"),
            # A group, since Polars treats a bare literal's value as literal.
            pl.col("s").str.replace_all(f"(?:{pat})", "<$0>").alias("replaced"),
        ).row(0, named=True)
    except Exception:
        return None
    # A pattern with no group has no struct to read.
    try:
        unnested = frame.select(pl.col("s").str.extract_groups(pat).struct.unnest())
        captures = unnested.row(0) if unnested.width > 0 else []
    except pl.exceptions.ComputeError:
        captures = []
    fields = [
        hexed(pat),
        hexed(text),
        "1" if out["contains"] else "0",
        hexed(out["whole"]),
        ";".join(hexed(g) for g in captures),
        ",".join(hexed(m) for m in out["all"]),
        hexed(out["replaced"]),
    ]
    return " ".join(f or "." for f in fields)


def main() -> int:
    rng = random.Random(0x51c3)
    patterns = [pattern(rng, rng.randint(1, 4)) for _ in range(900)]
    # Patterns chosen for the rule they state, ahead of the random ones.
    rules = [
        "x*",
        "a||b",
        "||b",
        "a*?",
        "(?:|a)*",
        r"\b",
        r"\B",
        "(a)|(b)",
        "(a|ab)(c|bcd)?",
        "^",
        "$",
        "(?m)^",
        "(?m)$",
        ".*",
        ".+?",
        r"\w+",
        "(?i)k+",
        "(?i)σ",
        "(?i)ß",
        "é",
        ".",
        "😀",
        "[^a]",
        "(?i)[^a]",
    ]
    lines = []
    base = texts(rng, 5)
    seen = set()
    for pat in rules + patterns:
        if pat in seen:
            continue
        seen.add(pat)
        for t in (base if pat in rules else sampled(rng, base)):
            line = row(pat, t)
            if line is not None:
                lines.append(line)
    header = (
        "# pattern text contains whole group;... all,... replaced: each field hex of\n"
        "# UTF-8, = the empty text, - a null, a . an empty list of groups or matches.\n"
        "# From Rust's regex through polars 1.35.2.\n"
    )
    body = header + "\n".join(lines) + "\n"
    target = Path(__file__).resolve().parents[1] / "golden" / "regex.txt"
    if "--check" in sys.argv[1:]:
        with tempfile.TemporaryDirectory() as scratch:
            fresh = Path(scratch) / "regex.txt"
            fresh.write_text(body, encoding="utf-8", newline="\n")
            if not filecmp.cmp(fresh, target, shallow=False):
                print("golden/regex.txt differs from what the generator writes",
                      file=sys.stderr)
                return 1
        return 0
    target.write_text(body, encoding="utf-8", newline="\n")
    print(f"wrote golden/regex.txt: {len(lines)} lines, {len(seen)} patterns")
    return 0


if __name__ == "__main__":
    sys.exit(main())
