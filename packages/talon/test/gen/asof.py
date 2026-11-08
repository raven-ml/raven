# /// script
# requires-python = ">=3.11"
# dependencies = ["polars==1.35.2", "pandas==2.3.3"]
# ///
"""Writes golden/asof.txt, the oracle rows of the As-of suite.

    uv run packages/talon/test/gen/asof.py [--check]

With --check nothing is written, and the run fails if golden/asof.txt would
change.

Lines `left g t` and `right g rt` hold two tables, each sorted by its integer
key with ties, since both libraries require sorted keys, and grouped by `g`.
A line `pick direction exact j...` holds, per left row, the row of the right
table that pandas' merge_asof picks, or null, for a direction and whether
exact matches are allowed. Polars' join_asof must agree, and the run fails
otherwise, except where it departs from pandas: on a nearest distance tie,
which it breaks toward the later right row, and in a strict nearest, where it
still picks exact matches.
"""

import argparse
import random
import sys
from pathlib import Path

import pandas as pd
import polars as pl

GOLDEN = Path(__file__).resolve().parent.parent / "golden" / "asof.txt"


def table(rng, rows, key):
    pairs = sorted(
        ((rng.randrange(3), rng.randrange(30)) for _ in range(rows)),
        key=lambda r: r[1],
    )
    return {"g": [g for g, _ in pairs], key: [k for _, k in pairs]}


def picks(left, right, direction, exact):
    lp = pd.DataFrame(left)
    rp = pd.DataFrame({**right, "j": range(len(right["rt"]))})
    got = pd.merge_asof(
        lp, rp, left_on="t", right_on="rt", by="g", direction=direction,
        allow_exact_matches=exact,
    )["j"].tolist()
    got = [None if j != j else int(j) for j in got]
    if direction == "nearest" and not exact:
        return got
    polars = (
        pl.DataFrame(left)
        .join_asof(
            pl.DataFrame(rp), left_on="t", right_on="rt", by="g",
            strategy=direction, allow_exact_matches=exact,
        )["j"]
        .to_list()
    )
    for i, (p, q) in enumerate(zip(got, polars)):
        if p == q:
            continue
        t, rt = left["t"][i], right["rt"]
        tie = (
            direction == "nearest" and p is not None and q is not None
            and abs(t - rt[p]) == abs(rt[q] - t)
        )
        if not tie:
            sys.exit(f"pandas and Polars disagree at left row {i}: {p} {q}")
    return got


def generate():
    rng = random.Random(20261108)
    left = table(rng, 60, "t")
    right = table(rng, 40, "rt")
    for g, t in zip(left["g"], left["t"]):
        yield f"left {g} {t}\n"
    for g, rt in zip(right["g"], right["rt"]):
        yield f"right {g} {rt}\n"
    for direction in ["backward", "forward", "nearest"]:
        for exact in [True, False]:
            js = picks(left, right, direction, exact)
            js = " ".join("null" if j is None else str(j) for j in js)
            yield f"pick {direction} {int(exact)} {js}\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = "".join(generate())
    if not args.check:
        GOLDEN.write_text(text)
    elif not GOLDEN.exists() or GOLDEN.read_text() != text:
        sys.exit("golden/asof.txt is stale: run gen/asof.py")


if __name__ == "__main__":
    main()
