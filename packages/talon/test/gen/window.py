# /// script
# requires-python = ">=3.11"
# dependencies = ["polars==1.35.2", "pandas==2.3.3"]
# ///
"""Writes golden/window.txt, the oracle rows of the Window suite.

    uv run packages/talon/test/gen/window.py [--check]

With --check nothing is written, and the run fails if golden/window.txt would
change.

Two series of rows, in an order that is not sorted, with tied keys: datetimes
in microseconds with a window of seven days, and integers with a window of 3.
Each row is `key x right_sum right_max none_mean`:
- right_sum, right_max: Polars' rolling_sum_by and rolling_max_by with
  closed="right", the keys in (k - w, k], every row of the row's key included;
- none_mean: Polars' rolling_mean_by with closed="none", the keys in
  (k - w, k), every row of the row's key excluded, null over no row. pandas'
  rolling(closed="neither") over the rows sorted by key must agree, and the
  run fails otherwise.
"""

import argparse
import datetime as dt
import math
import random
import sys
from pathlib import Path

import pandas as pd
import polars as pl

GOLDEN = Path(__file__).resolve().parent.parent / "golden" / "window.txt"


def series(keys, x, window, by_time):
    df = pl.DataFrame({"k": keys, "x": x})
    col = pl.col("x")
    out = df.select(
        right_sum=col.rolling_sum_by("k", window, closed="right"),
        right_max=col.rolling_max_by("k", window, closed="right"),
        none_mean=col.rolling_mean_by("k", window, closed="none"),
    )
    order = sorted(range(len(x)), key=lambda i: keys[i])
    if by_time:
        index = pd.DatetimeIndex([keys[i] for i in order])
        s = pd.Series([x[i] for i in order], index=index)
        neither = s.rolling(window.upper(), closed="neither").mean().tolist()
        for j, i in enumerate(order):
            want, got = out["none_mean"][i], neither[j]
            if (want is None) != math.isnan(got) or (
                want is not None and abs(want - got) > 1e-9 * max(1, abs(got))
            ):
                sys.exit(f"pandas and Polars disagree at row {i}: {got} {want}")
    return out


def lines(name, keys, x, out, key_text):
    yield f"# {name}\n"
    for i in range(len(x)):
        row = [key_text(keys[i]), repr(x[i])]
        for c in ["right_sum", "right_max", "none_mean"]:
            v = out[c][i]
            row.append("null" if v is None else repr(float(v)))
        yield " ".join(row) + "\n"


def generate():
    rng = random.Random(20261108)
    epoch = dt.datetime(2024, 1, 1)
    days = [rng.randrange(0, 60 * 24) for _ in range(160)]
    times = [epoch + dt.timedelta(hours=h) for h in days]
    times += [times[i] for i in range(0, 160, 7)]
    rng.shuffle(times)
    tx = [float(rng.randrange(-100, 100)) / 4 for _ in times]
    us = lambda t: str((t - dt.datetime(1970, 1, 1)) // dt.timedelta(microseconds=1))
    ints = [rng.randrange(0, 40) for _ in range(80)]
    ix = [float(rng.randrange(-100, 100)) / 4 for _ in ints]
    yield from lines("datetime", times, tx, series(times, tx, "7d", True), us)
    yield from lines("int", ints, ix, series(ints, ix, "3i", False), str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = "".join(generate())
    if not args.check:
        GOLDEN.write_text(text)
    elif not GOLDEN.exists() or GOLDEN.read_text() != text:
        sys.exit("golden/window.txt is stale: run gen/window.py")


if __name__ == "__main__":
    main()
