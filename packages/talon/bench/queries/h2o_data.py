# /// script
# requires-python = "==3.13.*"
# dependencies = ["duckdb==1.5.6", "numpy==2.5.3", "polars==1.44.2", "pyarrow==25.0.1"]
# [tool.uv]
# exclude-newer = "2026-10-01T00:00:00Z"
# ///
"""Generate the H2O db-benchmark group-by and join data as CSV.

usage: uv run h2o_data.py SIZE DATA

SIZE is 1e6, 1e7 or 1e8 rows. The files land in DATA/h2o-SIZE/ under
db-benchmark's names: G1_SIZE_1e2_0_0.csv for group-by, and the four
J1_SIZE_*_0_0.csv tables for join.

This is a port of db-benchmark's groupby-datagen.R and join-datagen.R, with
no missing values and unsorted rows. It draws from numpy's PCG64 seeded with
108, so its values differ from R's, but they follow the same distributions and
are the same on every machine for the pinned numpy. Join's key split is
written with integer counts, which also covers 1e6 rows: the R script refuses
fewer than 1e7, because its small table would hold a single row.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

import h2o

SEED = 108
K = 100


def write(frame: pl.DataFrame, path: Path) -> None:
    frame.write_csv(path, float_precision=6)
    print(f"wrote {path} ({frame.height} rows)")


def label(name: str, width: int) -> pl.Expr:
    """[label name width] prints integer column [name] as db-benchmark's
    string identifiers: [id], then the integer zero-padded to [width]."""
    return (pl.lit("id") + pl.col(name).cast(pl.String).str.zfill(width)).alias(name)


def groupby(n: int, path: Path) -> None:
    rng = np.random.default_rng(SEED)
    highs = {"id1": K, "id2": K, "id3": n // K, "id4": K, "id5": K, "id6": n // K}
    highs |= {"v1": 5, "v2": 15}
    columns = {c: rng.integers(1, h + 1, n, dtype=np.int32) for c, h in highs.items()}
    columns["v3"] = np.round(rng.uniform(0, 100, n), 6)
    frame = pl.DataFrame(columns).with_columns(
        label("id1", 3), label("id2", 3), label("id3", 10)
    )
    write(frame, path)


def join(n: int, files: dict[str, str], root: Path) -> None:
    rng = np.random.default_rng(SEED)

    def split(m: int) -> dict[str, np.ndarray]:
        """[split m] permutes [m + m/10] keys into [x], shared by both sides,
        [l], only on the left, and [r], only on the right."""
        keys = rng.permutation(np.arange(1, m + m // 10 + 1, dtype=np.int64))
        common = m - m // 10
        return {"x": keys[:common], "l": keys[common:m], "r": keys[m:]}

    def sample_all(keys: np.ndarray, size: int) -> np.ndarray:
        """[sample_all keys size] is [size] keys in random order, holding
        every one of [keys] at least once."""
        extra = rng.choice(keys, size - len(keys), replace=True)
        return rng.permutation(np.concatenate([keys, extra])).astype(np.int32)

    key1, key2, key3 = split(n // 10**6), split(n // 10**3), split(n)
    left = [np.concatenate([k["x"], k["l"]]) for k in (key1, key2, key3)]
    right = [np.concatenate([k["x"], k["r"]]) for k in (key1, key2, key3)]

    def table(ids: list[np.ndarray], measure: str) -> pl.DataFrame:
        """[table ids measure] has the integer keys [ids] as [id1], [id2], …,
        their string forms as [id4], [id5], …, and the column [measure],
        drawn after the keys."""
        keys = range(1, len(ids) + 1)
        frame = pl.DataFrame({f"id{i}": column for i, column in zip(keys, ids)})
        frame = frame.with_columns(
            pl.format("id{}", f"id{i}").alias(f"id{i + 3}") for i in keys
        )
        values = np.round(rng.uniform(0, 100, frame.height), 6)
        return frame.with_columns(pl.Series(measure, values))

    small, medium = n // 10**6, n // 10**3
    write(table([sample_all(k, n) for k in left], "v1"), root / files["x"])
    write(table([sample_all(right[0], small)], "v2"), root / files["small"])
    write(
        table([sample_all(right[0], medium), sample_all(right[1], medium)], "v2"),
        root / files["medium"],
    )
    write(table([sample_all(k, n) for k in right], "v2"), root / files["big"])


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    size, data = sys.argv[1], Path(sys.argv[2])
    n = h2o.rows(size)
    root = h2o.directory(data, size)
    root.mkdir(parents=True, exist_ok=True)
    groupby(n, root / h2o.groupby_file(size))
    join(n, h2o.join_files(size), root)


if __name__ == "__main__":
    main()
