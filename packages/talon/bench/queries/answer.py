"""Answers: their canonical form, their committed files, and how two compare.

An answer is committed thinned: the rows at positions 0, s, 2s, ... of its
canonical form, where the stride s keeps at most [LIMIT] rows, so an answer of
at most [LIMIT] rows is committed whole. [answers/index.csv] records each
answer's full row count and stride. Because the rows are taken by position,
a row missing or added anywhere shifts every later sample.
"""

from pathlib import Path

import polars as pl
import polars.selectors as cs

LIMIT = 200
REL = 1e-9
ABS = 1e-12


def canonical(frame: pl.DataFrame, ordered: bool) -> pl.DataFrame:
    """[canonical frame ordered] is [frame] with integers (and integer-valued
    decimals, which DuckDB's sums of integers return) as int64 and floats as
    float64. Unless the question orders the rows, they are sorted by every
    column in turn, ascending, nulls last."""
    integral = [n for n, t in frame.schema.items() if t == pl.Decimal(38, 0)]
    frame = frame.with_columns(
        (cs.integer() | cs.by_name(integral)).cast(pl.Int64),
        cs.float().cast(pl.Float64),
    )
    return frame if ordered else frame.sort(frame.columns, nulls_last=True)


def stride(rows: int) -> int:
    return max(1, -(-rows // LIMIT))


def thin(frame: pl.DataFrame, stride: int) -> pl.DataFrame:
    return frame.gather_every(stride)


def compare(expected: pl.DataFrame, got: pl.DataFrame) -> list[str]:
    """[compare expected got] is the differences between two canonical
    answers. Floats agree within a relative [REL] or an absolute [ABS], nulls
    agree with nulls and NaNs with NaNs; everything else agrees exactly."""
    if expected.schema != got.schema:
        return [f"schema {dict(got.schema)}, expected {dict(expected.schema)}"]
    if expected.height != got.height:
        return [f"{got.height} rows, expected {expected.height}"]
    problems = []
    for name, dtype in expected.schema.items():
        a, b = expected[name], got[name]
        differ = ~close(a, b) if dtype == pl.Float64 else ~a.eq_missing(b)
        if n := int(differ.sum()):
            row = int(differ.arg_true()[0])
            problems.append(
                f"column {name}: {n} rows differ, first at row {row}:"
                f" {b[row]!r}, expected {a[row]!r}"
            )
    return problems


def close(a: pl.Series, b: pl.Series) -> pl.Series:
    x, y = pl.col("a"), pl.col("b")
    tolerance = pl.max_horizontal(
        pl.lit(ABS), REL * pl.max_horizontal(x.abs(), y.abs())
    )
    return (
        pl.DataFrame({"a": a, "b": b})
        .select(
            (x.is_null() & y.is_null())
            | (x.is_nan() & y.is_nan()).fill_null(False)
            | ((x - y).abs() <= tolerance).fill_null(False)
        )
        .to_series()
    )


# Files


def path(root: Path, id: str) -> Path:
    """[path root id] is the file of the answer to [id], e.g.
    [root/groupby-1e6/q01.csv] for [groupby/1e6/q01]."""
    workload, question = id.rsplit("/", 1)
    return root / workload.replace("/", "-") / f"{question}.csv"


INDEX_SCHEMA = {"question": pl.String, "rows": pl.Int64, "stride": pl.Int64}


def read_index(root: Path) -> dict[str, tuple[int, int]]:
    """[read_index root] maps each committed answer's id to its full row count
    and stride, and is empty when nothing is committed."""
    index = root / "index.csv"
    if not index.exists():
        return {}
    frame = pl.read_csv(index, schema=INDEX_SCHEMA)
    return {q: (rows, s) for q, rows, s in frame.iter_rows()}


def write_index(root: Path, index: dict[str, tuple[int, int]]) -> None:
    rows = [(q, rows, s) for q, (rows, s) in sorted(index.items())]
    pl.DataFrame(rows, schema=INDEX_SCHEMA, orient="row").write_csv(root / "index.csv")


def write(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(path)


def read(path: Path, schema: pl.Schema) -> pl.DataFrame:
    return pl.read_csv(path, schema=schema)
