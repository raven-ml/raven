"""The H2O db-benchmark questions: ten group-by questions and five joins.

The SQL is the db-benchmark DuckDB solution's, and the Polars queries name
their columns as the SQL does. Every engine loads the same CSV files with the
same declared schema: identifiers as strings, other integers as int32, and
measures as float64.
"""

from pathlib import Path

import polars as pl

from workload import Question, Table, Workload

SIZES = ["1e6", "1e7", "1e8"]


def pretty(n: int) -> str:
    """[pretty n] is db-benchmark's name for the power of ten [n]: 1e7."""
    return f"1e{len(str(n)) - 1}"


def rows(size: str) -> int:
    if size not in SIZES:
        raise SystemExit(f"unknown H2O size {size!r}; expected one of {SIZES}")
    return int(float(size))


def groupby_file(size: str) -> str:
    return f"G1_{size}_1e2_0_0.csv"


def join_files(size: str) -> dict[str, str]:
    n = rows(size)
    return {
        "x": f"J1_{size}_NA_0_0.csv",
        "small": f"J1_{size}_{pretty(n // 10**6)}_0_0.csv",
        "medium": f"J1_{size}_{pretty(n // 10**3)}_0_0.csv",
        "big": f"J1_{size}_{size}_0_0.csv",
    }


def directory(data: Path, size: str) -> Path:
    return data / f"h2o-{size}"


# Group-by

GROUPBY_SCHEMA = {
    "id1": pl.String,
    "id2": pl.String,
    "id3": pl.String,
    "id4": pl.Int32,
    "id5": pl.Int32,
    "id6": pl.Int32,
    "v1": pl.Int32,
    "v2": pl.Int32,
    "v3": pl.Float64,
}

GROUPBY = [
    Question(
        "q01",
        "SELECT id1, sum(v1) AS v1 FROM x GROUP BY id1",
        lambda t: t.x.group_by("id1").agg(v1=pl.sum("v1")),
    ),
    Question(
        "q02",
        "SELECT id1, id2, sum(v1) AS v1 FROM x GROUP BY id1, id2",
        lambda t: t.x.group_by("id1", "id2").agg(v1=pl.sum("v1")),
    ),
    Question(
        "q03",
        "SELECT id3, sum(v1) AS v1, avg(v3) AS v3 FROM x GROUP BY id3",
        lambda t: t.x.group_by("id3").agg(v1=pl.sum("v1"), v3=pl.mean("v3")),
    ),
    Question(
        "q04",
        "SELECT id4, avg(v1) AS v1, avg(v2) AS v2, avg(v3) AS v3 FROM x GROUP BY id4",
        lambda t: t.x.group_by("id4").agg(
            v1=pl.mean("v1"), v2=pl.mean("v2"), v3=pl.mean("v3")
        ),
    ),
    Question(
        "q05",
        "SELECT id6, sum(v1) AS v1, sum(v2) AS v2, sum(v3) AS v3 FROM x GROUP BY id6",
        lambda t: t.x.group_by("id6").agg(
            v1=pl.sum("v1"), v2=pl.sum("v2"), v3=pl.sum("v3")
        ),
    ),
    Question(
        "q06",
        "SELECT id4, id5, quantile_cont(v3, 0.5) AS median_v3, stddev(v3) AS sd_v3"
        " FROM x GROUP BY id4, id5",
        lambda t: t.x.group_by("id4", "id5").agg(
            median_v3=pl.median("v3"), sd_v3=pl.std("v3")
        ),
    ),
    Question(
        "q07",
        "SELECT id3, max(v1) - min(v2) AS range_v1_v2 FROM x GROUP BY id3",
        lambda t: t.x.group_by("id3").agg(range_v1_v2=pl.max("v1") - pl.min("v2")),
    ),
    Question(
        "q08",
        "SELECT id6, unnest(max(v3, 2)) AS largest2_v3"
        " FROM x WHERE v3 IS NOT NULL GROUP BY id6",
        lambda t: (
            t.x.drop_nulls("v3")
            .group_by("id6")
            .agg(largest2_v3=pl.col("v3").top_k(2))
            .explode("largest2_v3")
        ),
    ),
    Question(
        "q09",
        "SELECT id2, id4, pow(corr(v1, v2), 2) AS r2 FROM x GROUP BY id2, id4",
        lambda t: t.x.group_by("id2", "id4").agg(r2=pl.corr("v1", "v2") ** 2),
    ),
    Question(
        "q10",
        "SELECT id1, id2, id3, id4, id5, id6, sum(v3) AS v3, count(*) AS count"
        " FROM x GROUP BY id1, id2, id3, id4, id5, id6",
        lambda t: t.x.group_by("id1", "id2", "id3", "id4", "id5", "id6").agg(
            v3=pl.sum("v3"), count=pl.len()
        ),
    ),
]


def groupby(data: Path, size: str) -> Workload:
    path = directory(data, size) / groupby_file(size)
    return Workload(
        id=f"groupby/{size}",
        tables={"x": Table(path, GROUPBY_SCHEMA)},
        questions=GROUPBY,
        ordered=False,
        spill=data / "duckdb-tmp",
    )


# Join

JOIN_SCHEMAS = {
    "x": {
        "id1": pl.Int32,
        "id2": pl.Int32,
        "id3": pl.Int32,
        "id4": pl.String,
        "id5": pl.String,
        "id6": pl.String,
        "v1": pl.Float64,
    },
    "small": {"id1": pl.Int32, "id4": pl.String, "v2": pl.Float64},
    "medium": {
        "id1": pl.Int32,
        "id2": pl.Int32,
        "id4": pl.String,
        "id5": pl.String,
        "v2": pl.Float64,
    },
    "big": {
        "id1": pl.Int32,
        "id2": pl.Int32,
        "id3": pl.Int32,
        "id4": pl.String,
        "id5": pl.String,
        "id6": pl.String,
        "v2": pl.Float64,
    },
}


def prefixed(frame: pl.LazyFrame, table: str, keep: list[str]) -> pl.LazyFrame:
    """[prefixed frame table keep] renames every column of [frame] outside
    [keep] to [table_column], as the SQL's [AS] clauses do."""
    return frame.rename(
        {c: f"{table}_{c}" for c in frame.collect_schema() if c not in keep}
    )


JOIN = [
    Question(
        "q01",
        "SELECT x.*, small.id4 AS small_id4, v2 FROM x JOIN small USING (id1)",
        lambda t: t.x.join(prefixed(t.small, "small", ["id1", "v2"]), on="id1"),
    ),
    Question(
        "q02",
        "SELECT x.*, medium.id1 AS medium_id1, medium.id4 AS medium_id4,"
        " medium.id5 AS medium_id5, v2 FROM x JOIN medium USING (id2)",
        lambda t: t.x.join(prefixed(t.medium, "medium", ["id2", "v2"]), on="id2"),
    ),
    Question(
        "q03",
        "SELECT x.*, medium.id1 AS medium_id1, medium.id4 AS medium_id4,"
        " medium.id5 AS medium_id5, v2 FROM x LEFT JOIN medium USING (id2)",
        lambda t: t.x.join(
            prefixed(t.medium, "medium", ["id2", "v2"]), on="id2", how="left"
        ),
    ),
    Question(
        "q04",
        "SELECT x.*, medium.id1 AS medium_id1, medium.id2 AS medium_id2,"
        " medium.id4 AS medium_id4, v2 FROM x JOIN medium USING (id5)",
        lambda t: t.x.join(prefixed(t.medium, "medium", ["id5", "v2"]), on="id5"),
    ),
    Question(
        "q05",
        "SELECT x.*, big.id1 AS big_id1, big.id2 AS big_id2, big.id4 AS big_id4,"
        " big.id5 AS big_id5, big.id6 AS big_id6, v2 FROM x JOIN big USING (id3)",
        lambda t: t.x.join(prefixed(t.big, "big", ["id3", "v2"]), on="id3"),
    ),
]


def join(data: Path, size: str) -> Workload:
    root = directory(data, size)
    return Workload(
        id=f"join/{size}",
        tables={
            name: Table(root / file, JOIN_SCHEMAS[name])
            for name, file in join_files(size).items()
        },
        questions=JOIN,
        ordered=False,
        spill=data / "duckdb-tmp",
    )


def workloads(data: Path, size: str) -> list[Workload]:
    rows(size)
    return [groupby(data, size), join(data, size)]
