"""Workloads, and the two engines that run them.

A workload is a set of tables and the questions asked of them. Each question
is written twice, once as DuckDB SQL and once as a Polars lazy query, and both
spellings produce the same columns under the same names.

A table either loads before timing (H2O's CSV files, read into memory with a
declared schema) or is scanned inside the timed region (TPC-H's Parquet files).
"""

import gc
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import duckdb
import polars as pl


@dataclass(frozen=True)
class Table:
    path: Path
    schema: dict[str, pl.DataType] | None
    """The declared schema of a CSV table loaded before timing, or [None] for
    a Parquet table scanned inside the timed region."""


@dataclass(frozen=True)
class Question:
    name: str
    sql: str
    polars: Callable[[SimpleNamespace], pl.LazyFrame]


@dataclass(frozen=True)
class Workload:
    id: str
    """The case id prefix, e.g. [groupby/1e7] or [tpch/sf1]."""
    tables: dict[str, Table]
    questions: list[Question]
    ordered: bool
    """Whether answers come in an order the question fixes. Unordered answers
    are compared after sorting their rows."""
    spill: Path
    """The directory where DuckDB writes what does not fit in memory, beside
    the data."""


def check_tables(workload: Workload) -> None:
    missing = [str(t.path) for t in workload.tables.values() if not t.path.exists()]
    if missing:
        raise SystemExit(
            f"{workload.id}: missing data files (generate them first):\n  "
            + "\n  ".join(missing)
        )


# Engines

_SQL_TYPES = {pl.String: "VARCHAR", pl.Int32: "INTEGER", pl.Float64: "DOUBLE"}


class Duckdb:
    name = "duckdb"

    def __init__(self, workload: Workload):
        self.con = duckdb.connect(":memory:")
        self.con.execute(f"SET threads = {os.cpu_count()}")
        self.con.execute(f"SET temp_directory = '{workload.spill}'")
        # Every Parquet read pays for its bytes, as Polars' does.
        self.con.execute("SET enable_external_file_cache = false")
        for name, table in workload.tables.items():
            if table.schema is None:
                self.con.execute(
                    f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{table.path}')"
                )
            else:
                columns = ", ".join(
                    f"'{c}': '{_SQL_TYPES[t]}'" for c, t in table.schema.items()
                )
                self.con.execute(
                    f"CREATE TABLE {name} AS SELECT * FROM read_csv('{table.path}',"
                    f" header = true, columns = {{{columns}}})"
                )
        self.result = None

    def clear(self) -> None:
        self.result = None

    def run(self, question: Question) -> None:
        self.result = self.con.execute(question.sql).to_arrow_table()

    def answer(self) -> pl.DataFrame:
        return pl.from_arrow(self.result)


class Polars:
    name = "polars"

    def __init__(self, workload: Workload):
        frames = {}
        for name, table in workload.tables.items():
            if table.schema is None:
                frames[name] = pl.scan_parquet(table.path)
            else:
                frames[name] = pl.read_csv(table.path, schema=table.schema).lazy()
        self.tables = SimpleNamespace(**frames)
        self.result = None

    def clear(self) -> None:
        self.result = None

    def run(self, question: Question) -> None:
        self.result = question.polars(self.tables).collect()

    def answer(self) -> pl.DataFrame:
        return self.result


ENGINES = [Duckdb, Polars]


def check_engines() -> None:
    """[check_engines ()] refuses an environment that changes Polars' engine or
    thread pool, before any engine measures."""
    if "POLARS_ENGINE_AFFINITY" in os.environ:
        raise SystemExit(
            "POLARS_ENGINE_AFFINITY selects another Polars engine; unset it"
        )
    if pl.thread_pool_size() != os.cpu_count():
        raise SystemExit(
            f"Polars runs {pl.thread_pool_size()} threads on {os.cpu_count()} cores;"
            " unset POLARS_MAX_THREADS"
        )


def versions() -> str:
    return f"duckdb {duckdb.__version__} · polars {pl.__version__}"


def timed(engine, question: Question) -> float:
    """[timed engine question] runs [question] once and is its wall time in
    seconds. The previous answer is freed and the collector runs before the
    clock starts."""
    engine.clear()
    gc.collect()
    start = time.perf_counter()
    engine.run(question)
    return time.perf_counter() - start
