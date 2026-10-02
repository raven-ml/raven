# /// script
# requires-python = "==3.13.*"
# dependencies = ["duckdb==1.5.6", "numpy==2.5.3", "polars==1.44.2", "pyarrow==25.0.1"]
# [tool.uv]
# exclude-newer = "2026-10-01T00:00:00Z"
# ///
"""Generate the TPC-H tables as snappy Parquet, with money as float64.

usage: uv run tpch_data.py SF DATA

SF is the scale factor, 0.1, 1 or 10. The eight tables land in
DATA/tpch-sfSF/TABLE.parquet. They come from DuckDB's tpch extension, which
runs the reference dbgen, so they are the same on every machine. Every
DECIMAL(15,2) column (prices, costs, balances, quantities, discounts, taxes)
is written as DOUBLE; the other columns keep dbgen's types.
"""

import sys
from pathlib import Path

import duckdb

import tpch


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    sf, data = tpch.scale(sys.argv[1]), Path(sys.argv[2])
    root = tpch.directory(data, sf)
    root.mkdir(parents=True, exist_ok=True)
    # On disk, so that scale factor 10 does not have to fit in memory twice.
    scratch = root / "dbgen.duckdb"
    scratch.unlink(missing_ok=True)
    con = duckdb.connect(str(scratch))
    con.execute("INSTALL tpch")
    con.execute("LOAD tpch")
    con.execute(f"CALL dbgen(sf = {sf})")
    for table in tpch.TABLES:
        columns = ", ".join(
            f"CAST({name} AS DOUBLE) AS {name}" if kind.startswith("DECIMAL") else name
            for name, kind in con.execute(
                "SELECT column_name, data_type FROM information_schema.columns"
                " WHERE table_name = ? ORDER BY ordinal_position",
                [table],
            ).fetchall()
        )
        path = root / f"{table}.parquet"
        con.execute(
            f"COPY (SELECT {columns} FROM {table}) TO '{path}'"
            " (FORMAT parquet, COMPRESSION snappy)"
        )
        print(f"wrote {path}")
    con.close()
    scratch.unlink()
    Path(f"{scratch}.wal").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
