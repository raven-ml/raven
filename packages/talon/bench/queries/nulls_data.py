# /// script
# requires-python = "==3.13.*"
# dependencies = ["numpy==2.5.3", "pyarrow==25.0.1"]
# [tool.uv]
# exclude-newer = "2026-10-01T00:00:00Z"
# ///
"""Generate the nullable Parquet fixture of talon's bench.

usage: uv run nulls_data.py FILE
       uv run nulls_data.py --check

FILE holds 2^20 rows of two columns, each with 10% nulls: [x], a double
rounded to one decimal in [0, 100), and [b], a boolean. talon has no Parquet
writer, so the bench reads this file, which pyarrow writes with its defaults:
snappy pages of format 1.0, [x] dictionary-encoded, the definition levels of
both columns RLE and bit-packed. It draws from numpy's PCG64 seeded with 22,
so the file is the same on every machine for the pinned versions. With
--check, the file is written to a temporary directory, and the run fails
unless it and the committed ../data/nulls.parquet both have the SHA-256
recorded here.
"""

import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SEED = 22
ROWS = 1 << 20
NULLS = 0.1
SHA256 = "dd79afb8ddec9ff50870181e2b9a01979ca2edacee17c1b3fbfa6de251579b14"
COMMITTED = Path(__file__).resolve().parent.parent / "data" / "nulls.parquet"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: str) -> None:
    rng = np.random.default_rng(SEED)
    x = np.round(rng.uniform(0, 100, ROWS), 1)
    x_null = rng.random(ROWS) < NULLS
    b = rng.random(ROWS) < 0.5
    b_null = rng.random(ROWS) < NULLS
    table = pa.table(
        {
            "x": pa.array(x, mask=x_null, type=pa.float64()),
            "b": pa.array(b, mask=b_null, type=pa.bool_()),
        }
    )
    pq.write_table(table, path)


def check() -> None:
    with tempfile.TemporaryDirectory() as d:
        made = Path(d) / "nulls.parquet"
        write(str(made))
        for name, path in [("written", made), ("committed", COMMITTED)]:
            if (got := sha256(path)) != SHA256:
                sys.exit(f"the {name} file has SHA-256 {got}, expected {SHA256}")
    print(f"nulls.parquet matches its SHA-256 {SHA256}")


def main() -> None:
    if sys.argv[1:] == ["--check"]:
        check()
        return
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    write(sys.argv[1])
    print(f"wrote {sys.argv[1]} ({ROWS} rows)")


if __name__ == "__main__":
    main()
