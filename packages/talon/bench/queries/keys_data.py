# /// script
# requires-python = "==3.13.*"
# dependencies = ["numpy==2.5.3", "pyarrow==25.0.1"]
# [tool.uv]
# exclude-newer = "2026-10-01T00:00:00Z"
# ///
"""Generate the sorted-keys Parquet fixture of talon's bench.

usage: uv run keys_data.py FILE
       uv run keys_data.py --check

FILE holds 2^17 rows of two int64 columns without nulls, [k] and [p], the
same ascending keys, each repeated 1 to 7 times, as an order's line items
repeat its key. pyarrow writes it with its defaults but for [p], which is not
dictionary-encoded: snappy pages of format 1.0, [k]'s dictionary of its keys
and its indices bit-packed, [p]'s values PLAIN. Both compress to copies of a
key's high bytes from 8 bytes back. It draws from numpy's PCG64 seeded with 25,
so the file is the same on every machine for the pinned versions. With
--check, the file is written to a temporary directory, and the run fails
unless it and the committed ../data/keys.parquet both have the SHA-256
recorded here.
"""

import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SEED = 25
ROWS = 1 << 17
SHA256 = "afab11e078f1bf5142c689d176785baec0fbb85fcb074f7a794b9f7a8bade385"
COMMITTED = Path(__file__).resolve().parent.parent / "data" / "keys.parquet"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: str) -> None:
    rng = np.random.default_rng(SEED)
    repeats = rng.integers(1, 8, ROWS)
    keys = np.repeat(np.arange(len(repeats), dtype=np.int64) * 4, repeats)[:ROWS]
    table = pa.table({"k": pa.array(keys), "p": pa.array(keys)})
    pq.write_table(table, path, use_dictionary=["k"])


def check() -> None:
    with tempfile.TemporaryDirectory() as d:
        made = Path(d) / "keys.parquet"
        write(str(made))
        for name, path in [("written", made), ("committed", COMMITTED)]:
            if (got := sha256(path)) != SHA256:
                sys.exit(f"the {name} file has SHA-256 {got}, expected {SHA256}")
    print(f"keys.parquet matches its SHA-256 {SHA256}")


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
