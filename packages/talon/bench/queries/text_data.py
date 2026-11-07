# /// script
# requires-python = "==3.13.*"
# dependencies = ["numpy==2.5.3", "pyarrow==25.0.1"]
# [tool.uv]
# exclude-newer = "2026-10-01T00:00:00Z"
# ///
"""Generate the dictionary-encoded text Parquet fixture of talon's bench.

usage: uv run text_data.py FILE
       uv run text_data.py --check

FILE holds 2^18 rows of one column [s] without nulls: words of 8 to 27 bytes,
as short as the codes and modes of decision-support tables, about half of
them past 16 bytes, drawn uniformly from 100. pyarrow writes it with its
defaults: snappy pages of format 1.0 and [s] dictionary-encoded. It draws from
numpy's PCG64 seeded with 24, so the file is the same on every machine for the
pinned versions. With --check, the file is written to a temporary directory,
and the run fails unless it and the committed ../data/text.parquet both have
the SHA-256 recorded here.
"""

import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SEED = 24
ROWS = 1 << 18
WORDS = [f"word {i:02d} " + "-" * (i % 20) for i in range(100)]
SHA256 = "0f6c1c95ce0ee1052c02a7fe1e0a154a5db877c9869a7c36b53e33deb023cadf"
COMMITTED = Path(__file__).resolve().parent.parent / "data" / "text.parquet"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: str) -> None:
    rng = np.random.default_rng(SEED)
    codes = rng.integers(0, len(WORDS), ROWS)
    words = np.array(WORDS, dtype=object)
    table = pa.table({"s": pa.array(words[codes], type=pa.string())})
    pq.write_table(table, path)


def check() -> None:
    with tempfile.TemporaryDirectory() as d:
        made = Path(d) / "text.parquet"
        write(str(made))
        for name, path in [("written", made), ("committed", COMMITTED)]:
            if (got := sha256(path)) != SHA256:
                sys.exit(f"the {name} file has SHA-256 {got}, expected {SHA256}")
    print(f"text.parquet matches its SHA-256 {SHA256}")


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
