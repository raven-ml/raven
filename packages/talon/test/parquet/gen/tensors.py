# /// script
# requires-python = "==3.13.*"
# dependencies = ["pyarrow==25.0.1"]
# [tool.uv]
# exclude-newer = "2026-10-01T00:00:00Z"
# ///
"""Writes the tensor fixture of the Parquet suite.

Run from this directory with `uv run tensors.py [--check]`. The fixture is
committed, so the suite itself needs neither Python nor the network. With
--check nothing is written, and the run fails if the fixture would change.

../support/tensors.parquet holds 8 rows in two row groups of 4, as pyarrow
writes images of a survey: each column a list of lists of lists of float32,
optional at every level, in data pages of version 1 of about one row each.
Cell (b, y, x) of row i holds i * 100 + b * 12 + y * 4 + x
in every column but `bands`:

- `image`: rows of 2 x 3 x 4, row 2 null;
- `ragged`: as `image`, but list 1 of row 5's band 0 holds 3 values;
- `wide`: as `image`, but every list of row 6's values holds 5 values, cell
  (b, y, x) still i * 100 + b * 12 + y * 4 + x;
- `null_list`: as `image`, but row 1's band 1 is null;
- `null_value`: as `image`, but cell (1, 2, 3) of row 7 is null;
- `order`: as `image`, but list 1 of row 1's band 0 holds 3 values and row 3
  holds 1 band, so that an earlier row fails deeper than a later one;
- `bands`: row i holds i % 3 lists of 3 x 4, cell (b, y, x) holding
  i * 100 + b * 12 + y * 4 + x, row 4 null.
"""

import argparse
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).parent
OUT = HERE.parent / "support"
NAME = "tensors.parquet"
ROWS = 8
GROUP = 4
BANDS, HEIGHT, WIDTH = 2, 3, 4
# Bytes of a data page smaller than a row: pyarrow ends a page of a nested
# column at a row's end, so each holds about one row.
SMALL_PAGE = 64
SMALL_BATCH = 5


def cell(i, b, y, x):
    return float(i * 100 + b * 12 + y * 4 + x)


def image(i, bands=BANDS, width=WIDTH):
    return [[[cell(i, b, y, x) for x in range(width)] for y in range(HEIGHT)] for b in range(bands)]


def column(name, i):
    if name == "bands":
        return None if i == 4 else image(i, bands=i % 3)
    if i == 2:
        return None
    v = image(i, width=5 if (name, i) == ("wide", 6) else WIDTH)
    if (name, i) == ("ragged", 5):
        v[0][1] = v[0][1][:3]
    if (name, i) == ("null_list", 1):
        v[1] = None
    if (name, i) == ("null_value", 7):
        v[1][2][3] = None
    if (name, i) == ("order", 1):
        v[0][1] = v[0][1][:3]
    if (name, i) == ("order", 3):
        v = v[:1]
    return v


NAMES = ["image", "ragged", "wide", "null_list", "null_value", "order", "bands"]
TYPE = pa.list_(pa.list_(pa.list_(pa.float32())))


def generate(out):
    columns = {n: pa.array([column(n, i) for i in range(ROWS)], type=TYPE) for n in NAMES}
    pq.write_table(
        pa.table(columns),
        out / NAME,
        version="2.6",
        data_page_version="1.0",
        use_dictionary=False,
        compression="NONE",
        row_group_size=GROUP,
        data_page_size=SMALL_PAGE,
        write_batch_size=SMALL_BATCH,
        write_statistics=False,
        store_schema=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    if not parser.parse_args().check:
        generate(OUT)
        return
    with tempfile.TemporaryDirectory() as tmp:
        generate(Path(tmp))
        if (Path(tmp) / NAME).read_bytes() != (OUT / NAME).read_bytes():
            raise SystemExit(f"stale: {NAME}; run gen/tensors.py")


if __name__ == "__main__":
    main()
