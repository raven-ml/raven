# /// script
# requires-python = "==3.13.*"
# dependencies = ["numpy==2.5.4", "h5py==3.16.0"]
# ///
"""Writes rows.h5, the HDF5 file of bench_nx's io rows: a dataset of 8192
rows of 8 float32 values, in chunks of 64 rows, each a deflate stream.
h5py 3.16, with the libhdf5 2.0 its wheel bundles, writes it in its default
format. Run as uv run packages/nx/bench/nx/gen/rows.py."""

from pathlib import Path

import h5py
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "rows.h5"

with h5py.File(OUT, "w") as f:
    x = (np.arange(8192 * 8, dtype="<f4") % 97).reshape(8192, 8)
    f.create_dataset("x", data=x, chunks=(64, 8), compression="gzip")
