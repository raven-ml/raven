#!/usr/bin/env python3
"""Parity case: dst[idx[k], j] = src[k, j] with distinct row indices.

No two updates share a row, so their order is free and the index range is a
launch dimension next to the column range. A row index outside [0, 16) gates
the store off. No range covers the 16 rows of dst.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import dump  # noqa: E402

from tinygrad.uop.ops import UOp, KernelInfo, AxisType  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402

N, D, K = 16, 8, 5


def kernel():
    dst = UOp.param(0, dtypes.float32, shape=(-1,))
    idx = UOp.param(1, dtypes.int32, shape=(-1,))
    src = UOp.param(2, dtypes.float32, shape=(-1,))
    j = UOp.range(D, 0, AxisType.WEAK)
    k = UOp.range(K, 1, AxisType.WEAK)
    row = idx.index(k).load()
    in_bounds = (row >= 0) & (row < N)
    target = (row.cast(dtypes.weakint) * D + j).valid(in_bounds)
    st = dst.index(target).store(src.index(k * D + j).load())
    end = st.end(j, k)
    return UOp.sink(
        end,
        arg=KernelInfo(
            name="indexed_store_unique",
            axis_types=(AxisType.WEAK, AxisType.WEAK),
            opts_to_apply=(),
        ),
    )


if __name__ == "__main__":
    dump(kernel(), os.path.dirname(os.path.abspath(__file__)),
         stages=("stage5", "stage7"))
