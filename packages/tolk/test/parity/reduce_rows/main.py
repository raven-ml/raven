#!/usr/bin/env python3
"""Parity case: b[i] = sum_j(a[i*32+j]), 1 Global + 1 Reduce range."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import dump  # noqa: E402

from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402


def kernel():
    ROWS, COLS = 8, 32
    p0 = UOp.param(0, dtypes.float32, shape=(-1,))
    p1 = UOp.param(1, dtypes.float32, shape=(-1,))
    ri = UOp.range(ROWS, 0, AxisType.GLOBAL)
    rj = UOp.range(COLS, 1, AxisType.REDUCE)
    flat = ri * COLS + rj
    ld = p0.index(flat).load()
    red = UOp(Ops.REDUCE, dtypes.float32, (ld, rj), (Ops.ADD, 0))
    st = p1.index(ri).store(red)
    end = st.end(ri)
    return UOp.sink(
        end,
        arg=KernelInfo(
            name="reduce_rows",
            axis_types=(AxisType.GLOBAL, AxisType.REDUCE),
            opts_to_apply=(),
        ),
    )


if __name__ == "__main__":
    dump(kernel(), os.path.dirname(os.path.abspath(__file__)),
         stages=("stage5", "stage7"))
