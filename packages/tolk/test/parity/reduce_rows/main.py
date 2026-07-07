#!/usr/bin/env python3
"""Parity case: b[i] = sum_j(a[i*32+j]), 1 Global + 1 Reduce range."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import dump  # noqa: E402

from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType, ParamArg  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402


def kernel():
    ROWS, COLS = 8, 32
    p0 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(0))
    p1 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(1))
    ri = UOp.range(ROWS, 0, AxisType.GLOBAL)
    rj = UOp.range(COLS, 1, AxisType.REDUCE)
    flat = ri * COLS + rj
    ld = p0.index(flat, ptr=True).load()
    red = UOp(Ops.REDUCE, dtypes.float32, (ld, rj), (Ops.ADD, ()))
    st = p1.index(ri, ptr=True).store(red)
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
