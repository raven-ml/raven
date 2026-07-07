#!/usr/bin/env python3
"""Parity case: c[i*16+j] = a[i*16+j] + b[i*16+j], 2 Global ranges. GPU-only."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import dump, GPU_BACKENDS  # noqa: E402

from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType, ParamArg  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402


def kernel():
    ROWS, COLS = 8, 16
    p0 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(0))
    p1 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(1))
    p2 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(2))
    ri = UOp.range(ROWS, 0, AxisType.GLOBAL)
    rj = UOp.range(COLS, 1, AxisType.GLOBAL)
    flat = ri * COLS + rj
    ld_a = p0.index(flat, ptr=True).load()
    ld_b = p1.index(flat, ptr=True).load()
    add = ld_a + ld_b
    st = p2.index(flat, ptr=True).store(add)
    end = st.end(ri, rj)
    return UOp.sink(
        end,
        arg=KernelInfo(
            name="elementwise_2d",
            axis_types=(AxisType.GLOBAL, AxisType.GLOBAL),
            opts_to_apply=(),
        ),
    )


if __name__ == "__main__":
    dump(kernel(), os.path.dirname(os.path.abspath(__file__)),
         stages=("stage5", "stage7"), backends=GPU_BACKENDS)
