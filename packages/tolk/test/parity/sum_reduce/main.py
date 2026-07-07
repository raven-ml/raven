#!/usr/bin/env python3
"""Parity case: b[0] = sum(a[i]) over a single Reduce range of 256 elements."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import dump  # noqa: E402

from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType, ParamArg  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402


def kernel():
    p0 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(0))
    p1 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(1))
    r0 = UOp.range(256, 0, AxisType.REDUCE)
    ld = p0.index(r0, ptr=True).load()
    red = UOp(Ops.REDUCE, dtypes.float32, (ld, r0), (Ops.ADD, ()))
    c0 = UOp.const(dtypes.weakint, 0)
    st = p1.index(c0, ptr=True).store(red)
    return UOp.sink(
        st,
        arg=KernelInfo(
            name="sum_reduce",
            axis_types=(AxisType.REDUCE,),
            opts_to_apply=(),
        ),
    )


if __name__ == "__main__":
    dump(kernel(), os.path.dirname(os.path.abspath(__file__)),
         stages=("stage5", "stage7"))
