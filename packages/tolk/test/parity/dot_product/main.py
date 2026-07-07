#!/usr/bin/env python3
"""Parity case: c[0] = sum_k(a[k] * b[k]) over a single Reduce range of 128."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import dump  # noqa: E402

from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType, ParamArg  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402


def kernel():
    p0 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(0))
    p1 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(1))
    p2 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(2))
    r0 = UOp.range(128, 0, AxisType.REDUCE)
    ld_a = p0.index(r0, ptr=True).load()
    ld_b = p1.index(r0, ptr=True).load()
    mul = ld_a * ld_b
    red = UOp(Ops.REDUCE, dtypes.float32, (mul, r0), (Ops.ADD, ()))
    c0 = UOp.const(dtypes.weakint, 0)
    st = p2.index(c0, ptr=True).store(red)
    return UOp.sink(
        st,
        arg=KernelInfo(
            name="dot_product",
            axis_types=(AxisType.REDUCE,),
            opts_to_apply=(),
        ),
    )


if __name__ == "__main__":
    dump(kernel(), os.path.dirname(os.path.abspath(__file__)),
         stages=("stage5", "stage7"))
