#!/usr/bin/env python3
"""Parity case: c[0] = sum_k(a[k] * b[k]) over a single Reduce range of 128."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import dump  # noqa: E402

from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402


def kernel():
    p0 = UOp.param(0, dtypes.float32, shape=(-1,))
    p1 = UOp.param(1, dtypes.float32, shape=(-1,))
    p2 = UOp.param(2, dtypes.float32, shape=(-1,))
    r0 = UOp.range(128, 0, AxisType.REDUCE)
    ld_a = p0.index(r0).load()
    ld_b = p1.index(r0).load()
    mul = ld_a * ld_b
    red = UOp(Ops.REDUCE, dtypes.float32, (mul, r0), (Ops.ADD, 0))
    c0 = UOp.const(dtypes.weakint, 0)
    st = p2.index(c0).store(red)
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
