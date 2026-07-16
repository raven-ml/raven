#!/usr/bin/env python3
"""Parity case: b[0] = sum(a[i]); c[0] = sum(a[i]*a[i]), 1 Reduce range, 2 stores."""

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
    ld = p0.index(r0).load()
    red1 = UOp(Ops.REDUCE, dtypes.float32, (ld, r0), (Ops.ADD, 0))
    red2 = UOp(Ops.REDUCE, dtypes.float32, (ld * ld, r0), (Ops.ADD, 0))
    c0 = UOp.const(dtypes.weakint, 0)
    st1 = p1.index(c0).store(red1)
    st2 = p2.index(c0).store(red2)
    return UOp.sink(
        st1, st2,
        arg=KernelInfo(
            name="parallel_reduce",
            axis_types=(AxisType.REDUCE,),
            opts_to_apply=(),
        ),
    )


if __name__ == "__main__":
    dump(kernel(), os.path.dirname(os.path.abspath(__file__)),
         stages=("stage5", "stage7"))
