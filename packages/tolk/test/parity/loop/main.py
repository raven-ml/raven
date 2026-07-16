#!/usr/bin/env python3
"""Parity case: for loop with load/store over 10 elements."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import dump_stage7_program  # noqa: E402

from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402


def kernel():
    sink = UOp(Ops.SINK, dtypes.void, (), arg=KernelInfo())
    a = UOp.param(0, dtypes.float32, shape=(-1,))
    ten = UOp.const(dtypes.int, 10)
    ridx = UOp(Ops.RANGE, dtypes.int, (ten,), (0, AxisType.LOOP))
    idx_ld = a.index(ridx)
    ld = UOp(Ops.LOAD, dtypes.float32, (idx_ld,))
    idx_st = a.index(ridx)
    store = UOp(Ops.STORE, dtypes.void, (idx_st, ld))
    end = UOp(Ops.END, dtypes.void, (ridx,))
    return [sink, a, ten, ridx, idx_ld, ld, idx_st, store, end]


if __name__ == "__main__":
    dump_stage7_program(kernel(), os.path.dirname(os.path.abspath(__file__)))
