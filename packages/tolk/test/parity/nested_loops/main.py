#!/usr/bin/env python3
"""Parity case: two nested loops."""

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
    five = UOp.const(dtypes.int, 5)
    ridx0 = UOp(Ops.RANGE, dtypes.int, (ten,), (0, AxisType.LOOP))
    ridx1 = UOp(Ops.RANGE, dtypes.int, (five,), (1, AxisType.LOOP))
    combined = ridx0 + ridx1
    idx_ld = a.index(combined)
    ld = UOp(Ops.LOAD, dtypes.float32, (idx_ld,))
    idx_st = a.index(combined)
    store = UOp(Ops.STORE, dtypes.void, (idx_st, ld))
    end1 = UOp(Ops.END, dtypes.void, (ridx1,))
    end0 = UOp(Ops.END, dtypes.void, (ridx0,))
    return [sink, a, ten, five, ridx0, ridx1, combined, idx_ld, ld, idx_st,
            store, end1, end0]


if __name__ == "__main__":
    dump_stage7_program(kernel(), os.path.dirname(os.path.abspath(__file__)))
