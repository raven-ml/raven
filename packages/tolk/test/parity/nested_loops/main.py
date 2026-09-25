#!/usr/bin/env python3
"""Parity case: two nested loops."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import dump_stage7_program  # noqa: E402

from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402


def kernel():
    sink = UOp(Ops.SINK, src=(), arg=KernelInfo())
    a = UOp.param(0, dtypes.float32, shape=(-1,))
    ten = UOp.cconst(10, dtypes.int)
    five = UOp.cconst(5, dtypes.int)
    ridx0 = UOp(Ops.RANGE, src=(ten,), arg=(0, AxisType.WEAK))
    ridx1 = UOp(Ops.RANGE, src=(five,), arg=(1, AxisType.WEAK))
    combined = ridx0 + ridx1
    idx_ld = a.index(combined)
    ld = UOp(Ops.LOAD, src=(idx_ld,))
    idx_st = a.index(combined)
    store = UOp(Ops.STORE, src=(idx_st, ld))
    end1 = UOp(Ops.END, src=(store, ridx1))
    end0 = UOp(Ops.END, src=(end1, ridx0))
    return [sink, a, ten, five, ridx0, ridx1, combined, idx_ld, ld, idx_st,
            store, end1, end0]


if __name__ == "__main__":
    dump_stage7_program(kernel(), os.path.dirname(os.path.abspath(__file__)))
