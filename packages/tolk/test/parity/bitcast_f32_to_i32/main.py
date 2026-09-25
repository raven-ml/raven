#!/usr/bin/env python3
"""Parity case: bitcast float32 to int32."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import dump_stage7_program  # noqa: E402

from tinygrad.uop.ops import UOp, Ops, KernelInfo  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402


def kernel():
    sink = UOp(Ops.SINK, src=(), arg=KernelInfo())
    a = UOp.param(0, dtypes.float32, shape=(-1,))
    b = UOp.param(1, dtypes.int32, shape=(-1,))
    idx = UOp.cconst(0, dtypes.int)
    idx_a = a.index(idx)
    ld = UOp(Ops.LOAD, src=(idx_a,))
    bc = UOp(Ops.BITCAST, src=(ld,), arg=dtypes.int32)
    idx_b = b.index(idx)
    store = UOp(Ops.STORE, src=(idx_b, bc))
    return [sink, a, b, idx, idx_a, ld, bc, idx_b, store]


if __name__ == "__main__":
    dump_stage7_program(kernel(), os.path.dirname(os.path.abspath(__file__)))
