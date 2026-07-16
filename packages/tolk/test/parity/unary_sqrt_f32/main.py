#!/usr/bin/env python3
"""Parity case: sqrt on float32."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import dump_stage7_program  # noqa: E402

from tinygrad.uop.ops import UOp, Ops, KernelInfo  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402


def kernel():
    sink = UOp(Ops.SINK, dtypes.void, (), arg=KernelInfo())
    a = UOp.param(0, dtypes.float32, shape=(-1,))
    b = UOp.param(1, dtypes.float32, shape=(-1,))
    idx = UOp.const(dtypes.int, 0)
    idx_a = a.index(idx)
    ld = UOp(Ops.LOAD, dtypes.float32, (idx_a,))
    sq = UOp(Ops.SQRT, dtypes.float32, (ld,))
    idx_b = b.index(idx)
    store = UOp(Ops.STORE, dtypes.void, (idx_b, sq))
    return [sink, a, b, idx, idx_a, ld, sq, idx_b, store]


if __name__ == "__main__":
    dump_stage7_program(kernel(), os.path.dirname(os.path.abspath(__file__)))
