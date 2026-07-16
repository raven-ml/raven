#!/usr/bin/env python3
"""Parity case: two float32 loads, add, store at index 0.

Paired with main.ml. Run to regenerate *.expected files. Flat pre-linearized
UOp program — no full_rewrite_to_sink, rendered directly.
"""

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
    c = UOp.param(2, dtypes.float32, shape=(-1,))
    idx = UOp.const(dtypes.int, 0)
    idx_a = a.index(idx)
    ld_a = UOp(Ops.LOAD, dtypes.float32, (idx_a,))
    idx_b = b.index(idx)
    ld_b = UOp(Ops.LOAD, dtypes.float32, (idx_b,))
    add = ld_a + ld_b
    idx_c = c.index(idx)
    store = UOp(Ops.STORE, dtypes.void, (idx_c, add))
    return [sink, a, b, c, idx, idx_a, ld_a, idx_b, ld_b, add, idx_c, store]


if __name__ == "__main__":
    dump_stage7_program(kernel(), os.path.dirname(os.path.abspath(__file__)))
