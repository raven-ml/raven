#!/usr/bin/env python3
"""Parity case: 4 params, add two and store in the 4th (third is unused)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import dump_stage7_program  # noqa: E402

from tinygrad.uop.ops import UOp, Ops, KernelInfo, ParamArg  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402


def kernel():
    sink = UOp(Ops.SINK, dtypes.void, (), arg=KernelInfo())
    a = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(0))
    b = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(1))
    c = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(2))
    d = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(3))
    idx = UOp.const(dtypes.int, 0)
    idx_a = a.index(idx, ptr=True)
    ld_a = UOp(Ops.LOAD, dtypes.float32, (idx_a,))
    idx_b = b.index(idx, ptr=True)
    ld_b = UOp(Ops.LOAD, dtypes.float32, (idx_b,))
    add = ld_a + ld_b
    idx_d = d.index(idx, ptr=True)
    store = UOp(Ops.STORE, dtypes.void, (idx_d, add))
    return [sink, a, b, c, d, idx, idx_a, ld_a, idx_b, ld_b, add, idx_d, store]


if __name__ == "__main__":
    dump_stage7_program(kernel(), os.path.dirname(os.path.abspath(__file__)))
