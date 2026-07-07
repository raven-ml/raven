#!/usr/bin/env python3
"""Parity case: two int32 loads, multiply, store at index 0."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import dump_stage7_program  # noqa: E402

from tinygrad.uop.ops import UOp, Ops, KernelInfo, ParamArg  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402


def kernel():
    sink = UOp(Ops.SINK, dtypes.void, (), arg=KernelInfo())
    a = UOp(Ops.PARAM, dtypes.int32.ptr(), (), ParamArg(0))
    b = UOp(Ops.PARAM, dtypes.int32.ptr(), (), ParamArg(1))
    c = UOp(Ops.PARAM, dtypes.int32.ptr(), (), ParamArg(2))
    idx = UOp.const(dtypes.int, 0)
    idx_a = a.index(idx, ptr=True)
    ld_a = UOp(Ops.LOAD, dtypes.int32, (idx_a,))
    idx_b = b.index(idx, ptr=True)
    ld_b = UOp(Ops.LOAD, dtypes.int32, (idx_b,))
    mul = ld_a * ld_b
    idx_c = c.index(idx, ptr=True)
    store = UOp(Ops.STORE, dtypes.void, (idx_c, mul))
    return [sink, a, b, c, idx, idx_a, ld_a, idx_b, ld_b, mul, idx_c, store]


if __name__ == "__main__":
    dump_stage7_program(kernel(), os.path.dirname(os.path.abspath(__file__)))
