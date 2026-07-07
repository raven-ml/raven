#!/usr/bin/env python3
"""Parity case: If/Endif control flow."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import dump_stage7_program  # noqa: E402

from tinygrad.uop.ops import UOp, Ops, KernelInfo, ParamArg  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402


def kernel():
    sink = UOp(Ops.SINK, dtypes.void, (), arg=KernelInfo())
    a = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(0))
    idx = UOp.const(dtypes.int, 0)
    cond = UOp.const(dtypes.bool, True)
    if_op = UOp(Ops.IF, dtypes.void, (cond,))
    idx_a = a.index(idx, ptr=True)
    one = UOp.const(dtypes.float32, 1.0)
    store = UOp(Ops.STORE, dtypes.void, (idx_a, one))
    endif = UOp(Ops.ENDIF, dtypes.void, (if_op,))
    return [sink, a, idx, cond, if_op, idx_a, one, store, endif]


if __name__ == "__main__":
    dump_stage7_program(kernel(), os.path.dirname(os.path.abspath(__file__)))
