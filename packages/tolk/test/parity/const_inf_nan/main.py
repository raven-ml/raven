#!/usr/bin/env python3
"""Parity case: special float constants — infinity and NaN."""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import dump_stage7_program  # noqa: E402

from tinygrad.uop.ops import UOp, Ops, KernelInfo  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402


def kernel():
    sink = UOp(Ops.SINK, dtypes.void, (), arg=KernelInfo())
    a = UOp.param(0, dtypes.float32, shape=(-1,))
    idx0 = UOp.const(dtypes.int, 0)
    idx1 = UOp.const(dtypes.int, 1)
    inf_val = UOp.const(dtypes.float32, math.inf)
    nan_val = UOp.const(dtypes.float32, math.nan)
    idx_a0 = a.index(idx0)
    store0 = UOp(Ops.STORE, dtypes.void, (idx_a0, inf_val))
    idx_a1 = a.index(idx1)
    store1 = UOp(Ops.STORE, dtypes.void, (idx_a1, nan_val))
    return [sink, a, idx0, idx1, inf_val, nan_val, idx_a0, store0, idx_a1,
            store1]


if __name__ == "__main__":
    dump_stage7_program(kernel(), os.path.dirname(os.path.abspath(__file__)))
