#!/usr/bin/env python3
"""Parity case: b[i] = a[i] + 1.0; c[i] = a[i] * 2.0, 1 Global range, 2 stores."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import dump  # noqa: E402

from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402


def kernel():
    p0 = UOp.param(0, dtypes.float32, shape=(256,))
    p1 = UOp.param(1, dtypes.float32, shape=(256,))
    p2 = UOp.param(2, dtypes.float32, shape=(256,))
    r0 = UOp.range(256, 0, AxisType.GLOBAL)
    ld_a = p0.index(r0).load()
    st1 = p1.index(r0).store(ld_a + UOp.const(1.0, dtypes.float32))
    st2 = p2.index(r0).store(ld_a * UOp.const(2.0, dtypes.float32))
    return UOp.sink(UOp.group(st1, st2).end(r0), arg=KernelInfo(name='multi_output', opts_to_apply=()))


if __name__ == "__main__":
    dump(kernel(), os.path.dirname(os.path.abspath(__file__)),
         stages=("stage5", "stage7"))
