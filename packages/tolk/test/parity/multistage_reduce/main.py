#!/usr/bin/env python3
"""Parity case: c = a.sum(2).relu().sum(1), shape [32,32,32]."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import dump_tensor, mk_param, wrap_sink  # noqa: E402
from tinygrad.uop.ops import UOp, Ops  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402


def build():
    a = mk_param(0, 32, 32, 32)
    red1 = a._rop(Ops.ADD, (2,))
    zero = UOp.const(dtypes.float32, 0.0)
    zero_bc = zero.reshape((1, 1)).expand((32, 32))
    relu = red1.alu(Ops.MAX, zero_bc)
    red2 = relu._rop(Ops.ADD, (1,))
    return wrap_sink(red2)


if __name__ == "__main__":
    dump_tensor(build(), os.path.dirname(os.path.abspath(__file__)),
                stages=("stage5", "stage7"))
