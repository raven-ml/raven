#!/usr/bin/env python3
"""Parity case: c = a.sum(1)[:16] + b, shape [32,32], b=[16]."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import dump_tensor, mk_param, wrap_sink  # noqa: E402
from tinygrad.uop.ops import UOp, Ops, shape_to_shape_arg  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402


def build():
    a = mk_param(0, 32, 32)
    b = mk_param(1, 16)
    red = a._rop(Ops.ADD, (1,))
    reshaped = UOp(Ops.RESHAPE, dtypes.float32,
                   (red, shape_to_shape_arg((32,))))
    shrunk = reshaped.shrink(((0, 16),))
    return wrap_sink(shrunk + b)


if __name__ == "__main__":
    dump_tensor(build(), os.path.dirname(os.path.abspath(__file__)),
                stages=("stage5", "stage7"))
