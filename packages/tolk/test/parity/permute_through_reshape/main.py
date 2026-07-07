#!/usr/bin/env python3
"""Parity case: c = (a+b).reshape(4,4,4,4).permute(2,3,0,1)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import dump_tensor, mk_param, wrap_sink  # noqa: E402
from tinygrad.uop.ops import UOp, Ops, shape_to_shape_arg  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402


def build():
    a = mk_param(0, 16, 16)
    b = mk_param(1, 16, 16)
    add = a + b
    reshaped = UOp(Ops.RESHAPE, dtypes.float32, (add, shape_to_shape_arg((4, 4, 4, 4))))
    permed = UOp(Ops.PERMUTE, dtypes.float32, (reshaped,), (2, 3, 0, 1))
    return wrap_sink(permed)


if __name__ == "__main__":
    dump_tensor(build(), os.path.dirname(os.path.abspath(__file__)),
                stages=("stage5", "stage7"))
