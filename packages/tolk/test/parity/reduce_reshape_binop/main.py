#!/usr/bin/env python3
"""Parity case: c = a.sum(0).reshape(10) + b."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import dump_tensor, mk_param, wrap_sink  # noqa: E402
from tinygrad.uop.ops import UOp, Ops, shape_to_shape_arg  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402


def build():
    a = mk_param(0, 10, 10)
    b = mk_param(1, 10)
    red = a._rop(Ops.ADD, (0,))
    reshaped = UOp(Ops.RESHAPE, dtypes.float32, (red, shape_to_shape_arg((10,))))
    return wrap_sink(reshaped + b)


if __name__ == "__main__":
    dump_tensor(build(), os.path.dirname(os.path.abspath(__file__)),
                stages=("stage5", "stage7"))
