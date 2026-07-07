#!/usr/bin/env python3
"""Parity case: c = a.reshape(16).reshape(2,8) + b, shape [4,4], b=[2,8]."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import dump_tensor, mk_param, wrap_sink  # noqa: E402
from tinygrad.uop.ops import UOp, Ops, shape_to_shape_arg  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402


def build():
    a = mk_param(0, 4, 4)
    b = mk_param(1, 2, 8)
    r1 = UOp(Ops.RESHAPE, dtypes.float32, (a, shape_to_shape_arg((16,))))
    r2 = UOp(Ops.RESHAPE, dtypes.float32, (r1, shape_to_shape_arg((2, 8))))
    return wrap_sink(r2 + b)


if __name__ == "__main__":
    dump_tensor(build(), os.path.dirname(os.path.abspath(__file__)),
                stages=("stage5", "stage7"))
