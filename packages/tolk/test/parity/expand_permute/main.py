#!/usr/bin/env python3
"""Parity case: d = (a+b).expand(10,10,10) + (a+b).permute(2,1,0).expand(10,10,10)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import dump_tensor, mk_param, wrap_sink  # noqa: E402
from tinygrad.uop.ops import UOp, Ops, shape_to_shape_arg  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402


def build():
    a = mk_param(0, 10, 10, 1)
    b = mk_param(1, 10, 10, 1)
    ab = a + b
    expanded = UOp(Ops.EXPAND, dtypes.float32, (ab, shape_to_shape_arg((10, 10, 10))))
    permed = UOp(Ops.PERMUTE, dtypes.float32, (ab,), (2, 1, 0))
    permed_expanded = UOp(Ops.EXPAND, dtypes.float32,
                          (permed, shape_to_shape_arg((10, 10, 10))))
    return wrap_sink(expanded + permed_expanded)


if __name__ == "__main__":
    dump_tensor(build(), os.path.dirname(os.path.abspath(__file__)),
                stages=("stage5", "stage7"))
