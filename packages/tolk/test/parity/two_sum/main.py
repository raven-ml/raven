#!/usr/bin/env python3
"""Parity case: c = a.sum(0) + a.sum(1), shape [64,64]."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import dump_tensor, mk_param, wrap_sink  # noqa: E402
from tinygrad.uop.ops import UOp, Ops, shape_to_shape_arg  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402


def build():
    a = mk_param(0, 64, 64)
    red0 = UOp(Ops.REDUCE, dtypes.float32, (a,), (Ops.ADD, (0,)))
    red1 = UOp(Ops.REDUCE, dtypes.float32, (a,), (Ops.ADD, (1,)))
    reshaped0 = UOp(Ops.RESHAPE, dtypes.float32,
                    (red0, shape_to_shape_arg((64,))))
    reshaped1 = UOp(Ops.RESHAPE, dtypes.float32,
                    (red1, shape_to_shape_arg((64,))))
    return wrap_sink(reshaped0 + reshaped1)


if __name__ == "__main__":
    dump_tensor(build(), os.path.dirname(os.path.abspath(__file__)),
                stages=("stage5", "stage7"))
