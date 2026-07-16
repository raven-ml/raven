#!/usr/bin/env python3
"""Parity case: c = a.sum(0).permute(1,0) + b."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import dump_tensor, mk_param, wrap_sink  # noqa: E402
from tinygrad.uop.ops import UOp, Ops  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402


def build():
    a = mk_param(0, 10, 10, 10)
    b = mk_param(1, 10, 10)
    red = a._rop(Ops.ADD, (0,))
    permed = UOp(Ops.PERMUTE, dtypes.float32, (red,), (1, 0))
    return wrap_sink(permed + b)


if __name__ == "__main__":
    dump_tensor(build(), os.path.dirname(os.path.abspath(__file__)),
                stages=("stage5", "stage7"))
