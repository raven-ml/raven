#!/usr/bin/env python3
"""Parity case: d = (a + b).permute(1, 0) + c."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import dump_tensor, mk_param, wrap_sink  # noqa: E402
from tinygrad.uop.ops import UOp, Ops  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402


def build():
    a = mk_param(0, 2, 5)
    b = mk_param(1, 2, 5)
    c = mk_param(2, 5, 2)
    add = a + b
    permed = UOp(Ops.PERMUTE, dtypes.float32, (add,), (1, 0))
    return wrap_sink(permed + c)


if __name__ == "__main__":
    dump_tensor(build(), os.path.dirname(os.path.abspath(__file__)),
                stages=("stage5", "stage7"))
