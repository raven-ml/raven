#!/usr/bin/env python3
"""Parity case: c = sum(a * b), shape [256] -> scalar."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import dump_tensor, mk_param, wrap_sink  # noqa: E402
from tinygrad.uop.ops import UOp, Ops  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402


def build():
    a = mk_param(0, 256)
    b = mk_param(1, 256)
    mul = a * b
    red = mul._rop(Ops.ADD, (0,))
    return wrap_sink(red)


if __name__ == "__main__":
    dump_tensor(build(), os.path.dirname(os.path.abspath(__file__)),
                stages=("stage5", "stage7"))
