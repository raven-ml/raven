#!/usr/bin/env python3
"""Parity case: c = neg(sqrt(sum(a))), shape [16] -> scalar."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import dump_tensor, mk_param, wrap_sink  # noqa: E402
from tinygrad.uop.ops import UOp, Ops  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402


def build():
    a = mk_param(0, 16)
    red = UOp(Ops.REDUCE, dtypes.float32, (a,), (Ops.ADD, (0,)))
    sq = UOp(Ops.SQRT, dtypes.float32, (red,))
    neg = UOp(Ops.NEG, dtypes.float32, (sq,))
    return wrap_sink(neg)


if __name__ == "__main__":
    dump_tensor(build(), os.path.dirname(os.path.abspath(__file__)),
                stages=("stage5", "stage7"))
