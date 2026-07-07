#!/usr/bin/env python3
"""Parity case: d = (x+y).contiguous() + z, produces 2 kernels."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import dump_tensor, mk_param, wrap_sink  # noqa: E402
from tinygrad.uop.ops import UOp, Ops  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402


def build():
    x = mk_param(0, 32)
    y = mk_param(1, 32)
    z = mk_param(2, 32)
    add = x + y
    contig = UOp(Ops.CONTIGUOUS, dtypes.float32, (add,))
    return wrap_sink(contig + z)


if __name__ == "__main__":
    dump_tensor(build(), os.path.dirname(os.path.abspath(__file__)),
                stages=("stage5", "stage7"))
