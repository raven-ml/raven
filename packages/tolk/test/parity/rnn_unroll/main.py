#!/usr/bin/env python3
"""Parity case: an unrolled recurrence, forward only.

Two steps of `h <- x@W + h@U` over 4x4 matrices, each step contributing a
squared-magnitude scalar loss to a running sum. Each matmul is a
broadcast-multiply contracted over the trailing axis; every hidden state
feeds both the next step and its own loss term, so the chain fuses into a
small number of kernels whose split is decided by rangeify.

Paired with main.ml. Run to regenerate *.expected files.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import dump_tensor, mk_param, wrap_sink  # noqa: E402

from tinygrad.uop.ops import UOp, Ops, shape_to_shape_arg  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402

BATCH = DIM = 4
HORIZON = 2


def reshape(x, shape):
    return UOp(Ops.RESHAPE, dtypes.float32, (x, shape_to_shape_arg(shape)))


def transpose(x):
    return UOp(Ops.PERMUTE, dtypes.float32, (x,), (1, 0))


# out[m, n] = sum_k lhs[m, k] * rhs[k, n]
def matmul(lhs, rhs, m, k, n):
    ae = reshape(lhs, (m, 1, k)).expand((m, n, k))
    be = reshape(transpose(rhs), (1, n, k)).expand((m, n, k))
    return (ae * be)._rop(Ops.ADD, (2,))


def build():
    b, d = BATCH, DIM
    w_in, w_rec = mk_param(0, d, d), mk_param(1, d, d)
    h, acc = mk_param(2, b, d), None
    for t in range(HORIZON):
        x = mk_param(3 + t, b, d)
        h = matmul(x, w_in, b, d, d) + matmul(h, w_rec, b, d, d)
        loss = (h * h)._rop(Ops.ADD, (0, 1))
        acc = loss if acc is None else acc + loss
    return wrap_sink(acc)


if __name__ == "__main__":
    dump_tensor(build(), os.path.dirname(os.path.abspath(__file__)),
                stages=("stage7",))
