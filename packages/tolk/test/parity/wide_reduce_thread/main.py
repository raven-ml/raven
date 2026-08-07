#!/usr/bin/env python3
"""Parity case: a contraction sum whose iteration space exceeds 2^62.

Twelve independent contractions over a batch axis of 32 are summed into
one 8x8 result, which fuses into a single kernel carrying twelve reduce
axes on top of the output: 2^66 points. This is the shape a weight
gradient takes when a recurrence is unrolled -- one contraction per step,
all accumulated -- reduced to the smallest graph that reaches it.

The host-threading heuristic sizes its thread count from the product of
the full shape, so a 63-bit reconstruction of that product wraps and
silently drops the thread split. This pins that the split survives an
iteration space that does not fit in a machine word.

CPU only -- threading is a host-renderer feature -- and CPU_COUNT is
pinned so the chosen thread count does not follow the machine.

Paired with main.ml. Run to regenerate *.expected files.
"""

import os
import sys

os.environ["CPU_COUNT"] = "8"

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from helpers import ALL_BACKENDS, dump_tensor, mk_param, wrap_sink  # noqa: E402

from tinygrad.uop.ops import UOp, Ops, shape_to_shape_arg  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402

BACKENDS = {k: v for k, v in ALL_BACKENDS.items() if k == "cpu"}
BATCH, DIM, TERMS = 32, 8, 12


def reshape(x, shape):
    return UOp(Ops.RESHAPE, dtypes.float32, (x, shape_to_shape_arg(shape)))


def transpose(x):
    return UOp(Ops.PERMUTE, dtypes.float32, (x,), (1, 0))


# out[m, n] = sum_k lhs[k, m] * rhs[k, n]
def matmul_tn(lhs, rhs, m, k, n):
    ae = reshape(transpose(lhs), (m, 1, k)).expand((m, n, k))
    be = reshape(transpose(rhs), (1, n, k)).expand((m, n, k))
    return (ae * be)._rop(Ops.ADD, (2,))


def build():
    b, d = BATCH, DIM
    acc = None
    for t in range(TERMS):
        term = matmul_tn(mk_param(2 * t, b, d), mk_param(2 * t + 1, b, d),
                         d, b, d)
        acc = term if acc is None else acc + term
    return wrap_sink(acc)


if __name__ == "__main__":
    dump_tensor(build(), os.path.dirname(os.path.abspath(__file__)),
                stages=("stage7",), backends=BACKENDS)
