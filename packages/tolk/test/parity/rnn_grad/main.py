#!/usr/bin/env python3
"""Parity case: the reverse pass of an unrolled recurrence.

The forward chain is the one in `rnn_unroll` — two steps of
`h <- x@W + h@U` over 4x4 matrices under a squared-magnitude loss — and
this case adds the gradient sweep back through it, which is the graph a
training step compiles. It exercises the three matmul orientations
together (`a@b` forward, `a@b'` for the gradient flowing back through a
weight, `a'@b` for a weight's own gradient), keeps every forward hidden
state live across both sweeps, and accumulates each weight gradient as a
sum of one contraction per step.

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


def contract(ae, be, m, n, k):
    return (ae.expand((m, n, k)) * be.expand((m, n, k)))._rop(Ops.ADD, (2,))


# out[m, n] = sum_k lhs[m, k] * rhs[k, n]
def matmul_nn(lhs, rhs, m, k, n):
    return contract(reshape(lhs, (m, 1, k)),
                    reshape(transpose(rhs), (1, n, k)), m, n, k)


# out[m, n] = sum_k lhs[m, k] * rhs[n, k]
def matmul_nt(lhs, rhs, m, k, n):
    return contract(reshape(lhs, (m, 1, k)), reshape(rhs, (1, n, k)), m, n, k)


# out[m, n] = sum_k lhs[k, m] * rhs[k, n]
def matmul_tn(lhs, rhs, m, k, n):
    return contract(reshape(transpose(lhs), (m, 1, k)),
                    reshape(transpose(rhs), (1, n, k)), m, n, k)


def build():
    b, d = BATCH, DIM
    w_in, w_rec, h0 = mk_param(0, d, d), mk_param(1, d, d), mk_param(2, b, d)
    xs = [mk_param(3 + t, b, d) for t in range(HORIZON)]
    h = [h0]
    for x in xs:
        h.append(matmul_nn(x, w_in, b, d, d) + matmul_nn(h[-1], w_rec, b, d, d))
    two = UOp.const(2.0, dtypes.float32).expand((b, d))
    g = [None] * (HORIZON + 1)
    g[HORIZON] = two * h[HORIZON]
    for t in range(HORIZON - 1, -1, -1):
        carried = matmul_nt(g[t + 1], w_rec, b, d, d)
        g[t] = carried if t == 0 else two * h[t] + carried

    def sum_steps(operands):
        acc = None
        for t, o in enumerate(operands):
            term = matmul_tn(o, g[t + 1], d, b, d)
            acc = term if acc is None else acc + term
        return acc

    return wrap_sink(sum_steps(xs), sum_steps(h[:HORIZON]), g[0])


if __name__ == "__main__":
    dump_tensor(build(), os.path.dirname(os.path.abspath(__file__)),
                stages=("stage7",))
