#!/usr/bin/env python3
"""Parity case: two reduce accumulators in one kernel.

`out = A'@B + C@D` over 16x16. The two contractions fuse into one kernel,
but only the first has its contraction axis mapped onto the local index,
so it is staged through shared memory and group-reduced while the second
stays an ordinary loop. Each reduce therefore needs its own accumulator
register; giving them the same one makes the closing add read a single
accumulator twice and doubles the result.

CPU is excluded: without local dimensions there is no group reduce, so
the kernel has one accumulator and the case proves nothing there.

Paired with main.ml. Run to regenerate *.expected files.
"""

import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from helpers import GPU_BACKENDS, dump_tensor, mk_param, wrap_sink  # noqa: E402

from tinygrad.uop.ops import UOp, Ops, shape_to_shape_arg  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402

N = 16


def reshape(x, shape):
    return UOp(Ops.RESHAPE, dtypes.float32, (x, shape_to_shape_arg(shape)))


def transpose(x):
    return UOp(Ops.PERMUTE, dtypes.float32, (x,), (1, 0))


def contract(ae, be):
    return (ae.expand((N, N, N)) * be.expand((N, N, N)))._rop(Ops.ADD, (2,))


# out[m, n] = sum_k lhs[m, k] * rhs[k, n]
def matmul_nn(lhs, rhs):
    return contract(reshape(lhs, (N, 1, N)), reshape(transpose(rhs), (1, N, N)))


# out[m, n] = sum_k lhs[k, m] * rhs[k, n]
def matmul_tn(lhs, rhs):
    return contract(reshape(transpose(lhs), (N, 1, N)),
                    reshape(transpose(rhs), (1, N, N)))


def build():
    return wrap_sink(matmul_tn(mk_param(0, N, N), mk_param(1, N, N))
                     + matmul_nn(mk_param(2, N, N), mk_param(3, N, N)))


if __name__ == "__main__":
    dump_tensor(build(), os.path.dirname(os.path.abspath(__file__)),
                stages=("stage7",), backends=GPU_BACKENDS)
