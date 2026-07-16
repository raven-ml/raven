#!/usr/bin/env python3
"""Parity case: vectorize 4 floats, then index element 2."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import dump_stage7_program, ALL_BACKENDS  # noqa: E402

from tinygrad.uop.ops import UOp, Ops, KernelInfo  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402


def kernel():
    sink = UOp(Ops.SINK, dtypes.void, (), arg=KernelInfo())
    a = UOp.param(0, dtypes.float32, shape=(-1,))
    b = UOp.param(1, dtypes.float32, shape=(-1,))
    idx0 = UOp.const(dtypes.int, 0)
    idx1 = UOp.const(dtypes.int, 1)
    idx2 = UOp.const(dtypes.int, 2)
    idx3 = UOp.const(dtypes.int, 3)
    ia0 = a.index(idx0)
    ia1 = a.index(idx1)
    ia2 = a.index(idx2)
    ia3 = a.index(idx3)
    v0 = UOp(Ops.LOAD, dtypes.float32, (ia0,))
    v1 = UOp(Ops.LOAD, dtypes.float32, (ia1,))
    v2 = UOp(Ops.LOAD, dtypes.float32, (ia2,))
    v3 = UOp(Ops.LOAD, dtypes.float32, (ia3,))
    vec = UOp.stack(v0, v1, v2, v3)
    lane = vec.index(idx2).simplify()
    idx_b = b.index(idx0)
    store = UOp(Ops.STORE, dtypes.void, (idx_b, lane))
    return [sink, a, b, idx0, idx1, idx2, idx3,
            ia0, ia1, ia2, ia3, v0, v1, v2, v3,
            vec, lane, idx_b, store]


def kernel_scalarized():
    sink = UOp(Ops.SINK, dtypes.void, (), arg=KernelInfo())
    a = UOp.param(0, dtypes.float32, shape=(-1,))
    b = UOp.param(1, dtypes.float32, shape=(-1,))
    idx0 = UOp.const(dtypes.int, 0)
    idx1 = UOp.const(dtypes.int, 1)
    idx2 = UOp.const(dtypes.int, 2)
    idx3 = UOp.const(dtypes.int, 3)
    ia0 = a.index(idx0)
    ia1 = a.index(idx1)
    ia2 = a.index(idx2)
    ia3 = a.index(idx3)
    v0 = UOp(Ops.LOAD, dtypes.float32, (ia0,))
    v1 = UOp(Ops.LOAD, dtypes.float32, (ia1,))
    v2 = UOp(Ops.LOAD, dtypes.float32, (ia2,))
    v3 = UOp(Ops.LOAD, dtypes.float32, (ia3,))
    idx_b = b.index(idx0)
    store = UOp(Ops.STORE, dtypes.void, (idx_b, v2))
    return [sink, a, b, idx0, idx1, idx2, idx3,
            ia0, ia1, ia2, ia3, v0, v1, v2, v3, idx_b, store]


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    scalar_backends = {k: v for k, v in ALL_BACKENDS.items()
                       if k in {"cpu", "cuda"}}
    vector_backends = {k: v for k, v in ALL_BACKENDS.items()
                       if k in {"metal", "opencl"}}
    dump_stage7_program(kernel_scalarized(), out_dir, backends=scalar_backends)
    dump_stage7_program(kernel(), out_dir, backends=vector_backends)
