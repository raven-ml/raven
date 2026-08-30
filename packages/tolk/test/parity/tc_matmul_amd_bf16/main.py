#!/usr/bin/env python3
"""Parity case: C = A @ B, bfloat16 inputs, float32 accumulate, M=N=K=16.

Rendered for gfx1100 (RDNA3) and gfx942 (CDNA3): the heuristic optimizer
engages the 16x16x16 bfloat16 tensor core — a WMMA kernel on RDNA3, an MFMA
kernel (with the CDNA string rewrite) on CDNA3.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import _HipNoComgr, dump_tensor, mk_param, wrap_sink  # noqa: E402
from tinygrad.helpers import Target  # noqa: E402
from tinygrad.uop.ops import UOp, Ops, shape_to_shape_arg  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402

BACKENDS = {
    "amd_gfx1100": _HipNoComgr(Target("AMD", arch="gfx1100")),
    "amd_gfx942": _HipNoComgr(Target("AMD", arch="gfx942")),
}


def build():
    M, N, K = 16, 16, 16
    a = mk_param(0, M, K, dtype=dtypes.bfloat16)
    b = mk_param(1, K, N, dtype=dtypes.bfloat16)
    # dot: a.reshape(M,1,K) * b.permute(1,0).reshape(1,N,K), summed over K.
    ar = UOp(Ops.RESHAPE, a.dtype, (a, shape_to_shape_arg((M, 1, K))))
    ae = ar.expand((M, N, K))
    bt = UOp(Ops.PERMUTE, b.dtype, (b,), (1, 0))
    br = UOp(Ops.RESHAPE, bt.dtype, (bt, shape_to_shape_arg((1, N, K))))
    be = br.expand((M, N, K))
    mul = (ae * be).cast(dtypes.float32)
    red = mul._rop(Ops.ADD, (2,))
    return wrap_sink(red)


if __name__ == "__main__":
    dump_tensor(build(), os.path.dirname(os.path.abspath(__file__)),
                stages=("stage5", "stage7"), backends=BACKENDS)
