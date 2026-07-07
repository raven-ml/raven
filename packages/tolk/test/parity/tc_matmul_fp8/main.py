#!/usr/bin/env python3
"""Parity case: C = A @ B, fp8e4m3 inputs, float32 accumulate, M=16 N=8 K=32.

Rendered for both sm_80 and sm_90: the heuristic optimizer engages the
8x16x32 fp8 tensor core on sm_90 (mma.sync kernel) while sm_80, which has no
fp8 tensor core, renders a plain reduce loop.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import _CudaNoNvrtc, dump_tensor, mk_param, wrap_sink  # noqa: E402
from tinygrad.helpers import Target  # noqa: E402
from tinygrad.uop.ops import UOp, Ops, shape_to_shape_arg  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402

BACKENDS = {
    "cuda_sm80": _CudaNoNvrtc(Target("CUDA", arch="sm_80")),
    "cuda_sm90": _CudaNoNvrtc(Target("CUDA", arch="sm_90")),
}


def build():
    M, N, K = 16, 8, 32
    a = mk_param(0, M, K, dtype=dtypes.fp8e4m3)
    b = mk_param(1, K, N, dtype=dtypes.fp8e4m3)
    # dot: a.reshape(M,1,K) * b.permute(1,0).reshape(1,N,K), summed over K.
    ar = UOp(Ops.RESHAPE, a.dtype, (a, shape_to_shape_arg((M, 1, K))))
    ae = UOp(Ops.EXPAND, a.dtype, (ar, shape_to_shape_arg((M, N, K))))
    bt = UOp(Ops.PERMUTE, b.dtype, (b,), (1, 0))
    br = UOp(Ops.RESHAPE, bt.dtype, (bt, shape_to_shape_arg((1, N, K))))
    be = UOp(Ops.EXPAND, bt.dtype, (br, shape_to_shape_arg((M, N, K))))
    mul = (ae * be).cast(dtypes.float32)
    red = UOp(Ops.REDUCE, dtypes.float32, (mul,), (Ops.ADD, (2,)))
    return wrap_sink(red)


if __name__ == "__main__":
    dump_tensor(build(), os.path.dirname(os.path.abspath(__file__)),
                stages=("stage5", "stage7"), backends=BACKENDS)
