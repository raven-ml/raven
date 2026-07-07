#!/usr/bin/env python3
"""Parity case: C = A * B, M=N=K=4. GPU-only."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import dump, GPU_BACKENDS  # noqa: E402

from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType, ParamArg  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402


def kernel():
    M, N, K = 4, 4, 4
    pA = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(0))
    pB = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(1))
    pC = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(2))
    ri = UOp.range(M, 0, AxisType.GLOBAL)
    rj = UOp.range(N, 1, AxisType.GLOBAL)
    rk = UOp.range(K, 2, AxisType.REDUCE)
    a_idx = ri * K + rk
    b_idx = rk * N + rj
    c_idx = ri * N + rj
    ld_a = pA.index(a_idx, ptr=True).load()
    ld_b = pB.index(b_idx, ptr=True).load()
    mul = ld_a * ld_b
    red = UOp(Ops.REDUCE, dtypes.float32, (mul, rk), (Ops.ADD, ()))
    st = pC.index(c_idx, ptr=True).store(red)
    end = st.end(ri, rj)
    return UOp.sink(
        end,
        arg=KernelInfo(
            name="matmul_small",
            axis_types=(AxisType.GLOBAL, AxisType.GLOBAL, AxisType.REDUCE),
            opts_to_apply=(),
        ),
    )


if __name__ == "__main__":
    dump(kernel(), os.path.dirname(os.path.abspath(__file__)),
         stages=("stage5", "stage7"), backends=GPU_BACKENDS)
