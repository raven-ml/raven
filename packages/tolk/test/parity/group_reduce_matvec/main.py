#!/usr/bin/env python3
"""Parity case: matvec y[j] = sum_k w[j*64+k] * x[k], j=256, k=64,
with GROUP(0,8) + LOCAL(0,4) + UPCAST(0,4) applied explicitly.

The matvec grouping the heuristic picks for gpt2's decode step: a partial
reduce accumulated through a shared-memory tile indexed by (local, group,
upcast-lane). Pins the local-buffer materialization for a Stage carrying
both an upstream local range and a vectorized value. GPU backends only.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import ALL_BACKENDS, dump  # noqa: E402

from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType, ParamArg  # noqa: E402
from tinygrad.codegen.opt import Opt, OptOps  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402

BACKENDS = {k: v for k, v in ALL_BACKENDS.items() if k in ("cuda", "opencl")}


def kernel():
    N, K = 256, 64
    pW = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(0))
    pX = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(1))
    pY = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(2))
    rj = UOp.range(N, 0, AxisType.GLOBAL)
    rk = UOp.range(K, 1, AxisType.REDUCE)
    ld_w = pW.index(rj * K + rk, ptr=True).load()
    ld_x = pX.index(rk, ptr=True).load()
    red = UOp(Ops.REDUCE, dtypes.float32, (ld_w * ld_x, rk), (Ops.ADD, ()))
    st = pY.index(rj, ptr=True).store(red)
    end = st.end(rj)
    return UOp.sink(
        end,
        arg=KernelInfo(
            name="group_reduce_matvec",
            axis_types=(AxisType.GLOBAL, AxisType.REDUCE),
            opts_to_apply=(
                Opt(op=OptOps.GROUP, axis=0, arg=8),
                Opt(op=OptOps.LOCAL, axis=0, arg=4),
                Opt(op=OptOps.UPCAST, axis=0, arg=4),
            ),
        ),
    )


if __name__ == "__main__":
    dump(kernel(), os.path.dirname(os.path.abspath(__file__)),
         stages=("stage5", "stage7"), backends=BACKENDS)
