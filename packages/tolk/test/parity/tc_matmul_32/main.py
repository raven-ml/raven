#!/usr/bin/env python3
"""Parity case: C = A @ B, float16 inputs, float32 accumulate, M=N=K=32,
scheduled by the explicit opts TC:0:-1:0:1 and UNROLL:0:0.

Metal (Apple7) and CUDA (SM80) both carry tensor cores, and the shape is
large enough that the accumulator holds several WMMAs — so this case
observes the WMMA operand widths and the order the contracted axes reach
the register array, neither of which a single-WMMA kernel can see.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import ALL_BACKENDS, dump  # noqa: E402

from dataclasses import replace
from tinygrad.codegen.opt.postrange import Scheduler
from tinygrad.codegen.opt import Opt, OptOps  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp  # noqa: E402

BACKENDS = {k: ALL_BACKENDS[k] for k in ("cuda", "metal")}


def kernel(renderer):
    M, N, K = 32, 32, 32
    pA = UOp.param(0, dtypes.float16, shape=(M * K,))
    pB = UOp.param(1, dtypes.float16, shape=(K * N,))
    pC = UOp.param(2, dtypes.float32, shape=(M * N,))
    ri = UOp.range(M, 0, AxisType.GLOBAL)
    rj = UOp.range(N, 1, AxisType.GLOBAL)
    rk = UOp.range(K, 2, AxisType.REDUCE)
    ld_a = pA.index(ri * K + rk).load()
    ld_b = pB.index(rk * N + rj).load()
    mul = (ld_a * ld_b).cast(dtypes.float32)
    red = UOp(Ops.REDUCE, src=(mul, rk), arg=(Ops.ADD, 0))
    st = pC.index(ri * N + rj).store(red)
    tc = Opt(OptOps.TC, 0, (-1, 0, 1))
    ast = UOp.sink(st.end(ri, rj), arg=KernelInfo(name="tc_matmul_32", opts_to_apply=(tc,)))
    scheduler = Scheduler(ast, renderer)
    scheduler.apply_opt(tc)
    axis = scheduler.unrollable_dims[0]
    return ast.replace(arg=replace(ast.arg, opts_to_apply=(tc, Opt(OptOps.SPLIT, axis, (0, AxisType.UNROLL)))))


if __name__ == "__main__":
    for name, renderer in BACKENDS.items():
        dump(kernel(renderer), os.path.dirname(os.path.abspath(__file__)),
             stages=("stage5", "stage7"), backends={name: renderer})
