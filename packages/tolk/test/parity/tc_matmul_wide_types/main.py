#!/usr/bin/env python3
# Reference: tinygrad d0c9745274335e44b5dd7c15b8422c2b684a3ed9.
"""128-cubed BF16/FNUZ tensor-core contractions with fully unrolled reduction."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import ALL_BACKENDS, hip_renderer, dump  # noqa: E402

from dataclasses import replace
from tinygrad.helpers import Target
from tinygrad.codegen.opt.postrange import Scheduler
from tinygrad.codegen.opt import Opt, OptOps  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp  # noqa: E402

BACKENDS = {
    "metal_bf16": (ALL_BACKENDS["metal"], dtypes.bfloat16),
    "cuda_bf16": (ALL_BACKENDS["cuda"], dtypes.bfloat16),
    "amd_gfx1100_bf16": (hip_renderer(Target("AMD", arch="gfx1100")), dtypes.bfloat16),
    "amd_gfx1201_bf16": (hip_renderer(Target("AMD", arch="gfx1201")), dtypes.bfloat16),
    "amd_gfx942_bf16": (hip_renderer(Target("AMD", arch="gfx942")), dtypes.bfloat16),
    "amd_gfx950_bf16": (hip_renderer(Target("AMD", arch="gfx950")), dtypes.bfloat16),
    "amd_gfx942_e4m3fnuz": (hip_renderer(Target("AMD", arch="gfx942")), dtypes.fp8e4m3fnuz),
    "amd_gfx942_e5m2fnuz": (hip_renderer(Target("AMD", arch="gfx942")), dtypes.fp8e5m2fnuz),
}

def kernel(renderer, dtype):
    M, N, K = 128, 128, 128
    pA = UOp.param(0, dtype, shape=(M * K,))
    pB = UOp.param(1, dtype, shape=(K * N,))
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
    ast = UOp.sink(st.end(ri, rj), arg=KernelInfo(name="tc_matmul_wide_types", opts_to_apply=(tc,)))
    scheduler = Scheduler(ast, renderer)
    scheduler.apply_opt(tc)
    axis = scheduler.unrollable_dims[0]
    return ast.replace(arg=replace(ast.arg, opts_to_apply=(tc, Opt(OptOps.SPLIT, axis, (0, AxisType.UNROLL)))))


if __name__ == "__main__":
    for name, (renderer, dtype) in BACKENDS.items():
        dump(kernel(renderer, dtype), os.path.dirname(os.path.abspath(__file__)),
             stages=("stage5", "stage7"), backends={name: renderer})
