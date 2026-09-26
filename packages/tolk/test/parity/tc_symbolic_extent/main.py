#!/usr/bin/env python3
# Reference: tinygrad d0c9745274335e44b5dd7c15b8422c2b684a3ed9.
"""Tensor-core eligibility and full lowering for a symbolic reduction extent."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import ALL_BACKENDS, dump  # noqa: E402

from tinygrad.codegen.opt.postrange import Scheduler
from tinygrad.codegen.opt import Opt, OptOps, KernelOptError  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp  # noqa: E402

BACKENDS = {k: ALL_BACKENDS[k] for k in ("metal", "cuda", "amd")}


def kernel(aligned):
    M, N, K = 128, 128, 128
    pA = UOp.param(0, dtypes.float16, shape=(M * K,))
    pB = UOp.param(1, dtypes.float16, shape=(K * N,))
    pC = UOp.param(2, dtypes.float32, shape=(M * N,))
    ri = UOp.range(M, 0, AxisType.GLOBAL)
    rj = UOp.range(N, 1, AxisType.GLOBAL)
    k = UOp.variable("tc_k", 1, 2, param=True)
    rk = UOp.range(k * (8 if aligned else 1), 2, AxisType.REDUCE)
    ld_a = pA.index(ri * K + rk).load()
    ld_b = pB.index(rk * N + rj).load()
    mul = (ld_a * ld_b).cast(dtypes.float32)
    red = UOp(Ops.REDUCE, src=(mul, rk), arg=(Ops.ADD, 0))
    st = pC.index(ri * N + rj).store(red)
    tc = Opt(OptOps.TC, 0, (-1, 0, 1))
    ast = UOp.sink(st.end(ri, rj), arg=KernelInfo(name="tc_symbolic_extent", opts_to_apply=(tc,)))
    return ast


def generate(out_dir):
    lines = []
    for label, aligned in (("aligned", True), ("unaligned", False)):
        for name, renderer in BACKENDS.items():
            ast = kernel(aligned)
            scheduler = Scheduler(ast, renderer)
            try:
                scheduler.apply_opt(ast.arg.opts_to_apply[0])
            except KernelOptError:
                lines.append(f"{label} {name}: rejected")
            else:
                dims = scheduler.tensor_core.dims
                lines.append(f"{label} {name}: {dims[0]}x{dims[1]}x{dims[2]}")
                dump(ast, out_dir, stages=("stage5", "stage7"),
                     backends={f"{label}_{name}": renderer})
    with open(os.path.join(out_dir, "eligibility.expected"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    generate(os.path.dirname(os.path.abspath(__file__)))
