#!/usr/bin/env python3
"""Parity case: GPU special dimensions (group_id, local_id). Metal/OpenCL."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import dump_stage7_program, ALL_BACKENDS  # noqa: E402

from tinygrad.uop.ops import UOp, Ops, KernelInfo, ParamArg  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402


def kernel():
    sink = UOp(Ops.SINK, dtypes.void, (), arg=KernelInfo())
    a = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(0))
    bound = UOp.const(dtypes.int, 32)
    gid = UOp(Ops.SPECIAL, dtypes.int, (bound,), "gidx0")
    lid = UOp(Ops.SPECIAL, dtypes.int, (bound,), "lidx0")
    combined = gid + lid
    idx_a = a.index(combined, ptr=True)
    ld = UOp(Ops.LOAD, dtypes.float32, (idx_a,))
    idx_st = a.index(combined, ptr=True)
    store = UOp(Ops.STORE, dtypes.void, (idx_st, ld))
    return [sink, a, bound, gid, lid, combined, idx_a, ld, idx_st, store]


if __name__ == "__main__":
    backends = {k: v for k, v in ALL_BACKENDS.items() if k in ("metal", "opencl")}
    dump_stage7_program(kernel(), os.path.dirname(os.path.abspath(__file__)),
                        backends=backends)
