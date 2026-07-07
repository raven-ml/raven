#!/usr/bin/env python3
"""Parity case: shared memory + barrier (GPU backends only)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import dump_stage7_program, GPU_BACKENDS  # noqa: E402

from tinygrad.uop.ops import UOp, Ops, KernelInfo, ParamArg  # noqa: E402
from tinygrad.dtype import dtypes, AddrSpace  # noqa: E402


def kernel():
    local_ptr = dtypes.float32.ptr(size=256, addrspace=AddrSpace.LOCAL)
    sink = UOp(Ops.SINK, dtypes.void, (), arg=KernelInfo())
    a = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(0))
    temp = UOp.placeholder((256,), dtypes.float32, 0, AddrSpace.LOCAL)
    idx = UOp.const(dtypes.int, 0)
    zero = UOp.const(dtypes.float32, 0.0)
    idx_local = temp.index(idx, ptr=True)
    store_local = UOp(Ops.STORE, dtypes.void, (idx_local, zero))
    barrier = UOp(Ops.BARRIER, dtypes.void, (store_local,))
    after = UOp(Ops.AFTER, local_ptr, (temp, barrier))
    idx_local2 = after.index(idx, ptr=True)
    ld = UOp(Ops.LOAD, dtypes.float32, (idx_local2,))
    idx_global = a.index(idx, ptr=True)
    store_global = UOp(Ops.STORE, dtypes.void, (idx_global, ld))
    return [sink, a, temp, idx, zero, idx_local, store_local, barrier, after,
            idx_local2, ld, idx_global, store_global]


if __name__ == "__main__":
    dump_stage7_program(kernel(), os.path.dirname(os.path.abspath(__file__)),
                        backends=GPU_BACKENDS)
