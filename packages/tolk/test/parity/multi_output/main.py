#!/usr/bin/env python3
"""Parity case: b[i] = a[i] + 1.0; c[i] = a[i] * 2.0, 1 Global range, 2 stores."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import dump  # noqa: E402

from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType, ParamArg  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402


def kernel():
    p0 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(0))
    p1 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(1))
    p2 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(2))
    r0 = UOp.range(256, 0, AxisType.GLOBAL)
    ld_a = p0.index(r0, ptr=True).load()
    st1 = p1.index(r0, ptr=True).store(ld_a + UOp.const(dtypes.float32, 1.0))
    e1 = st1.end(r0)
    st2 = p2.index(r0, ptr=True).store(ld_a * UOp.const(dtypes.float32, 2.0))
    e2 = st2.end(r0)
    return UOp.sink(
        e1, e2,
        arg=KernelInfo(
            name="multi_output",
            axis_types=(AxisType.GLOBAL,),
            opts_to_apply=(),
        ),
    )


if __name__ == "__main__":
    dump(kernel(), os.path.dirname(os.path.abspath(__file__)),
         stages=("stage5", "stage7"))
