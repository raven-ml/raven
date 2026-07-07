#!/usr/bin/env python3
"""Parity case: c[i] = sqrt(a[i]), 1 Global range, unary SQRT."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import dump  # noqa: E402

from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType, ParamArg  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402


def kernel():
    p0 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(0))
    p1 = UOp(Ops.PARAM, dtypes.float32.ptr(), (), ParamArg(1))
    r0 = UOp.range(256, 0, AxisType.GLOBAL)
    ld = p0.index(r0, ptr=True).load()
    sq = UOp(Ops.SQRT, dtypes.float32, (ld,))
    st = p1.index(r0, ptr=True).store(sq)
    end = st.end(r0)
    return UOp.sink(
        end,
        arg=KernelInfo(
            name="elementwise_sqrt",
            axis_types=(AxisType.GLOBAL,),
            opts_to_apply=(),
        ),
    )


if __name__ == "__main__":
    dump(kernel(), os.path.dirname(os.path.abspath(__file__)),
         stages=("stage5", "stage7"))
