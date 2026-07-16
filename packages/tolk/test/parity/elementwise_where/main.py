#!/usr/bin/env python3
"""Parity case: c[i] = (a[i] > 0) ? a[i] : 0.0 (ReLU), 1 Global range."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import dump  # noqa: E402

from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402


def kernel():
    p0 = UOp.param(0, dtypes.float32, shape=(-1,))
    p1 = UOp.param(1, dtypes.float32, shape=(-1,))
    r0 = UOp.range(256, 0, AxisType.GLOBAL)
    ld = p0.index(r0).load()
    zero = UOp.const(dtypes.float32, 0.0)
    cond = zero.alu(Ops.CMPLT, ld)  # 0.0 < a[i] => a[i] > 0
    val = cond.where(ld, zero)
    st = p1.index(r0).store(val)
    end = st.end(r0)
    return UOp.sink(
        end,
        arg=KernelInfo(
            name="elementwise_where",
            axis_types=(AxisType.GLOBAL,),
            opts_to_apply=(),
        ),
    )


if __name__ == "__main__":
    dump(kernel(), os.path.dirname(os.path.abspath(__file__)),
         stages=("stage5", "stage7"))
