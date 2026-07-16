#!/usr/bin/env python3
"""Parity case: c[i] = (float32)a_f16[i] + b[i], 1 Global range, mixed dtypes."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import dump  # noqa: E402

from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402


def kernel():
    p0 = UOp.param(0, dtypes.half, shape=(-1,))
    p1 = UOp.param(1, dtypes.float32, shape=(-1,))
    p2 = UOp.param(2, dtypes.float32, shape=(-1,))
    r0 = UOp.range(256, 0, AxisType.GLOBAL)
    ld_a = p0.index(r0).load()
    cast_a = ld_a.cast(dtypes.float32)
    ld_b = p1.index(r0).load()
    add = cast_a + ld_b
    st = p2.index(r0).store(add)
    end = st.end(r0)
    return UOp.sink(
        end,
        arg=KernelInfo(
            name="elementwise_cast_f16",
            axis_types=(AxisType.GLOBAL,),
            opts_to_apply=(),
        ),
    )


if __name__ == "__main__":
    dump(kernel(), os.path.dirname(os.path.abspath(__file__)),
         stages=("stage5", "stage7"))
