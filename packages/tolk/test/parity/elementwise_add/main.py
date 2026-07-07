#!/usr/bin/env python3
"""Parity case: c[i] = a[i] + b[i] over a single Global range of 256 elements.

Paired with main.ml. Run to regenerate *.expected files for all backends.
"""

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
    add = p0.index(r0, ptr=True).load() + p1.index(r0, ptr=True).load()
    end = p2.index(r0, ptr=True).store(add).end(r0)
    return UOp.sink(
        end,
        arg=KernelInfo(
            name="elementwise_add",
            axis_types=(AxisType.GLOBAL,),
            opts_to_apply=(),
        ),
    )


if __name__ == "__main__":
    dump(kernel(), os.path.dirname(os.path.abspath(__file__)),
         stages=("stage5", "stage7"))
