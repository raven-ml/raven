#!/usr/bin/env python3
"""Parity case: c[i] = a[i] + b[i] over a single Global range of 256 elements.

Paired with main.ml. Run to regenerate *.expected files for all backends.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import dump  # noqa: E402

from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType  # noqa: E402
from tinygrad.dtype import dtypes  # noqa: E402


def kernel():
    p0 = UOp.param(0, dtypes.float32, shape=(-1,))
    p1 = UOp.param(1, dtypes.float32, shape=(-1,))
    p2 = UOp.param(2, dtypes.float32, shape=(-1,))
    r0 = UOp.range(256, 0, AxisType.GLOBAL)
    add = p0.index(r0).load() + p1.index(r0).load()
    end = p2.index(r0).store(add).end(r0)
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
