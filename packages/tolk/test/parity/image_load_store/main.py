#!/usr/bin/env python3
# Reference: tinygrad d0c9745274335e44b5dd7c15b8422c2b684a3ed9.
"""Image reads and writes through coalescing, coordinate selection and rendering."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import dump
from tinygrad.uop.ops import UOp, AxisType, KernelInfo
from tinygrad.dtype import dtypes
from tinygrad.helpers import Context, Target
from tinygrad.renderer.cstyle import OpenCLRenderer

RENDERER = OpenCLRenderer(Target("CL", arch="IMAGE_PITCH_ALIGNMENT=8"))
DTYPES = {"opencl_float": dtypes.float, "opencl_half": dtypes.half}

def kernel(dtype):
    src = UOp.param(0, dtype, shape=(256,))
    dst = UOp.param(1, dtype, shape=(256,))
    y = UOp.range(8, 0, AxisType.GLOBAL)
    x = UOp.range(8, 1, AxisType.GLOBAL)
    c = UOp.range(4, 2, AxisType.UPCAST)
    index = y * 32 + x * 4 + c
    value = src.index(index).load() + UOp.const(0.5, dtype)
    return UOp.sink(dst.index(index).store(value).end(y, x, c),
                   arg=KernelInfo(name="image_copy", opts_to_apply=()))

def generate(out_dir):
    with Context(IMAGE=2):
        for name, dtype in DTYPES.items():
            dump(kernel(dtype), out_dir, stages=("stage5", "stage7"),
                 backends={name: RENDERER}, optimize=False)

if __name__ == "__main__":
    generate(os.path.dirname(os.path.abspath(__file__)))
