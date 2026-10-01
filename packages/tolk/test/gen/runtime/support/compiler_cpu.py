"""Goldens of tinygrad/runtime/support/compiler_cpu.py: the kernels that the
compiler suite renders with Clang, compiles and runs.

The input golden `kernels` holds one `Ops.LINEAR` per kernel, whose arg is
the kernel's name and whose sources are its nodes, linearized as `to_program`
linearizes them for Clang:
- `add`, the sum of two buffers of 16 floats, which test/device/cpu/test_cpu.py
  compiles for x86_64 with and without AVX;
- `sqrt`, the square roots of a buffer of 16 floats;
- `half_add`, the sum of two buffers of one half, which
  test/null/test_compile_failures.py compiles for Apple's processors;
- `muladd`, a product plus a buffer, 16 floats, whose rendered expression a
  compiler that contracts would fuse into one multiply-add.
"""

from golden import graph
from tinygrad import Tensor, dtypes
from tinygrad.codegen import full_rewrite_to_sink, line_rewrite, pm_alloc_to_buf, pm_linearize_cleanups
from tinygrad.codegen.late.linearizer import linearize
from tinygrad.helpers import Target
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.uop.ops import Ops, UOp


def empty(*shape, dtype=dtypes.float):
    return Tensor.empty(*shape, device="NULL", dtype=dtype)


def linearized(out, ren):
    lin, _ = Tensor.linear_with_vars(out)
    (ast,) = [c.src[0] for c in lin.src if c.op is Ops.CALL and c.src[0].op is Ops.SINK]
    full = full_rewrite_to_sink(ast, ren, optimize=True)
    return line_rewrite(linearize(full), pm_linearize_cleanups + pm_alloc_to_buf)


KERNELS = [
    ("add", lambda: empty(16) + empty(16)),
    ("sqrt", lambda: empty(16).sqrt()),
    ("half_add", lambda: empty(1, dtype=dtypes.half) + empty(1, dtype=dtypes.half)),
    ("muladd", lambda: empty(16) * empty(16) + empty(16)),
]


@graph
def kernels():
    ren = ClangRenderer(Target("CPU", "CLANG", "x86_64,x86-64"))
    return UOp.sink(*[UOp(Ops.LINEAR, src=tuple(linearized(build(), ren)), arg=name) for name, build in KERNELS])
