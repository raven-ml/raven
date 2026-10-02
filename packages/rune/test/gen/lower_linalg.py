"""The kernels tinygrad schedules for the linear algebra whose lowering agrees
with it: a float32 matrix product with the +0 the lowering adds to every float
sum. The factorizations repeat their steps in a loop, which tinygrad's
schedules do not hold. Each operand is a buffer on the CPU."""

from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp

from golden import graph


def operand(*shape):
    return Tensor.empty(*shape, dtype=dtypes.float32, device="CPU")


def kernels(t):
    """The kernel sinks of the schedule that realizes `t` into a buffer of its
    own shape, in order."""
    linear, _ = t.contiguous().linear_with_vars()
    return UOp.sink(*[call.src[0] for call in linear.src])


@graph
def matmul(): return kernels(operand(4, 3) @ operand(3, 5) + 0.0)
@graph
def matmul_batched(): return kernels(operand(2, 1, 4, 3) @ operand(3, 3, 5) + 0.0)
