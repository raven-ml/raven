"""The kernels tinygrad schedules for the index operations whose lowering
agrees with it: a pad whose fill is not -0., a concatenation of pieces of one
length, and the windows of an unfold. Each operand is a buffer on the CPU."""

from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp

from golden import graph


def operand(*shape):
    return Tensor.empty(*shape, dtype=dtypes.float32, device="CPU")


def kernels(t):
    """The kernel sinks of the schedule that realizes `t` into a buffer of its
    own shape, in order: a result ending in a movement is a view of the buffer
    realized before it, so it is made contiguous."""
    linear, _ = t.contiguous().linear_with_vars()
    return UOp.sink(*[call.src[0] for call in linear.src])


@graph
def pad_zero(): return kernels(operand(4, 4).pad(((1, 2), (0, 3))))
@graph
def pad_value(): return kernels(operand(4, 4).pad(((1, 2), (0, 3)), value=1.5))
@graph
def cat_equal(): return kernels(operand(2, 4).cat(operand(2, 4), operand(2, 4), dim=0))
@graph
def unfold():
    # extract_patches ~kernel_size:[|2; 3|] ~stride:[|1; 2|] ~dilation:[|2; 1|]
    # ~padding:[|(1, 0); (1, 1)|] of a [|2; 5; 6|] operand.
    windows = operand(2, 5, 6).pad(((0, 0), (1, 0), (1, 1)))._pool((2, 3), (1, 2), (2, 1))
    return kernels(windows.permute(0, 3, 4, 1, 2).reshape(2, 6, -1))
