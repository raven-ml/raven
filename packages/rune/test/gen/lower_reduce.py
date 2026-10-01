"""The kernels tinygrad schedules for the reductions, scans and arg-reductions
whose lowering agrees with it: sums and products whose accumulator is the
operand's dtype, a float sum with the +0 the lowering adds, and the extremes and
arg-reductions of integers; and for the positions that sort integers, the
lowering's packing of each key above its position around tinygrad's sorting
network. Each operand is a 4x4 buffer on the CPU, but for a scan long enough to
run in two stages, whose result is made contiguous as the lowering's is."""

from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp

from golden import graph


def operand(dtype=dtypes.float32, shape=(4, 4)):
    return Tensor.empty(*shape, dtype=dtype, device="CPU")


def kernels(t):
    """The kernel sinks of the schedule that realizes `t`, in order."""
    linear, _ = t.linear_with_vars()
    return UOp.sink(*[call.src[0] for call in linear.src])


@graph
def sum_axis(): return kernels(operand().sum(1) + 0.0)
@graph
def sum_all(): return kernels(operand().sum() + 0.0)
@graph
def sum_uint(): return kernels(operand(dtypes.uint32).sum(1))
@graph
def prod_axis(): return kernels(operand().prod(1))
@graph
def max_int(): return kernels(operand(dtypes.int32).max(1))
@graph
def min_int(): return kernels(operand(dtypes.int32).min(1))
@graph
def cumsum(): return kernels(operand().cumsum(1) + 0.0)
@graph
def cumsum_uint(): return kernels(operand(dtypes.uint32).cumsum(1))
@graph
def cumsum_long(): return kernels(operand(dtypes.uint32, (2, 600)).cumsum(1).contiguous())
@graph
def cumprod(): return kernels(operand().cumprod(1))
@graph
def cummax_int(): return kernels(operand(dtypes.int32).cummax(1)[0])
@graph
def cummin_int(): return kernels(operand(dtypes.int32).cummin(1)[0])
@graph
def argmax_int(): return kernels(operand(dtypes.int32).argmax(1).cast(dtypes.int64))
@graph
def argmin_int(): return kernels(operand(dtypes.int32).argmin(1).cast(dtypes.int64))


def packed_positions(x, descending):
    """The positions that sort the int32 `x` along its last axis: each key, its
    sign bit flipped, packed above its position, complemented when descending,
    sorted by tinygrad's network, and the positions read back."""
    n = x.shape[-1]
    low = (n - 1).bit_length()
    mask = (1 << low) - 1
    keys = (x ^ dtypes.int32.min).bitcast(dtypes.uint32).cast(dtypes.int64)
    ranks = Tensor.arange(n, dtype=dtypes.int64).reshape(1, n)
    tie = (lambda r: mask - r) if descending else (lambda r: r)
    packed = ((keys << low) | tie(ranks)).contiguous()
    return tie(packed.sort(-1, descending=descending)[0] & mask)


@graph
def argsort_int(): return kernels(packed_positions(operand(dtypes.int32), False))
@graph
def argsort_int_descending(): return kernels(packed_positions(operand(dtypes.int32), True))
