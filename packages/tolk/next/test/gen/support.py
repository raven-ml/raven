"""Goldens of the test-support library: sample graphs in the graph format."""

from golden import graph
from graph import boundary
from tinygrad import Tensor, dtypes
from tinygrad.schedule.prepare import prepare_rangeify
from tinygrad.schedule.rangeify import get_kernel_graph


def sum_of_cast_program():
    x = Tensor.empty(16, device="CPU", dtype=dtypes.half)
    return boundary(x.float().sum())


@graph
def matmul():
    a, b = Tensor.empty(4, 8, device="CPU"), Tensor.empty(8, 3, device="CPU")
    return boundary(a @ b)


@graph
def sum_of_cast():
    return sum_of_cast_program()


@graph
def sum_of_cast_kernels():
    return get_kernel_graph(prepare_rangeify(sum_of_cast_program()))
