"""Goldens of tinygrad/uop/movement.py: graphs before and after mop_cleanup.

A golden `<name>.golden` is the sink of an input graph and of its rewrite by
mop_cleanup to a fixed point, in that order. The suite builds the same input
with tolk's constructors and compares its rewrite with the golden.
"""

from golden import graph
from tinygrad.dtype import dtypes
from tinygrad.uop.movement import mop_cleanup
from tinygrad.uop.ops import Ops, UOp, graph_rewrite


def rewritten(u):
    return UOp.sink(u, graph_rewrite(u, mop_cleanup))


def storage(*shape):
    return UOp.param(0, dtypes.float32, shape)


def element(src, i):
    return UOp(Ops.INDEX, src=(src, UOp.const(i)))


# Shrinks

@graph
def merge_two_shrinks():
    inner = storage(8, 10)._mop(Ops.SHRINK, ((1, 6), (2, 7)))
    return rewritten(inner._mop(Ops.SHRINK, ((2, 2), (3, 4))))


@graph
def merge_three_shrinks():
    x = storage(16)._mop(Ops.SHRINK, ((1, 12),))._mop(Ops.SHRINK, ((2, 8),))
    return rewritten(x._mop(Ops.SHRINK, ((3, 4),)))


@graph
def merge_shrinks_of_symbolic_starts():
    offset, size = UOp.variable("o", 0, 4), UOp.variable("n", 1, 3)
    inner = storage(16)._mop(Ops.SHRINK, ((offset, 8),))
    return rewritten(inner._mop(Ops.SHRINK, ((2, size),)))


# Reshapes

@graph
def merge_two_reshapes():
    return rewritten(storage(8, 10)._mop(Ops.RESHAPE, (80,))._mop(Ops.RESHAPE, (4, 20)))


@graph
def merge_reshapes_back_to_the_source_shape():
    x = storage(8, 10)
    return rewritten(x._mop(Ops.RESHAPE, (80,))._mop(Ops.RESHAPE, (8, 10)))


@graph
def remove_a_reshape_to_the_source_shape():
    x = storage(8, 10)._mop(Ops.PERMUTE, (1, 0))
    return rewritten(x._mop(Ops.RESHAPE, (10, 8)))


# Permutes

@graph
def merge_two_permutes():
    return rewritten(storage(2, 3, 4)._mop(Ops.PERMUTE, (1, 2, 0))._mop(Ops.PERMUTE, (0, 2, 1)))


@graph
def merge_inverse_permutes_into_the_source():
    return rewritten(storage(2, 3, 4)._mop(Ops.PERMUTE, (1, 2, 0))._mop(Ops.PERMUTE, (2, 0, 1)))


@graph
def remove_the_identity_permute():
    return rewritten(storage(2, 3)._mop(Ops.PERMUTE, (0, 1)))


@graph
def keep_a_permute_that_moves_an_axis():
    return rewritten(storage(2, 3)._mop(Ops.PERMUTE, (1, 0)))


# Stacks of elements

@graph
def stack_the_elements_of_a_node_in_order():
    x = storage(3)
    return rewritten(UOp(Ops.STACK, src=tuple(element(x, i) for i in range(3))))


@graph
def keep_a_stack_of_elements_out_of_order():
    x = storage(3)
    return rewritten(UOp(Ops.STACK, src=tuple(element(x, i) for i in (1, 0, 2))))


@graph
def keep_a_stack_of_some_elements():
    x = storage(3)
    return rewritten(UOp(Ops.STACK, src=tuple(element(x, i) for i in range(2))))


# Indexing

@graph
def index_a_stack_by_a_constant():
    a, b = UOp.variable("a", 0, 9), UOp.variable("b", 0, 9)
    return rewritten(UOp(Ops.INDEX, src=(UOp.stack(a, b), UOp.const(1))))


@graph
def index_a_stack_by_a_constant_and_further_indices():
    rows = UOp.stack(storage(4), UOp.param(1, dtypes.float32, 4))
    return rewritten(UOp(Ops.INDEX, src=(rows, UOp.const(1), UOp.variable("j", 0, 3))))


@graph
def index_an_index_by_scalars():
    i, j = UOp.variable("i", 0, 3), UOp.variable("j", 0, 4)
    return rewritten(UOp(Ops.INDEX, src=(UOp(Ops.INDEX, src=(storage(4, 5), i)), j)))


@graph
def index_an_index_of_a_shaped_index():
    table = UOp.param(1, dtypes.int32, (4, 5))
    i, j = UOp.variable("i", 0, 3), UOp.variable("j", 0, 4)
    return rewritten(UOp(Ops.INDEX, src=(UOp(Ops.INDEX, src=(storage(20), table)), i, j)))


@graph
def keep_an_index_of_a_shaped_index_by_fewer_indices():
    table = UOp.param(1, dtypes.int32, (4, 5))
    return rewritten(UOp(Ops.INDEX, src=(UOp(Ops.INDEX, src=(storage(20), table)), UOp.variable("i", 0, 3))))
