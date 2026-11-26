"""Goldens of Shape's calls in tinygrad: the graphs tinygrad/uop/ops.py's
movement, construction and storage builders make, and graphs before and after
tinygrad/uop/movement.py's mop_cleanup.

A graph golden named after a builder is a sink in the graph format of
test/README.md. A golden of mop_cleanup is the sink of an input graph and of
its rewrite by mop_cleanup to a fixed point, in that order. The suite builds
the same input with tolk's constructors and compares its rewrite with the
golden.
"""

import dataclasses
import itertools

from golden import graph
from graph import toposort
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.uop.movement import mop_cleanup
from tinygrad.uop.ops import AxisType, Ops, UOp, graph_rewrite
from tinygrad.uop.weak import pm_uncast_const


def rewritten(u):
    return UOp.sink(u, graph_rewrite(u, mop_cleanup))


def stored(*shape):
    return UOp.param(0, dtypes.float32, shape)


def element(src, i):
    return UOp(Ops.INDEX, src=(src, UOp.const(i)))


# Shrinks

@graph
def merge_two_shrinks():
    inner = stored(8, 10)._mop(Ops.SHRINK, ((1, 6), (2, 7)))
    return rewritten(inner._mop(Ops.SHRINK, ((2, 2), (3, 4))))


@graph
def merge_three_shrinks():
    x = stored(16)._mop(Ops.SHRINK, ((1, 12),))._mop(Ops.SHRINK, ((2, 8),))
    return rewritten(x._mop(Ops.SHRINK, ((3, 4),)))


@graph
def merge_shrinks_of_symbolic_starts():
    offset, size = UOp.variable("o", 0, 4), UOp.variable("n", 1, 3)
    inner = stored(16)._mop(Ops.SHRINK, ((offset, 8),))
    return rewritten(inner._mop(Ops.SHRINK, ((2, size),)))


# Reshapes

@graph
def merge_two_reshapes():
    return rewritten(stored(8, 10)._mop(Ops.RESHAPE, (80,))._mop(Ops.RESHAPE, (4, 20)))


@graph
def merge_reshapes_back_to_the_source_shape():
    x = stored(8, 10)
    return rewritten(x._mop(Ops.RESHAPE, (80,))._mop(Ops.RESHAPE, (8, 10)))


@graph
def remove_a_reshape_to_the_source_shape():
    x = stored(8, 10)._mop(Ops.PERMUTE, (1, 0))
    return rewritten(x._mop(Ops.RESHAPE, (10, 8)))


# Permutes

@graph
def merge_two_permutes():
    return rewritten(stored(2, 3, 4)._mop(Ops.PERMUTE, (1, 2, 0))._mop(Ops.PERMUTE, (0, 2, 1)))


@graph
def merge_inverse_permutes_into_the_source():
    return rewritten(stored(2, 3, 4)._mop(Ops.PERMUTE, (1, 2, 0))._mop(Ops.PERMUTE, (2, 0, 1)))


@graph
def remove_the_identity_permute():
    return rewritten(stored(2, 3)._mop(Ops.PERMUTE, (0, 1)))


@graph
def keep_a_permute_that_moves_an_axis():
    return rewritten(stored(2, 3)._mop(Ops.PERMUTE, (1, 0)))


# Stacks of elements

@graph
def stack_the_elements_of_a_node_in_order():
    x = stored(3)
    return rewritten(UOp(Ops.STACK, src=tuple(element(x, i) for i in range(3))))


@graph
def keep_a_stack_of_elements_out_of_order():
    x = stored(3)
    return rewritten(UOp(Ops.STACK, src=tuple(element(x, i) for i in (1, 0, 2))))


@graph
def keep_a_stack_of_some_elements():
    x = stored(3)
    return rewritten(UOp(Ops.STACK, src=tuple(element(x, i) for i in range(2))))


# Indexing

@graph
def index_a_stack_by_a_constant():
    a, b = UOp.variable("a", 0, 9), UOp.variable("b", 0, 9)
    return rewritten(UOp(Ops.INDEX, src=(UOp.stack(a, b), UOp.const(1))))


@graph
def index_a_stack_by_a_constant_and_further_indices():
    rows = UOp.stack(stored(4), UOp.param(1, dtypes.float32, 4))
    return rewritten(UOp(Ops.INDEX, src=(rows, UOp.const(1), UOp.variable("j", 0, 3))))


@graph
def index_an_index_by_scalars():
    i, j = UOp.variable("i", 0, 3), UOp.variable("j", 0, 4)
    return rewritten(UOp(Ops.INDEX, src=(UOp(Ops.INDEX, src=(stored(4, 5), i)), j)))


@graph
def index_an_index_of_a_shaped_index():
    table = UOp.param(1, dtypes.int32, (4, 5))
    i, j = UOp.variable("i", 0, 3), UOp.variable("j", 0, 4)
    return rewritten(UOp(Ops.INDEX, src=(UOp(Ops.INDEX, src=(stored(20), table)), i, j)))


@graph
def keep_an_index_of_a_shaped_index_by_fewer_indices():
    table = UOp.param(1, dtypes.int32, (4, 5))
    return rewritten(UOp(Ops.INDEX, src=(UOp(Ops.INDEX, src=(stored(20), table)), UOp.variable("i", 0, 3))))


# Graphs: what the constructors build

def named(u, name):
    """`u`, a define, with the name `name`."""
    return u.replace(arg=dataclasses.replace(u.arg, name=name))


@graph
def movement():
    p = UOp.param(0, dtypes.float, (2, 3, 4))
    return UOp.sink(p.reshape((6, 4)), p.reshape((-1, 2)), p.reshape((2, 3, 4)), p.permute((2, 0, 1)), p.permute((0, -1, 1)),
                    p.permute((0, 1, 2)), p.flip(0, 2), p.flip(-1), p.shrink(((0, 1), None, (1, 3))),
                    p.shrink((None, None, None)), p.shrink_to((1, None, 2)), p.pad(((1, 0), None, (0, 2))),
                    p.pad_to((None, 5, 6)), p.flatten(), p.flatten(1), p.flatten(0, 1), p.reshape((2, 12)).unflatten(1, (3, 4)))


@graph
def expansion():
    p = UOp.param(0, dtypes.float, (3, 1))
    s = UOp.param(1, dtypes.float, (1, 4, 1))
    return UOp.sink(p.expand((2, 3, 1)), p.expand((3, 5)), p.expand((2, 3, 5)), p.expand((-1, 5)), s.expand((2, 4, 3)),
                    UOp.const(1.0).expand((4, 8)), UOp.const(1.0).expand(()))


@graph
def squeezes():
    p = UOp.param(0, dtypes.float, (1, 3, 1, 2))
    return UOp.sink(p.squeeze(), p.squeeze(0), p.squeeze(2), p.squeeze(1), p.squeeze(-2))


@graph
def axes():
    p = UOp.param(0, dtypes.float, (2, 3, 4))
    return UOp.sink(p.unsqueeze(0), p.unsqueeze(2), p.unsqueeze(-1), p.transpose(), p.transpose(0, 2), p.transpose(-1, -2),
                    *p.split([1, 2], 1), *p.split([2, 2], dim=-1), *p.split([2]))


@graph
def stacks():
    a, b = UOp.param(0, dtypes.float, (2, 3)), UOp.param(1, dtypes.float, (2, 3))
    h = UOp.param(2, dtypes.half, (2, 3))
    return UOp.sink(a.stack(b), a.stack(b, dim=1), a.stack(b, dim=-1), a.stack(h), a.stack(b, a), UOp.stack(UOp.invalid(), UOp.const(1.0).cast(dtypes.float)),
                    UOp.const(1).broadcast(3), UOp.const(1).broadcast(1))


@graph
def concatenation():
    a, b = UOp.param(0, dtypes.float, (2, 3)), UOp.param(1, dtypes.float, (2, 3))
    c = UOp.param(2, dtypes.float, (4, 3))
    return UOp.sink(a.cat(b), a.cat(b, dim=1), a.cat(c), a.cat(c, b), a.cat(b, dim=-1))


@graph
def pools():
    p = UOp.param(0, dtypes.float, (2, 5, 6))
    return UOp.sink(p._pool((3,)), p._pool((3,), 2), p._pool((2, 2), (1, 2), (2, 1)), p._pool((5, 6)), p._pool((2,), 3),
                    p.repeat((2, 1, 1)), p.repeat((3, 1, 1, 2)))


@graph
def running():
    p = UOp.param(0, dtypes.float, (3, 5))
    return UOp.sink(p._split_cumalu(1, Ops.ADD), p._split_cumalu(0, Ops.MAX), p._split_cumalu(-1, Ops.MUL))


@graph
def aranges():
    return UOp.sink(UOp.arange(5), UOp.arange(2, 9, 3), UOp.arange(4, 0, -1), UOp.arange(3, dtype=dtypes.int64),
                    UOp.arange(3, dtype=dtypes.float), UOp.arange(0))


@graph
def padding():
    p = UOp.param(0, dtypes.float, (4, 4))
    return UOp.sink(p.pad(((1, 1), (2, 0)), value=0.0), p.pad(((1, 1), None), value=1.5), p.pad(((-1, 2), (0, -2))),
                    p.pad_to((6, None), value=2.0), p.pad(((1, -1), (0, 2)), value=-1.0))


@graph
def reductions():
    p = UOp.param(0, dtypes.float, (2, 3, 4))
    q = UOp.param(1, dtypes.float, (2, 1, 4))
    return UOp.sink(p._rop(Ops.ADD, (0,)), p._rop(Ops.MAX, (2, 0)), p._rop(Ops.MUL, (1,)), q._rop(Ops.ADD, (1,)),
                    q._rop(Ops.ADD, (1, 2)), p._rop(Ops.ADD, ()))


@graph
def constants_like():
    p = UOp.param(0, dtypes.float, (2, 3))
    r = UOp.range(4, 0)
    return UOp.sink(p.const_like(0), p.const_like(True, dtypes.bool), r.const_like(3), UOp.const(1).const_like(2),
                    UOp.stack(UOp.const(1.0).cast(dtypes.half), UOp.const(2.0).cast(dtypes.half)).vconst_like(0))


@graph
def contraction():
    r0, r1 = UOp.range(2, 0, AxisType.UPCAST), UOp.range(3, 1, AxisType.UPCAST)
    return UOp.sink((r0 * 3 + r1).contract(r0, r1), (r0 + 1).contract(r0))


@graph
def storage():
    return UOp.sink(UOp.param(2, dtypes.float, (2, 3, 4), device="CPU"),
                    UOp.param(3, dtypes.half), UOp.param(4, dtypes.float, (4,), vmin_vmax=(0.0, 1.0), name="w", volatile=True),
                    UOp.placeholder((2, 3), dtypes.weakint, 5), UOp.placeholder((8,), dtypes.float, 6, addrspace=AddrSpace.LOCAL),
                    named(UOp.placeholder((4,), dtypes.half, 7, addrspace=AddrSpace.REG), "acc"),
                    UOp.alloc((), dtypes.int, slot=9),
                    UOp.new_buffer("CPU", 4, dtypes.float, 11).view_as((2, 2)))


@graph
def storage_like():
    p = UOp.param(0, dtypes.float, (2, 3), device="CPU")
    v = UOp.variable("n", 1, 8)
    return UOp.sink(p.param_like(5), v.param_like(6), v.bind(3).param_like(7), p.placeholder_like(9),
                    p.alloc_like(10), p.alloc_like(11, addrspace=AddrSpace.LOCAL))


@graph
def calls():
    body = UOp.param(0, dtypes.float, 4).index(UOp.range(4, 0)).store(1.0).end(UOp.range(4, 0)).sink()
    a = UOp.param(0, dtypes.float, 4, device="CPU")
    return UOp.sink(body.call(a), body.call(a, name="fill", precompile=True), UOp.custom_function("f", UOp.const(0, dtypes.uint64), dtype=dtypes.int).call(),
                    UOp.param(0, dtypes.float, 4, device="CPU").store_call(UOp.param(1, dtypes.float, 4, device="CPU")))


@graph
def custom_kernels():
    a, b = UOp.param(0, dtypes.float, 4, device="CPU"), UOp.param(1, dtypes.float, 4, device="CPU")
    return UOp.sink(*UOp.custom_kernel(a, b, fxn=lambda x, y: x.index(UOp.range(4, 0)).store(y.index(UOp.range(4, 0)).load()).end(UOp.range(4, 0)).sink()))


@graph
def variables_and_binding():
    n, m = UOp.variable("n", 1, 8), UOp.variable("m", 0, 4)
    e = n.bind(3) * m.bind(2) + n
    unbound, _ = e.unbind_all()
    return UOp.sink(n.bind(3), n.bind(3).unbound(), unbound, *e.variables())


# Storage without a given slot takes the next number of a counter. These graphs
# number theirs from MINTED, in the order the graph lists them, as the suite
# renumbers its own.
MINTED = 1000


def minted(fn):
    def body():
        UOp.unique_num = itertools.count(MINTED)
        sink = fn()
        slots = [u.arg.slot for u in toposort(sink) if u.op is Ops.ALLOC]
        if list(dict.fromkeys(slots)) != list(range(MINTED, MINTED + len(set(slots)))):
            raise RuntimeError(f"{fn.__name__} lists its minted slots out of order: {slots}")
        return sink
    body.__name__ = fn.__name__
    return graph(body)


@minted
def clones():
    p = UOp.param(0, dtypes.float, (2, 3), device="CPU")
    return UOp.sink(p.clone(), p.clone("CUDA"), p.empty_like(), p.empty_like(dtypes.half, "CUDA"),
                    UOp.empty((2, 3), dtype=dtypes.int, device="CPU"))


@minted
def outputs():
    a = UOp.param(0, dtypes.float, 4, device="CPU")
    return UOp.sink(*UOp.call_with_outputs((a + 1.0, a * 2.0), a, name="two"), a.call_with_output(a, precompile=True))


# Symbolic sizes: their shape arguments are simplified by the symbolic rules

@graph
def symbolic_storage():
    n = UOp.variable("n", 1, 8)
    return UOp.sink(UOp.param(0, dtypes.float, (2, n)), UOp.param(1, dtypes.int, n), UOp.alloc((2, n), dtypes.weakfloat, slot=8))


@minted
def symbolic_outputs():
    a = UOp.param(0, dtypes.float, 4, device="CPU")
    n = UOp.variable("n", 1, 4)
    b = UOp.param(1, dtypes.float, (n,), device="CPU")
    formal = UOp.param(0, dtypes.int, vmin_vmax=(1, 8), name="d", addrspace=AddrSpace.ALU)
    return UOp.sink(*UOp.call_with_outputs((b + 1.0,), a, b, output_pos=(0,)),
                    UOp.const(1.0).cast(dtypes.float).expand((formal,)).call_with_output(UOp.const(5)),
                    UOp.empty((2, n), dtype=dtypes.int, device="CPU"))


# Weak constants: graphs before and after pm_uncast_const

def uncast(u):
    return UOp.sink(u, graph_rewrite(u, pm_uncast_const))


def small(name="x", dtype=dtypes.int8):
    return UOp.variable(name, 0, 10, dtype)


def weak(name="w"):
    return UOp.variable(name, 0, 4)


@graph
def uncast_a_committed_literal():
    return uncast(small(dtype=dtypes.int32) + UOp.const(1, dtypes.int32))


@graph
def uncast_keeps_literals_with_no_committed_peer():
    return uncast(UOp.const(1, dtypes.int32) + UOp.const(2, dtypes.int32))


@graph
def uncast_keeps_a_shifted_literal():
    count = UOp.variable("s", 0, 31, dtypes.uint32)
    return uncast(UOp(Ops.SHL, src=(UOp.const(1, dtypes.int32), count)))


@graph
def uncast_keeps_a_cast_of_an_expression():
    return uncast(small(dtype=dtypes.int32) + (weak() + 1).cast(dtypes.int32))


@graph
def uncast_keeps_a_shifted_literal_of_its_peer_s_type():
    count = UOp.variable("s", 0, 31, dtypes.int32)
    return uncast(UOp(Ops.SHL, src=(UOp.const(1, dtypes.int32), count)))


@graph
def uncast_keeps_a_literal_that_widens_a_comparison():
    return uncast(UOp(Ops.CMPLT, src=(small(dtype=dtypes.int16), UOp.const(1, dtypes.int32))))


@graph
def uncast_drops_a_cast_that_fits():
    return uncast(small("u", dtypes.uint8) < UOp.cconst(44, dtypes.uint8))
