"""Goldens of tinygrad/codegen/late/gater.py: moving gates from indices onto
loads and stores.

`<kernel>.golden` is the sink that `pm_move_gates_from_index` receives when a
kernel is compiled, and `<kernel>_moved.golden` its result. The hand-built
cases come as a table naming each case by its position (`moves`), an input
sink whose sources are the cases (`moves_input`), and an output sink whose
sources are their results, in the same order (`moves_output`).
"""

from golden import graph, table
from graph import kernels, stage
from tinygrad import Tensor, dtypes
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.late.gater import pm_move_gates_from_index
from tinygrad.dtype import Invalid
from tinygrad.helpers import Target
from tinygrad.renderer.cstyle import ClangRenderer, CUDARenderer
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp, graph_rewrite

CPU = ClangRenderer(Target("CPU", "CLANG", "x86_64,x86-64"))
CUDA = CUDARenderer(Target("CUDA", "CUDA", "sm_80"))


def move(u): return graph_rewrite(u, pm_move_gates_from_index)


def noopt(kernel): return kernel.replace(arg=KernelInfo(opts_to_apply=()))


# The host upcasts no kernel without a reduce, so a padding asks for its
# vectors of 4, whose loads the pass gates.
def vectors(kernel): return kernel.replace(arg=KernelInfo(opts_to_apply=(Opt(OptOps.SPLIT, 0, (4, AxisType.UPCAST)),)))


KERNELS = {
    "pad": (lambda: vectors(kernels(Tensor.empty(5, device="CPU").pad((2, 1)) + 1)[-1]), CPU),
    "pad_value": (lambda: vectors(kernels(Tensor.empty(5, device="CPU").pad((2, 1), value=1.0) * 2)[-1]), CPU),
    "conv": (lambda: noopt(kernels(Tensor.empty(1, 2, 6, 6, device="CPU").conv2d(
        Tensor.empty(3, 2, 3, 3, device="CPU"), padding=1))[-1]), CPU),
    "pad_cuda": (lambda: kernels(Tensor.empty(30, device="CPU").pad((2, 1)) + 1)[-1], CUDA),
    "sum_group": (lambda: kernels(Tensor.empty(4096, device="CPU").sum())[-1], CUDA),
}


def declare(name, kernel, renderer):
    def given(): return stage("move gates from index", kernel(), renderer)
    def moved(): return move(given())
    given.__name__, moved.__name__ = name, f"{name}_moved"
    graph(given)
    graph(moved)


for name, (kernel, renderer) in KERNELS.items():
    declare(name, kernel, renderer)


# Hand-built cases

def cases():
    buf = UOp.param(0, dtypes.float, 64)
    half = UOp.param(1, dtypes.half, 64)
    g = UOp.variable("g", False, True, dtypes.bool)
    h = UOp.variable("h", False, True, dtypes.bool)
    i = UOp.variable("i", 0, 63)
    j = UOp.variable("j", 0, 7)
    x = UOp.variable("x", -1.0, 1.0, dtypes.float)
    y = UOp.variable("y", -1.0, 1.0, dtypes.double)
    one = UOp.const(1.0, dtypes.float)
    gated = lambda c, v: c.where(v, UOp.invalid())
    at = lambda b, *idx: UOp(Ops.INDEX, src=(b, *idx))
    load = at(buf, i).load(UOp.const(0.0, dtypes.float), g)
    return [
        # a storage indexed by two gated indices under one gate
        ("two_indices_load", at(buf, gated(g, j), gated(g, i)).load()),
        ("two_indices_store", at(buf, gated(g, j), gated(g, i)).store(one)),
        ("two_indices_two_gates", at(buf, gated(g, j), gated(h, i)).load()),
        ("second_index_gated", at(buf, j, gated(g, i)).load()),
        ("three_indices", at(buf, gated(g, j), gated(g, i), j).load()),
        # the first index gated
        ("index_load", at(buf, gated(g, i)).load()),
        ("index_store", at(buf, gated(g, i)).store(x)),
        ("shrink_load", UOp(Ops.SHRINK, src=(buf, gated(g, i), UOp.const(4))).load()),
        ("shrink_store", UOp(Ops.SHRINK, src=(buf, gated(g, i), UOp.const(4))).store(x)),
        ("comparison_gate", at(buf, gated(i < 32, i)).load()),
        ("half_load", at(half, gated(g, i)).load()),
        ("gated_load_kept", at(buf, gated(g, i)).load(UOp.const(0.0, dtypes.float), g)),
        ("valid_index_kept", at(buf, i).load()),
        ("gated_value_kept", at(buf, i).store(gated(g, x))),
        # a selection around a load gated by its condition
        ("where_invalid", g.where(load, UOp.invalid())),
        ("where_constant", g.where(load, UOp.const(2.5, dtypes.float))),
        ("where_literal", UOp(Ops.WHERE, src=(g, load, UOp.const(2.5)))),
        ("where_node", g.where(load, x)),
        ("where_of_cast_back", g.where(load.cast(dtypes.double), x.cast(dtypes.double))),
        ("where_cast_other", g.where(load, y.cast(dtypes.float))),
        ("where_wider", g.where(load, y.cast(dtypes.float)).cast(dtypes.double)),
        ("where_of_cast", g.where(load.cast(dtypes.double), y)),
        ("where_of_cast_constant", g.where(load.cast(dtypes.double), UOp.const(-0.5, dtypes.double))),
        ("where_of_cast_literal", UOp(Ops.WHERE, src=(g, load.cast(dtypes.double), UOp.const(-0.5)))),
        ("where_of_half", g.where(at(half, i).load(UOp.const(0.0, dtypes.half), g).cast(dtypes.float), x)),
        ("where_negated", g.where(x, at(buf, i).load(UOp.const(0.0, dtypes.float), g.logical_not()))),
        ("where_negated_invalid", g.where(UOp.invalid(), at(buf, i).load(UOp.const(0.0, dtypes.float), g.logical_not()))),
        ("where_other_gate_kept", h.where(load, x)),
        ("where_ungated_kept", g.where(at(buf, i).load(), x)),
        ("where_negated_other_gate_kept", g.where(x, at(buf, i).load(UOp.const(0.0, dtypes.float), h.logical_not()))),
        # gates moved, then selections folded
        ("index_then_where", g.where(at(buf, gated(g, i)).load(), x)),
        ("index_then_where_negated", g.where(x, at(buf, gated(g.logical_not(), i)).load())),
    ]


@table
def moves():
    return ["case", "src"], [(case, str(k)) for k, (case, _) in enumerate(cases())]


@graph
def moves_input(): return UOp.sink(*[u for _, u in cases()])


@graph
def moves_output(): return UOp.sink(*[move(u) for _, u in cases()])
