"""Goldens of tinygrad/codegen/simplify.py: range and reduction simplification.

Recorded kernels: `<kernel>.golden` is the sink that a pass running one of the
matchers receives when tinygrad compiles a `Tensor` program, and
`<kernel>_<result>.golden` what the matcher alone makes of it. The codegen
passes run on the CPU renderer; `pm_reduce_simplify` runs in the scheduler, on
the graph that `get_kernel_graph` hands to its reduce collapse.

Hand-built cases: for each matcher, a table naming each case by its position
(`<matcher>`), an input sink whose sources are the cases (`<matcher>_input`),
and an output sink whose sources are what the matcher makes of each case alone,
in the same order (`<matcher>_output`). A matcher that reads a context starts
each case with an empty one.
"""

from golden import graph, table
from graph import kernels, stage
from tinygrad import Tensor, dtypes
from tinygrad.codegen.simplify import (pm_flatten_range, pm_load_collapse, pm_reduce_collapse, pm_reduce_simplify,
                                       pm_reduce_unparented, pm_simplify_ranges, pm_split_ranges)
from tinygrad.dtype import Invalid
from tinygrad.helpers import DEV, Target
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp, graph_rewrite
import tinygrad.schedule.rangeify

DEV.value = "CPU"
CPU = ClangRenderer(Target("CPU", "CLANG", "x86_64,x86-64"))

SPLIT = pm_split_ranges + pm_flatten_range
SIMPLIFY = pm_flatten_range + pm_simplify_ranges


def split(u): return graph_rewrite(u, SPLIT, ctx={})
def simplify(u): return graph_rewrite(u, SIMPLIFY, ctx={})
def flatten(u): return graph_rewrite(u, pm_flatten_range)
def unparent(u): return graph_rewrite(u, pm_reduce_unparented)
def collapse(u): return graph_rewrite(u, pm_reduce_collapse)
def reduce_simplify(u): return graph_rewrite(u, pm_reduce_simplify)
def load_collapse(u): return graph_rewrite(u, pm_load_collapse)


def empty(*shape, dtype=dtypes.float): return Tensor.empty(*shape, dtype=dtype)


def at_reduce_collapse(program):
    """The graph that `get_kernel_graph` hands to its reduce collapse when the
    tensor `program()` is scheduled."""

    class Captured(Exception):
        pass

    rewrite = tinygrad.schedule.rangeify.graph_rewrite

    def capture(sink, *args, name=None, **kwargs):
        if name == "symbolic+reduce_collapse+debuf": raise Captured(sink)
        return rewrite(sink, *args, name=name, **kwargs)

    tinygrad.schedule.rangeify.graph_rewrite = capture
    try:
        program().schedule_linear()
    except Captured as e:
        return e.args[0]
    finally:
        tinygrad.schedule.rangeify.graph_rewrite = rewrite
    raise RuntimeError("scheduling the tensors runs no reduce collapse")


def codegen(name):
    return lambda program: stage(name, kernels(program())[-1], CPU)


# Recorded kernels: (the graph a pass receives, the matcher's result's name,
# the matcher).
KERNELS = {
    # a gather by an index tensor, and an embedding: a sum over the table's rows
    # of the row the index selects
    "gather": (codegen("load collapse"), lambda: empty(10, 4)[empty(3, dtype=dtypes.int32)], "collapsed",
               load_collapse),
    "embedding": (codegen("load collapse"), lambda: empty(10, 4)[empty(2, 3, dtype=dtypes.int32)], "collapsed",
                  load_collapse),
    # an index taken modulo a divisor of its range's size
    "transpose": (codegen("split ranges"), lambda: empty(6, 4).permute(1, 0).reshape(24) + 1, "split", split),
    "repeat": (codegen("split ranges"), lambda: empty(3, 4).repeat(2, 2) + 1, "split", split),
    # adjacent loops that index contiguously, and kernels whose ranges stay
    "elementwise": (codegen("simplify ranges"), lambda: empty(4, 6) + 1, "simplified", simplify),
    "embedding_ranges": (codegen("simplify ranges"), lambda: empty(10, 4)[empty(2, 3, dtype=dtypes.int32)],
                         "simplified", simplify),
    "matmul": (codegen("simplify ranges"), lambda: empty(4, 8) @ empty(8, 3), "simplified", simplify),
    "conv": (codegen("simplify ranges"), lambda: empty(1, 2, 6, 6).conv2d(empty(3, 2, 3, 3), padding=1),
             "simplified", simplify),
    # sums of values that are functions of the range: aranges, masks, one-hot
    # encodings, a sum of zeros
    "arange": (at_reduce_collapse, lambda: Tensor.arange(10).cast(dtypes.float).clone(), "collapsed",
               reduce_simplify),
    "arange_sum": (at_reduce_collapse, lambda: Tensor.arange(6).reshape(3, 2).sum(axis=1).clone(), "collapsed",
                   reduce_simplify),
    "arange_index": (at_reduce_collapse, lambda: Tensor.arange(10)[empty(3, dtype=dtypes.int32)], "collapsed",
                     reduce_simplify),
    "triu": (at_reduce_collapse, lambda: Tensor.ones(8, 8).contiguous().triu(), "collapsed", reduce_simplify),
    "one_hot": (at_reduce_collapse, lambda: empty(4, dtype=dtypes.int32).one_hot(10).cast(dtypes.float),
                "collapsed", reduce_simplify),
    "sum_zero": (at_reduce_collapse, lambda: (empty(4, 8) * 0).sum(1), "collapsed", reduce_simplify),
    "arange_transposed": (at_reduce_collapse, lambda: (Tensor.arange(4) * empty(4)).reshape(4, 1).T.sum(),
                          "collapsed", reduce_simplify),
    "cumsum": (at_reduce_collapse, lambda: empty(8).cumsum(), "collapsed", reduce_simplify),
}


def declare(name, capture, program, result, rewrite):
    def given(): return capture(program)
    def made(): return rewrite(given())
    given.__name__, made.__name__ = name, f"{name}_{result}"
    graph(given)
    graph(made)


for name, (capture, program, result, rewrite) in KERNELS.items():
    declare(name, capture, program, result, rewrite)


# Hand-built cases

def rng(n, axis, kind=AxisType.WEAK): return UOp.range(n, axis, kind)
def red(n, axis): return rng(n, axis, AxisType.REDUCE)
def buf(slot=0, size=1024, dtype=dtypes.float): return UOp.param(slot, dtype, size)
def kernel(*srcs): return UOp.sink(*srcs, arg=KernelInfo())
def f32(x): return UOp.const(x, dtypes.float)
def i32(x): return UOp.const(x, dtypes.int32)
def gated_load(valid, idx, slot=0): return buf(slot).index(idx.valid(valid)).load()
def store_at(idx, value, slot=0): return buf(slot).index(idx).store(value)


def declare_cases(matcher, rewrite, cases):
    def listed(): return ["case", "src"], [(name, i) for i, (name, _) in enumerate(cases)]
    def given(): return UOp.sink(*(u for _, u in cases))
    def made(): return UOp.sink(*(rewrite(u) for _, u in cases))
    listed.__name__, given.__name__, made.__name__ = matcher, f"{matcher}_input", f"{matcher}_output"
    table(listed)
    graph(given)
    graph(made)


def flatten_cases():
    r0, r1, r2 = rng(3, 0), rng(4, 1), rng(5, 2)
    s0, s1 = red(3, 3), red(4, 4)
    dependent = UOp.range(r0 + 1, 5)
    value = f32(1.0)
    return [
        ("end_of_ranges", store_at(r0 * 4 + r1, value).end(r0, r1)),
        ("end_of_an_expression", store_at(r0 * 4 + r1, value).end(r0 * 4 + r1)),
        ("end_of_an_expression_and_a_range", store_at(r0 * 20 + r1 * 5 + r2, value).end(r0 * 4 + r1, r2)),
        ("end_of_a_repeated_range", store_at(r0, value).end(r0, r0 + 1)),
        ("end_of_a_dependent_range_first", store_at(r0 * 4 + dependent, value).end(dependent, r0)),
        ("end_of_nothing", UOp(Ops.END, src=(store_at(UOp.const(0), value),))),
        ("end_of_a_constant", UOp(Ops.END, src=(store_at(UOp.const(0), value), UOp.const(3)))),
        ("reduce_of_ranges", (s0 * 4 + s1).cast(dtypes.float).reduce(s0, s1, arg=Ops.ADD)),
        ("reduce_of_an_expression", (s0 * 4 + s1).cast(dtypes.float).reduce(s0 * 4 + s1, arg=Ops.ADD)),
        ("reduce_of_a_repeated_range", s0.cast(dtypes.float).reduce(s0, s0, arg=Ops.MAX)),
        ("reduce_of_nothing", UOp(Ops.REDUCE, src=(value,), arg=(Ops.ADD, 0))),
        ("reduce_of_an_expression_inside_a_range", (s0 + r0).cast(dtypes.float).reduce(s0 + r0, arg=Ops.ADD)),
    ]


def split_cases():
    def mod_kernel(r, c, op=Ops.FLOORMOD):
        return kernel(store_at(r, UOp(op, src=(r, UOp.const(c)))).end(r))
    r8, r12 = rng(8, 0), rng(12, 1)
    nested = rng(8, 0)
    n = UOp.variable("n", 1, 16)
    s8 = red(8, 1)
    return [
        # test_uop_symbolic.py::TestRangeSplitting::test_range_split_on_mod
        ("nested_sink", kernel(store_at(UOp.const(0), (nested % 2).cast(dtypes.int32), slot=0).sink().end(nested))),
        ("mod_2_of_8", mod_kernel(r8, 2)),
        ("mod_4_of_12", mod_kernel(r12, 4)),
        ("mod_3_of_7", mod_kernel(rng(7, 0), 3)),
        ("mod_3_of_8", mod_kernel(r8, 3)),
        ("mod_8_of_8", mod_kernel(r8, 8)),
        ("mod_1_of_8", mod_kernel(r8, 1)),
        ("mod_16_of_8", mod_kernel(r8, 16)),
        ("cmod_4_of_12", mod_kernel(r12, 4, Ops.CMOD)),
        ("mod_of_a_symbolic_range", mod_kernel(UOp.range(n, 0), 2)),
        ("mod_of_a_warp_range", mod_kernel(rng(8, 0, AxisType.WARP), 2)),
        ("mod_of_a_device_range", mod_kernel(rng(8, 0, AxisType.DEVICE), 2)),
        ("mod_of_a_local_range", mod_kernel(rng(8, 0, AxisType.LOCAL), 2)),
        ("mod_by_a_variable", kernel(store_at(r8, r8 % UOp.variable("c", 1, 4)).end(r8))),
        ("two_mods_of_one_range", kernel(store_at(r12, r12 % 2 + r12 % 4).end(r12))),
        ("mod_and_div", kernel(store_at(r12, r12 % 4 + r12 // 4 * 10).end(r12))),
        ("mod_in_a_reduce", kernel(store_at(UOp.const(0), (s8 % 2).cast(dtypes.float).reduce(s8, arg=Ops.ADD)))),
        ("mod_of_two_ranges", kernel(store_at(r8 * 12 + r12, (r8 % 2) + (r12 % 3)).end(r8, r12))),
        ("sink_without_kernel", UOp.sink(store_at(r8, r8 % 2).end(r8))),
    ]


def simplify_cases():
    r0, r1, r2 = rng(3, 0), rng(4, 1), rng(5, 2)
    s0, s1, s2 = red(3, 3), red(4, 4), red(5, 5)
    one = f32(1.0)
    r = rng(204, 0)
    q = rng(8, 1)
    ra, rb = rng(3, 6), rng(4, 7, AxisType.LOOP)
    mask = (r < 4).where(one, UOp.invalid())
    return [
        # merging
        ("adjacent_contiguous", kernel(store_at(r0 * 4 + r1, one).end(r0, r1))),
        ("adjacent_three", kernel(store_at(r0 * 20 + r1 * 5 + r2, one).end(r0, r1, r2))),
        ("adjacent_unused", kernel(store_at(UOp.const(0), one).end(r0, r1, r2))),
        ("adjacent_one_used", kernel(store_at(r0, one).end(r0, r1))),
        ("adjacent_transposed", kernel(store_at(r1 * 3 + r0, one).end(r0, r1))),
        ("adjacent_different_types", kernel(store_at(ra * 4 + rb, one).end(ra, rb))),
        ("not_adjacent", kernel(store_at(r0 * 20 + r2 * 4 + r1, one).end(r0, r1, r2))),
        ("reduce_contiguous", kernel(store_at(UOp.const(0), (s0 * 4 + s1).cast(dtypes.float).reduce(s0, s1, arg=Ops.ADD)))),
        ("reduce_transposed", kernel(store_at(UOp.const(0), (s1 * 3 + s0).cast(dtypes.float).reduce(s0, s1, arg=Ops.ADD)))),
        ("reduce_three", kernel(store_at(UOp.const(0), (s0 * 20 + s1 * 5 + s2).cast(dtypes.float)
                                         .reduce(s0, s1, s2, arg=Ops.ADD)))),
        ("ranges_in_different_reduces", kernel(store_at(UOp.const(0),
                                         (s0.cast(dtypes.float).reduce(s0, arg=Ops.ADD) + s1.cast(dtypes.float))
                                         .reduce(s1, arg=Ops.ADD)))),
        ("loop_and_reduce", kernel(store_at(r0, (r0 * 4 + s1).cast(dtypes.float).reduce(s1, arg=Ops.ADD)).end(r0))),
        # shrinking
        ("single_guard", kernel(gated_load(r < 4, r).end(r))),
        ("two_guards", kernel(gated_load(r < 4, r).sink().end(r), gated_load(r < 8, r, slot=1).sink().end(r))),
        ("guard_beyond_size", kernel(gated_load(r < 300, r).sink().end(r))),
        ("guard_of_one", kernel(gated_load(r < 1, r).sink().end(r))),
        ("guarded_and_unguarded", kernel((gated_load(r < 4, r) + buf(1).index(r).load()).sink().end(r))),
        ("guard_by_a_variable", kernel(gated_load(r < UOp.variable("c", 1, 8), r).sink().end(r))),
        ("guard_by_a_conjunction", kernel(gated_load((r < 4) & (q < 2), r * 8 + q).sink().end(r, q))),
        ("guard_of_a_reduce_range", kernel(store_at(UOp.const(0), (s2.cast(dtypes.float) + gated_load(s2 < 2, s2))
                                                    .reduce(s2, arg=Ops.ADD)))),
        ("guard_on_a_later_index", kernel(buf().index(UOp.const(0), r.valid(r < 4)).load().sink().end(r))),
        ("guards_in_a_stack", kernel(buf().index(UOp(Ops.STACK, src=(r.valid(r < 4), r.valid(r < 8))))
                                     .load().sink().end(r))),
        ("store_gate", kernel(buf().index(r).store(one, r < 200).end(r))),
        ("store_where_invalid", kernel(buf().index(r).store((r < 4).where(mask, Invalid)).end(r))),
        ("store_where_invalid_flipped", kernel(buf().index(r).store((r >= 4).where(Invalid, mask)).end(r))),
        ("store_through_a_gated_index", kernel(buf().index(r.valid(r < 4)).store(one).end(r))),
        ("nested_sink", kernel(gated_load(r < 4, r).sink().end(r))),
        ("sink_without_kernel", UOp.sink(gated_load(r < 4, r).sink().end(r))),
    ]


def unparented_cases():
    r0, r1 = red(4, 0), red(5, 1)
    n = UOp.range(UOp.variable("n", 1, 16), 2, AxisType.REDUCE)
    x = r0.cast(dtypes.float)
    return [
        ("sum_over_an_unused_range", x.reduce(r0, r1, arg=Ops.ADD)),
        ("product_over_an_unused_range", x.reduce(r0, r1, arg=Ops.MUL)),
        ("maximum_over_an_unused_range", x.reduce(r0, r1, arg=Ops.MAX)),
        ("sum_over_used_ranges", (x + r1.cast(dtypes.float)).reduce(r0, r1, arg=Ops.ADD)),
        ("sum_of_a_constant", f32(3.0).reduce(r0, r1, arg=Ops.ADD)),
        ("product_of_a_constant", f32(3.0).reduce(r0, arg=Ops.MUL)),
        ("maximum_of_a_constant", f32(3.0).reduce(r0, arg=Ops.MAX)),
        ("integer_sum_of_a_constant", i32(3).reduce(r0, arg=Ops.ADD)),
        ("sum_over_a_symbolic_range", x.reduce(r0, n, arg=Ops.ADD)),
        ("product_over_a_symbolic_range", x.reduce(r0, n, arg=Ops.MUL)),
        # test_uop_graph.py::TestReduceCollapse::test_reduce_shapeless_const_unroll
        ("sum_over_an_unroll_range", UOp.const(3.0).cast(dtypes.float).reduce(UOp.range(4, 0, AxisType.UNROLL),
                                                                              arg=(Ops.ADD, 0))),
        ("sum_of_nothing", UOp(Ops.REDUCE, src=(x,), arg=(Ops.ADD, 0))),
    ]


def collapse_cases():
    r = red(10, 0)
    x = UOp.variable("x", 0, 20, dtypes.int32)
    y = UOp.variable("y", 0, 20, dtypes.int32)
    lo, hi = UOp.variable("lo", 0, 20), UOp.variable("hi", 0, 20)
    two = f32(2.0)
    zero = UOp.const(0.0)
    o = rng(6, 1)
    p = UOp.param(0, dtypes.bool)
    return [
        ("sum_below_a_bound", (r < 3).where(two, zero).reduce(r, arg=Ops.ADD)),
        ("sum_above_a_bound", (r < 3).where(zero, two).reduce(r, arg=Ops.ADD)),
        ("sum_between_bounds", ((r < 2).logical_not() & (r < 7)).where(two, zero).reduce(r, arg=Ops.ADD)),
        ("sum_below_a_variable", (r < hi).where(two, zero).reduce(r, arg=Ops.ADD)),
        ("sum_above_a_variable", (r < lo).where(zero, two).reduce(r, arg=Ops.ADD)),
        ("sum_between_variables", ((r < lo).logical_not() & (r < hi)).where(two, zero).reduce(r, arg=Ops.ADD)),
        ("sum_below_a_bound_beyond_the_range", (r < 30).where(two, zero).reduce(r, arg=Ops.ADD)),
        ("sum_of_a_variable", (r < 3).where(UOp.variable("v", 0, 9, dtypes.float), zero).reduce(r, arg=Ops.ADD)),
        ("sum_of_the_range", (r < 3).where(r.cast(dtypes.float), zero).reduce(r, arg=Ops.ADD)),
        ("sum_below_an_outer_range", (r < o).where(two, zero).reduce(r, arg=Ops.ADD)),
        ("comparison_of_a_sum", (r + 2) < 7),
        ("comparison_of_a_sum_of_ranges", (r + o) < 7),
        ("comparison_of_a_sum_with_a_variable", (r + y) < x),
        ("comparison_of_a_product", (r * 3) < 15),
        ("comparison_of_a_product_by_a_variable", (r * (y + 1)) < x),
        ("comparison_of_a_product_by_zero_or_more", (r * y) < x),
        ("comparison_of_a_float_product", (r.cast(dtypes.float) * two) < f32(7.0)),
        ("sum_below_a_shifted_range", ((r + 2) < 7).where(two, zero).reduce(r, arg=Ops.ADD)),
        ("sum_below_a_cast_shifted_range", ((r.cast(dtypes.int32) + i32(2)).cast(dtypes.weakint) < 7).where(two, zero)
         .reduce(r, arg=Ops.ADD)),
        ("sum_below_a_scaled_range", ((r * 3) < 15).where(two, zero).reduce(r, arg=Ops.ADD)),
        ("sum_of_a_sum", (r.cast(dtypes.float) + two).reduce(r, arg=Ops.ADD)),
        # test_uop_graph.py::TestReduceCollapse::test_multi_range_reduce_add
        ("sum_of_a_sum_over_two_ranges", (red(3, 0).cast(dtypes.float) + red(4, 1).cast(dtypes.float))
         .reduce(red(3, 0), red(4, 1), arg=Ops.ADD)),
        ("sum_gated_by_a_parameter", (p & (r < 3)).where(two, zero).reduce(r, arg=Ops.ADD)),
        ("product_by_a_cast_comparison", two * (r < 3).cast(dtypes.float)),
        ("sum_of_a_product_by_a_cast_comparison", (two * (r < 3).cast(dtypes.float)).reduce(r, arg=Ops.ADD)),
        ("integer_sum_below_a_bound", (r < 5).where(i32(1), UOp.const(0)).reduce(r, arg=Ops.ADD)),
        ("sum_below_a_bound_of_a_typed_zero", (r < 3).where(two, f32(0.0)).reduce(r, arg=Ops.ADD)),
    ]


def load_collapse_cases():
    r = red(10, 0)
    table_ = buf(0, 10)
    idx = UOp.param(1, dtypes.int32, 4).index(UOp.const(0))
    label = UOp.variable("label", 0, 9)
    ri = r.cast(dtypes.int32)
    x = UOp.variable("x", 0, 20, dtypes.int32)
    loaded = idx.cast(dtypes.weakint)
    return [
        ("sum_selected_by_a_constant", (UOp.const(3) != r).where(UOp.const(0.0), table_.index(r)).reduce(r, arg=Ops.ADD)),
        ("sum_selected_by_a_variable", (label != r).where(UOp.const(0.0), table_.index(r)).reduce(r, arg=Ops.ADD)),
        ("sum_selected_by_a_load", (loaded != r).where(UOp.const(0.0), table_.index(r)).reduce(r, arg=Ops.ADD)),
        ("sum_selected_by_a_cast_range", (idx != ri).where(UOp.const(0.0), table_.index(r)).reduce(r, arg=Ops.ADD)),
        ("sum_selected_by_a_shifted_load", ((idx + i32(2)) != ri).where(UOp.const(0.0), table_.index(r)).reduce(r, arg=Ops.ADD)),
        ("sum_selected_by_the_range", (r != r).where(UOp.const(0.0), table_.index(r)).reduce(r, arg=Ops.ADD)),
        ("sum_of_a_one_hot_product", (table_.index(r) * (label.eq(r)).cast(dtypes.float)).reduce(r, arg=Ops.ADD)),
        ("sum_selected_over_two_ranges", (label != r).where(UOp.const(0.0), table_.index(r)).reduce(r, red(3, 1), arg=Ops.ADD)),
        ("sum_selected_with_a_typed_zero", (label != r).where(f32(0.0), table_.index(r)).reduce(r, arg=Ops.ADD)),
        ("maximum_selected", (label != r).where(UOp.const(0.0), table_.index(r)).reduce(r, arg=Ops.MAX)),
        ("sum_selected_by_a_shifted_cast", ((x + i32(2)).cast(dtypes.weakint) != r).where(UOp.const(0.0), table_.index(r))
         .reduce(r, arg=Ops.ADD)),
        ("sum_selected_by_a_shifted_range", ((r + 2) != label).where(UOp.const(0.0), table_.index(r)).reduce(r, arg=Ops.ADD)),
        ("sum_selected_by_a_shifted_cast_range", ((ri + i32(2)).cast(dtypes.weakint) != label)
         .where(UOp.const(0.0), table_.index(r)).reduce(r, arg=Ops.ADD)),
        ("inequality_of_a_shifted_cast", (x + i32(2)).cast(dtypes.weakint) != r),
        ("comparison_of_a_shifted_load", (loaded + 5) < 20),
        ("comparison_of_a_shifted_load_by_variables", (loaded + UOp.variable("y", 0, 5)) < UOp.variable("c", 0, 30)),
        ("comparison_of_a_shifted_int32_load", (idx + i32(5)) < i32(20)),
        ("comparison_of_a_load_plus_a_load", (loaded + loaded) < 20),
        ("comparison_of_a_shifted_variable", (UOp.variable("v", 0, 9) + 5) < 20),
    ]


declare_cases("flatten_range", flatten, flatten_cases())
declare_cases("split_ranges", split, split_cases())
declare_cases("simplify_ranges", simplify, simplify_cases())
declare_cases("reduce_unparented", unparent, unparented_cases())
declare_cases("reduce_collapse", collapse, collapse_cases())
declare_cases("reduce_simplify", reduce_simplify, collapse_cases() + unparented_cases())
declare_cases("load_collapse", load_collapse, load_collapse_cases())


# tinygrad's test_simplify_valid_idx.py::TestRangeShrink, whose cases compile
# a sink: each is recorded where the kernel reaches `simplify ranges`, and the
# matcher's result.

def shrink_cases():
    r = rng(204, 0)
    x = (r < 4).where(UOp.const(1.0), Invalid)
    return {
        "shrink_single_guard": lambda: gated_load(r < 4, r).sink(),
        "shrink_picks_max_guard": lambda: UOp.sink(gated_load(r < 4, r), gated_load(r < 8, r)),
        "shrink_guard_ge_max": lambda: gated_load(r < 300, r).sink(),
        "shrink_unguarded_elsewhere": lambda: UOp.sink(gated_load(r < 4, r), buf(1, 204).index(r).load()),
        "shrink_used_in_reduce": lambda: (r.cast(dtypes.float) + gated_load(r < 4, r)).reduce(r, arg=Ops.ADD).sink(),
        "shrink_to_single_iteration": lambda: gated_load(r < 1, r).sink(),
        "shrink_store_where_invalid": lambda: buf(0, 204).index(r).store((r < 4).where(x, Invalid)).sink(),
        "shrink_store_where_invalid_flipped": lambda: buf(0, 204).index(r).store((r >= 4).where(Invalid, x)).sink(),
    }


for name, sink in shrink_cases().items():
    declare(name, lambda program: stage("simplify ranges", program().replace(arg=KernelInfo()), CPU), sink,
            "simplified", simplify)
