"""Goldens of tinygrad/codegen/late/linearizer.py: splitting loop ends,
chaining loops, and linearizing kernels.

A kernel is compiled for a renderer, and each of its goldens is a graph the
pipeline builds on the way: `<kernel>.golden` is the sink `linearize` receives
and `<kernel>_linear.golden` the list it returns, as the sources of a
`LINEAR`; `<kernel>_unsplit.golden` is the sink the final rewrite receives and
`<kernel>_split.golden` that sink after `pm_split_ends` alone;
`<kernel>_unchained.golden` is the sink `pm_add_control_flow` receives and
`<kernel>_chained.golden` its result. `<kernel>_linear_toposort.golden` is the
list without the structural tie-break (TUPLE_ORDER=0).

Hand-built cases come as three goldens: a table naming each case by its
position, an input sink whose sources are the cases, and an output sink whose
sources are their results, in the same order.
"""

import contextlib
import io
import os

from golden import graph, table, text
from graph import kernels, stage
from tinygrad import Tensor, Variable, dtypes
from tinygrad.codegen.late.linearizer import CFGContext, linearize, pm_add_control_flow, pm_split_ends
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import Context, Target
from tinygrad.renderer import Renderer
from tinygrad.renderer.cstyle import ClangRenderer, CUDARenderer
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp, graph_rewrite
from test.runtime.test_wait_loop import loop_in_loop_kernel, nested_loop_kernel, two_loops_kernel, wait_loop_kernel

CPU = ClangRenderer(Target("CPU", "CLANG", "x86_64,x86-64"))
CUDA = CUDARenderer(Target("CUDA", "CUDA", "sm_80"))
NULL = Renderer(Target())


def with_opts(kernel, opts):
    return kernel.replace(arg=KernelInfo(opts_to_apply=tuple(opts)))


def last(*tensors): return kernels(*tensors)[-1]


def dependent_loop_bound():
    # test/null/test_linearizer_rewrite.py::test_dependent_loop_bound
    buf, out, counts = UOp.param(0, dtypes.int, 16), UOp.param(1, dtypes.int, 4), UOp.param(2, dtypes.int, 4)
    outer = UOp.range(4, 0, AxisType.LOOP)
    inner = UOp.range(counts.index(outer).load().maximum(0).minimum(4), 1)
    store = buf.index(outer * 4 + inner).store(UOp.const(1, dtypes.int)).end(inner)
    return UOp.sink(out.after(store).index(outer).store(UOp.const(2, dtypes.int)).end(outer))


def wait_loop(fxn):
    # test/runtime/test_wait_loop.py, compiled for one int of output
    return lambda: fxn(UOp.param(0, dtypes.int, 1))


def cpu(t): return lambda: last(t())


# name: (kernel, renderer, stages beyond linearize)
KERNELS = {
    "matmul": (cpu(lambda: Tensor.empty(4, 8, device="CPU") @ Tensor.empty(8, 3, device="CPU")), CPU, "sc"),
    "matmul_noopt": (lambda: with_opts(last(Tensor.empty(4, 8, device="CPU") @ Tensor.empty(8, 3, device="CPU")), []),
                     CPU, "sc"),
    "softmax": (lambda: with_opts(last(Tensor.empty(4, 256, device="CPU").softmax(-1)), []), CPU, "sc"),
    "conv": (lambda: with_opts(last(Tensor.empty(1, 2, 6, 6, device="CPU").conv2d(
        Tensor.empty(3, 2, 3, 3, device="CPU"), padding=1)), []), CPU, "sc"),
    "two_sums": (cpu(lambda: Tensor.empty(64, device="CPU").sum() + Tensor.empty(128, device="CPU").max()), CPU, "sc"),
    "transpose": (cpu(lambda: Tensor.empty(3, 5, 7, device="CPU").permute(2, 0, 1).contiguous()), CPU, "sc"),
    "variable": (lambda: [c.src[0] for c in Tensor.linear_with_vars(
        Tensor.empty(10, device="CPU")[:Variable("n", 1, 10).bind(4)].contiguous() + 1)[0].src
        if c.src[0].op is Ops.SINK][-1], CPU, ""),
    "dependent_loop_bound": (dependent_loop_bound, NULL, "c"),
    "sum_group": (cpu(lambda: Tensor.empty(4096, device="CPU").sum()), CUDA, "sc"),
    "matmul_local": (lambda: with_opts(last(Tensor.empty(64, 64, device="CPU") @ Tensor.empty(64, 64, device="CPU")),
                                       [Opt(OptOps.SPLIT, 0, (4, AxisType.LOCAL)), Opt(OptOps.SPLIT, 1, (4, AxisType.LOCAL))]),
                     CUDA, "s"),
    # test/runtime/test_linearizer.py, whose claims are about the order
    "late_bias_load": (lambda: with_opts(last(Tensor.empty(1, 3, 16, 16, device="CPU").conv2d(
        Tensor.empty(16, 3, 3, 3, device="CPU"), Tensor.empty(16, device="CPU"))), []), CPU, ""),
    "two_nested_range_alt_indexing": (lambda: with_opts(last(Tensor.empty(2, device="CPU").reshape(2, 1).pad(
        ((1, 1), (1, 1)), value=2).sum()), []), CPU, ""),
    "range_outer_op_before_phi": (lambda: with_opts(last(
        (Tensor.empty(4, 1, device="CPU") + (b := Tensor.empty(1, 1, device="CPU"))[0]).sum() + b[0]), []), CPU, ""),
    "simple_unroll": (lambda: with_opts(last((Tensor.empty(64, 64, device="CPU") @ Tensor.empty(64, 64, device="CPU")).relu()),
                                        [Opt(OptOps.SPLIT, 2, (4, AxisType.UNROLL)), Opt(OptOps.SPLIT, 0, (4, AxisType.UPCAST))]),
                      CPU, ""),
    "wait_loop": (wait_loop(wait_loop_kernel), CPU, ""),
    "nested_loop": (wait_loop(nested_loop_kernel), CPU, "c"),
    "two_loops": (wait_loop(two_loops_kernel), CPU, "c"),
    "loop_in_loop": (wait_loop(loop_in_loop_kernel), CPU, "c"),
}


def linear(lst): return UOp(Ops.LINEAR, src=tuple(lst))


def chain(sink): return graph_rewrite(sink, pm_add_control_flow, ctx=CFGContext(sink), bottom_up=True)


def declare(name, kernel, renderer, stages):
    def at(pass_name): return lambda: stage(pass_name, kernel(), renderer)

    def named(fn, suffix):
        fn.__name__ = name + suffix
        return fn

    graph(named(at("linearize"), ""))
    graph(named(lambda: linear(linearize(at("linearize")())), "_linear"))
    if "s" in stages:
        graph(named(at("final rewrite"), "_unsplit"))
        graph(named(lambda: graph_rewrite(at("final rewrite")(), pm_split_ends), "_split"))
    if "c" in stages:
        graph(named(at("add control flow"), "_unchained"))
        graph(named(lambda: chain(at("add control flow")()), "_chained"))


for name, (kernel, renderer, stages) in KERNELS.items():
    declare(name, kernel, renderer, stages)


@graph
def matmul_linear_toposort():
    with Context(TUPLE_ORDER=0): return linear(linearize(stage("linearize", KERNELS["matmul"][0](), CPU)))


@graph
def conv_linear_toposort():
    with Context(TUPLE_ORDER=0): return linear(linearize(stage("linearize", KERNELS["conv"][0](), CPU)))


@text
def debug_linearize():
    os.environ["DEBUG_LINEARIZE"] = "1"
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        linearize(stage("linearize", KERNELS["dependent_loop_bound"][0](), NULL))
    return out.getvalue()


# Hand-built cases

def cases(name, build, result):
    """Declare the goldens `<name>`, `<name>_input` and `<name>_output` of the
    cases `build()`, a list of (name, node), and of `result` on each."""
    def names(): return ["case", "src"], [(case, str(i)) for i, (case, _) in enumerate(build())]
    def inputs(): return UOp.sink(*[u for _, u in build()])
    def outputs(): return UOp.sink(*[result(u) for _, u in build()])
    names.__name__, inputs.__name__, outputs.__name__ = name, f"{name}_input", f"{name}_output"
    table(names)
    graph(inputs)
    graph(outputs)


def fbuf(slot, size=16): return UOp.param(slot, dtypes.float, size)


def store_in(slot, *ranges):
    """A store into the parameter `slot`, at the sum of `ranges`."""
    return fbuf(slot).index(sum(ranges[1:], ranges[0]) if ranges else UOp.const(0)).store(UOp.const(1.0, dtypes.float))


# Priorities: each case is a sink, and its result the list linearize returns.

def orders():
    r0, r1 = UOp.range(4, 0, AxisType.LOOP), UOp.range(8, 1, AxisType.LOOP)
    empty = UOp.range(0, 2, AxisType.LOOP)
    p0, p1, p2 = fbuf(0), fbuf(1), fbuf(2)
    load = lambda p, i: p.index(UOp.const(i)).load()
    reg = UOp.placeholder((1,), dtypes.float, 0, addrspace=AddrSpace.REG)
    local = UOp.placeholder((4,), dtypes.float, 1, addrspace=AddrSpace.LOCAL)
    return [
        ("empty", UOp.sink()),
        ("params_by_slot", UOp.sink(load(p2, 0) + load(p0, 0) + load(p1, 0))),
        ("variables_by_name", UOp.sink(UOp.variable("b", 0, 9, dtypes.int) + UOp.variable("a", 0, 9, dtypes.int))),
        ("params_before_constants", UOp.sink(p1.index(UOp.const(3)).store(UOp.const(2.0, dtypes.float)))),
        ("buffers_before_local_buffers", UOp.sink(
            local.index(UOp.const(0)).store(UOp.const(0.0, dtypes.float)),
            reg.index(UOp.const(0)).store(UOp.const(0.0, dtypes.float)))),
        ("loads_early_stores_late", UOp.sink(
            p0.index(UOp.const(0)).store(load(p1, 0) + 1.0), p0.index(UOp.const(1)).store(load(p2, 0) * 2.0))),
        ("structure_breaks_ties", UOp.sink(UOp.const(1) + UOp.const(2), UOp.const(2) - UOp.const(1))),
        ("constants_by_their_text", UOp.sink(load(p0, 2), load(p0, 10), load(p0, 100))),
        ("hoisted_load", UOp.sink(p1.index(r0).store(load(p0, 0) + r0.cast(dtypes.float)).end(r0))),
        ("run_count", UOp.sink(p1.index(r0 * 8 + r1).store(
            p0.index(r0).load() + p0.index(r1).load()).end(r0, r1))),
        ("empty_range_runs_none", UOp.sink(p1.index(empty).store(load(p0, 0) + 1.0).end(empty), load(p2, 0))),
        ("end_before_load", UOp.sink(p1.index(r0).store(1.0).end(r0), load(p2, 3))),
        ("unended_range", UOp.sink(p1.index(r0).store(load(p0, 0)))),
        ("after_value", UOp.sink(load(p0, 0).after(UOp.const(1.0, dtypes.float)))),
        ("after_store", UOp.sink(p0.index(UOp.const(0)).store(1.0).after(p0.index(UOp.const(1)).store(2.0)))),
        ("empty_group", UOp.sink(UOp.group())),
    ]


cases("orders", orders, lambda u: linear(linearize(u)))


@graph
def orders_output_toposort():
    with Context(TUPLE_ORDER=0): return UOp.sink(*[linear(linearize(u)) for _, u in orders()])


# Splitting ends: each case is an END, and its result the END rewritten.

def splits():
    r0, r1, r2 = (UOp.range(4, i, AxisType.LOOP) for i in range(3))
    g, lp, rd = UOp.range(4, 0, AxisType.GLOBAL), UOp.range(4, 1, AxisType.LOOP), UOp.range(4, 2, AxisType.REDUCE)
    a, b, c = (UOp(Ops.RANGE, src=(UOp.const(4),), arg=(*i, AxisType.LOOP)) for i in ((0, 1), (0, 2), (1,)))
    u = store_in(0, r0, r1, r2)
    return [
        ("one_range", store_in(0, r0).end(r0)),
        ("ranges_in_order", u.end(r0, r1, r2)),
        ("ranges_out_of_order", u.end(r1, r2, r0)),
        ("axis_types", store_in(0, g, lp, rd).end(rd, g, lp)),
        ("identities_of_parts", store_in(0, a, b, c).end(c, a, b)),
        ("repeated_range", u.end(r1, r0, r1, r2)),
        ("ranges_of_a_source", UOp(Ops.END, src=(u, r0 + r2, r1))),
        ("source_and_its_range", UOp(Ops.END, src=(u, r0 * 2, r0, r1, r2))),
        ("source_without_ranges", UOp(Ops.END, src=(store_in(0), UOp.const(3)))),
        ("no_range", UOp(Ops.END, src=(store_in(0),))),
        ("closed_ranges_of_a_source", UOp(Ops.END, src=(u.end(r2), store_in(1, r1, r2).end(r2), r0))),
        ("ended_twice", u.end(r2, r1).end(r0)),
    ]


cases("splits", splits, lambda u: graph_rewrite(u, pm_split_ends))


# Chaining loops: each case is a sink, and its result the sink with its ranges
# chained.

def loop(slot, axis, *outer, after=None):
    """The END of a loop of range `axis` storing into `slot`, inside `outer`,
    reading the parameter `after` after its loop if given."""
    r = UOp.range(4, axis, AxisType.LOOP)
    value = UOp.const(1.0, dtypes.float) if after is None else fbuf(after[0]).after(after[1]).index(r).load()
    return fbuf(slot).index(sum(outer, r)).store(value).end(r)


def chains():
    e0 = loop(0, 0)
    e1 = loop(1, 1)
    o = UOp.range(4, 9, AxisType.LOOP)
    return [
        ("one_loop", UOp.sink(e0)),
        ("siblings", UOp.sink(e0, e1)),
        ("siblings_reversed", UOp.sink(e1, e0)),
        ("three_siblings", UOp.sink(e0, e1, loop(2, 2))),
        ("dependent_sibling_first", UOp.sink(loop(1, 1, after=(0, e0)), e0)),
        ("dependency_chain", UOp.sink(loop(2, 2, after=(1, loop(1, 1, after=(0, e0)))), loop(1, 1, after=(0, e0)))),
        ("nested", UOp.sink(UOp.group(loop(0, 0, o), loop(1, 1, o)).end(o))),
        ("nested_dependent", UOp.sink(UOp.group(loop(1, 1, o, after=(0, loop(0, 0, o))), loop(0, 0, o)).end(o))),
        ("nested_and_sibling", UOp.sink(UOp.group(loop(0, 0, o), loop(1, 1, o)).end(o), loop(2, 2))),
        ("unchained_range", UOp.sink(store_in(0, UOp.range(4, 0, AxisType.LOOP)))),
    ]


cases("chains", chains, chain)


@graph
def cyclic():
    """A loop whose range runs after the loop it closes: no order runs it."""
    child = UOp.range(4, 1, AxisType.LOOP)
    parent = UOp.range(4, 0, AxisType.LOOP).replace(src=(UOp.const(4), child))
    return UOp.sink(fbuf(0).index(parent + child).store(1.0).end(child).end(parent))
