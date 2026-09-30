"""Goldens of tinygrad/codegen/gpudims.py: the launch dimensions that
get_grouped_dims makes of loop sizes, and kernels rewritten by pm_add_gpudims.

`grouped_dims.golden` holds the cases of tinygrad's test_gpudims.py, then a
sweep of sizes drawn from a fixed seed against the bounds of real targets. A
row gives the sizes of the hardware indices, by name, and each loop's index as
an expression, or the exception tinygrad raises. The goldens named after a
claim hold one graph each, which the suite builds with tolk.next's
constructors.
"""

import random

from golden import graph, table
from tinygrad.codegen.gpudims import add_gpudims, get_grouped_dims, pm_add_gpudims
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.helpers import Target
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp, graph_rewrite

BIG = (0x8FFFFFFF,) * 3


def renderer(global_max=BIG, local_max=BIG, global_prod_max=None):
    class R(Renderer): pass
    R.global_max, R.local_max, R.global_prod_max = global_max, local_max, global_prod_max
    return R(Target())


def declare(name, fn):
    fn.__name__ = name
    graph(fn)


# get_grouped_dims

TINYGRAD_CASES = [
    ((2,), (16, 16, 16), False), ((2, 3), (16, 16, 16), False), ((2, 3), (16, 16, 16), True),
    ((2, 3, 4), (16, 16, 16), False), ((64, 3, 4), (16, 16, 16), False), ((64, 3, 4), (16, 4, 16), False),
    ((64, 3, 4), (16, 16, 16), True), ((128, 3, 4), (16, 4, 256), False), ((4, 4, 512), (16, 4, 256), False),
    ((5, 12, 7), (8, 4, 16), False), ((512, 4, 2), (8192, 2, 2), False), ((128,), (16, 16, 256), False),
    ((65536,), (16, 16, 256), False), ((65536, 2), (65535, 65535, 65535), False), ((121,), (12, 12, 12), False),
    ((128, 128), (16, 16, 256), False), ((2, 3, 4, 5), (16, 16, 16), False), ((2, 3, 4, 5), (32, 16, 16), True),
    ((2, 3, 4, 5), (4, 16, 16), False), ((2, 3, 4, 5), (16, 16, 16), True), ((23,), (16, 16, 16), False),
    ((128, 3, 4), (16, 2, 2), False), ((2, 3, 4, 5, 6), (16, 16, 16), False), ((4, 4, 4, 4), (16, 16), False),
    ((2, 3, 4, 5), None, False), ((2, 2, 2, 2, 2), (4, 4, 4), False), ((2, 2, 2, 2, 2, 2), (8, 8, 8), False),
    ((2, 3, 4), None, False), ((100,), None, False),
    # the old tolk's cases
    ((7, 7), (49, 1, 1), False), ((2**61, 2**61, 2), (2**60, 2**61, 2**60), False),
    ((2**32, 2**32), (0x7fffffff,), False),
]

SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 16, 23, 32, 64, 100, 121, 128, 256, 512, 1024, 4096, 65536]
BOUNDS = [None, (16, 16, 16), (8, 4, 16), (16, 4, 256), (4, 16, 16), (12, 12, 12), (65535, 65535, 65535),
          (1024, 1024, 64), (256, 256, 256), (16, 16), (64,)]


def sweep():
    rng = random.Random(0)
    cases = [(tuple(rng.choice(SIZES) for _ in range(rng.randint(1, 5))), rng.choice(BOUNDS), rng.random() < 0.5)
             for _ in range(400)]
    return [c for c in dict.fromkeys(cases) if c not in TINYGRAD_CASES]


def render(idx):
    """An index as an expression; ssimplify leaves a constant index as an int."""
    return idx.render(simplify=False) if isinstance(idx, UOp) else str(idx)


def specials(idxs):
    found = {u.arg: u.src[0].arg for idx in idxs if isinstance(idx, UOp) for u in idx.toposort() if u.op is Ops.SPECIAL}
    return " ".join(f"{name}={size}" for name, size in sorted(found.items()))


@table
def grouped_dims():
    rows = []
    for dims, max_sizes, reverse in TINYGRAD_CASES + sweep():
        try:
            idxs = get_grouped_dims("gidx", dims, max_sizes, reverse)
            result, exprs = specials(idxs), "[" + ", ".join(render(i) for i in idxs) + "]"
        except Exception as e:
            result, exprs = type(e).__name__, ""
        rows.append((repr(dims), repr(max_sizes), repr(reverse), result, exprs))
    return ["dims", "max_sizes", "reverse", "specials", "idxs"], rows


def grouped(dims, max_sizes, reverse=False, prefix="gidx"):
    return UOp.sink(*get_grouped_dims(prefix, dims, max_sizes, reverse))


@graph
def grouped_symbolic_crosses_a_limit_by_merging():
    return grouped((1, UOp.variable("n", 1, 4)), (4, 3))


@graph
def grouped_symbolic_merges_into_a_symbolic_size():
    return grouped((UOp.variable("n", 1, 4), UOp.variable("m", 1, 8)), (64,))


@graph
def grouped_symbolic_keeps_a_fitting_size():
    return grouped((UOp.variable("n", 1, 16),), (16, 16, 16))


@graph
def grouped_symbolic_without_bounds():
    return grouped((UOp.variable("n", 1, 10),), None)


@graph
def grouped_symbolic_merges_committed_sizes():
    return grouped((UOp.variable("a", 2, 4, dtypes.int16), UOp.variable("b", 2, 4, dtypes.int32)), (16,))


@graph
def grouped_thread_indices():
    return grouped((2, 3, 4, 5), (16, 16, 16), prefix="lidx")


@graph
def grouped_reversed_merge():
    return grouped((2, 3, 4, 5), (32, 16, 16), reverse=True)


@table
def grouped_symbolic_failures():
    rows = []
    for name, dims, max_sizes in [("split_at_its_maximum", (UOp.variable("n", 1, 32),), (16, 16, 16)),
                                  ("split_above_its_bound", (UOp.variable("n", 17, 32),), (16, 16, 16))]:
        try:
            get_grouped_dims("gidx", dims, max_sizes)
            result = "None"
        except Exception as e:
            result = type(e).__name__
        rows.append((name, result))
    return ["case", "raises"], rows


# add_gpudims

def rng(size, axis, axis_type):
    return UOp.range(size, axis, axis_type)


def kernel(index, value, ranges, size=4096):
    return UOp.param(0, dtypes.float, size).index(index).store(value).end(*ranges).sink(arg=KernelInfo())


def rewritten(sink, ren=None):
    return UOp.sink(sink, graph_rewrite(sink, pm_add_gpudims, ctx=ren or renderer()))


def globals_kernel():
    g0, g1 = rng(32, 0, AxisType.GLOBAL), rng(16, 1, AxisType.GLOBAL)
    return kernel(g0 * 16 + g1, UOp.const(1.0, dtypes.float), (g0, g1))


@graph
def add_gpudims_globals():
    return rewritten(globals_kernel())


@graph
def add_gpudims_globals_by_axis_order():
    g3, g1 = rng(32, 3, AxisType.GLOBAL), rng(16, 1, AxisType.GLOBAL)
    return rewritten(kernel(g3 * 16 + g1, UOp.const(1.0, dtypes.float), (g3, g1)))


@graph
def add_gpudims_globals_and_locals():
    g, l = rng(32, 0, AxisType.GLOBAL), rng(8, 1, AxisType.LOCAL)
    return rewritten(kernel(g * 8 + l, UOp.const(1.0, dtypes.float), (g, l)))


@graph
def add_gpudims_merges_four_globals():
    gs = [rng(n, i, AxisType.GLOBAL) for i, n in enumerate((2, 3, 4, 5))]
    return rewritten(kernel(((gs[0] * 3 + gs[1]) * 4 + gs[2]) * 5 + gs[3], UOp.const(1.0, dtypes.float), gs, 120))


@graph
def add_gpudims_splits_a_global():
    g = rng(1024, 0, AxisType.GLOBAL)
    return rewritten(kernel(g, UOp.const(1.0, dtypes.float), (g,)), renderer(global_max=(256, 256, 256)))


@graph
def add_gpudims_keeps_the_warp_apart():
    w = rng(32, 0, AxisType.WARP)
    ls = [rng(2, i, AxisType.LOCAL) for i in (1, 2, 3)]
    g = rng(4, 4, AxisType.GLOBAL)
    index = (((g * 32 + w) * 2 + ls[0]) * 2 + ls[1]) * 2 + ls[2]
    return rewritten(kernel(index, UOp.const(1.0, dtypes.float), (w, *ls, g)), renderer(local_max=(1024, 1024, 64)))


def symbolic_warp_kernel():
    w = rng(UOp.variable("n", 1, 32), 0, AxisType.WARP)
    l, g = rng(4, 1, AxisType.LOCAL), rng(4, 2, AxisType.GLOBAL)
    return kernel((g * 32 + w) * 4 + l, UOp.const(1.0, dtypes.float), (w, l, g))


@table
def add_gpudims_symbolic_warp():
    """tinygrad puts a symbolic warp's size, a node, among the integers of
    local_max, and fails comparing with it."""
    try:
        add_gpudims(renderer(local_max=(1024, 1024, 64)), symbolic_warp_kernel())
        result = "None"
    except Exception as e:
        result = type(e).__name__
    return ["case", "raises"], [("warp_of_n_in_1_to_32", result)]


@graph
def add_gpudims_bounds_globals_by_threads():
    g, l = rng(256, 0, AxisType.GLOBAL), rng(256, 1, AxisType.LOCAL)
    ren = renderer(global_max=(256, 256, 256), local_max=(128, 128, 128), global_prod_max=(128, 128, 128))
    return rewritten(UOp.param(0, dtypes.float, 512).index(g + l).store(UOp.const(1.0)).end(g, l).sink(arg=KernelInfo()),
                     ren)


@graph
def add_gpudims_bounds_globals_by_merged_threads():
    g = rng(256, 0, AxisType.GLOBAL)
    ls = [rng(4, i, AxisType.LOCAL) for i in (1, 2, 3, 4)]
    ren = renderer(global_max=(256, 256, 256), local_max=(16, 16, 16), global_prod_max=(1024, 1024, 1024))
    index = (((g * 4 + ls[0]) * 4 + ls[1]) * 4 + ls[2]) * 4 + ls[3]
    return rewritten(kernel(index, UOp.const(1.0, dtypes.float), (g, *ls), 65536), ren)


@graph
def add_gpudims_bounds_globals_by_threads_alone():
    g, l = rng(256, 0, AxisType.GLOBAL), rng(256, 1, AxisType.LOCAL)
    ren = renderer(global_max=None, local_max=(128, 128, 128), global_prod_max=(128, 128, 128))
    return rewritten(kernel(g + l, UOp.const(1.0, dtypes.float), (g, l), 512), ren)


@graph
def add_gpudims_keeps_an_end_of_a_variable():
    n = UOp.variable("n", 0, 3)
    return rewritten(UOp.param(0, dtypes.float, 4).index(n).store(UOp.const(1.0, dtypes.float)).end(n).sink())


def missing_locals_kernel(sizes):
    g = rng(32, 0, AxisType.GLOBAL)
    ls = [rng(n, i + 1, AxisType.LOCAL) for i, n in enumerate(sizes)]
    buf = UOp.param(0, dtypes.float, 64)
    loaded = UOp.param(1, dtypes.float, 64).index(g + sum(ls[1:], ls[0])).load()
    return buf.index(g).store(loaded).end(g, *ls).sink(arg=KernelInfo())


@graph
def add_gpudims_masks_a_store_by_its_missing_local():
    return rewritten(missing_locals_kernel((8,)))


@graph
def add_gpudims_masks_a_store_by_its_missing_locals():
    return rewritten(missing_locals_kernel((8, 4)))


@graph
def add_gpudims_keeps_a_reduce_range():
    g, r = rng(16, 0, AxisType.GLOBAL), rng(8, 1, AxisType.REDUCE)
    loaded = UOp.param(1, dtypes.float, 128).index(g * 8 + r).load()
    return rewritten(UOp.param(0, dtypes.float, 16).index(g).store(loaded).end(g, r).sink(arg=KernelInfo()))


@graph
def add_gpudims_leaves_a_local_store_unmasked():
    g, l = rng(32, 0, AxisType.GLOBAL), rng(8, 1, AxisType.LOCAL)
    shared = UOp.param(0, dtypes.float, 32, addrspace=AddrSpace.LOCAL)
    loaded = UOp.param(1, dtypes.float, 256).index(g * 8 + l).load()
    return rewritten(shared.index(g).store(loaded).end(g, l).sink(arg=KernelInfo()))


@graph
def add_gpudims_symbolic_global():
    n = UOp.variable("n", 1, 64)
    g = rng(n, 0, AxisType.GLOBAL)
    return rewritten(kernel(g, UOp.const(1.0, dtypes.float), (g,), 64))


@graph
def add_gpudims_device_range():
    d, g = rng(2, 0, AxisType.DEVICE), rng(4, 1, AxisType.GLOBAL)
    return rewritten(UOp.param(0, dtypes.float, 4).index(g).store(d.cast(dtypes.float)).end(g, d).sink(arg=KernelInfo()))


@graph
def add_gpudims_device_range_without_kernel():
    d = rng(2, 0, AxisType.DEVICE)
    return rewritten(UOp.param(0, dtypes.float, 4).index(d).store(UOp.const(1.0, dtypes.float)).end(d).sink())


@table
def add_gpudims_declines():
    r = rng(4, 0, AxisType.REDUCE)
    g = rng(32, 0, AxisType.GLOBAL)
    s = UOp.special(32, "gidx0")
    cases = {
        "no_kernel_info": kernel(g, UOp.const(1.0, dtypes.float), (g,)).replace(arg=None),
        "hardware_indices": UOp.param(0, dtypes.float, 32).index(s).store(UOp.const(1.0, dtypes.float)).sink(arg=KernelInfo()),
        "no_global_or_local": kernel(r, UOp.const(1.0, dtypes.float), (r,)),
    }
    return ["case", "result"], [(name, repr(add_gpudims(renderer(), sink))) for name, sink in cases.items()]


@table
def add_gpudims_failures():
    g, l = rng(32, 0, AxisType.GLOBAL), rng(8, 1, AxisType.LOCAL)
    loaded = UOp.param(1, dtypes.float, 64).index(g + l).load()
    two = UOp.param(0, dtypes.float, (32, 2)).index(g, UOp.const(0)).store(loaded).end(g, l).sink(arg=KernelInfo())
    try:
        add_gpudims(renderer(), two)
        result = "None"
    except Exception as e:
        result = type(e).__name__
    return ["case", "raises"], [("missing_local_on_a_two_index_store", result)]
