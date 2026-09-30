"""Goldens of tinygrad/codegen/late/coalesce.py: simplifying a gated index,
and merging accesses to consecutive elements into vector accesses.

`<kernel>.golden` is the sink that `memory_coalescing` receives when a kernel is
compiled, and `<kernel>_coalesced.golden` its result, for a renderer that
supports vector accesses. `<kernel>_scalar.golden` is the result for one that
does not, and `<kernel>_allow_half8.golden` the result with ALLOW_HALF8=1.

The hand-built cases come as a table naming each case by its position
(`accesses`), an input sink whose sources are the cases (`accesses_input`),
each case a sink of its own, and the sinks of their results in the same order
(`accesses_coalesced`, `accesses_scalar`, `accesses_allow_half8`). The cases of
`indexing_simplify` follow the same shape (`indices`, `indices_input`,
`indices_simplified`).
"""

import os

import tinygrad.codegen
from golden import graph, table
from graph import kernels
from tinygrad import Tensor, Variable, dtypes
from tinygrad.codegen.late.coalesce import indexing_simplify, memory_coalescing
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import Target, getenv
from tinygrad.renderer.cstyle import ClangRenderer, CUDARenderer
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp, graph_rewrite

CPU = ClangRenderer(Target("CPU", "CLANG", "x86_64,x86-64"))
CUDA = CUDARenderer(Target("CUDA", "CUDA", "sm_80"))
SCALAR = ClangRenderer(Target("CPU", "CLANG", "x86_64,x86-64"))
SCALAR.supports_float4 = False


def allow_half8(sink):
    os.environ["ALLOW_HALF8"] = "1"
    getenv.cache_clear()
    return memory_coalescing(sink, CPU)


def given(kernel, renderer):
    """The sink that compiling `kernel` for `renderer` hands to
    memory_coalescing."""
    class Captured(Exception):
        pass

    def capture(sink, ctx): raise Captured(sink)

    coalesce, tinygrad.codegen.memory_coalescing = tinygrad.codegen.memory_coalescing, capture
    try:
        tinygrad.codegen.full_rewrite_to_sink(kernel, renderer, optimize=kernel.tag is None)
    except Captured as e:
        return e.args[0]
    finally:
        tinygrad.codegen.memory_coalescing = coalesce
    raise RuntimeError("compiling the kernel does not coalesce")


def empty(*shape, dtype=dtypes.float): return Tensor.empty(*shape, device="CPU", dtype=dtype)
def split(axis, amount, axis_type=AxisType.UPCAST): return Opt(OptOps.SPLIT, axis, (amount, axis_type))


def with_opts(*opts):
    def apply(kernel): return kernel.replace(arg=KernelInfo(opts_to_apply=opts))
    return apply


def last(tensor): return kernels(tensor)[-1]


def last_with_vars(tensor):
    return [call.src[0] for call in tensor.linear_with_vars()[0].src if call.src[0].op is Ops.SINK][-1]


def add(shape, dtype): return lambda: last(empty(*shape, dtype=dtype) + empty(*shape, dtype=dtype))


def shifted(n): return empty(n + 1).shrink(((1, n + 1),))


x4 = Variable("x", 0, 4, multiple_of=4).bind(4)
x2 = Variable("x", 0, 4, multiple_of=2).bind(4)
conv = lambda n: last(empty(1, 1, n).conv2d(empty(1, 1, 5).shrink(((0, 1), (0, 1), (1, 5)))))

# The kernels of test_gen_float4.py, each with its opts, two of
# test_linearizer.py, then one elementwise kernel per element type, a padded
# one, a reduction and a CUDA kernel.
KERNELS = {
    "basic": (add((2, 8), dtypes.float), [split(0, 4)], CPU),
    "multidim": (add((2, 8), dtypes.float), [split(0, 4), split(0, 2)], CPU),
    "unaligned_load": (lambda: last(shifted(8) + shifted(8)), [split(0, 4)], CPU),
    "multidim_unaligned_load": (lambda: last(empty(2, 9).shrink(((0, 2), (1, 9))) + empty(2, 9).shrink(((0, 2), (1, 9)))),
                                [split(1, 4), split(1, 2)], CPU),
    "sometimes_unaligned": (lambda: conv(8), [split(1, 4, AxisType.UNROLL)], CPU),
    "multidim_sometimes_unaligned": (lambda: conv(7), [split(0, 0), split(1, 0, AxisType.UNROLL)], CPU),
    "expand": (lambda: last(shifted(8) + empty(2).reshape((2, 1)).expand((2, 4)).reshape((8,))), [split(0, 4)], CPU),
    "heterogeneous": (lambda: last(empty(8) + shifted(8)), [split(0, 4)], CPU),
    "aligned_variable": (lambda: last_with_vars(empty(4) + empty(12).shrink(((x4, x4 + 4),))), [split(0, 4)], CPU),
    "unaligned_variable": (lambda: last_with_vars(empty(4) + empty(12).shrink(((x2, x2 + 4),))), [split(0, 4)], CPU),
    "load_dedup": (lambda: (lambda a: last(a[:-1] + a[1:]))(empty(4)), [split(0, 0)], CPU),
    "grouped_store": (lambda: last(empty(64, 64) @ empty(64, 64)),
                      [split(0, 4, AxisType.LOCAL), Opt(OptOps.SPLIT, 3, (8, AxisType.LOCAL, True)),
                       split(3, 4, AxisType.UNROLL), split(0, 4), split(1, 2)], CUDA),
    "half": (add((2, 8), dtypes.half), [split(0, 4)], CPU),
    "half8": (add((2, 16), dtypes.half), [split(0, 8)], CPU),
    "int": (add((2, 8), dtypes.int), [split(0, 4)], CPU),
    "uint": (add((2, 8), dtypes.uint), [split(0, 4)], CPU),
    "fp8e4m3": (add((2, 8), dtypes.fp8e4m3), [split(0, 4)], CPU),
    "fp8e5m2fnuz": (add((2, 8), dtypes.fp8e5m2fnuz), [split(0, 4)], CPU),
    "char": (add((2, 8), dtypes.char), [split(0, 4)], CPU),
    "long": (add((2, 8), dtypes.long), [split(0, 4)], CPU),
    "double": (add((2, 8), dtypes.double), [split(0, 4)], CPU),
    "bfloat16": (add((2, 8), dtypes.bfloat16), [split(0, 4)], CPU),
    "pad": (lambda: last(empty(12).pad(((2, 2),)) + 1), [split(0, 4)], CPU),
    "sum": (lambda: last(empty(4, 16).sum(1)), [split(1, 4, AxisType.UNROLL)], CPU),
    "cuda": (lambda: last(empty(64, 64) + empty(64, 64)), None, CUDA),
}


def declare(name, kernel, opts, renderer):
    def sink():
        k = kernel()
        return given(with_opts(*opts)(k) if opts is not None else k, renderer)
    def coalesced(): return memory_coalescing(sink(), CPU)
    def scalar(): return memory_coalescing(sink(), SCALAR)
    def half8(): return allow_half8(sink())
    sink.__name__, coalesced.__name__, scalar.__name__, half8.__name__ = \
        name, f"{name}_coalesced", f"{name}_scalar", f"{name}_allow_half8"
    graph(sink)
    graph(coalesced)
    graph(scalar)
    if name.startswith("half"): graph(half8)


for name, (kernel, opts, renderer) in KERNELS.items():
    declare(name, kernel, opts, renderer)


# Hand-built accesses

def at(buf, idx): return UOp(Ops.INDEX, src=(buf, idx))


def loads(buf, idxs, arg=None): return [at(buf, i).load(arg=arg) for i in idxs]


def stores(buf, idxs):
    return [at(buf, i).store(UOp.variable(f"v{k}", -1.0, 1.0, buf.dtype)) for k, i in enumerate(idxs)]


def access_cases():
    buf = UOp.param(0, dtypes.float, 64)
    other = UOp.param(1, dtypes.float, 64)
    r = UOp.range(16, 0)
    i = UOp.variable("i", 0, 60)
    m4 = UOp.variable("m4", 0, 60, multiple_of=4)
    m2 = UOp.variable("m2", 0, 60, multiple_of=2)
    g = UOp.variable("g", False, True, dtypes.bool)
    h = UOp.variable("h", False, True, dtypes.bool)
    c = UOp.const
    param = lambda dtype: UOp.param(2, dtype, 64)
    return [
        # constant offsets: runs of four, two and one, each starting where its
        # length divides the offset
        ("four_constants", loads(buf, [c(k) for k in range(4)])),
        ("eight_constants", loads(buf, [c(k) for k in range(8)])),
        ("three_constants", loads(buf, [c(k) for k in range(3)])),
        ("one_constant", loads(buf, [c(5)])),
        ("from_one_to_four", loads(buf, [c(k) for k in range(1, 5)])),
        ("from_two_to_seven", loads(buf, [c(k) for k in range(2, 8)])),
        ("with_a_gap", loads(buf, [c(0), c(1), c(3)])),
        ("reversed", loads(buf, [c(k) for k in reversed(range(4))])),
        ("two_runs", loads(buf, [c(k) for k in [0, 1, 2, 3, 8, 9]])),
        # offsets from a node: node + c, c + node, and the node itself
        ("range_base", loads(buf, [r * 4 + k if k else r * 4 for k in range(4)])),
        ("constant_first", loads(buf, [UOp(Ops.ADD, src=(c(k), r * 4)) if k else r * 4 for k in range(4)])),
        ("mixed_spellings", loads(buf, [r * 4, r * 4 + 1, UOp(Ops.ADD, src=(c(2), r * 4)), r * 4 + 3])),
        ("unknown_divisibility", loads(buf, [i + k if k else i for k in range(4)])),
        ("multiple_of_four", loads(buf, [m4 + k if k else m4 for k in range(4)])),
        ("multiple_of_two", loads(buf, [m2 + k if k else m2 for k in range(4)])),
        ("range_base_from_one", loads(buf, [r * 4 + k for k in range(1, 5)])),
        ("two_bases", loads(buf, [r * 4 + k if k else r * 4 for k in range(2)] + [r * 8 + k if k else r * 8 for k in range(2)])),
        # stores
        ("four_stores", stores(buf, [c(k) for k in range(4)])),
        ("three_stores", stores(buf, [r * 4 + k if k else r * 4 for k in range(3)])),
        ("loads_and_stores", loads(buf, [c(k) for k in range(4)]) + stores(buf, [c(k) for k in range(4, 8)])),
        ("load_and_store_one_element", loads(buf, [c(k) for k in range(2)]) + stores(buf, [c(k) for k in range(2)])),
        ("two_buffers", loads(buf, [c(0), c(1)]) + loads(other, [c(2), c(3)])),
        # gates
        ("one_gate", loads(buf, [(r * 4 + k if k else r * 4).valid(g) for k in range(4)])),
        ("two_gates", loads(buf, [(r * 4 + k if k else r * 4).valid(g if k < 2 else h) for k in range(4)])),
        ("gated_stores", [at(buf, (r * 4 + k if k else r * 4).valid(g)).store(c(float(k), dtypes.float)) for k in range(4)]),
        ("gate_on_some", loads(buf, [(c(k)).valid(g) if k % 2 else c(k) for k in range(4)])),
        ("invalid_index", [at(buf, UOp.invalid()).load(), *loads(buf, [c(0), c(1)])]),
        # the argument of an access
        ("nontemporal", loads(buf, [c(k) for k in range(4)], arg="nontemporal")),
        ("half_nontemporal", loads(buf, [c(0), c(1)], arg="nontemporal") + loads(buf, [c(2), c(3)])),
        # element types
        *[(f"four_{repr(dt).removeprefix('dtypes.')}", loads(param(dt), [c(k) for k in range(4)]))
          for dt in (dtypes.half, dtypes.int, dtypes.uint, *dtypes.fp8s, dtypes.bfloat16, dtypes.double,
                     dtypes.char, dtypes.uchar, dtypes.short, dtypes.long, dtypes.ulong, dtypes.bool)],
        ("eight_half", loads(param(dtypes.half), [c(k) for k in range(8)])),
        ("sixteen_half", loads(param(dtypes.half), [c(k) for k in range(16)])),
        ("six_half_from_two", loads(param(dtypes.half), [c(k) for k in range(2, 8)])),
        ("eight_half_stores", stores(param(dtypes.half), [c(k) for k in range(8)])),
        # storage left alone, and storage that is not
        ("register", loads(UOp.placeholder((16,), dtypes.float, 3, addrspace=AddrSpace.REG), [c(k) for k in range(4)])),
        ("local", loads(UOp.placeholder((16,), dtypes.float, 3, addrspace=AddrSpace.LOCAL), [c(k) for k in range(4)])),
        ("volatile", loads(UOp.param(4, dtypes.float, 16, volatile=True), [c(k) for k in range(4)])),
        ("volatile_view", loads(UOp.param(4, dtypes.uint32, 4, volatile=True).bitcast(dtypes.int32), [c(k) for k in range(4)])),
        ("volatile_stores", stores(UOp.param(4, dtypes.float, 16, volatile=True), [c(k) for k in range(4)])),
        ("bitcast_view", loads(UOp.param(4, dtypes.uint32, 4).bitcast(dtypes.int32), [c(k) for k in range(4)])),
    ]


def cases(): return [(name, UOp.sink(*us)) for name, us in access_cases()]


@table
def accesses():
    return ["case", "src"], [(case, str(k)) for k, (case, _) in enumerate(cases())]


@graph
def accesses_input(): return UOp.sink(*[u for _, u in cases()])


@graph
def accesses_coalesced(): return UOp.sink(*[memory_coalescing(u, CPU) for _, u in cases()])


@graph
def accesses_scalar(): return UOp.sink(*[memory_coalescing(u, SCALAR) for _, u in cases()])


@graph
def accesses_allow_half8(): return UOp.sink(*[allow_half8(u) for _, u in cases()])


# indexing_simplify

def Special(expr, nmax): return UOp.special(nmax, expr)
def Range(n, nmax): return UOp.range(nmax, n)


def gated_load(valid, idx): return UOp.param(0, dtypes.float, 1024).index(idx.valid(valid)).load()


def index_cases():
    gidx0, lidx0 = Special("gidx0", 5), Special("lidx0", 4)
    r0, r1, r2, r3 = Range(0, 4), Range(1, 4), Range(2, 4), Range(3, 4)
    s0, s1, s2 = Range(0, 30), Range(1, 7), Range(2, 2)
    t0, t1, t2 = Range(0, 1024), Range(1, 4), Range(2, 4)
    g = UOp.variable("g", False, True, dtypes.bool)
    i = UOp.variable("i", 0, 60)
    return [
        # the gated loads of test_simplify_valid_idx.py
        ("cumsum", gated_load((gidx0 * 4 + lidx0 < 19).ne(True), gidx0 * 4 + lidx0 - 19)),
        ("within_valid", gated_load(((r0 * 3 + r1) < 8) & ((((r0 * 3 + r1) // 8 + r2 * 3 + r3) % 4) < 2), r0 + r1 + r2 + r3)),
        ("becomes_constant", gated_load((s2 < 1) & (s1 < 6),
                                        (((s1 + s2) + 1) // 7) * -31 + ((((s1 + s2) + 218) // 224 + s0) % 30) * 1568)),
        ("becomes_constant_nested", gated_load(((r0 + r1) < 1).ne(True) & ((r2 + r3) < 1).ne(True),
                                               ((r0 + r1) + (r2 + r3) + 28) // 30)),
        ("non_constant_bound", gated_load((t0 < (t1 * 4 + t2)) & (t0 < -1).ne(True), t0)),
        # a gate that simplifies the index no further than it simplifies alone
        ("unrelated_gate", gated_load(g, i + 1)),
        ("true_gate", UOp.param(0, dtypes.int, 1).index(UOp.const(0).valid(UOp.const(True)))),
        ("index_simplifies_alone", gated_load(g, i * 1 + 0)),
        ("bound_already_known", gated_load(i < 100, i * 1 + 0)),
        ("store", UOp.param(0, dtypes.float, 1024).index((i + 4).valid(i < 4)).store(UOp.const(1.0, dtypes.float))),
        # an ungated index. An image access, through two indices, is an excluded
        # path (README), which the suite tests by itself.
        ("ungated", UOp.param(0, dtypes.float, 1024).index(r0 * 4).load()),
    ]


@table
def indices():
    return ["case", "src"], [(case, str(k)) for k, (case, _) in enumerate(index_cases())]


@graph
def indices_input(): return UOp.sink(*[u for _, u in index_cases()])


@graph
def indices_simplified(): return UOp.sink(*[graph_rewrite(u, indexing_simplify) for _, u in index_cases()])
