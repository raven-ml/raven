"""Goldens of tinygrad/renderer/__init__.py: the cost estimates of linearized
kernels, accesses restated at another data type, and the data types a renderer
supports.

A kernel is linearized as `to_program` linearizes it, and an input golden holds
one `Ops.LINEAR` per kernel, whose sources are its nodes in order; a table
golden gives each kernel's position among them and its estimates. A symbolic
kernel's estimates are written at each value of its variable `n`.
"""

from dataclasses import replace

from golden import graph, table
from tinygrad import Tensor, Variable, dtypes
from tinygrad.codegen import full_rewrite_to_sink, line_rewrite, pm_alloc_to_buf, pm_linearize_cleanups
from tinygrad.codegen.late.linearizer import linearize
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import EMULATED_DTYPES, Context, Target
from tinygrad.renderer import Estimates, Renderer, with_storage
from tinygrad.renderer.cstyle import ClangRenderer, CUDARenderer
from tinygrad.uop.ops import AxisType, Ops, UOp, sym_infer

CPU = ClangRenderer(Target("CPU", arch="arm64,generic"))
CUDA = CUDARenderer(Target("CUDA", arch="sm_80"))
N = 64


def empty(*shape, dtype=dtypes.float):
    return Tensor.empty(*shape, device="NULL", dtype=dtype)


def kernel(*tensors, renderer=CPU, opts=None):
    """The one kernel that realizing `tensors` compiles, linearized by
    `renderer`, with `opts` in place of the heuristic's if given, and the
    iterations of the loop its launch splits into blocks, or None."""
    linear, _ = Tensor.linear_with_vars(*tensors)
    (ast,) = [c.src[0] for c in linear.src if c.op is Ops.CALL and c.src[0].op is Ops.SINK]
    if opts is not None: ast = ast.replace(arg=replace(ast.arg, opts_to_apply=tuple(opts)))
    sink = full_rewrite_to_sink(ast, renderer, optimize=ast.tag is None)
    lin = UOp(Ops.LINEAR, src=tuple(line_rewrite(linearize(sink), pm_linearize_cleanups + pm_alloc_to_buf)))
    return lin, sink.arg.split


def gemm(dtype=dtypes.float):
    return empty(N, N, dtype=dtype) @ empty(N, N, dtype=dtype)


def self_assign():
    a = empty(1024, 1024, dtype=dtypes.uint8).realize()
    return a.assign(a + a)


def up(axis, amount): return Opt(OptOps.SPLIT, axis, (amount, AxisType.UPCAST))
def local(axis, amount): return Opt(OptOps.SPLIT, axis, (amount, AxisType.LOCAL))


# The kernels of null/test_uops_stats.py
KERNELS = [
    ("add_uint8", lambda: kernel(empty(1024, 1024, dtype=dtypes.uint8) + empty(1024, 1024, dtype=dtypes.uint8))),
    ("add_const_uint8", lambda: kernel(empty(1024, 1024, dtype=dtypes.uint8) + 3)),
    ("add_expanded_uint8", lambda: kernel(empty(1024, 1, dtype=dtypes.uint8).expand(1024, 1024)
                                          + empty(1024, 1024, dtype=dtypes.uint8))),
    ("self_add_uint8", lambda: kernel((a := empty(1024, 1024, dtype=dtypes.uint8)) + a)),
    ("self_add_transposed_uint8", lambda: kernel((a := empty(1024, 1024, dtype=dtypes.uint8)) + a.T)),
    ("self_add_assign_uint8", lambda: kernel(self_assign())),
    ("simple_add", lambda: kernel(empty(100, 100) + empty(100, 100))),
    ("simple_add_sq", lambda: kernel(((a := empty(100, 100)) + (b := empty(100, 100))) * (a + b))),
    ("cat_equal_pieces", lambda: kernel(Tensor.cat(*[empty(256, 128) for _ in range(4)], dim=1))),
    ("cat_unequal_pieces", lambda: kernel(Tensor.cat(*[empty(256, 128) for _ in range(3)], empty(256, 129), dim=1))),
    ("simple_matmul", lambda: kernel(empty(1024, 1024) @ empty(1024, 1024))),
    ("simple_matmul_8192", lambda: kernel(empty(8192, 8192) @ empty(8192, 8192))),
    ("gemm", lambda: kernel(gemm(), opts=[])),
    ("gemm_one_upcasted", lambda: kernel(gemm(), opts=[up(0, 4)])),
    ("gemm_upcasted", lambda: kernel(gemm(), opts=[up(0, 4), up(1, 4), Opt(OptOps.SPLIT, 4, (4, AxisType.UNROLL))])),
    ("gemm_upcasted_locals", lambda: kernel(gemm(), renderer=CUDA, opts=[up(0, 4), up(1, 4), local(0, 4), local(1, 4)])),
    ("gemm_group", lambda: kernel(gemm(), renderer=CUDA, opts=[Opt(OptOps.SPLIT, 2, (4, AxisType.LOCAL))])),
    ("gemm_tc_half", lambda: kernel(gemm(dtypes.half), renderer=CUDA, opts=[Opt(OptOps.TC, 0, (-1, 0, 1))])),
    ("reduce", lambda: kernel(empty(N * N).sum(), opts=[])),
]

n = Variable("n", 1, 64)

SYMBOLIC_KERNELS = [
    ("shrunk_add", lambda: kernel(empty(64, 4)[:n.bind(10)] + 1)),
    ("shrunk_sum", lambda: kernel(empty(64, 4)[:n.bind(10)].sum(axis=0), opts=[])),
    ("shrunk_sum_on_threads", lambda: kernel(empty(64, 256)[:n.bind(10)].sum(axis=0), renderer=CUDA)),
]


def built(kernels): return [(name, *make()) for name, make in kernels]


def linears(kernels): return UOp.sink(*[lin for _, lin, _ in built(kernels)])


# A split kernel's counts are those of one block; the tables count the block
# that runs the whole loop.
def whole(split, var_vals):
    if split is None: return var_vals, "-"
    extent = sym_infer(split, var_vals) if isinstance(split, UOp) else split
    return {**var_vals, "block_lo": 0, "block_hi": extent}, str(extent)


def count(x, var_vals): return sym_infer(x, var_vals) if isinstance(x, UOp) else x


@graph
def kernels(): return linears(KERNELS)


@table
def estimates():
    rows = []
    for i, (name, lin, split) in enumerate(built(KERNELS)):
        e, ignoring = Estimates.from_uops(lin.src), Estimates.from_uops(lin.src, ignore_indexing=True)
        var_vals, blocks = whole(split, {})
        rows.append((name, str(i), blocks, count(e.ops, var_vals), count(ignoring.ops, var_vals),
                     count(e.lds, var_vals), count(e.mem, var_vals)))
    return ["case", "src", "split", "ops", "ops_ignoring_indexing", "lds", "mem"], rows


@graph
def symbolic_kernels(): return linears(SYMBOLIC_KERNELS)


@table
def symbolic_estimates():
    rows = []
    for i, (name, lin, split) in enumerate(built(SYMBOLIC_KERNELS)):
        e, ignoring = Estimates.from_uops(lin.src), Estimates.from_uops(lin.src, ignore_indexing=True)
        for value in (1, 10, 64):
            var_vals, blocks = whole(split, {"n": value})
            rows.append((name, str(i), str(value), blocks, count(e.ops, var_vals), count(ignoring.ops, var_vals),
                         count(e.lds, var_vals), count(e.mem, var_vals)))
    return ["case", "src", "n", "split", "ops", "ops_ignoring_indexing", "lds", "mem"], rows


# Accesses restated at another data type, as renderers restate a bool access as
# a byte access

param = UOp.param(0, dtypes.bool, 16)
r = UOp.range(16, 0)
store = UOp.param(1, dtypes.float, 16).index(r).store(1.0)

ACCESSES = [
    ("index_of_param", param.index(r), dtypes.uint8),
    ("gated_index_of_param", param.index(r, r < 8), dtypes.uint8),
    ("load_of_param", param.index(r).load(), dtypes.uint8),
    ("index_after_store", UOp.param(1, dtypes.float, 16).after(store).index(r), dtypes.int),
    ("index_of_buffer", UOp.new_buffer("CPU", 16, dtypes.half, 3).index(r), dtypes.ushort),
    ("index_of_local_alloc", UOp.alloc((16,), dtypes.float, slot=2, addrspace=AddrSpace.LOCAL).index(r), dtypes.uint),
    ("index_of_register_alloc", UOp.alloc((4,), dtypes.float, slot=3, addrspace=AddrSpace.REG).index(UOp.const(1)),
     dtypes.int),
    ("index_at_its_own_type", param.index(r), dtypes.bool),
]


@graph
def accesses(): return UOp.sink(*[u for _, u, _ in ACCESSES])


@table
def restatements(): return ["case", "src", "dtype"], [(name, str(i), dt) for i, (name, _, dt) in enumerate(ACCESSES)]


@graph
def restated(): return UOp.sink(*[with_storage(u, dt) for _, u, dt in ACCESSES])


# The base renderer's data types

@table
def supported_dtypes():
    rows = []
    for emulated in ["", "long", "int64", "double", "half,long", "ulong", ",long,"]:
        with Context(EMULATED_DTYPES=emulated):
            supported = Renderer(Target("CPU")).supported_dtypes()
        rows.append((repr(emulated), " ".join(repr(d) for d in dtypes.all if d in supported)))
    return ["emulated", "supported"], rows
