"""Goldens of tinygrad/codegen/opt/heuristic.py: the optimisations that
hand_coded_optimizations chooses for real kernels, per renderer.

A kernel golden, named after the kernel, is the sink that compiling a real
kernel hands to `apply_opts`. `cases.golden` lists each case: a kernel, a
renderer, the settings it runs under, and the optimisations chosen. The graph
golden named after a case is what `apply_opts` returns when it applies the
hand-coded optimisations. A setting whose name starts with MV is an
environment variable, which tinygrad reads through the cached `getenv`. `renderers.golden` is what each renderer tells the
heuristic.
"""

import functools
import os

from golden import graph, table
from graph import kernels, stage
from tinygrad import Tensor, UOp, dtypes
from tinygrad.codegen.opt.heuristic import hand_coded_optimizations
from tinygrad.codegen.opt.postrange import Scheduler, apply_opts
from tinygrad.helpers import Context, Target, getenv
from tinygrad.renderer import Renderer, tc
from tinygrad.renderer.cstyle import ClangRenderer, CUDARenderer, HIPRenderer, MetalRenderer
from tinygrad.uop.ops import Ops


def hip(arch):
    """The HIP renderer of `arch`, without its compiler, which needs comgr."""
    r = HIPRenderer.__new__(HIPRenderer)
    Renderer.__init__(r, Target("AMD", "HIP", arch))
    r.tensor_cores = tc.get_amd(arch)
    return r


RENDERERS = {
    "cpu": ClangRenderer(Target("CPU", "CLANG", "x86_64,x86-64")),
    "metal": MetalRenderer(Target("METAL", "METAL", "Apple9")),
    "cuda": CUDARenderer(Target("CUDA", "CUDA", "sm_89")),
    "amd": hip("gfx1100"),
}


@table
def renderers():
    return ["renderer", "device", "arch", "has_local", "has_shared", "shared_max", "tensor_cores"], [
        (name, r.target.device, r.target.arch, r.has_local, r.has_shared, r.shared_max, len(r.tensor_cores))
        for name, r in RENDERERS.items()]


# Kernels

def last(*tensors): return kernels(*tensors)[-1]


def empty(*shape, dtype=dtypes.float): return Tensor.empty(*shape, dtype=dtype)


def decoded(n, k):
    """An [n, k] matrix decoded by a table: each code byte is a row of two
    values, scaled by its group of 32."""
    table, codes, scales = empty(256, 2), empty(n, k // 2, dtype=dtypes.uint8), empty(n, k // 32)
    rows = codes.cast(dtypes.int32).maximum(0).minimum(255).cast(dtypes.weakint)
    values = Tensor(table.uop.index(rows.uop))
    return (values.reshape(n, k // 32, 32) * scales.reshape(n, k // 32, 1)).reshape(n, k)


def routed(blocks, rows, n, k, experts):
    """Blocks of `rows` rows, each block's rows multiplied by the [n, k] matrix of its expert among `experts`, decoded from 4-bit
    codes, two a byte, and a scale per 32: a product of positions sorted by expert, the expert of row r its block's, at r // rows."""
    owner = empty(blocks, dtype=dtypes.int).maximum(0).minimum(experts - 1).cast(dtypes.weakint)
    owner = owner.reshape(blocks, 1).expand(blocks, rows).reshape(blocks * rows)
    def gather(t, m): return Tensor(t.reshape(experts, m).uop.index(owner.uop))
    codes = gather(empty(experts * n * k // 2, dtype=dtypes.uint8), n * k // 2)
    scales = gather(empty(experts * n * k // 32), n * k // 32)
    values = Tensor.stack(codes & 15, codes >> 4, dim=-1).cast(dtypes.float).reshape(blocks * rows, n, k // 32, 32)
    w = (values * scales.reshape(blocks * rows, n, k // 32, 1)).reshape(blocks * rows, n, k)
    return empty(blocks * rows, 1, k) @ w.transpose(1, 2)


def routed_bfloat16(blocks, rows, n, k, experts):
    """`routed` at bfloat16, as a prompt's experts multiply: bfloat16 rows, and codes decoded to bfloat16 values, which hold them
    exactly, each operand widened to float32 for the product."""
    owner = empty(blocks, dtype=dtypes.int).maximum(0).minimum(experts - 1).cast(dtypes.weakint)
    owner = owner.reshape(blocks, 1).expand(blocks, rows).reshape(blocks * rows)
    def gather(t, m): return Tensor(t.reshape(experts, m).uop.index(owner.uop))
    codes = gather(empty(experts * n * k // 2, dtype=dtypes.uint8), n * k // 2)
    scales = gather(empty(experts * n * k // 32, dtype=dtypes.bfloat16), n * k // 32)
    values = Tensor.stack(codes & 15, codes >> 4, dim=-1).cast(dtypes.bfloat16).reshape(blocks * rows, n, k // 32, 32)
    w = (values * scales.reshape(blocks * rows, n, k // 32, 1)).reshape(blocks * rows, n, k)
    return empty(blocks * rows, 1, k, dtype=dtypes.bfloat16).float() @ w.float().transpose(1, 2)


def normed(k):
    """A [1, k] activation divided by its scale and multiplied by a gain, as a normalisation leaves it."""
    return empty(1, k) / empty(1, 1) * empty(k)


def with_vars(*tensors):
    return [c.src[0] for c in Tensor.linear_with_vars(*tensors)[0].src if c.src[0].op is Ops.SINK][-1]


KERNELS = {
    "add": lambda: last(empty(64, 64) + empty(64, 64)),
    "add_broadcast": lambda: last(empty(64, 64) + empty(64, 1)),
    "add_small": lambda: last(empty(4) + 1),
    "add_large": lambda: last(empty(1024, 1024) + 1),
    "add_odd": lambda: last(empty(3, 5, 7) + 1),
    "transpose": lambda: last(empty(64, 128).T.contiguous()),
    "sum": lambda: last(empty(4096).sum()),
    "sum_rows": lambda: last(empty(32, 32).sum(1)),
    "sum_rows_wide": lambda: last(empty(4096, 64).sum(1)),
    "sum_cols": lambda: last(empty(64, 4096).sum(0)),
    "sum_17": lambda: last(empty(512, 17).sum(1)),
    "sum_100": lambda: last(empty(4096, 100).sum(1)),
    "sum_3x3": lambda: last(empty(4096, 3, 3).sum((1, 2))),
    "sum_two_axes": lambda: last(empty(16, 5, 7, 32).sum((1, 3))),
    "outer_add": lambda: last(empty(4, 1) + empty(1, 256)),
    "max_rows": lambda: last(empty(2048, 16).max(1)),
    "matmul": lambda: last(empty(128, 128) @ empty(128, 128)),
    "matmul_half": lambda: last(empty(128, 128, dtype=dtypes.half).matmul(empty(128, 128, dtype=dtypes.half), dtype=dtypes.float)),
    "matmul_half_ragged": lambda: last(empty(17, 23, dtype=dtypes.half).matmul(empty(23, 29, dtype=dtypes.half), dtype=dtypes.float)),
    "batched_matmul_half": lambda: last(empty(4, 64, 64, dtype=dtypes.half).matmul(empty(4, 64, 64, dtype=dtypes.half))),
    "vecmat": lambda: last(empty(1, 4096) @ empty(4096, 1024)),
    "vecmat_of_exp": lambda: last(empty(1, 4096).exp() @ empty(4096, 1024)),
    "vecmat_1000": lambda: last(empty(1, 4096) @ empty(4096, 1000)),
    "matvec": lambda: last(empty(1024, 4096) @ empty(4096, 1)),
    "vecmat_of_cast": lambda: last(empty(1, 4096, dtype=dtypes.bfloat16).float() @ empty(4096, 1024)),
    "vecmat_decoded": lambda: last(empty(1, 4096) @ decoded(1024, 4096).T),
    "vecmat_normed": lambda: last((empty(1, 4096) / empty(1, 1) * empty(4096)) @ empty(4096, 1024)),
    "vecmat_wide": lambda: last(empty(1, 256) @ empty(256, 65536)),
    # gpt-oss-20b's decode products, the experts already selected: a projection of a normalised activation by a bfloat16
    # matrix, and four experts' MXFP4 products, gate and up of the normalised activation, then down of each expert's own
    # activation, scaled and summed over the experts
    "gpt_oss_qkv": lambda: last(normed(2880) @ empty(2880, 4096, dtype=dtypes.bfloat16).float()),
    "gpt_oss_kv": lambda: last(normed(2880) @ empty(2880, 512, dtype=dtypes.bfloat16).float()),
    "gpt_oss_router": lambda: last(normed(2880) @ empty(2880, 32, dtype=dtypes.bfloat16).float()),
    "gpt_oss_gate_up": lambda: last(normed(2880) @ decoded(4 * 5760, 2880).T),
    "gpt_oss_down": lambda: last(((empty(4, 1, 2880) @ decoded(4 * 2880, 2880).reshape(4, 2880, 2880).transpose(1, 2)).relu()
                                  * empty(4, 1, 1)).sum(0)),
    # a prompt's routed product, its positions sorted by expert into blocks of 2 rows that share their expert's matrix
    "routed_blocks": lambda: last(routed(8, 2, 64, 64, 4)),
    # the same at bfloat16 in blocks of 16 rows, a tensor core's tile: the rows are the core's M, each block's matrix its B
    "routed_tiles": lambda: last(routed_bfloat16(4, 16, 64, 64, 4)),
    # a bfloat16 product of a value another kernel stores: attention's heads, merged, then projected
    "projection_of_stored": lambda: last((empty(4, 32, 16, dtype=dtypes.bfloat16).float() @ empty(4, 16, 16, dtype=dtypes.bfloat16).float())
                                         .cast(dtypes.bfloat16).permute(1, 0, 2).reshape(32, 64).float()
                                         @ empty(64, 64, dtype=dtypes.bfloat16).float()),
    # attention's scores, 8 query heads sharing 2 key heads: an operand read through h // 4 that one buffer holds
    "shared_keys": lambda: last(empty(8, 1, 64) @ empty(2, 1, 64, 16).expand(2, 4, 64, 16).reshape(8, 64, 16)),
    "experts_down": lambda: last(((empty(2, 1, 32) @ empty(2, 16, 32).transpose(1, 2)).relu() * empty(2, 1, 1)).sum(0)),
    "conv": lambda: last(empty(1, 16, 32, 32).conv2d(empty(32, 16, 3, 3), padding=1)),
    "conv_half": lambda: last(empty(1, 16, 32, 32, dtype=dtypes.half).conv2d(empty(32, 16, 3, 3, dtype=dtypes.half), padding=1)),
    "stack": lambda: last(Tensor.stack(empty(1024), empty(1024), empty(1024))),
    "pad": lambda: last(empty(1024, 3).pad(((0, 0), (1, 1))) + 1),
    "stack_7": lambda: last(Tensor.stack(*[empty(1024) for _ in range(7)])),
    "stack_8": lambda: last(Tensor.stack(*[empty(1024) for _ in range(8)])),
    "pad_7x7": lambda: last(empty(64, 5, 5).pad(((0, 0), (1, 1), (1, 1))) + 1),
    "pad_7x8": lambda: last(empty(64, 5, 6).pad(((0, 0), (1, 1), (1, 1))) + 1),
    "sum_5_by_3": lambda: last(empty(512, 5, 7, 3).sum((1, 3))),
    "cumsum": lambda: last(empty(256).cumsum()),
    "softmax": lambda: last(empty(32, 256).softmax(-1)),
    # each pair of axes indexes a buffer that the third does not: on a GPU three axes upcast by 4, 64 lanes
    "paired_products": lambda: last((empty(32, 32, 1, 16) * empty(1, 32, 32, 16) + empty(32, 1, 32, 16) * empty(32, 32, 1, 16)).sum(3)),
    # two products each summed over 4, as a tangent of a layer sums its input's tangent by the weights and its input by the
    # weights' tangent: on the host the lanes are filled by an upcast by 2, and the kernel of two reduces is not unrolled past them
    "tangent_products": lambda: last((empty(32, 64, 1, 4) * empty(1, 1, 100, 4)).sum(3) + (empty(1, 64, 1, 4) * empty(32, 1, 100, 4)).sum(3)),
    "variable": lambda: with_vars(empty(1024)[:UOp.variable("n", 1, 1024).bind(512)].contiguous() + 1),
}

# Kernels without a reduce on the host, whose output is upcast until one value fills a vector of 64 bytes, unless their
# operations already hold 5 independent operations for each step of their longest chain
def tree(xs): return xs[0] if len(xs) == 1 else tree(xs[:len(xs) // 2]) + tree(xs[len(xs) // 2:])


HOST_KERNELS = {
    "exp_float": lambda: last(empty(4096).exp()),
    "exp_double": lambda: last(empty(4096, dtype=dtypes.double).exp()),
    "add_half": lambda: last(empty(4096, dtype=dtypes.half) + 1),
    "add_uchar": lambda: last(empty(4096, dtype=dtypes.uint8) + 1),
    "cast_to_double": lambda: last(empty(4096).cast(dtypes.double) + 1),
    "compare": lambda: last(empty(4096) < 1),
    "add_24": lambda: last(empty(24) + 1),
    "add_17": lambda: last(empty(17) + 1),
    "wide_tree": lambda: last(tree([empty(4096) * (1.5 + k) for k in range(64)])),
    "long_chain": lambda: last(functools.reduce(lambda x, k: (x * (1.5 + k)).exp2(), range(64), empty(4096))),
}
KERNELS.update(HOST_KERNELS)

CASES = []


def case(kernel, renderer, suffix="", **context):
    CASES.append((f"{kernel}_{renderer}{suffix}", kernel, renderer, context))


for kernel in KERNELS:
    for renderer in ["cpu"] if kernel in HOST_KERNELS else ["cpu", "metal", "cuda", "amd"]:
        case(kernel, renderer)
for renderer in ["metal", "cuda", "amd"]:
    case("matmul_half", renderer, "_no_tc", TC=0)
    case("matmul_half", renderer, "_tc_shape", TC=2)
    case("matmul_half", renderer, "_tc_min_globals", TC_MIN_GLOBALS=64)
    case("matmul_half", renderer, "_tc_min_globals_1", TC_MIN_GLOBALS=1)
    case("matmul_half_ragged", renderer, "_tc_opt_2", TC_OPT=2)
    case("conv_half", renderer, "_tc_opt_1", TC_OPT=1)
case("matmul_half", "metal", "_tc_select_2", TC_SELECT=2)
case("matmul", "cuda", "_tf32", ALLOW_TF32=1)
# the matrix-vector layouts' switch, from the environment
for renderer in ["metal", "cuda", "amd"]:
    case("vecmat", renderer, "_mv_0", MV=0)


INPUTS = {}


def kernel_input(kernel):
    if kernel not in INPUTS:
        with Context(DEV="CPU"): INPUTS[kernel] = stage("apply_opts", KERNELS[kernel](), RENDERERS["cpu"])
    return INPUTS[kernel]


def optimize(kernel, renderer, context):
    environment = {k: str(v) for k, v in context.items() if k.startswith("MV")}
    settings = {k: v for k, v in context.items() if not k.startswith("MV")}
    kernel = kernel_input(kernel)
    os.environ.update(environment)
    getenv.cache_clear()
    try:
        with Context(**settings):
            return apply_opts(kernel, RENDERERS[renderer])
    finally:
        for k in environment: del os.environ[k]
        getenv.cache_clear()


def context_cell(context): return " ".join(f"{k}={v}" for k, v in context.items())


OUTCOMES = {name: optimize(kernel, renderer, context) for name, kernel, renderer, context in CASES}


@table
def cases():
    return ["case", "kernel", "renderer", "context", "opts"], [
        (name, kernel, renderer, context_cell(context), repr(OUTCOMES[name].arg.applied_opts))
        for name, kernel, renderer, context in CASES]


def declare(name, fn):
    fn.__name__ = name
    graph(fn)


for kernel in KERNELS:
    declare(kernel, lambda kernel=kernel: kernel_input(kernel))
for name, optimized in OUTCOMES.items():
    declare(name, lambda optimized=optimized: optimized)
