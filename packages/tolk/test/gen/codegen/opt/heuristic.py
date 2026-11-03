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
    "avx2": ClangRenderer(Target("CPU", "CLANG", "x86_64,x86-64-v3")),
    "metal": MetalRenderer(Target("METAL", "METAL", "Apple9")),
    "cuda": CUDARenderer(Target("CUDA", "CUDA", "sm_89")),
    "amd": hip("gfx1100"),
}


@table
def renderers():
    return ["renderer", "device", "arch", "has_local", "has_shared", "shared_max", "tensor_cores", "registers", "per_byte_first"], [
        (name, r.target.device, r.target.arch, r.has_local, r.has_shared, r.shared_max, len(r.tensor_cores),
         "" if r.registers is None else f"{r.registers[0]}x{r.registers[1]}", r.per_byte_first)
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


def routed_by_argmax(rows, n, k, experts):
    """Each of `rows` rows multiplied by the [n, k] matrix of its expert, which the kernel finds as the argmax over the experts of a
    score of the row's position and the expert's: a reduction an operand reads that runs in none of the contraction's ranges."""
    position = Tensor.arange(rows, dtype=dtypes.int).reshape(rows, 1)
    scores = (position * 7 + Tensor.arange(experts, dtype=dtypes.int).reshape(1, experts) * 3) % 11
    owner = scores.argmax(1).maximum(0).minimum(experts - 1).cast(dtypes.weakint)
    w = Tensor(empty(experts, n * k).uop.index(owner.uop)).reshape(rows, n, k)
    return empty(rows, 1, k) @ w.transpose(1, 2)


def q8_0(n, k):
    """An [n, k] matrix of Q8_0 blocks: each block of 32 values is 34 bytes, a float16 scale, then 32 signed codes."""
    blocks = empty(n, k // 32, 34, dtype=dtypes.uint8)
    d = (blocks[:, :, 0].cast(dtypes.uint16) | (blocks[:, :, 1].cast(dtypes.uint16) * 256)).bitcast(dtypes.half)
    q = blocks[:, :, 2:].bitcast(dtypes.int8).cast(dtypes.float)
    return (q * d.cast(dtypes.float).reshape(n, k // 32, 1)).reshape(n, k)


def gguf_mxfp4(n, k):
    """An [n, k] matrix of GGUF's MXFP4 blocks: each block of 32 values is 17 bytes, a scale byte, then 16 bytes of codes, value
    j in the low half of byte j and value j + 16 in its high half."""
    blocks = empty(n, k // 32, 17, dtype=dtypes.uint8)
    codes = blocks[:, :, 1:]
    values = Tensor.cat(codes & 15, codes >> 4, dim=-1).cast(dtypes.float)
    return (values * blocks[:, :, 0].cast(dtypes.float).reshape(n, k // 32, 1)).reshape(n, k)


def rows(m):
    """m rows of a bfloat16 activation by a [32, 256] bfloat16 matrix, each widened to float32."""
    return empty(m, 256, dtype=dtypes.bfloat16).float() @ empty(32, 256, dtype=dtypes.bfloat16).float().T


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
    # a tensor core leaves 8192, 2048, 512 and 32 workgroups of a warp on CUDA: its upcasts and local split apply while
    # they leave 100 workgroups, all three, both upcasts, the upcast of M alone, none
    "matmul_half_1024": lambda: last(empty(1024, 1024, dtype=dtypes.half).matmul(empty(1024, 1024, dtype=dtypes.half), dtype=dtypes.float)),
    "matmul_half_512": lambda: last(empty(512, 512, dtype=dtypes.half).matmul(empty(512, 512, dtype=dtypes.half), dtype=dtypes.float)),
    "matmul_half_256": lambda: last(empty(256, 256, dtype=dtypes.half).matmul(empty(256, 256, dtype=dtypes.half), dtype=dtypes.float)),
    # N's upcast by 2 more: a 512-token prompt's product by a 5760-column bfloat16 matrix holds 4 by 4 by 5 by 2, 160 outputs a
    # lane of CUDA's core, and takes it; a product of 1280 by 2560 holds 4 by 5 by 5, 100, leaves the workgroups for it, and 200
    # would pass the bound
    "matmul_bf16_5760": lambda: last(empty(512, 2880, dtype=dtypes.bfloat16).float() @ empty(2880, 5760, dtype=dtypes.bfloat16).float()),
    "matmul_half_1280x2560": lambda: last(empty(1280, 64, dtype=dtypes.half).matmul(empty(64, 2560, dtype=dtypes.half), dtype=dtypes.float)),
    # float32 contractions on a GPU's registers: a product whose rows and columns each one operand does not read holds 8 by 8 outputs
    # a thread, attention's scores per head likewise; a product of 1000 by 1000 leaves its halved axes no power of two of threads
    "matmul_256": lambda: last(empty(256, 64) @ empty(64, 256)),
    "scores": lambda: last(empty(8, 128, 16) @ empty(8, 128, 16).transpose(1, 2)),
    "matmul_1000": lambda: last(empty(1000, 16) @ empty(16, 1000)),
    # gpt-oss-20b's attention values at a decode step: 64 by 64 outputs over 1024 slots, from bfloat16 widened
    "attention_values": lambda: last(empty(64, 1024, dtype=dtypes.bfloat16).float() @ empty(1024, 64, dtype=dtypes.bfloat16).float()),
    # products a variable counts: the workgroups are not known, and every split applies
    "matmul_half_variable": lambda: with_vars(empty(64, 128, 128, dtype=dtypes.half)[:UOp.variable("n", 1, 64).bind(8)]
                                              .matmul(empty(128, 128, dtype=dtypes.half), dtype=dtypes.float)),
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
    # a contraction's operand of least reuse is streamed: a prompt's rows share each element of the matrix, 4, 12 and 32 times
    "rows_m4": lambda: last(rows(4)),
    "rows_m12": lambda: last(rows(12)),
    "rows_m32": lambda: last(rows(32)),
    # 16 rows summing 64 values each: a sum short enough that a tensor core takes it first
    "rows_short": lambda: last(empty(16, 64, dtype=dtypes.bfloat16).float() @ empty(32, 64, dtype=dtypes.bfloat16).float().T),
    # quantised matrices whose codes run along a range other than the first of the reduce: Q8_0's 32 values of a block of 34
    # bytes, and GGUF's MXFP4 blocks of 17 bytes, 16 bytes of codes each, at one token; MXFP4 decoded by a table at 12 tokens
    "q8_0_blocks": lambda: last(empty(1, 256) @ q8_0(64, 256).T),
    "gguf_mxfp4": lambda: last(empty(1, 256) @ gguf_mxfp4(64, 256).T),
    "mxfp4_m12": lambda: last(empty(12, 256) @ decoded(64, 256).T),
    # a prompt's routed product, its positions sorted by expert into blocks of 2 rows that share their expert's matrix
    "routed_blocks": lambda: last(routed(8, 2, 64, 64, 4)),
    # the same at bfloat16 in blocks of 64 rows, tiles of a tensor core: the rows are the core's M, each block's matrix its B
    "routed_tiles": lambda: last(routed_bfloat16(4, 64, 64, 64, 4)),
    # blocks of 16 rows over 256 values, as a prompt's routed products: 16 blocks sharing 2 experts' matrices do 227 multiply-adds
    # per byte of them and take a tensor core, 4 blocks each of its own expert do 28 and stream
    "routed_shared": lambda: last(routed_bfloat16(16, 16, 64, 256, 2)),
    "routed_own": lambda: last(routed_bfloat16(4, 16, 64, 256, 4)),
    # either side of the bound: 9 blocks over 4 experts do 63.9 multiply-adds a byte and stream, 10 blocks do 71.0 and take a core
    "routed_at_bound": lambda: last(routed_bfloat16(9, 16, 64, 256, 4)),
    "routed_past_bound": lambda: last(routed_bfloat16(10, 16, 64, 256, 4)),
    # routed rows whose experts the kernel finds by an argmax over the experts, as a router computed in the same program would
    "routed_by_argmax": lambda: last(routed_by_argmax(16, 64, 256, 8)),
    # 1280 rows in blocks of 16 to 128 over 4 experts, on a core: the core's M upcast keeps a warp's rows in one block where one of
    # its candidates does, 2 rows of tiles at 32 and 4 at 64 and 128 where 5 divides the tiles, and takes 5 at 16
    **{f"routed_run_{c}": (lambda c=c: last(routed_bfloat16(1280 // c, c, 64, 256, 4))) for c in [16, 32, 64, 128]},
    # a bfloat16 product of a value another kernel stores: attention's heads, merged, then projected
    "projection_of_stored": lambda: last((empty(4, 64, 16, dtype=dtypes.bfloat16).float() @ empty(4, 16, 16, dtype=dtypes.bfloat16).float())
                                         .cast(dtypes.bfloat16).permute(1, 0, 2).reshape(64, 64).float()
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


def mxfp4(n, k, dtype):
    """An [n, k] MXFP4 matrix decoded to `dtype`: value j of a row is the low or high half of the row's code byte j // 2, an
    E2M1 value of a sign bit, two exponent bits and a mantissa bit, times its group of 32's scale byte e, 2^(e - 127)."""
    codes = empty(n, k // 2, dtype=dtypes.uint8)
    nibbles = Tensor.stack(codes & 15, codes >> 4, dim=-1).reshape(n, k).cast(dtypes.uint32)
    e, m = (nibbles >> 1) & 3, nibbles & 1
    normal = (((e + 126) << 23) | (m << 22)).bitcast(dtypes.float)
    magnitude = (e == 0).where(m.cast(dtypes.float) * 0.5, normal)
    values = (magnitude.bitcast(dtypes.uint32) | ((nibbles >> 3) << 31)).bitcast(dtypes.float)
    s = empty(n, k // 32, dtype=dtypes.uint8).cast(dtypes.uint32)
    scales = (s == 255).where(float("nan"), (s == 0).where(0x400000, s << 23).bitcast(dtypes.float))
    return (values.reshape(n, k // 32, 32) * scales.reshape(n, k // 32, 1)).reshape(n, k).cast(dtype)


def decoded_rows(rows, dtype, n=16384, k=16384):
    """`rows` rows of `dtype` by an MXFP4 matrix of n by k decoded to `dtype`, each operand widened to float32."""
    x, w = empty(rows, k, dtype=dtype), mxfp4(n, k, dtype)
    return last((x.float() @ w.float().T).cast(dtype))


# Decoded products of a dense MXFP4 matrix by rows of activations, which the Heuristic suite optimises on GPU renderers to
# count how many threads decode each value; kernels without cases
DECODED = {f"decoded_rows_{rows}_{name}": (lambda rows=rows, dtype=dtype: decoded_rows(rows, dtype))
           for rows in [1, 2, 12, 16, 17, 32, 64, 128, 256, 512]
           for name, dtype in [("bfloat16", dtypes.bfloat16), ("half", dtypes.half), ("float", dtypes.float)]}
# the same at 32 by 128, small enough for the suite to evaluate: rows padded into passes (17, 19) and float32 rows streamed past 32
DECODED.update({f"decoded_small_{rows}_{name}": (lambda rows=rows, dtype=dtype: decoded_rows(rows, dtype, 32, 128))
                for rows, name, dtype in [(17, "bfloat16", dtypes.bfloat16), (19, "half", dtypes.half), (64, "float", dtypes.float)]})

# Reductions of few outputs over long sums on the host, whose launch runs its output loop in blocks on the host's cores: 8
# products of 2^20 values, 64 of 2^15 widened from bfloat16, and the sums of 64 columns over 2^17 rows, each one kernel as rune
# lowers it rather than a sum split in two
def whole(f):
    with Context(SPLIT_REDUCEOP=0): return last(f())


HOST_SPLITS = {
    "matvec_tall": lambda: whole(lambda: empty(8, 1 << 20) @ empty(1 << 20, 1)),
    "vecmat_long": lambda: whole(lambda: empty(1, 32768) @ empty(32768, 64, dtype=dtypes.bfloat16).float()),
    "column_sums": lambda: whole(lambda: empty(131072, 64).sum(0)),
}
KERNELS.update(HOST_SPLITS)

CASES = []


def case(kernel, renderer, suffix="", **context):
    CASES.append((f"{kernel}_{renderer}{suffix}", kernel, renderer, context))


for kernel in KERNELS:
    for renderer in ["cpu", "avx2"] if kernel in HOST_KERNELS or kernel in HOST_SPLITS else ["cpu", "avx2", "metal", "cuda", "amd"]:
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
        with Context(DEV="CPU"): INPUTS[kernel] = stage("apply_opts", {**KERNELS, **DECODED}[kernel](), RENDERERS["cpu"])
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


for kernel in [*KERNELS, *DECODED]:
    declare(kernel, lambda kernel=kernel: kernel_input(kernel))
for name, optimized in OUTCOMES.items():
    declare(name, lambda optimized=optimized: optimized)
