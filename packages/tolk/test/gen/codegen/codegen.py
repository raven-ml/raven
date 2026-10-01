"""Goldens of tinygrad/codegen/__init__.py: kernels compiled to programs for
Clang, Metal, CUDA and HIP.

The input golden `kernels` is a sink whose sources are the kernels the suite
compiles, as the scheduler hands them to `to_program`. The table `cases` lists
each case: a kernel by its position among them, a target, the settings it
compiles under, and its outcome, `ok` or the error `to_program` raises. The
graph golden named after an `ok` case is a sink of two sources: the kernel
lowered by `full_rewrite_to_sink`, and the program `to_program` makes, less
its binary, whose sources are the lowered sink with its estimates, the
linearized instructions and the rendered source.

tolk writes each operation on a narrow scalar as a cast of itself to its
type (D17), so the source of a program with such an operation is tinygrad's
source for its instructions with those casts: the text golden
`<case>_narrowed` of a case whose column `narrowed` is `True`.

The kernels are those of tinygrad's tests of the pipeline (test_linearizer*.py,
test_custom_kernel.py, the rows its section of REGRESSIONS.md takes from other
files), the kernels of the old tolk's codegen goldens, and `Tensor` programs
chosen to cover each pass, with `Tensor.empty` in place of realized data,
which schedules the same kernels.
"""

from collections import Counter
from dataclasses import replace
import functools

import tinygrad.runtime.support.compiler_amd as compiler_amd

# Making a HIP renderer makes its compiler, which asserts that comgr is loaded;
# comgr is absent where the goldens are generated, and no case compiles.
compiler_amd.c.DLL._loaded_.add(compiler_amd.comgr.dll.nm)

import tinygrad.runtime.ops_metal as ops_metal

# Making a Metal renderer starts Metal's code generation service, whose threads
# make the processes forked for each golden crash at random.
ops_metal.MetalCompiler.support.MTLCodeGenServiceCreate = lambda name: None

from golden import graph, table, text
from graph import kernels as scheduled
from tinygrad import Tensor, Variable, dtypes, nn
from tinygrad.codegen import do_to_program, full_rewrite_to_sink
from tinygrad.codegen.opt import KernelOptError, Opt, OptOps
from tinygrad.dtype import AddrSpace, Invalid
from tinygrad.helpers import DEV, Context, Target
from tinygrad.renderer.cstyle import ClangRenderer, CUDARenderer, HIPRenderer, MetalRenderer
from tinygrad.schedule.rangeify import BufferizeOpts
from tinygrad.uop.ops import AxisType, GroupOp, KernelInfo, Ops, UOp

TARGETS = {
    "clang": (ClangRenderer, Target("CPU", "CLANG", "x86_64,x86-64")),
    "metal": (MetalRenderer, Target("METAL", "METAL", "Apple9")),
    "cuda": (CUDARenderer, Target("CUDA", "CUDA", "sm_89")),
    "hip": (HIPRenderer, Target("AMD", "HIP", "gfx1100")),
}
GPUS = ("metal", "cuda", "hip")
DEV.value = "CPU"


def renderer(target):
    cls, t = TARGETS[target]
    ren = cls(t)
    # No case compiles: the binary is the source's bytes.
    ren.compiler.compile_cached = lambda src: src.encode()
    return ren


# Optimisations

def upcast(axis, amount): return Opt(OptOps.SPLIT, axis, (amount, AxisType.UPCAST))
def unroll(axis, amount): return Opt(OptOps.SPLIT, axis, (amount, AxisType.UNROLL))
def local(axis, amount): return Opt(OptOps.SPLIT, axis, (amount, AxisType.LOCAL))
def group(axis, amount): return Opt(OptOps.SPLIT, axis, (amount, AxisType.LOCAL, True))
def tensor_core(axis=0): return Opt(OptOps.TC, axis, (-1, 0, 1))


def with_opts(kernel, opts): return kernel.replace(arg=replace(kernel.arg, opts_to_apply=tuple(opts)))


# Kernels of Tensor programs

def empty(*shape, dtype=dtypes.float): return Tensor.empty(*shape, dtype=dtype, device="CPU")


def last(*tensors): return scheduled(*tensors)[-1]


def with_vars(*tensors):
    """The last kernel of a program over bound variables."""
    linear, _ = Tensor.linear_with_vars(*tensors)
    return [c.src[0] for c in linear.src if c.op is Ops.CALL and c.src[0].op is Ops.SINK][-1]


def matmul(n, dt=dtypes.float): return last(empty(n, n, dtype=dt) @ empty(n, n, dtype=dt))


def custom(outputs, *inputs, fxn): return last(Tensor.custom_kernel(outputs, *inputs, fxn=fxn)[0])


def unoptimised(kernel): return with_opts(kernel, [])


def single_kernel_softmax(x):
    # runtime/test_softmax_fusion.py
    nr_dim, r_dim = x.shape
    inp = x.reshape(nr_dim, 1, 1, r_dim).expand(nr_dim, r_dim, 1, r_dim)
    imx = x.reshape(nr_dim, 1, r_dim, 1).expand(nr_dim, r_dim, r_dim, r_dim).max(axis=-2, keepdim=True)
    ss = (inp - imx.detach()).exp().sum(axis=-1, keepdim=True)
    inp = x.reshape(nr_dim, r_dim, 1, 1)
    imx = x.reshape(nr_dim, 1, r_dim, 1).expand(nr_dim, r_dim, r_dim, 1).max(axis=-2, keepdim=True)
    return (inp - imx.detach()).exp().div(ss).reshape(x.shape)


def where_fold():
    a = empty(4, 4)
    b = a.shrink(((1, 2), None)).pad(((1, 2), None)).bool()
    return unoptimised(last(a.assign(b.where(2, a))))


def cat():
    x = Tensor.arange(2**10) + Tensor.empty((), dtype=dtypes.uint32)
    return last(x.cat(x).cat(Tensor.empty(1, dtype=dtypes.uint32)))


n = Variable("n", 1, 64)

PROGRAMS = {
    "add": lambda: last(empty(16, 16) + empty(16, 16)),
    "add_big": lambda: last(empty(1024, 1024) + empty(1024, 1024)),
    "sum": lambda: last(empty(64, 64).sum(1)),
    "sum_all": lambda: last(empty(4096).sum()),
    "big_sum": lambda: last(empty(1 << 20).sum()),
    "max": lambda: last(empty(32, 32).max(0)),
    "matmul": lambda: matmul(16),
    "matmul_big": lambda: matmul(256),
    "matmul_half": lambda: matmul(64, dtypes.half),
    "matmul_half_big": lambda: matmul(256, dtypes.half),
    "matvec": lambda: last(empty(1, 1024) @ empty(1024, 1024)),
    "softmax": lambda: last(empty(8, 32).softmax(-1)),
    "exp_log": lambda: last(empty(16).exp() + empty(16).log() + empty(16).sin()),
    "pow": lambda: last(empty(16) ** empty(16)),
    "pad": lambda: last(empty(15, 15).pad(((1, 2), (0, 1))) * 2),
    "cast_half": lambda: last((empty(16, dtype=dtypes.half) * 3).cast(dtypes.float)),
    "half_math": lambda: last((empty(16, dtype=dtypes.half).exp2() + empty(16, dtype=dtypes.half).sqrt()).cast(dtypes.float)),
    "bf16": lambda: last((empty(16, dtype=dtypes.bfloat16) + 1).cast(dtypes.float)),
    "fp8": lambda: last((empty(16, dtype=dtypes.fp8e4m3).cast(dtypes.float) * 2).cast(dtypes.fp8e4m3)),
    "idiv": lambda: last(empty(64, dtype=dtypes.int) // 7 + empty(64, dtype=dtypes.int) % 5),
    "int8": lambda: last((empty(64, dtype=dtypes.int8) + empty(64, dtype=dtypes.int8)) * 3),
    "long": lambda: last(empty(16, dtype=dtypes.int64) * 3 + 1),
    "uint64": lambda: last(empty(16, dtype=dtypes.uint64) >> 3),
    "bool_chain": lambda: last(((empty(16) > 0) & (empty(16) < 1)) | (empty(16) == 0.5)),
    "where": lambda: last((empty(32) > 0).where(empty(32), 1.0)),
    "argmax": lambda: last(empty(32, 16).argmax(1)),
    "cumsum": lambda: last(empty(64).cumsum()),
    "conv": lambda: last(empty(1, 4, 8, 8).conv2d(empty(8, 4, 3, 3))),
    "conv_big": lambda: last(empty(2, 16, 32, 32).conv2d(empty(32, 16, 3, 3))),
    "max_pool": lambda: last(empty(1, 8, 16, 16).max_pool2d()),
    "reduce_expand": lambda: last(empty(16, 16).sum(1, keepdim=True) + empty(16, 16)),
    "rmsnorm": lambda: last((lambda x: x * ((x * x).mean(-1, keepdim=True) + 1e-5).rsqrt())(empty(4, 64))),
    "layernorm": lambda: last(empty(8, 128).layernorm()),
    "var_mean": lambda: scheduled(*empty(16, 64).var_mean(1))[-1],
    "ffn": lambda: last((lambda x: (x @ empty(64, 128)).silu() * (x @ empty(64, 128)))(empty(1, 64))),
    "attention": lambda: last((lambda q, k, v: (q @ k.T).softmax(-1) @ v)(empty(16, 32), empty(16, 32), empty(16, 32))),
    "embedding": lambda: last(empty(100, 32)[Tensor.empty(4, device="CPU", dtype=dtypes.int32)]),
    "gather": lambda: last(empty(16)[Tensor.empty(8, device="CPU", dtype=dtypes.int32)]),
    "multi_output": lambda: scheduled(*(lambda a: [a + 1, a * 2])(empty(32, 32)))[-1],
    "outer": lambda: last(empty(32)[:, None] * empty(48)[None, :]),
    "transpose": lambda: last(empty(64, 32).T.contiguous()),
    "flip": lambda: last(empty(8, 16).flip(1).contiguous()),
    "arange": lambda: last(Tensor.arange(32).clone()),
    "threefry": lambda: last(Tensor.rand(16)),
    "symbolic": lambda: with_vars(empty(64)[:n].contiguous() * 2 + 1),
    "symbolic_sum": lambda: with_vars(empty(64)[:n].sum()),
    "shard_add": lambda: last(Tensor.empty(16, 16, device="CPU").shard(("CPU:0", "CPU:1"), 0) + 1),
    "shard_sum": lambda: last(Tensor.empty(16, 16, device="CPU").shard(("CPU:0", "CPU:1"), 1).sum(1)),
    # null/test_linearizer.py
    "load_dedup": lambda: last((lambda a: a[:-1] + a[1:])(empty(4))),
    "reduce_upcast": lambda: last(Tensor.conv2d(empty(1, 1, 3), empty(1, 1, 2), padding=1).relu()),
    "zero_fold": lambda: last(Tensor.stack(empty(1), empty(1))),
    "sum_acc_bool": lambda: unoptimised(last(empty(3, dtype=dtypes.bool).sum())),
    "sum_acc_short": lambda: unoptimised(last(empty(3, dtype=dtypes.int16).sum())),
    "sum_acc_half": lambda: unoptimised(last(empty(3, dtype=dtypes.half).sum())),
    "sum_acc_bf16": lambda: unoptimised(last(empty(3, dtype=dtypes.bfloat16).sum())),
    "matmul_acc_half": lambda: unoptimised(last(empty(8, 8, dtype=dtypes.half).matmul(empty(8, 8, dtype=dtypes.half), dtype=dtypes.half))),
    "upcast_with_locals": lambda: last((empty(1, 128) @ empty(128, 128)).relu()),
    # null/test_linearizer_rewrite.py
    "rewrite_reduction": lambda: last((empty(64, 64) * 2).sum(axis=1)),
    # runtime/test_linearizer.py
    "two_nested_range": lambda: unoptimised(last(empty(2).reshape(2, 1).expand(2, 3).sum())),
    "three_nested_range": lambda: unoptimised(last(empty(2).reshape(2, 1).expand(2, 3).expand(2, 2, 3).sum())),
    "range_outer_op_before_phi_nested_range": lambda: unoptimised(last(
        (lambda a, b: (a.reshape(2, 1).expand(2, 3) + b[0]).sum() + b[0])(empty(2), empty(1, 1)))),
    "default_global_reversed": lambda: unoptimised(last(empty(5, 6, 7).shrink(((0, 4), (0, 5), (0, 6))) + 1)),
    "where_fold": where_fold,
    "phi_arange_float": lambda: last(Tensor.arange(5.5, (3.5 * 300), 3.5).clone()),
    "phi_arange_negative": lambda: last(Tensor.arange(-1, -100, -5).clone()),
    "phi_arange_255": lambda: last(Tensor.arange(255).clone()),
    "two_grouped_stores_local": lambda: with_opts(last(single_kernel_softmax(empty(32, 32))), [local(3, 4), local(5, 4)]),
    # null/test_dtype_weak.py::TestNoRedundantWide
    "fancy_index": lambda: last(empty(8, 9, 10, 11, 12)[1, Tensor([0, 1, 2]).reshape(3, 1), 2, Tensor([0, 1]).reshape(1, 2), 2]),
    # null/test_arange.py
    "cat": cat,
    "triu": lambda: last(empty(256, 256).triu()),
    "bitcast_narrower": lambda: last(empty(4).bitcast(dtypes.uint16)),
}

# The kernels of tinygrad's LLaMA with the old tolk's goldens' sizes, by the name
# tinygrad gives each once optimised for Clang.
LLAMA = {"E_8_2": "llama_embedding", "r_2_8": "llama_rmsnorm", "r_2_8_8": "llama_ffn_gate",
         "E_2_2_4": "llama_vector_scale", "r_2_32_8": "llama_output_projection"}


def llama_kernels():
    from extra.models.llama import Transformer
    model = Transformer(dim=8, hidden_dim=16, n_heads=2, n_kv_heads=1, n_layers=1, norm_eps=1e-5, vocab_size=32,
                        max_context=8, jit=False, disable_kv_cache=True)
    for p in nn.state.get_parameters(model): p.replace(Tensor.empty(p.shape, dtype=p.dtype, device="CPU"))
    logits = model.forward(Tensor.empty(1, 2, dtype=dtypes.int, device="CPU"), 0, float("nan"), 0, 1.0, 0.0, 0.0)
    found = {}
    for k in scheduled(logits):
        name = LLAMA.get(full_rewrite_to_sink(k, renderer("clang")).arg.name)
        if name is not None and name not in found: found[name] = k
    if missing := set(LLAMA.values()) - set(found): raise RuntimeError(f"LLaMA kernels not found: {missing}")
    return found


# Kernels built by hand

def ki(name=None, **kwargs): return KernelInfo(**({"name": name} if name else {}), **kwargs)


def param(slot, n, dt=dtypes.float): return UOp.param(slot, dt, n)


def elementwise(name, op, dt=dtypes.float, n=256):
    a, b, c = param(0, n, dt), param(1, n, dt), param(2, n, dt)
    r = UOp.range(n, 0, AxisType.LOOP)
    return c.index(r).store(op(a.index(r).load(), b.index(r).load())).end(r).sink(arg=ki(name, opts_to_apply=()))


def reduced(name, op, n, value=lambda x: x):
    a, out = param(0, n), param(1, 1)
    r = UOp.range(n, 0, AxisType.REDUCE)
    red = UOp(Ops.REDUCE, src=(value(a.index(r).load()), r), arg=(op, 0))
    return out.index(UOp.const(0, dtypes.int)).store(red).sink(arg=ki(name))


def dot_product():
    a, b, out = param(0, 128), param(1, 128), param(2, 1)
    r = UOp.range(128, 0, AxisType.REDUCE)
    red = UOp(Ops.REDUCE, src=(a.index(r).load() * b.index(r).load(), r), arg=(Ops.ADD, 0))
    return out.index(UOp.const(0, dtypes.int)).store(red).sink(arg=ki("dot_product"))


def matmul_small():
    a, b, c = param(0, 16), param(1, 16), param(2, 16)
    i, j, k = UOp.range(4, 0, AxisType.LOOP), UOp.range(4, 1, AxisType.LOOP), UOp.range(4, 2, AxisType.REDUCE)
    red = UOp(Ops.REDUCE, src=(a.index(i * 4 + k).load() * b.index(k * 4 + j).load(), k), arg=(Ops.ADD, 0))
    return c.index(i * 4 + j).store(red).end(i, j).sink(arg=ki("matmul_small"))


def elementwise_2d():
    a, b, c = param(0, 128), param(1, 128), param(2, 128)
    i, j = UOp.range(8, 0, AxisType.LOOP), UOp.range(16, 1, AxisType.LOOP)
    return c.index(i * 16 + j).store(a.index(i * 16 + j).load() + b.index(i * 16 + j).load()).end(i, j) \
        .sink(arg=ki("elementwise_2d"))


def reduce_rows():
    a, out = param(0, 256), param(1, 8)
    i, j = UOp.range(8, 0, AxisType.LOOP), UOp.range(32, 1, AxisType.REDUCE)
    red = UOp(Ops.REDUCE, src=(a.index(i * 32 + j).load(), j), arg=(Ops.ADD, 0))
    return out.index(i).store(red).end(i).sink(arg=ki("reduce_rows"))


def two_outputs():
    a, b, c = param(0, 256), param(1, 256), param(2, 256)
    r = UOp.range(256, 0, AxisType.LOOP)
    x = a.index(r).load()
    return UOp.group(b.index(r).store(x + 1.0), c.index(r).store(x * 2.0)).end(r).sink(arg=ki("two_outputs"))


def gated_store():
    a, b, c = param(0, 256), param(1, 256), param(2, 256)
    r = UOp.range(256, 0, AxisType.LOOP)
    value = (r < 200).where(a.index(r).load() + b.index(r).load(), UOp.invalid())
    return c.index(r).store(value).end(r).sink(arg=ki("gated_store"))


def no_optimize():
    a, b, c = param(0, 256), param(1, 256), param(2, 256)
    r = UOp.range(256, 0, AxisType.LOOP)
    return c.index(r).store(a.index(r).load() + b.index(r).load()).end(r).sink(arg=ki("no_optimize"), tag=1)


def relu():
    a, b = param(0, 256), param(1, 256)
    r = UOp.range(256, 0, AxisType.LOOP)
    x = a.index(r).load()
    return b.index(r).store((UOp.const(0.0, dtypes.float) < x).where(x, 0.0)).end(r).sink(arg=ki("relu"))


def unary(name, op, dt=dtypes.float):
    a, b = param(0, 256, dt), param(1, 256)
    r = UOp.range(256, 0, AxisType.LOOP)
    return b.index(r).store(op(a.index(r).load())).end(r).sink(arg=ki(name))


def mixed_cast():
    a, b, c = param(0, 256, dtypes.half), param(1, 256), param(2, 256)
    r = UOp.range(256, 0, AxisType.LOOP)
    return c.index(r).store(a.index(r).load().cast(dtypes.float) + b.index(r).load()).end(r) \
        .sink(arg=ki("elementwise_cast_f16"))


def two_reductions():
    a, b, c = param(0, 128), param(1, 1), param(2, 1)
    r = UOp.range(128, 0, AxisType.REDUCE)
    x = a.index(r).load()
    zero = UOp.const(0, dtypes.int)
    return UOp.sink(b.index(zero).store(UOp(Ops.REDUCE, src=(x, r), arg=(Ops.ADD, 0))),
                    c.index(zero).store(UOp(Ops.REDUCE, src=(x * x, r), arg=(Ops.ADD, 0))), arg=ki("parallel_reduce"))


def lorenz():
    x, y, z, out = (param(i, 16) for i in range(4))
    r = UOp.range(16, 0, AxisType.LOOP)
    x, y, z = x.index(r).load(), y.index(r).load(), z.index(r).load()
    for _ in range(3):
        dx, dy, dz = 10.0 * (y - x), x * (28.0 - z) - y, x * y - 2.5 * z
        x, y, z = x + 0.0625 * dx, y + 0.0625 * dy, z + 0.0625 * dz
    return out.index(r).store((x + y) + z).end(r).sink(arg=ki("lorenz_fold"))


def gated_loop():
    a, r = UOp.param(0, dtypes.int, 16), UOp.range(16, 0, AxisType.LOOP)
    return a.index(r.valid(r < 8)).store(r.cast(dtypes.int)).end(r).sink(arg=KernelInfo(), tag=1)


def gated_threads():
    a, lidx0 = UOp.param(0, dtypes.int, 4), UOp.special(4, "lidx0")
    return a.index(lidx0.valid(lidx0.ne(0))).store(UOp.const(1).cast(dtypes.int)).sink(arg=KernelInfo(), tag=1)


def inline_const_alu():
    a, b, zero = UOp.param(0, dtypes.int, 1), UOp.param(1, dtypes.int, 1), UOp.const(0)
    alu = b.index(zero).load().alu(Ops.MAX, UOp.const(dtypes.int.min + 1).cast(dtypes.int))
    return UOp.store(a.index(zero), alu).sink(arg=KernelInfo())


def linearizer_fail_1():
    # null/test_linearizer_failures.py::test_fail_1
    c0 = UOp.param(0, dtypes.float, 64)
    c1 = UOp.range(UOp.const(2), 1, AxisType.WEAK)
    c2 = UOp.range(UOp.const(32), 2, AxisType.WEAK)
    c3 = ((c1 * UOp.const(32)) + c2)
    c4 = UOp.param(1, dtypes.float, 163840)
    c5 = UOp.range(UOp.const(2560), 0, AxisType.REDUCE)
    c6 = c4.index(((((((c5 // UOp.const(8)) % UOp.const(8)) * UOp.const(8)) + (c5 % UOp.const(8))) +
                   (((c2 * UOp.const(40)) + (c5 // UOp.const(64))) * UOp.const(64))) + (c1 * UOp.const(81920))))
    c7 = UOp.param(2, dtypes.float, 64)
    c8 = c7.index(c3)
    c9 = ((((c6 + (c8 * UOp.const(-1.0))) * (c6 + (c8 * UOp.const(-1.0)))).reduce(c5, arg=Ops.ADD) *
           UOp.const(0.000390625)) + UOp.const(1e-05)).sqrt().reciprocal()
    return c0.index(c3).store(c9).end(c1, c2).sink(arg=KernelInfo())


def failure_beam_mnist():
    # runtime/test_linearizer_dumb.py::test_failure_beam_mnist
    c0 = UOp.param(0, dtypes.uchar, 4014080)
    c1 = UOp.range(UOp.const(512), 0, AxisType.GLOBAL)
    c2 = UOp.range(UOp.const(784), 1, AxisType.GLOBAL)
    c3 = UOp.range(UOp.const(10), 3, AxisType.GLOBAL)
    c4 = UOp.param(1, dtypes.int, 512)
    c5 = c4.index(c1.valid(UOp.const(True)))
    c6 = UOp.range(UOp.const(6000), 1004, AxisType.REDUCE)
    c7 = UOp.range(UOp.const(3750), 2006, AxisType.REDUCE)
    c8 = UOp.range(UOp.const(16), 2007, AxisType.LOCAL)
    c9 = UOp.param(2, dtypes.uchar, 47040000)
    c10 = c9.index((((c3 * UOp.const(4704000)) + c2) + (c6 * UOp.const(784))).valid(UOp.const(True)))
    c11 = c5.alu(Ops.CMPNE, ((((c3 * UOp.const(6000)) + c6) + ((c7 * UOp.const(16)) + c8)).alu(Ops.CMPLT, UOp.const(59999))
                            .where(UOp.const(0).cast(dtypes.int), UOp.const(1).cast(dtypes.int)).reduce(c7, c8, arg=Ops.ADD) +
                            UOp.const(-1).cast(dtypes.int))).where(UOp.const(0).cast(dtypes.uchar), c10).reduce(c6, arg=Ops.ADD)
    c12 = c0.index((((c1 * UOp.const(7840)) + (c2 * UOp.const(10))) + c3).valid(UOp.const(True))).store(c11).end(c1, c2, c3)
    return c12.sink(arg=KernelInfo(name='test', applied_opts=(local(4, 16),), opts_to_apply=None))


def reduce_shapeless_const_unroll():
    # null/test_uop_graph.py::TestReduceCollapse::test_reduce_shapeless_const_unroll
    out = UOp.param(0, dtypes.float, 1)
    red = UOp.const(3.0).cast(dtypes.float).reduce(UOp.range(4, 0, AxisType.UNROLL), arg=(Ops.ADD, 0))
    return UOp.sink(out.index(UOp.const(0)).store(red)).replace(arg=KernelInfo())


def sqrt_of_int():
    """A float operation of an int operand, which the lowering casts first."""
    a, out = UOp.param(0, dtypes.int, 16), UOp.param(1, dtypes.float, 16)
    r = UOp.range(16, 0, AxisType.LOOP)
    return out.index(r).store(UOp(Ops.SQRT, src=(a.index(r).load(),))).end(r).sink(arg=KernelInfo())


def invalid_lanes():
    """rune's fold of a float32 [2; 1] with output size [1; 1], kernel [1; 2], stride [1; 1], dilation [1; 2] and
    padding [(0, 0); (1, 1)]: every window lies in the padding, so each lane of the unrolled reduce reads Invalid and
    the vector folds to a scalar that the reduce's lanes still index, which the renderers write as components of a
    scalar that no compiler takes. rune now lowers such a fold to zeros."""
    out, x = UOp.param(0, dtypes.float, 1, device="CPU"), UOp.param(1, dtypes.float, 2, device="CPU")
    r1, r0 = UOp.range(4, 1, AxisType.REDUCE), UOp.range(2, 0, AxisType.REDUCE)
    j = r1 * 3 + 1
    value = ((r1 < 3) & ((r0 < 1) & (j % 5 < 1))).where(x.index((r1 < 3).where(j // 5, UOp.const(Invalid))), 0.0)
    return out.index(UOp.const(0)).store(value.reduce(r0, r1, arg=Ops.ADD) + 0.0).sink(arg=KernelInfo())



def invalid_lanes_int8():
    """rune's fold of an int8 [2; 1] with output size [1], kernel [2], dilation [2] and padding [(1, 1)], summed in
    uint: every window lies in the padding, so the unrolled sum folds to a constant that its lanes still index, as the
    float32 fold's lanes do."""
    out, x = UOp.param(0, dtypes.char, 1, device="CPU"), UOp.param(1, dtypes.char, 2, device="CPU")
    r = UOp.range(4, 0, AxisType.REDUCE)
    j = r * 3 + 1
    value = ((r < 3) & (j % 5 < 1)).where(x.index((r < 3).where(j // 5, UOp.const(Invalid))), UOp.const(0))
    return out.index(UOp.const(0)).store(value.cast(dtypes.uint).reduce(r, arg=Ops.ADD).cast(dtypes.char)).sink(arg=KernelInfo())

def dependent_loop_bound():
    # null/test_linearizer_rewrite.py::test_dependent_loop_bound
    buf, out, counts = UOp.param(0, dtypes.int, 16), UOp.param(1, dtypes.int, 4), UOp.param(2, dtypes.int, 4)
    outer = UOp.range(4, 0, AxisType.LOOP)
    inner = UOp.range(counts.index(outer).load().maximum(0).minimum(4), 1)
    store = buf.index(outer * 4 + inner).store(UOp.const(1, dtypes.int)).end(inner)
    return out.after(store).index(outer).store(UOp.const(2, dtypes.int)).end(outer) \
        .sink(arg=KernelInfo(opts_to_apply=()))


# The old tolk's lowering tests, as kernels compiled without optimising

def unoptimised_kernel(*stores): return UOp.sink(*stores, arg=KernelInfo(), tag=1)


def shared_loop(unbounded, same_buffer, barrier):
    """A loop that writes a local buffer, then reads it or another one back."""
    loop = UOp.loop(19) if unbounded else UOp.range(2, 19, AxisType.LOOP)
    def local(slot): return UOp.alloc((1,), dtypes.int, slot=slot, addrspace=AddrSpace.LOCAL)
    shared, zero = local(10), UOp.const(0)
    inp, out = UOp.param(0, dtypes.int, 1), UOp.param(1, dtypes.int, 1)
    write = shared.after(loop).index(zero).store(inp.index(zero).load())
    read = (shared if same_buffer else local(11)).after(write).index(zero).load()
    body = out.index(zero).store(read)
    if barrier: body = UOp(Ops.BARRIER, src=(body,))
    cond = UOp.param(2, dtypes.bool)
    return unoptimised_kernel(body.backedge(loop, cond) if unbounded else body.end(loop))


def shared_local_axis_loop():
    """A loop over threads that writes local memory and reads it back."""
    zero, loop = UOp.const(0), UOp.range(2, 19, AxisType.LOCAL)
    shared = UOp.alloc((1,), dtypes.int, slot=10, addrspace=AddrSpace.LOCAL)
    inp, out = UOp.param(0, dtypes.int, 1), UOp.param(1, dtypes.int, 1)
    write = shared.after(loop).index(zero).store(inp.index(zero).load())
    return unoptimised_kernel(out.index(zero).store(shared.after(write).index(zero).load()).end(loop))


def shared_reduce_range():
    """Two sums over one reduce range, each inside a loop of its own."""
    a, b = UOp.param(0, dtypes.float, 32), UOp.param(1, dtypes.float, 32)
    o1, o2 = UOp.param(2, dtypes.float, 4), UOp.param(3, dtypes.float, 4)
    i, j, k = UOp.range(4, 0, AxisType.LOOP), UOp.range(4, 1, AxisType.LOOP), UOp.range(8, 2, AxisType.REDUCE)
    s1 = a.index(i * 8 + k).load().reduce(k, arg=Ops.ADD)
    s2 = b.index(j * 8 + k).load().reduce(k, arg=Ops.ADD)
    return UOp.sink(o1.index(i).store(s1).end(i), o2.index(j).store(s2).end(j), arg=KernelInfo(opts_to_apply=()))


def shared_reduce_ranges():
    """Three sums over two reduce ranges, each inside a loop of its own."""
    k, l = UOp.range(4, 3, AxisType.REDUCE), UOp.range(4, 4, AxisType.REDUCE)
    stores = []
    for n in range(3):
        a, o, i = UOp.param(n, dtypes.float, 64), UOp.param(3 + n, dtypes.float, 4), UOp.range(4, n, AxisType.LOOP)
        stores.append(o.index(i).store(a.index(i * 16 + k * 4 + l).load().reduce(k, l, arg=Ops.ADD)).end(i))
    return UOp.sink(*stores, arg=KernelInfo(opts_to_apply=()))


def explicit_local_slots():
    """A reduction staged in local memory, next to a local buffer of slot 17."""
    zero = UOp.const(0)
    explicit = UOp.alloc((1,), dtypes.int, slot=17, addrspace=AddrSpace.LOCAL)
    initialized = explicit.after(explicit.index(zero).store(UOp.const(3, dtypes.int)))
    inp, out = UOp.param(0, dtypes.int, 4), UOp.param(1, dtypes.int, 1)
    r = UOp.range(4, 0, AxisType.REDUCE)
    staged = inp.index(r).load().reduce(r, arg=Ops.ADD).bufferize(arg=BufferizeOpts(None, AddrSpace.LOCAL))
    return unoptimised_kernel(out.index(zero).store(staged + initialized.index(zero).load()))


def anonymous_local():
    zero = UOp.const(0)
    local = UOp.alloc((4,), dtypes.float, slot=17, addrspace=AddrSpace.LOCAL)
    stored = local.index(zero).store(UOp.const(3.0, dtypes.float))
    return unoptimised_kernel(UOp.param(0, dtypes.float, 1).index(zero).store(local.after(stored).index(zero).load()))


def weak_index_store():
    p, r = UOp.param(0, dtypes.float, 8), UOp.range(8, 0)
    return unoptimised_kernel(p.index(r).store(UOp.const(1.0, dtypes.float)).end(r))


def negated_index():
    zero = UOp.const(0)
    return unoptimised_kernel(UOp.param(1, dtypes.float, 1).index(zero).store(-UOp.param(0, dtypes.float, 1).index(zero)))


def numbered_variables():
    n, m = UOp.variable("n", 0, 8), UOp.variable("m", 0, 8)
    out = UOp.param(0, dtypes.int, 1)
    return unoptimised_kernel(out.index(UOp.const(0)).store((n + m).cast(dtypes.int)))


def comparison_extrema():
    """Comparisons of a long with the bounds of its type, each stored."""
    x, out = UOp.param(1, dtypes.int64), UOp.param(0, dtypes.bool, 4)
    lo, hi = UOp.const(-2**63), UOp.const(2**63 - 1)
    predicates = [(hi < x) & (x < UOp.const(-2**63 + 1)), (x < lo).logical_not(), (hi < x).logical_not(), (x * -1) < lo]
    return UOp.sink(*[out.index(UOp.const(i)).store(p) for i, p in enumerate(predicates)], arg=KernelInfo())


# Custom kernels (runtime/test_custom_kernel.py)

def custom_arange(C):
    i = UOp.range(C.shape[0], 0)
    return C[i].store(i.cast(C.dtype)).end(i).sink(arg=KernelInfo(name=f"custom_arange_{C.shape[0]}"))


def custom_eye(C):
    i, j = UOp.range(C.shape[0], 0), UOp.range(C.shape[1], 1)
    return C[i, j].store((i.eq(j)).cast(C.dtype)).end(i, j).sink(arg=KernelInfo(name=f"custom_eye_{C.numel()}"))


def custom_sum(B, A):
    i = UOp.range(A.shape[0], 0, axis_type=AxisType.REDUCE)
    B = B[0].set(0.0)
    B = B[0].set(B.after(i)[0] + A[i], end=i)
    return B.sink(arg=KernelInfo(name=f"custom_sum_{A.shape[0]}", opts_to_apply=()))


def custom_gemm(C, A, B):
    i, j, k = UOp.range(C.shape[0], 0), UOp.range(C.shape[1], 1), UOp.range(A.shape[1], 2, axis_type=AxisType.REDUCE)
    C = C[i, j].set(0.0)
    prog = C[i, j].store(C.after(k)[i, j] + A[i, k] * B[k, j]).end(k).end(i, j)
    return prog.sink(arg=KernelInfo(name=f"custom_gemm_{C.shape[0]}_{C.shape[1]}_{A.shape[1]}", opts_to_apply=()))


def custom_add(C, A, B):
    C, A, B = C.flatten(), A.flatten(), B.flatten()
    i = UOp.range(C.numel(), 0)
    return C[i].store(A[i] + B[i]).end(i).sink(arg=KernelInfo(name=f"custom_add_kernel_{C.numel()}")).simplify()


def flip_contract(dest, src):
    i, j = UOp.range(dest.shape[0], 0), UOp.range(dest.shape[1], 1, AxisType.UPCAST)
    vec = src[i, j].contract(j)
    store = UOp.group(*[dest[i, k].store(vec.index(3 - k)) for k in range(4)])
    return store.end(i, j).sink(arg=KernelInfo(name=f"flip_contract_{dest.numel()}", opts_to_apply=()))


def slice_sum(dest, src):
    G = UOp.range(src.shape[0], 0, dtype=dtypes.int)
    reg = UOp.placeholder((1,), dest.dtype, 0, addrspace=AddrSpace.REG)
    reg = reg.after(G)[0].set(0)
    R = UOp.range(src.shape[1], 1, AxisType.REDUCE)
    reg = reg[0].set(reg.after(R)[0] + src[G, :][R], end=R)
    return dest[G].set(reg[0], end=G).sink(arg=KernelInfo(name=f"slice_sum_{src.shape[0]}_{src.shape[1]}", opts_to_apply=()))


def simple_qkv(O, Q, K, V):
    N, d = Q.shape[0], Q.shape[1]
    i, d_out = UOp.range(N, 0), UOp.range(d, 1)
    j, k_inner = UOp.range(N, 2, axis_type=AxisType.REDUCE), UOp.range(d, 3, axis_type=AxisType.REDUCE)
    qk_acc = UOp.placeholder((1,), Q.dtype, 0, addrspace=AddrSpace.REG)
    qk_acc = qk_acc.after(i, j)[0].set(0.0)
    qk_acc = qk_acc[0].set(qk_acc.after(k_inner)[0] + Q[i, k_inner] * K[j, k_inner], end=k_inner)
    out_acc = UOp.placeholder((1,), Q.dtype, 1, addrspace=AddrSpace.REG)
    out_acc = out_acc.after(i, d_out)[0].set(0.0)
    out_acc = out_acc[0].set(out_acc.after(j)[0] + qk_acc[0] / (d ** 0.5) * V[j, d_out], end=j)
    return O[i, d_out].store(out_acc[0]).end(d_out).end(i).sink(arg=KernelInfo(name=f"simple_qkv_{N}_{d}", opts_to_apply=()))


def group_reduce_split_range(C, A):
    i, j = UOp.range(4, 0), UOp.range(8, 1, AxisType.LOCAL)
    return C[i].store((A[i, j] * (j % 2).cast(A.dtype)).reduce(j, arg=Ops.ADD)).end(i).sink(arg=KernelInfo(opts_to_apply=()))


def nested_group_reduce(C, B):
    i, g1, g2 = UOp.range(4, 0), UOp.range(4, 1, AxisType.LOCAL), UOp.range(8, 2, AxisType.LOCAL)
    return C[i].store(B[i, g1, g2].reduce(g2, arg=Ops.ADD).reduce(g1, arg=Ops.ADD)).end(i).sink(arg=KernelInfo(opts_to_apply=()))


def thread_reduce(axis_type):
    def kernel(C, A):
        i, j = UOp.range(4, 0), UOp.range(8, 1, axis_type)
        return C[i].store(A[i, j].reduce(j, arg=Ops.ADD)).end(i).sink(arg=KernelInfo(opts_to_apply=()))
    return kernel


def stage_then_reduce(addrspace):
    def kernel(C, A):
        i, j, jj = UOp.range(4, 0), UOp.range(8, 1, AxisType.LOOP), UOp.range(8, 2, AxisType.LOOP)
        stage = (A[i, j] * 2).bufferize(j, arg=BufferizeOpts(None, addrspace))
        return C[i].store(stage.index(jj).reduce(jj, arg=Ops.ADD)).end(i).sink(arg=KernelInfo(opts_to_apply=()))
    return kernel


def reg_placeholder_then_reduce(C, A):
    i, j = UOp.range(4, 0), UOp.range(8, 1, AxisType.REDUCE)
    reg = UOp.placeholder((1,), dtypes.float, 0, addrspace=AddrSpace.REG)
    reg = reg.after(i)[0].set(A[i, 0])
    return C[i].store(A[i, j].reduce(j, arg=Ops.ADD) + reg[0]).end(i).sink(arg=KernelInfo(opts_to_apply=()))


def split_range_id_free_of_loop(C, A):
    r, l = UOp.range(4, 0), UOp.loop(1)
    cnt = UOp.placeholder((1,), dtypes.int, slot=0, addrspace=AddrSpace.REG)
    cnt = cnt.after(r)[0].set(0)
    cnt = cnt.after(cnt[0].store(nxt := cnt.after(l)[0] + 1).backedge(l, nxt < 3))
    return C[r].set(A[r] + cnt[0].cast(C.dtype), end=r).sink(arg=KernelInfo(opts_to_apply=(upcast(0, 2),)))


def sharded():
    devs = ("CPU:0", "CPU:1")
    c = Tensor(Tensor.empty(8, 16, device=devs).uop.unshard(0), device=devs)
    return custom(c, empty(16, 16).shard(devs, axis=0), empty(16, 16).shard(devs, axis=0), fxn=custom_add)


HAND = {
    # the old tolk's codegen goldens
    "elementwise_add": lambda: elementwise("elementwise_add", lambda a, b: a + b),
    "elementwise_int32": lambda: elementwise("elementwise_int32", lambda a, b: a + b, dtypes.int32),
    "sum_reduce": lambda: reduced("sum_reduce", Ops.ADD, 256),
    "max_reduce": lambda: reduced("max_reduce", Ops.MAX, 64),
    "dot_product": dot_product,
    "matmul_small": matmul_small,
    "elementwise_2d": elementwise_2d,
    "reduce_rows": reduce_rows,
    "no_optimize": no_optimize,
    "two_outputs": two_outputs,
    "gated_store": gated_store,
    "elementwise_where": relu,
    "elementwise_cast_f16": mixed_cast,
    "elementwise_sqrt": lambda: unary("elementwise_sqrt", lambda x: UOp(Ops.SQRT, src=(x,))),
    "parallel_reduce": two_reductions,
    "lorenz_fold": lorenz,
    # renderer tests of tinygrad and the Cstyle suite
    "gated_loop": gated_loop,
    "gated_threads": gated_threads,
    "inline_const_alu": inline_const_alu,
    "linearizer_fail_1": linearizer_fail_1,
    "dependent_loop_bound": dependent_loop_bound,
    "failure_beam_mnist": failure_beam_mnist,
    "shared_loop_barrier": lambda: shared_loop(False, True, False),
    "shared_backedge_barrier": lambda: shared_loop(True, True, False),
    "shared_backedge_two_buffers": lambda: shared_loop(True, False, False),
    "shared_backedge_barrier_kept": lambda: shared_loop(True, True, True),
    "explicit_local_slots": explicit_local_slots,
    "shared_local_axis_loop": shared_local_axis_loop,
    "shared_reduce_range": shared_reduce_range,
    "shared_reduce_ranges": shared_reduce_ranges,
    "anonymous_local": anonymous_local,
    "weak_index_store": weak_index_store,
    "negated_index": negated_index,
    "numbered_variables": numbered_variables,
    "comparison_extrema": comparison_extrema,
    "weak_product_permuted": lambda: last((empty(2, 4, dtype=dtypes.int32) * 2147483648).reshape(4, 2).permute(1, 0).contiguous()),
    "reduce_shapeless_const_unroll": reduce_shapeless_const_unroll,
    "sqrt_of_int": sqrt_of_int,
    "invalid_lanes": invalid_lanes,
    "invalid_lanes_int8": invalid_lanes_int8,
    # runtime/test_custom_kernel.py
    "custom_arange": lambda: custom(empty(16), fxn=custom_arange),
    "custom_eye": lambda: custom(empty(8, 8), fxn=custom_eye),
    "custom_sum": lambda: custom(empty(1), empty(64), fxn=custom_sum),
    "custom_gemm": lambda: custom(empty(16, 16), empty(16, 16), empty(16, 16), fxn=custom_gemm),
    "flip_contract": lambda: custom(empty(8, 4), empty(8, 4), fxn=flip_contract),
    "slice_sum": lambda: custom(empty(8), empty(8, 16), fxn=slice_sum),
    "simple_qkv": lambda: custom(empty(8, 4), empty(8, 4), empty(8, 4), empty(8, 4), fxn=simple_qkv),
    "group_reduce_split_range": lambda: custom(empty(4), empty(4, 8), fxn=group_reduce_split_range),
    "nested_group_reduce": lambda: custom(empty(4), empty(4, 4, 8), fxn=nested_group_reduce),
    "local_reduce": lambda: custom(empty(4), empty(4, 8), fxn=thread_reduce(AxisType.LOCAL)),
    "warp_reduce": lambda: custom(empty(4), empty(4, 8), fxn=thread_reduce(AxisType.WARP)),
    "stage_then_reduce": lambda: custom(empty(4), empty(4, 8), fxn=stage_then_reduce(AddrSpace.LOCAL)),
    "reg_stage_then_reduce": lambda: custom(empty(4), empty(4, 8), fxn=stage_then_reduce(AddrSpace.REG)),
    "reg_placeholder_then_reduce": lambda: custom(empty(4), empty(4, 8), fxn=reg_placeholder_then_reduce),
    "split_range_id_free_of_loop": lambda: custom(empty(4), empty(4), fxn=split_range_id_free_of_loop),
    "sharded_custom_add": sharded,
}

# Cases with the optimisations they ask for: a kernel of PROGRAMS or HAND, and
# the kernel's opts_to_apply.
OPTS = {
    "sum_group": ("sum", [group(1, 16)]),
    "sum_unroll": ("sum", [unroll(1, 4)]),
    "max_group": ("max", [local(1, 8)]),
    "matmul_opts": ("matmul", [unroll(2, 4), upcast(0, 4), upcast(1, 4)]),
    "matmul_local": ("matmul", [unroll(2, 4), local(0, 4), local(1, 4), upcast(0, 2)]),
    "matmul_tc": ("matmul_half", [tensor_core()]),
    "matmul_noopt": ("matmul", []),
    "add_up": ("add", [upcast(0, 4), local(0, 4)]),
    "load_dedup_upcast": ("load_dedup", [upcast(0, 0)]),
    "reduce_upcast_unroll": ("reduce_upcast", [upcast(0, 0), unroll(1, 0)]),
    "zero_fold_upcast": ("zero_fold", [upcast(0, 0)]),
    "upcast_with_locals_opts": ("upcast_with_locals", [local(1, 8), local(0, 4), upcast(0, 4)]),
    "rewrite_reduction_opts": ("rewrite_reduction", [upcast(0, 4), unroll(2, 4)]),
    "arange_upcast": ("arange", [upcast(0, 4)]),
    # runtime/test_tensor_cores.py::test_tensor_cores_unroll_phi: the reduce
    # axis after the tensor core is Metal's
    "matmul_tc_unroll": ("matmul_half", [tensor_core(), unroll(4, 2)]),
}

# Cases that compile a kernel under settings: a kernel and the settings.
SETTINGS = {
    "matmul_noopt_setting": ("matmul", "NOOPT=1"),
    "idiv_fast": ("idiv", "DISABLE_FAST_IDIV=0"),
    "exp_log_transcendental": ("exp_log", "TRANSCENDENTAL=2"),
    "long_emulated": ("long", "EMULATED_DTYPES=long"),
    "matmul_half_tc_shaped": ("matmul_half", "TC=2"),
    "matmul_half_no_tc": ("matmul_half", "TC=0"),
    "triu_noopt": ("triu", "NOOPT=1"),
}

# The targets of a case, where it is not every target: kernels with thread axes
# of their own need a target with threads.
ON_GPUS = {"gated_threads", "group_reduce_split_range", "nested_group_reduce", "local_reduce", "warp_reduce",
           "stage_then_reduce", "matmul_small", "elementwise_2d", "default_global_reversed", "two_grouped_stores_local",
           "failure_beam_mnist", "shared_loop_barrier", "shared_backedge_barrier", "shared_backedge_two_buffers",
           "shared_backedge_barrier_kept", "explicit_local_slots", "anonymous_local", "shared_local_axis_loop"}


def build():
    """Every case, and the kernels they compile, in order."""
    made = {name: make() for name, make in {**PROGRAMS, **HAND}.items()}
    made.update(llama_kernels())
    cases = [(name, name, "") for name in made]
    cases += [(name, base, "") for name, base in [(c, b) for c, (b, _) in OPTS.items()]]
    cases += [(name, base, setting) for name, (base, setting) in SETTINGS.items()]
    kernel_of = {name: made[base] if name not in OPTS else with_opts(made[base], OPTS[name][1])
                 for name, base, _ in cases}
    order = list(dict.fromkeys(kernel_of.values()))
    return [(name, order.index(kernel_of[name]), setting) for name, _, setting in cases], order


CASES, KERNELS = build()


ON_METAL = {"matmul_tc_unroll"}


def targets(case): return ("metal",) if case in ON_METAL else GPUS if case in ON_GPUS else tuple(TARGETS)


def context(setting):
    return Context(**{k: (v if k == "EMULATED_DTYPES" else int(v)) for k, v in (s.split("=") for s in setting.split())})


@functools.cache
def program(case, target):
    """The program of `case` for `target`, or the error `to_program` raises."""
    name, kernel, setting = next(c for c in CASES if c[0] == case)
    with context(setting):
        try:
            return do_to_program(KERNELS[kernel], renderer(target))
        except KernelOptError as e:
            return f"KernelOptError: {e}"


# D17: an operation is narrowed when it is inlined into its one user, which does
# not store it, and computes on a scalar that C promotes: a char or a short, or
# Clang's __fp16.

def narrowing(ren, uops):
    """`uops` with each operation D17 narrows followed by a cast to its type, or
    None if it narrows none."""
    promoted = (dtypes.char, dtypes.uchar, dtypes.short, dtypes.ushort) + \
        ((dtypes.half,) if isinstance(ren, ClangRenderer) else ())
    children = Counter(v for u in uops for v in u.src)
    user = {v: u for u in uops for v in u.src}
    def narrowed(u):
        return u.op in GroupOp.ALU - {Ops.WHERE} and children[u] == 1 and u.max_numel() == 1 and u.dtype in promoted \
            and not (user[u].op is Ops.STORE and user[u].src[1] is u)
    if not any(narrowed(u) for u in uops): return None
    out, new = [], {}
    for u in uops:
        nu = u.replace(src=tuple(new.get(v, v) for v in u.src))
        out.append(nu)
        new[u] = UOp(Ops.CAST, src=(nu,), arg=nu.dtype) if narrowed(u) else nu
        if new[u] is not nu: out.append(new[u])
    return out


def narrowed(case, target):
    prg = program(case, target)
    if isinstance(prg, str): return None
    return narrowing(renderer(target), list(prg.src[1].src))


@graph
def kernels():
    return UOp.sink(*KERNELS)


@table
def cases():
    rows = []
    for case, kernel, setting in CASES:
        for target in targets(case):
            prg = program(case, target)
            outcome = prg if isinstance(prg, str) else "ok"
            _, t = TARGETS[target]
            rows.append((f"{case}_{target}", case, str(kernel), target, t.device, t.renderer, t.arch, setting,
                         outcome, str(narrowed(case, target) is not None)))
    return ["name", "case", "kernel", "target", "device", "renderer", "arch", "setting", "outcome", "narrowed"], rows


def program_golden(case, target):
    def body():
        _, kernel, setting = next(c for c in CASES if c[0] == case)
        ast = KERNELS[kernel]
        with context(setting): lowered = full_rewrite_to_sink(ast, renderer(target), optimize=ast.tag is None)
        prg = program(case, target)
        return UOp.sink(lowered, prg.replace(src=prg.src[:3]))
    body.__name__ = f"{case}_{target}"
    return body


def narrowed_golden(case, target, uops):
    def body(): return renderer(target).render(uops)
    body.__name__ = f"{case}_{target}_narrowed"
    return body


for case, _, _ in CASES:
    for target in targets(case):
        if isinstance(program(case, target), str): continue
        graph(program_golden(case, target))
        if (uops := narrowed(case, target)) is not None: text(narrowed_golden(case, target, uops))
