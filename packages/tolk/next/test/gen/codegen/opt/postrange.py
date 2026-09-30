"""Goldens of tinygrad/codegen/opt/postrange.py: kernels optimised by a given
sequence of optimisations.

A kernel golden, named after the kernel, is the sink that compiling a real
kernel hands to `apply_opts`, the same for every renderer. `cases.golden`
lists each case: a kernel, a renderer, the optimisations asked for (the
kernel's `opts_to_apply`), the settings it runs under, and its outcome, `ok`
or the error `apply_opts` raises. `refused` is how many of the optimisations
apply before one is refused. The graph golden named after an `ok` case is what
`apply_opts` returns, and `axes.golden` is the scheduler's view of the kernel
after the optimisations, before it is flattened. `renderers.golden` is what
each renderer tells the scheduler.

The cases are those of tinygrad's test_kernel_opts.py, test_tensor_cores.py
and test_custom_kernel.py, with `Tensor.empty` in place of realized random
data, which schedules the same kernels.
"""

import dataclasses

from golden import graph, table
from graph import kernels, stage
from tinygrad import Tensor, UOp, dtypes
from tinygrad.codegen.opt import KernelOptError, Opt, OptOps
from tinygrad.codegen.opt.postrange import Scheduler, apply_opts
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import Context, Target
from tinygrad.renderer import Renderer, tc
from tinygrad.renderer.cstyle import ClangRenderer, CUDARenderer, HIPRenderer, MetalRenderer
from tinygrad.schedule.indexing import BufferizeOpts
from tinygrad.uop.ops import AxisType, KernelInfo, Ops


def hip(arch):
    """The HIP renderer of `arch`, without its compiler, which needs comgr."""
    r = HIPRenderer.__new__(HIPRenderer)
    Renderer.__init__(r, Target("AMD", "HIP", arch))
    r.tensor_cores = tc.get_amd(arch)
    return r


def small_shared():
    """A renderer whose workgroups share 64 bytes."""
    class SmallShared(Renderer): shared_max = 64
    return SmallShared(Target())


RENDERERS = {
    "cpu": ClangRenderer(Target("CPU", "CLANG", "x86_64,x86-64")),
    "metal": MetalRenderer(Target("METAL", "METAL", "Apple9")),
    "cuda": CUDARenderer(Target("CUDA", "CUDA", "sm_89")),
    "nv": CUDARenderer(Target("NV", "CUDA", "sm_89")),
    "amd": hip("gfx1100"),
    "small_shared": small_shared(),
}


@table
def renderers():
    return ["renderer", "device", "arch", "has_local", "has_shared", "shared_max", "tensor_cores"], [
        (name, r.target.device, r.target.arch, r.has_local, r.has_shared, r.shared_max, len(r.tensor_cores))
        for name, r in RENDERERS.items()]


# Optimisations

def split(axis, amount, target, top=False):
    return Opt(OptOps.SPLIT, axis, (amount, target, True) if top else (amount, target))


def local(axis, amount, top=False): return split(axis, amount, AxisType.LOCAL, top)
def upcast(axis, amount, top=False): return split(axis, amount, AxisType.UPCAST, top)
def unroll(axis, amount): return split(axis, amount, AxisType.UNROLL)
def padto(axis, amount): return Opt(OptOps.PADTO, axis, amount)
def swap(axis, other): return Opt(OptOps.SWAP, axis, other)
def tensor_core(axis=0, tc_select=-1, tc_opt=0, use_tc=1): return Opt(OptOps.TC, axis, (tc_select, tc_opt, use_tc))


# Kernels

def last(*tensors): return kernels(*tensors)[-1]


def empty(*shape, dtype=dtypes.float): return Tensor.empty(*shape, dtype=dtype, device="CPU")


def custom(outputs, *inputs, fxn):
    return last(Tensor.custom_kernel(outputs, *inputs, fxn=fxn)[0])


def custom_gemm(C, A, B):
    i, j, k = UOp.range(C.shape[0], 0), UOp.range(C.shape[1], 1), UOp.range(A.shape[1], 2, axis_type=AxisType.REDUCE)
    C = C[i, j].set(0.0)
    prog = C[i, j].store(C.after(k)[i, j] + A[i, k] * B[k, j]).end(k).end(i, j)
    return prog.sink(arg=KernelInfo(name=f"custom_gemm_{C.shape[0]}_{C.shape[1]}_{A.shape[1]}", opts_to_apply=()))


def loop_acc_gemm(ACC, A, B):
    t, j, k = UOp.range(A.shape[0], 0, AxisType.LOOP), UOp.range(B.shape[1], 1), UOp.range(A.shape[1], 2, AxisType.REDUCE)
    mm = (A[t, k] * B[k, j]).cast(dtypes.float).reduce(k, arg=Ops.ADD)
    return ACC[j].store(ACC.after(t)[j] + mm).end(t).end(j).sink(arg=KernelInfo(opts_to_apply=()))


def contracted_m(n, m, k, dtype_out):
    def kernel(C, A, B):
        i, j, r = UOp.range(m*2, 0, AxisType.WEAK), UOp.range(n*2, 1), UOp.range(k*2, 2, AxisType.REDUCE)
        out = (A[i, r]*B[r, j]).cast(dtype_out).reduce(i, r, arg=Ops.ADD)
        return C[j].store(out).end(j).sink(arg=KernelInfo(opts_to_apply=()))
    return kernel


def thread_sum(axis_type):
    # test_custom_kernel.py::test_local_reduce
    def kernel(C, A):
        i, j = UOp.range(4, 0), UOp.range(8, 1, axis_type)
        return C[i].store(A[i, j].reduce(j, arg=Ops.ADD)).end(i).sink(arg=KernelInfo(opts_to_apply=()))
    return kernel


def stage_then_reduce(C, A):
    i, j, jj = UOp.range(4, 0), UOp.range(8, 1, AxisType.LOOP), UOp.range(8, 2, AxisType.LOOP)
    staged = (A[i, j] * 2).bufferize(j, arg=BufferizeOpts(None, AddrSpace.LOCAL))
    return C[i].store(staged.index(jj).reduce(jj, arg=Ops.ADD)).end(i).sink(arg=KernelInfo(opts_to_apply=()))


def stage_outside_an_output(C, A):
    # k indexes the output but not the staged values, so it is never made global
    i, k, j, jj = UOp.range(4, 0), UOp.range(2, 3), UOp.range(8, 1, AxisType.LOOP), UOp.range(8, 2, AxisType.LOOP)
    staged = (A[i, j] * 2).bufferize(j, arg=BufferizeOpts(None, AddrSpace.LOCAL))
    return C[i, k].store(staged.index(jj).reduce(jj, arg=Ops.ADD)).end(i, k).sink(arg=KernelInfo(opts_to_apply=()))


# Kernels built by hand, for edges that no Tensor program reaches

def kernel_of(*stores): return UOp.sink(*stores, arg=KernelInfo())


def summing_local(size):
    """out[l] = sum over r of a[l*8+r], the thread axis l of size `size`."""
    out, a = UOp.param(0, dtypes.float, 16), UOp.param(1, dtypes.float, 128)
    l, r = UOp.range(size, 0, AxisType.LOCAL), UOp.range(8, 1, AxisType.REDUCE)
    return kernel_of(out.index(l).store(a.index(l*8+r).reduce(r, arg=Ops.ADD)).end(l))


def huge_axes(size, *types):
    out = UOp.param(0, dtypes.float, 1)
    rngs = [UOp.range(size, i, t) for i, t in enumerate(types)]
    return kernel_of(out.index(UOp.const(0, dtypes.weakint)).store(UOp.const(1.0, dtypes.float)).end(*rngs))


def loop_counter(C, A):
    # test_custom_kernel.py::test_split_range_id_free_of_loop
    r, l = UOp.range(4, 0), UOp.loop(1)
    cnt = UOp.placeholder((1,), dtypes.int, slot=0, addrspace=AddrSpace.REG)
    cnt = cnt.after(r)[0].set(0)
    cnt = cnt.after(cnt[0].store(nxt := cnt.after(l)[0] + 1).backedge(l, nxt < 3))
    return C[r].set(A[r] + cnt[0].cast(C.dtype), end=r).sink(arg=KernelInfo(opts_to_apply=()))


def tagged_swap():
    out = UOp.param(0, dtypes.float, 64)
    r0, r1 = UOp.range(8, 0, AxisType.GLOBAL), UOp.range(8, 1, AxisType.GLOBAL)
    return kernel_of(out.index((r0*8+r1).rtag("keep")).store(r0.cast(dtypes.float)).end(r0, r1))


def closed_extent():
    # the extent of the outer range is a reduction over a closed range
    inner = UOp.range(4, 0, AxisType.REDUCE)
    extent = inner.cast(dtypes.int).reduce(inner, arg=Ops.ADD)
    outer = UOp.range(extent, 1)
    out = UOp.param(0, dtypes.int, 32)
    return kernel_of(out.index(outer).store(UOp.const(1, dtypes.int)).end(outer))


def nested_end():
    outer, inner = UOp.range(8, 0), UOp.range(4, 1)
    out = UOp.param(0, dtypes.int, 32)
    return kernel_of(out.index(outer*4+inner).store(UOp.const(1, dtypes.int)).end(inner).end(outer))


def launched():
    # a kernel whose hardware indices are already given: they lead its name
    out = UOp.param(0, dtypes.float, 256)
    g, l, r = UOp.special(4, "gidx0"), UOp.special(8, "lidx0"), UOp.range(8, 0, AxisType.GLOBAL)
    return kernel_of(out.index(g*64 + l*8 + r).store(UOp.const(1.0, dtypes.float)).end(r))


def int32_axis():
    out = UOp.param(0, dtypes.float, 8)
    r = UOp.range(UOp.const(8, dtypes.int32), 0, AxisType.GLOBAL, dtype=dtypes.int32)
    return kernel_of(out.index(r).store(UOp.const(1.0, dtypes.float)).end(r))


def tensor_core_matmul(t, m, n, k, batch=()):
    return last(empty(*batch, m, k, dtype=t.dtype_in).matmul(empty(*batch, k, n, dtype=t.dtype_in), dtype=t.dtype_out))


def first_tc(renderer, dtypes_in=None):
    return next(t for t in RENDERERS[renderer].tensor_cores if dtypes_in is None or t.dtype_in in dtypes_in)


KERNELS = {
    # test_kernel_opts.py
    "sum_rows": lambda: last(empty(32, 32).sum(1)),
    "add_one": lambda: last(empty(32, 32) + 1),
    "local_and_grouped_reduce": lambda: last(empty(4, 4, 128).sqrt() + (empty(4, 4, 128, 128) + 1).sum(axis=3).exp()),
    "sum_and_max": lambda: last((a := empty(7, 11, 13)).sum((1, 2)) + a.max((1, 2))),
    "flip_pad_sum": lambda: last(empty(17, 19).flip(0).pad(((2, 3), (0, 0))).sum(0)),
    "strided_conv": lambda: last(empty(1, 3, 15, 15).conv2d(empty(4, 3, 3, 3), padding=1, stride=2)),
    "cumsum": lambda: last(empty(13, 17).cumsum(1)),
    "elementwise": lambda: last(((a := empty(16, 16)) + empty(16, 16)).sqrt() * (a + 1).exp()),
    "elementwise_4": lambda: last(((a := empty(4)) + empty(4)).sqrt() * (a + 1).exp()),
    "matmul": lambda: last(empty(128, 128) @ empty(128, 128)),
    "double_reduce": lambda: last(empty(8, 128, 8, 128).sum(axis=(1, 3))),
    "matmul_17x17": lambda: last(empty(17, 17) @ empty(17, 17)),
    "matmul_4x4": lambda: last(empty(4, 4) @ empty(4, 4)),
    "shrunk_sum_0": lambda: last((empty(18, 18).shrink(((0, 17), (0, 17))) * 100).sum(0)),
    "shrunk_sum_1": lambda: last((empty(18, 18).shrink(((0, 17), (0, 17))) * 100).sum(1)),
    "shrunk_sum": lambda: last((empty(18, 18).shrink(((0, 17), (0, 17))) * 100).sum()),
    "shrunk_bool_sum": lambda: last(empty(18, 18, dtype=dtypes.bool).shrink(((0, 17), (0, 17))).sum()),
    "shrunk_bool_sum_0": lambda: last(empty(18, 18, dtype=dtypes.bool).shrink(((0, 17), (0, 17))).sum(0)),
    "shrunk_bool_any": lambda: last(empty(18, 18, dtype=dtypes.bool).shrink(((0, 17), (0, 17))).sum(dtype=dtypes.bool)),
    "shrunk_bool_any_0": lambda: last(empty(18, 18, dtype=dtypes.bool).shrink(((0, 17), (0, 17))).sum(0, dtype=dtypes.bool)),
    "shrunk_bool_any_1": lambda: last(empty(18, 18, dtype=dtypes.bool).shrink(((0, 17), (0, 17))).sum(1, dtype=dtypes.bool)),
    "shrunk_sum_exp": lambda: last((empty(18, 18).shrink(((0, 17), (0, 17))) * 100).sum().exp()),
    "shrunk_sum_0_exp": lambda: last((empty(18, 18).shrink(((0, 17), (0, 17))) * 100).sum(0).exp()),
    "group_full_unroll_sum": lambda: last(((empty(2, 28, 4096) * 0.5).float().square()).sum(axis=(0, 2))),
    "row_sum": lambda: last(empty(4, 17).sum(1)),
    "row_max": lambda: last(empty(4, 17).max(1)),
    "row_prod": lambda: last(empty(4, 17).prod(1)),
    "max_then_sum": lambda: last(empty(2, 3).max(1).sum(0)),
    "neg_sum_then_max": lambda: last((-empty(2, 3)).sum(1).max(0)),
    "prod_then_sum": lambda: last(empty(2, 3).prod(1).sum(0)),
    "repeated_sum": lambda: last(empty(7, 5).repeat((3, 16)).sum(1)),
    "repeated_row_sum": lambda: last(empty(7, 5).repeat((1, 16)).sum(1)),
    "cat_sum": lambda: last((a := empty(7, 5)).cat(a, dim=1).sum(1)),
    "repeated_prod": lambda: last(empty(7, 5).repeat((1, 4)).prod(1)),
    "row_sum_7": lambda: last(empty(7, 5).sum(1)),
    "row_all": lambda: last((empty(7, 5) > 0).all(1)),
    "masked_sum": lambda: last((Tensor.arange(7).reshape(7, 1) % 2 == 0).expand(7, 17).where(empty(7, 17), 1.0).sum(1)),
    "prefix_masked_sum": lambda: last(
        (Tensor.arange(17).reshape(1, 17) < 5).expand(7, 17).where(empty(7, 17), 1.0).sum(1)),
    "masked_prod": lambda: last(
        (Tensor.arange(7).reshape(7, 1) % 2 == 0).expand(7, 17)[:, :11].where(Tensor.ones(7, 11, device="CPU"), 2.0).prod(1)),
    "exp_sum": lambda: last(empty(18, 18).shrink(((0, 17), (0, 17))).exp().exp().sum()),
    "exp_sum_0": lambda: last(empty(18, 18).shrink(((0, 17), (0, 17))).exp().exp().sum(0)),
    "compare_sum": lambda: last((empty(18, 18).shrink(((0, 17), (0, 17))).exp() < 1).sum()),
    "compare_sum_0": lambda: last((empty(18, 18).shrink(((0, 17), (0, 17))).exp() < 1).sum(0)),
    "max_0": lambda: last((-empty(18, 18).shrink(((0, 17), (0, 17))) * 100).max(0)),
    "max_1": lambda: last((-empty(18, 18).shrink(((0, 17), (0, 17))) * 100).max(1)),
    "max_all": lambda: last((-empty(18, 18).shrink(((0, 17), (0, 17))) * 100).max()),
    "where_max": lambda: last((empty(17, 17).max(axis=0, keepdim=True) > 1).where(1, 0).int().max(0)),
    "where_max_multioutput": lambda: last(
        (r := empty(17, 17).max(axis=0, keepdim=True) > 1).where(1, 0).int().max(0), r.where(2, 0).int().max(0)),
    "matmul_32x32": lambda: last(empty(32, 32) @ empty(32, 32)),
    "arange": lambda: last(Tensor.arange(128).clone()),
    "sum_rows_64": lambda: last(empty(64, 64).sum(1)),
    "double_sum": lambda: last(empty(4, 4, 4).sum((1, 2)).sum()),
    # test_custom_kernel.py
    "variable_add": lambda: [c.src[0] for c in Tensor.linear_with_vars(
        empty(10)[:UOp.variable("n", 1, 10).bind(4)].contiguous() + 1)[0].src if c.src[0].op is Ops.SINK][-1],
    "custom_warp_sum": lambda: custom(empty(4), empty(4, 8), fxn=thread_sum(AxisType.WARP)),
    "custom_local_sum": lambda: custom(empty(4), empty(4, 8), fxn=thread_sum(AxisType.LOCAL)),
    "custom_gemm": lambda: custom(empty(16, 16), empty(16, 16), empty(16, 16), fxn=custom_gemm),
    "stage_then_reduce": lambda: custom(empty(4), empty(4, 8), fxn=stage_then_reduce),
    "stage_outside_an_output": lambda: custom(empty(4, 2), empty(4, 8), fxn=stage_outside_an_output),
    "loop_counter": lambda: custom(empty(4), empty(4), fxn=loop_counter),
    # built by hand
    "huge_local": lambda: summing_local(2**60),
    "symbolic_local_8": lambda: summing_local(UOp.variable("n", 2, 8)),
    "symbolic_local_16": lambda: summing_local(UOp.variable("n", 2, 16)),
    "huge_global": lambda: huge_axes(2**62 - 1, AxisType.GLOBAL),
    "huge_upcasts": lambda: huge_axes(2**32, AxisType.UPCAST, AxisType.UPCAST),
    "tagged_swap": tagged_swap,
    "closed_extent": closed_extent,
    "nested_end": nested_end,
    "int32_axis": int32_axis,
    "launched": launched,
}

CASES = []


def case(name, kernel, renderer, opts, **context):
    CASES.append((name, kernel, renderer, opts, context))


# test_kernel_opts.py

for n, arg in enumerate([99, -1]):
    case(f"swap_invalid_arg_{n}", "add_one", "metal", [swap(0, arg)])

for n, opts in enumerate([
    [local(0, 2)], [local(0, 16)], [local(1, 2, top=True)], [local(1, 64, top=True)],
    [local(0, 2), local(2, 2, top=True)], [local(0, 32), local(2, 2, top=True)], [local(0, 2), local(2, 64, top=True)],
    [local(0, 2), local(2, 2, top=True), upcast(0, 8), unroll(4, 4)],
    [local(1, 2), local(2, 2), local(3, 2), local(4, 2)],
    [local(0, 2)] * 4,
    [local(0, 2), local(2, 2), local(0, 2), local(4, 2), local(0, 2), local(6, 2), local(0, 2), local(8, 2)],
]):
    case(f"local_and_grouped_reduce_{n}", "local_and_grouped_reduce", "metal", opts)

case("sum_and_max_grouped", "sum_and_max", "metal", [local(0, 0), unroll(1, 11), local(2, 0, top=True), padto(2, 32)])
case("flip_pad_sum_grouped", "flip_pad_sum", "metal", [local(1, 0, top=True), padto(0, 8), upcast(0, 12), local(0, 0)])
case("strided_conv_grouped", "strided_conv", "metal", [local(5, 0, top=True), local(1, 0)])
case("cumsum_unrolled_padded", "cumsum", "cpu", [unroll(2, 0), upcast(0, 0), padto(0, 4)])
for amount in (2, 4, 8):
    case(f"elementwise_upcast_{amount}", "elementwise", "cpu", [upcast(0, amount)])
case("elementwise_full_upcast", "elementwise_4", "cpu", [upcast(0, 4)])

for n, opts in enumerate([
    [local(1, 32)], [local(0, 4), local(1, 4)], [local(0, 16), local(1, 8)], [local(2, 32, top=True)],
    [local(2, 32, top=True), unroll(2, 4)], [local(0, 2), local(1, 2), local(4, 32, top=True)],
    [local(0, 4), local(0, 8), local(4, 4, top=True)],
    [local(0, 4), local(0, 4), local(4, 8, top=True), unroll(4, 4), upcast(0, 4), upcast(1, 2)],
    [local(0, 4), local(0, 4), local(4, 8, top=True), unroll(4, 4), upcast(0, 8)],
]):
    case(f"matmul_{n}", "matmul", "metal", opts)
case("matmul_upcast_group", "matmul", "metal", [local(2, 32, top=True), upcast(2, 4)])

for n, opts in enumerate([
    [local(2, 2, top=True)], [local(2, 32, top=True)], [local(3, 2, top=True)], [local(3, 32, top=True)],
    [local(2, 2, top=True), local(4, 2, top=True)], [local(2, 16, top=True), local(4, 2, top=True)],
    [local(2, 4, top=True), local(4, 64, top=True)], [local(2, 16, top=True), local(4, 2, top=True), unroll(2, 4)],
    [local(2, 2, top=True), local(4, 32, top=True), unroll(4, 4)],
    [local(0, 4), local(1, 4), local(4, 4, top=True), local(6, 4, top=True)],
    [local(0, 4), local(1, 4), local(4, 2, top=True), local(6, 32, top=True), unroll(5, 4)],
    [local(0, 2), local(1, 2), local(4, 8, top=True), local(6, 4, top=True), upcast(0, 2)],
    [local(0, 2), local(1, 2), local(4, 8, top=True), local(6, 4, top=True), upcast(0, 2), unroll(4, 4), unroll(5, 4)],
    [local(0, 4), local(1, 4), local(4, 4, top=True), local(6, 4, top=True), upcast(0, 2), upcast(0, 2)],
]):
    case(f"double_reduce_{n}", "double_reduce", "metal", opts)

for n, opts in enumerate([
    [padto(0, 32)], [padto(1, 32)], [padto(2, 32)], [padto(0, 32), padto(1, 32)],
    [padto(0, 32), padto(1, 32), padto(2, 32)], [padto(0, 32), padto(1, 32), upcast(0, 2), upcast(1, 2)],
]):
    case(f"padto_matmul_{n}", "matmul_17x17", "cpu", opts)

for n, opts in enumerate([
    [upcast(0, 0)], [upcast(1, 0)], [unroll(2, 0)], [padto(0, 8)], [padto(1, 8)], [padto(2, 8)],
    [upcast(0, 0), padto(1, 8)], [upcast(1, 0), padto(1, 8)], [unroll(2, 0), padto(2, 8)],
]):
    case(f"padto_upcasted_{n}", "matmul_4x4", "cpu", opts)

for kernel in ["shrunk_sum_0", "shrunk_sum_1"]:
    case(f"padto_{kernel}", kernel, "cpu", [padto(0, 32)])
    case(f"padto_{kernel}_upcast", kernel, "cpu", [padto(0, 32), upcast(0, 8)])
for axis in (0, 1):
    for kernel in ["shrunk_sum", "shrunk_sum_0", "shrunk_bool_sum", "shrunk_bool_sum_0", "shrunk_bool_any",
                   "shrunk_bool_any_0", "shrunk_bool_any_1"]:
        case(f"padto_{kernel}_axis_{axis}", kernel, "cpu", [padto(axis, 32)])
case("padto_shrunk_sum_exp", "shrunk_sum_exp", "cpu", [padto(0, 32)])
case("padto_shrunk_sum_0_exp", "shrunk_sum_0_exp", "cpu", [padto(1, 32)])

case("padto_group_full_unroll_sum", "group_full_unroll_sum", "metal", [local(2, 256, top=True), padto(3, 32), unroll(3, 0), upcast(0, 7)])
for amount in (4, 0):
    case(f"padto_unrolled_sum_{amount}", "row_sum", "cpu", [padto(1, 32), unroll(1, amount)])
    case(f"padto_unrolled_max_{amount}", "row_max", "cpu", [padto(1, 32), unroll(1, amount)])
case("padto_unrolled_upcast", "row_sum", "cpu", [padto(1, 32), unroll(1, 0), upcast(0, 2)])
for kernel in ["max_then_sum", "neg_sum_then_max", "prod_then_sum"]:
    case(f"padto_nested_{kernel}", kernel, "cpu", [padto(1, 4)])
case("padto_nested_both", "max_then_sum", "cpu", [padto(0, 4), padto(1, 4)])
case("padto_unindexed_repeat", "repeated_sum", "cpu", [padto(2, 32)])
case("padto_unindexed_unrolled", "repeated_row_sum", "cpu", [unroll(1, 4), padto(1, 3)])
case("padto_unindexed_cat", "cat_sum", "cpu", [padto(1, 4)])
case("padto_unindexed_prod", "repeated_prod", "cpu", [padto(1, 3)])
case("padto_twice", "row_sum_7", "cpu", [padto(1, 8), padto(1, 16)])
case("padto_all", "row_all", "cpu", [padto(1, 8)])
case("padto_masked_sum_8", "masked_sum", "cpu", [padto(1, 8)])
case("padto_masked_sum_32", "masked_sum", "cpu", [padto(1, 32)])
case("padto_prefix_masked_sum", "prefix_masked_sum", "cpu", [padto(1, 8)])
case("padto_masked_prod", "masked_prod", "cpu", [padto(1, 8)])
case("padto_unrolled_prod", "row_prod", "cpu", [padto(1, 32), unroll(1, 0), upcast(0, 2)])
for n, arg in enumerate([-4, 0, 1]):
    case(f"padto_arg_{n}", "row_sum", "cpu", [padto(1, arg)])
case("padto_exp_sum", "exp_sum", "cpu", [padto(0, 32)])
case("padto_exp_sum_0", "exp_sum_0", "cpu", [padto(1, 32)])
case("padto_compare_sum", "compare_sum", "cpu", [padto(0, 32)])
case("padto_compare_sum_0", "compare_sum_0", "cpu", [padto(1, 32)])
for kernel in ["max_0", "max_1"]:
    case(f"padto_{kernel}", kernel, "cpu", [padto(0, 32)])
    case(f"padto_{kernel}_upcast", kernel, "cpu", [padto(0, 32), upcast(0, 8)])
case("padto_max_all", "max_all", "cpu", [padto(0, 32)])
case("padto_max_0_axis_1", "max_0", "cpu", [padto(1, 32)])
case("padto_where", "where_max", "cpu", [padto(0, 32)])
case("padto_where_upcast", "where_max", "cpu", [padto(0, 32), upcast(0, 8)])
case("padto_where_multioutput", "where_max_multioutput", "cpu", [padto(0, 32)])
case("padto_where_multioutput_upcast", "where_max_multioutput", "cpu", [padto(0, 32), upcast(0, 8)])

for n, opts in enumerate([
    [local(0, 2)], [local(0, 2), local(3, 2)], [local(0, 2), unroll(3, 0)], [local(0, 2), local(3, 0), unroll(3, 2)],
    [local(2, 2), unroll(2, 0)],
]):
    case(f"color_shapes_{n}", "matmul_32x32", "metal", opts)
case("arange_local", "arange", "metal", [local(0, 8)])
case("arange_local_upcast", "arange", "metal", [local(0, 8), upcast(0, 0)])
case("top_split_upcast", "sum_rows_64", "cpu", [upcast(0, 16, top=True)])
case("double_sum_group", "double_sum", "metal", [local(0, 16, top=True)])
case("double_sum_unroll_group", "double_sum", "metal", [unroll(1, 4), local(0, 16, top=True)])
case("double_sum_group_twice", "double_sum", "metal", [local(1, 4, top=True), local(1, 16, top=True)])

# The refusals of apply_opt, one per check

case("split_axis_negative", "sum_rows", "cpu", [upcast(-1, 2)])
case("split_axis_past_end", "sum_rows", "cpu", [unroll(2, 2)])
case("split_last_axis", "sum_rows", "cpu", [unroll(1, 2)])
case("padto_axis_past_end", "sum_rows", "cpu", [padto(2, 64)])
case("swap_axis_past_end", "matmul", "metal", [swap(3, 0)])
case("split_amount_one", "sum_rows", "cpu", [upcast(0, 1)])
case("split_amount_negative", "sum_rows", "cpu", [upcast(0, -2)])
case("local_without_locals", "sum_rows", "cpu", [local(0, 2)])
case("unroll_over_32", "sum_rows_64", "cpu", [unroll(1, 64)])
case("unroll_32", "sum_rows_64", "cpu", [unroll(1, 32)])
case("upcast_over_16", "sum_rows_64", "cpu", [upcast(0, 32)])
case("upcast_16", "sum_rows_64", "cpu", [upcast(0, 16)])
case("upcast_reduce", "sum_rows", "cpu", [upcast(1, 2)])
case("unroll_global", "sum_rows", "cpu", [unroll(0, 2)])
case("split_not_dividing", "sum_rows", "cpu", [upcast(0, 3)])
case("group_over_shared_memory", "matmul", "metal", [local(2, 16, top=True), local(0, 16), upcast(0, 8), local(4, 8)])
case("group_at_shared_memory", "matmul", "metal", [local(2, 16, top=True), local(0, 16), upcast(0, 8), local(4, 4)])
case("local_after_group_over_shared_memory", "matmul", "metal", [local(2, 16, top=True), upcast(0, 8), local(0, 16), local(0, 16)])
case("unroll_without_reduce", "add_one", "cpu", [unroll(0, 2)])
case("padto_upcast_axis", "sum_rows", "cpu", [upcast(0, 2), padto(1, 32)])
case("padto_quadruple", "row_sum", "cpu", [padto(1, 68)])
case("padto_under_quadruple", "row_sum", "cpu", [padto(1, 64)])
case("padto_symbolic_axis", "variable_add", "cpu", [padto(0, 4)])
case("split_symbolic_axis", "variable_add", "cpu", [upcast(0, 2)])
case("padto_warp_axis", "custom_warp_sum", "metal", [padto(1, 16)])
case("swap_globals", "matmul", "metal", [swap(0, 1)])
case("swap_not_global", "matmul", "metal", [swap(0, 2)])
case("swap_then_split", "matmul", "metal", [swap(0, 1), local(0, 4), upcast(1, 4)])
case("swap_on_cpu", "matmul_4x4", "cpu", [swap(0, 1)])
case("swap_with_itself", "matmul", "metal", [swap(0, 0)])
case("swap_with_past_end", "matmul", "metal", [swap(0, 3)])
case("local_ignores_shared_memory", "matmul", "small_shared", [local(0, 32)])
case("tc_on_cpu", "matmul", "cpu", [tensor_core()])
case("tc_not_first", "matmul", "metal", [upcast(0, 4), tensor_core()])
case("tc_negative_axis", "matmul", "metal", [tensor_core(axis=-1)])
case("tc_select_out_of_range", "matmul", "metal", [tensor_core(tc_select=5)])
case("tc_select_negative", "matmul", "metal", [tensor_core(tc_select=-2)])
case("tc_opt_out_of_range", "matmul", "metal", [tensor_core(tc_opt=3)])
case("tc_use_tc_zero", "matmul", "metal", [tensor_core(use_tc=0)])
case("tc_use_tc_out_of_range", "matmul", "metal", [tensor_core(use_tc=3)])
case("tc_without_reduce", "add_one", "metal", [tensor_core()])
case("tc_on_max", "row_max", "metal", [tensor_core()])
case("tc_axis_out_of_choices", "matmul", "metal", [tensor_core(axis=1)])

# test_custom_kernel.py

case("custom_gemm_group", "custom_gemm", "amd", [local(2, 4)])
case("custom_gemm_unroll", "custom_gemm", "cpu", [unroll(2, 4)])
case("stage_then_reduce_globals", "stage_then_reduce", "metal", [])
case("stage_outside_an_output_globals", "stage_outside_an_output", "metal", [])
case("loop_counter_upcast", "loop_counter", "cpu", [upcast(0, 2)])
case("local_sum_axes", "custom_local_sum", "metal", [])
case("warp_sum_axes", "custom_warp_sum", "metal", [])

# Edges of host integers, symbolic sizes, tags and flattening

case("group_beyond_host_integers", "huge_local", "metal", [local(1, 2, top=True)])
case("group_within_symbolic_shared_memory", "symbolic_local_8", "small_shared", [local(1, 2, top=True)])
case("group_maybe_beyond_symbolic_shared_memory", "symbolic_local_16", "small_shared", [local(1, 2, top=True)])
case("padto_beyond_host_integers", "huge_global", "metal", [padto(0, 4)])
case("local_whole_axis_beyond_host_integers", "huge_global", "metal", [padto(0, 4), local(0, 0)])
case("upcasts_beyond_host_integers", "huge_upcasts", "metal", [])
case("swap_keeps_tags", "tagged_swap", "metal", [swap(0, 1)])
case("flatten_keeps_a_closed_extent", "closed_extent", "cpu", [])
case("nested_end_globals", "nested_end", "metal", [])
case("launched_upcast", "launched", "metal", [upcast(0, 2)])
case("split_int32_axis", "int32_axis", "metal", [upcast(0, 2)])


# test_tensor_cores.py, on each renderer with tensor cores

def tensor_core_cases(renderer):
    ren = RENDERERS[renderer]
    for i, t in enumerate(ren.tensor_cores):
        n, m, k = t.dims
        name = f"{renderer}_{i}"
        KERNELS[f"tc_{name}"] = lambda t=t, n=n, m=m, k=k: tensor_core_matmul(t, m, n, k)
        case(f"tc_{name}_basic", f"tc_{name}", renderer, [tensor_core()], ALLOW_TF32=1)
        if i > 0: continue
        KERNELS[f"tc_{name}_padded"] = lambda t=t, n=n, m=m, k=k: tensor_core_matmul(t, m+1, n+1, k+1)
        for tc_opt in (0, 1, 2):
            case(f"tc_{name}_padded_{tc_opt}", f"tc_{name}_padded", renderer, [tensor_core(tc_opt=tc_opt)], ALLOW_TF32=1)
        KERNELS[f"tc_{name}_small_n"] = lambda t=t, n=n, m=m, k=k: tensor_core_matmul(t, m, n//4, k)
        case(f"tc_{name}_small_n", f"tc_{name}_small_n", renderer, [tensor_core(tc_opt=2)], ALLOW_TF32=1)
        KERNELS[f"tc_{name}_small_m"] = lambda t=t, n=n, m=m, k=k: tensor_core_matmul(t, m//4, n, k)
        case(f"tc_{name}_small_m", f"tc_{name}_small_m", renderer, [tensor_core(tc_opt=2)], ALLOW_TF32=1)
        if k // 8 > 0:
            KERNELS[f"tc_{name}_small_k"] = lambda t=t, n=n, m=m, k=k: tensor_core_matmul(t, m, n, k//8)
            case(f"tc_{name}_small_k", f"tc_{name}_small_k", renderer, [tensor_core(tc_opt=2)], ALLOW_TF32=1)
    t = first_tc(renderer, [dtypes.half, dtypes.float])
    n, m, k = t.dims
    kernel = f"tc_{renderer}_tiled"
    KERNELS[kernel] = lambda: tensor_core_matmul(t, m*8, n*8, k)
    case(f"{kernel}_extra_locals", kernel, renderer, [tensor_core()] + [local(0, 2)] * 3, ALLOW_TF32=1)
    case(f"{kernel}_use_tc_2", kernel, renderer, [tensor_core(use_tc=2)], ALLOW_TF32=1)
    KERNELS[f"tc_{renderer}_batched"] = lambda: tensor_core_matmul(t, m*2, n*2, k*2, batch=(3,))
    case(f"tc_{renderer}_batched_upcast", f"tc_{renderer}_batched", renderer, [tensor_core(), upcast(0, 0)], ALLOW_TF32=1)
    KERNELS[f"tc_{renderer}_64"] = lambda: tensor_core_matmul(t, 64, 64, 64)
    warp = scheduled(f"tc_{renderer}_64", renderer, [tensor_core()]).axis_types.index(AxisType.WARP)
    case(f"tc_{renderer}_padto_warp", f"tc_{renderer}_64", renderer, [tensor_core(), padto(warp, 7)], ALLOW_TF32=1)
    KERNELS[f"tc_{renderer}_group"] = lambda: tensor_core_matmul(t, 16, 16, 64)
    reduce = scheduled(f"tc_{renderer}_group", renderer, [tensor_core()], ALLOW_TF32=1).axis_types.index(AxisType.REDUCE)
    for amount in (2, 4):
        for top in (False, True):
            case(f"tc_{renderer}_group_{amount}{'_top' if top else ''}", f"tc_{renderer}_group", renderer,
                 [tensor_core(), local(reduce, amount, top)], ALLOW_TF32=1)
    case(f"tc_{renderer}_unroll", f"tc_{renderer}_group", renderer, [tensor_core(), unroll(reduce, 2)], ALLOW_TF32=1)
    KERNELS[f"tc_{renderer}_failed_padto"] = lambda: tensor_core_matmul(t, m//4, n+n//2, k)
    case(f"tc_{renderer}_failed_padto", f"tc_{renderer}_failed_padto", renderer, [tensor_core(tc_opt=2)], ALLOW_TF32=1)
    KERNELS[f"tc_{renderer}_nested_reduce"] = lambda: last(
        empty(m*2, k, dtype=t.dtype_in).matmul(empty(k, n, dtype=t.dtype_in), dtype=t.dtype_out).sum(0))
    case(f"tc_{renderer}_nested_reduce", f"tc_{renderer}_nested_reduce", renderer, [tensor_core()], ALLOW_TF32=1)
    KERNELS[f"tc_{renderer}_contracted_m"] = lambda: custom(
        empty(n*2, dtype=t.dtype_out), empty(m*2, k*2, dtype=t.dtype_in), empty(k*2, n*2, dtype=t.dtype_in),
        fxn=contracted_m(n, m, k, t.dtype_out))
    case(f"tc_{renderer}_contracted_m", f"tc_{renderer}_contracted_m", renderer, [tensor_core()], ALLOW_TF32=1)
    KERNELS[f"tc_{renderer}_ragged"] = lambda: tensor_core_matmul(t, m*2+1, n*2+1, k*3-1)
    ragged = scheduled(f"tc_{renderer}_ragged", renderer, [tensor_core(tc_opt=2)], ALLOW_TF32=1).axis_types.index(AxisType.REDUCE)
    case(f"tc_{renderer}_padto_unroll", f"tc_{renderer}_ragged", renderer,
         [tensor_core(tc_opt=2), padto(ragged, 4), unroll(ragged, 2), unroll(ragged, 0)], ALLOW_TF32=1)
    KERNELS[f"tc_{renderer}_shifted"] = lambda: last(
        (empty(m*2+1, k*3-1, dtype=t.dtype_in) + 1).matmul(empty(k*3-1, n*2+1, dtype=t.dtype_in) + 1, dtype=t.dtype_out))
    case(f"tc_{renderer}_padto_shifted_operand", f"tc_{renderer}_shifted", renderer, [tensor_core(tc_opt=2)], ALLOW_TF32=1)
    KERNELS[f"tc_{renderer}_masked"] = lambda: last(
        (empty(m*2+1, 1) > 0.5).expand(m*2+1, k*3-1).where(empty(m*2+1, k*3-1, dtype=t.dtype_in), Tensor(1, dtype=t.dtype_in, device="CPU"))
        .matmul((empty(1, n*2+1) > 0.5).expand(k*3-1, n*2+1).where(empty(k*3-1, n*2+1, dtype=t.dtype_in),
                Tensor(1, dtype=t.dtype_in, device="CPU")), dtype=t.dtype_out))
    case(f"tc_{renderer}_padto_masked_operand", f"tc_{renderer}_masked", renderer, [tensor_core(tc_opt=2)], ALLOW_TF32=1)
    KERNELS[f"tc_{renderer}_17_23_29"] = lambda: tensor_core_matmul(t, 17, 29, 23)
    case(f"tc_{renderer}_shape_padded", f"tc_{renderer}_17_23_29", renderer, [tensor_core(tc_opt=2, use_tc=2)], ALLOW_TF32=1)
    case(f"tc_{renderer}_padto_full_upcast", f"tc_{renderer}_17_23_29", renderer,
         [tensor_core(tc_opt=2), padto(0, 4), upcast(0, 0)], ALLOW_TF32=1)
    KERNELS[f"tc_{renderer}_relu"] = lambda: last(
        empty(16, 64, dtype=t.dtype_in).matmul(empty(64, 16, dtype=t.dtype_in), dtype=t.dtype_out).relu())
    case(f"tc_{renderer}_unroll_relu", f"tc_{renderer}_relu", renderer, [tensor_core(), unroll(reduce, 2)], ALLOW_TF32=1)


def half_accumulate_cases(renderer):
    t = next((t for t in RENDERERS[renderer].tensor_cores if t.dtype_in == t.dtype_out == dtypes.half), None)
    if t is None: return
    kernel = f"tc_{renderer}_half_128"
    KERNELS[kernel] = lambda: last(empty(128, 128, dtype=dtypes.half).matmul(empty(128, 128, dtype=dtypes.half), dtype=dtypes.half))
    r = scheduled(kernel, renderer, [tensor_core()]).axis_types.index(AxisType.REDUCE)
    for n, opts in enumerate([
        [], [upcast(0, 4)], [upcast(1, 4)], [upcast(0, 4), upcast(1, 4)], [unroll(r, 2)], [upcast(0, 4), unroll(r+1, 2)],
        [upcast(0, 4), upcast(1, 4), unroll(r+2, 2)], [upcast(0, 4), upcast(1, 4), unroll(r+2, 4)],
    ]):
        case(f"{kernel}_{n}", kernel, renderer, [tensor_core()] + opts)


def multi_reduce_cases(renderer):
    t = first_tc(renderer, [dtypes.half])
    kernel = f"tc_{renderer}_conv"
    KERNELS[kernel] = lambda: last(empty(16, 16, 29, 29, dtype=t.dtype_in).conv2d(
        empty(32, 16, 16, 16, dtype=t.dtype_in), padding=1, dtype=t.dtype_out))
    for axis in range(9):
        case(f"{kernel}_{axis}", kernel, renderer, [tensor_core(axis=axis, tc_opt=2)])


def loop_acc_cases(renderer):
    ren = RENDERERS[renderer]
    i, t = next((i, t) for i, t in enumerate(ren.tensor_cores) if t.dtype_in is dtypes.half and t.dtype_out is dtypes.float)
    n, m, k = t.dims
    kernel = f"tc_{renderer}_loop_acc"
    KERNELS[kernel] = lambda: custom(empty(n), empty(m, k, dtype=dtypes.half), empty(k, n, dtype=dtypes.half), fxn=loop_acc_gemm)
    case(f"{kernel}_tc", kernel, renderer, [tensor_core(tc_select=i)])


def scheduled(kernel, renderer, opts, **context):
    """The scheduler of `kernel` after `opts`, as apply_opts leaves it."""
    with Context(**context):
        k = Scheduler(kernel_input(kernel), RENDERERS[renderer])
        k.convert_loop_to_global()
        for opt in opts: k.apply_opt(opt)
        return k


INPUTS = {}


def kernel_input(kernel):
    if kernel not in INPUTS:
        with Context(DEV="CPU"): INPUTS[kernel] = stage("apply_opts", KERNELS[kernel](), RENDERERS["cpu"])
    return INPUTS[kernel]


for renderer in ["metal", "cuda", "amd"]:
    tensor_core_cases(renderer)
    half_accumulate_cases(renderer)
multi_reduce_cases("metal")
loop_acc_cases("amd")
case("tc_cuda_float_without_tf32", "tc_cuda_5", "cuda", [tensor_core()])
case("tc_nv_float_without_tf32", "tc_cuda_5", "nv", [tensor_core()])
case("tc_nv_float", "tc_cuda_5", "nv", [tensor_core()], ALLOW_TF32=1)


# Outcomes

def with_opts(kernel, opts): return kernel.replace(arg=dataclasses.replace(kernel.arg, opts_to_apply=tuple(opts)))


def outcome(kernel, renderer, opts, context):
    """`ok` and the optimised kernel, or the error and how many opts applied."""
    with Context(**context):
        try:
            return "ok", apply_opts(with_opts(kernel_input(kernel), opts), RENDERERS[renderer])
        except (KernelOptError, ValueError) as e:
            k = Scheduler(kernel_input(kernel), RENDERERS[renderer])
            k.convert_loop_to_global()
            for applied, opt in enumerate(opts):
                try: k.apply_opt(opt)
                except type(e): return type(e).__name__, applied
            raise


def context_cell(context): return " ".join(f"{k}={v}" for k, v in context.items())


@table
def cases():
    return ["case", "kernel", "renderer", "opts", "context", "outcome", "refused"], [
        (name, kernel, renderer, repr(tuple(opts)), context_cell(context), result, "" if result == "ok" else value)
        for name, kernel, renderer, opts, context in CASES
        for result, value in [OUTCOMES[name]]]


def sint(x): return x.render() if isinstance(x, UOp) else str(x)
def sints(xs): return "[" + ", ".join(map(sint, xs)) + "]"


def rngs_cell(rngs):
    return " ".join("_".join(map(str, r.arg[:-1])) + f":{r.vmax+1}:{r.arg[-1].name}" for r in rngs)


def made(ret):
    if ret is None: return []
    return list(ret) if isinstance(ret, (list, tuple)) else [ret]


@table
def axes():
    rows = []
    for name, kernel, renderer, opts, context in CASES:
        if OUTCOMES[name][0] != "ok": continue
        with Context(**context):
            k = Scheduler(kernel_input(kernel), RENDERERS[renderer])
            k.convert_loop_to_global()
            ret = None
            for opt in opts: ret = k.apply_opt(opt)
        # a size that does not render on one line is pinned by the case's graph
        if any("\n" in sint(x) for x in k.full_shape): continue
        rows.append((name, k.shape_len, sints(k.full_shape), repr(k.axis_types), repr(k.reduce_axes), repr(k.upcastable_dims),
                     repr(k.unrollable_dims), k.upcasted, sint(k.upcast_size()), k.group_for_reduces, len(k.reduceops),
                     len(k.bufs), k.colored_shape(), rngs_cell(made(ret)) if opts else ""))
    return ["case", "shape_len", "full_shape", "axis_types", "reduce_axes", "upcastable_dims", "unrollable_dims",
            "upcasted", "upcast_size", "group_for_reduces", "reduceops", "bufs", "colored_shape", "made"], rows


def escaped(text): return text.replace("\x1b", "\\e")


# Weak axes: an output axis (CPU), one a buffered value keeps weak, and one
# that is no output
WEAK_COLORS = [("weak_output", "sum_rows_64", "cpu"), ("weak_outside_a_stage", "stage_outside_an_output", "metal"),
               ("weak_reduced", "tc_metal_contracted_m", "metal")]


@table
def colors():
    """The colored shape and name of the kernels of test_color_shapes_with_local,
    of a kernel with hardware indices, and of weak axes, with every escape
    character written as `\\e`."""
    shown = [(name, kernel, renderer, opts) for name, kernel, renderer, opts, _ in CASES
             if name.startswith("color_shapes_") or name == "launched_upcast"]
    rows = []
    for name, kernel, renderer, opts in shown + [(n, k, r, []) for n, k, r in WEAK_COLORS]:
        with Context(NO_COLOR=0):
            k = scheduled(kernel, renderer, opts)
            rows.append((name, kernel, renderer, repr(tuple(opts)), escaped(k.colored_shape()),
                         escaped(k.get_optimized_ast().arg.name)))
    return ["case", "kernel", "renderer", "opts", "colored_shape", "name"], rows


def declare(name, fn):
    fn.__name__ = name
    graph(fn)


OUTCOMES = {name: outcome(kernel, renderer, opts, context) for name, kernel, renderer, opts, context in CASES}
for kernel in list(KERNELS):
    declare(kernel, lambda kernel=kernel: kernel_input(kernel))
for name, (result, optimized) in OUTCOMES.items():
    if result == "ok": declare(name, lambda optimized=optimized: optimized)
