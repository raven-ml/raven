"""Goldens of tinygrad/schedule/rangeify.py: from tensor graphs to kernel graphs.

Recorded programs: `<program>.golden` is the graph that `get_kernel_graph`
receives when the tensors of a `Tensor` program are scheduled on the CPU, and
`<program>_kernels.golden` the kernel graph it returns. A program that
schedules several graphs, such as an allreduce compiled as a function of its
own, records the later ones as `<program>_<n>.golden` and
`<program>_<n>_kernels.golden`.

`kernel_counts.golden` is the number of kernels of each recorded graph.
`<program>_debug.golden` is what `get_kernel_graph` prints of the first graph
with DEBUG_RANGEIFY.
"""

import contextlib
import io
import os

from golden import graph, table, text
from tinygrad import Tensor, Variable, dtypes, function
from tinygrad.helpers import DEBUG_RANGEIFY, DEV, MAX_KERNEL_BUFFERS, SPLIT_REDUCEOP
from tinygrad.dtype import Invalid
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp
import tinygrad.schedule
import tinygrad.schedule.prepare
import tinygrad.schedule.rangeify

DEV.value = "CPU"
D2 = ("CPU:0", "CPU:1")
D4 = tuple(f"CPU:{k}" for k in range(4))
DISK = "DISK:/tmp/tolk-rangeify"


def scheduled(program):
    """The (input, output) pairs of each `get_kernel_graph` that scheduling the
    tensors of `program()` runs, in order."""
    seen, kernel_graph = [], tinygrad.schedule.get_kernel_graph

    def capture(tsink):
        out = kernel_graph(tsink)
        seen.append((tsink, out))
        return out

    tinygrad.schedule.get_kernel_graph = capture
    try:
        first, *rest = tensors if isinstance(tensors := program(), tuple) else (tensors,)
        first.linear_with_vars(*rest)
    finally:
        tinygrad.schedule.get_kernel_graph = kernel_graph
    if not seen: raise RuntimeError("scheduling the tensors runs no get_kernel_graph")
    return seen


def empty(*shape, dtype=dtypes.float, device=None): return Tensor.empty(*shape, dtype=dtype, device=device)


def sharded(*shape, axis, devices=D2):
    shard = tuple(n // len(devices) if a == axis else n for a, n in enumerate(shape))
    return Tensor(empty(*shard, device=devices).uop.unshard(axis), device=devices)


def setitem(index, value, shape=(8, 8)):
    t = empty(*shape).contiguous()
    t[index] = value
    return t


def add_one(C, A):
    C, A = C.flatten(), A.flatten()
    i = UOp.range(A.numel(), 0)
    return C[i].store(A[i] + 1).end(i).sink(arg=KernelInfo(name=f"add_one_{A.numel()}"))


def custom_kernel():
    return Tensor.custom_kernel(empty(4, 4), empty(4, 4) * 2, fxn=add_one)[0] + 1


def inline_function():
    @function
    def f(a, b): return a * 2 + b
    return f(empty(4, 8), empty(4, 8)).sum(1)


def assign_double_diamond():
    # each assign reads, through a sum, the buffer the other overwrites
    b0, b1 = empty(16).contiguous(), empty(16).contiguous()
    r0, r1 = (empty(16, 16) - b1.contiguous()).sum(1), (empty(16, 16) - b0.contiguous()).sum(1)
    return b0.assign(r0 * b0), b1.assign(r1 * b1)


def stage_of_assigned():
    # storage written by an assign, then materialised again
    a = empty(8).contiguous()
    a.assign(empty(8) + 1)
    return a.contiguous().reshape(2, 4).permute(1, 0) + 1


def assigned_read_twice():
    # assigned storage read through two indexings
    a = empty(4, 4).contiguous()
    a.assign(empty(4, 4) + 1)
    return a + a.permute(1, 0)


def assigned_contiguous():
    # assigned storage materialised again, and returned too
    a = empty(8).contiguous()
    a.assign(empty(8) + 1)
    return a.contiguous() + 1, a


def partially_invalid():
    t = empty(8).contiguous()
    t[0:4] = Tensor.invalids(4, dtype=dtypes.float)
    return t + 1


def symbolic_kept():
    # a value of symbolic shape, of four storages, read through two indexings
    v = Variable("v", 1, 10).bind(3)
    x = [empty(4, 10)[:, :v] for _ in range(4)]
    y = (x[0] + x[1] + x[2] + x[3]).exp2()
    return y + y.flip(0)


def custom_kernel_of_views():
    # a custom kernel whose arguments are a view at an offset and one at 0
    x = empty(16).contiguous()
    return Tensor.custom_kernel(empty(4), x[4:8], fxn=add_one)[0] + x[0:4]


def precompiled_function():
    @function(precompile=True)
    def f(a): return (a * 2 + 1).sum(0)
    return f(empty(4, 4)) + 1


def mesh(*shape):
    """A value of `shape` sharded on its first two axes across four devices."""
    rng = UOp.range(4, -1, AxisType.DEVICE)
    rows, cols = rng // 2, rng % 2
    return Tensor(empty(*shape).uop.copy_to_device(D4)._shard(0, rows)._shard(1, cols).unshard((0, 1), (rows, cols)))


def limited(program, n):
    def limited_program():
        # each golden runs in a process of its own, which the setting outlives
        MAX_KERNEL_BUFFERS.value = n
        return program()
    return limited_program


def split(program):
    def split_program():
        SPLIT_REDUCEOP.value = 1
        return program()
    return split_program


def many_inputs(n, *shape): return sum((empty(*(shape or (8,))) for _ in range(n)), start=empty(*(shape or (8,))))


PROGRAMS = {
    # elementwise, reductions and matmuls
    "add": lambda: empty(4, 4) + empty(4, 4),
    "sum": lambda: empty(4, 8).sum(1),
    "sum_all": lambda: empty(4, 8).sum(),
    "matmul": lambda: empty(4, 8) @ empty(8, 3),
    "double_matmul": lambda: empty(16, 16) @ empty(16, 16) @ empty(16, 16),
    "where": lambda: (empty(4, 4) > 0).where(empty(4, 4), 0),
    "cast_half": lambda: empty(4, 4, dtype=dtypes.half).float().sum(),
    "softmax": lambda: empty(4, 8).softmax(-1),
    "layernorm": lambda: empty(4, 8).layernorm(),
    "attention": lambda: empty(2, 2, 8, 4).scaled_dot_product_attention(empty(2, 2, 8, 4), empty(2, 2, 8, 4)),
    "conv": lambda: empty(1, 2, 8, 8).conv2d(empty(3, 2, 3, 3)),
    "maxpool": lambda: empty(1, 2, 8, 8).max_pool2d(),
    "cat": lambda: Tensor.cat(empty(2, 4), empty(3, 4)),
    "stack": lambda: Tensor.stack(empty(4), empty(4), empty(4)),
    "pad": lambda: empty(4, 4).pad(((1, 2), (0, 3))) + 1,
    "arange": lambda: Tensor.arange(16).reshape(4, 4) + 1,
    "cumsum": lambda: empty(8).cumsum(),
    "sort": lambda: empty(8).sort()[0],
    "embedding": lambda: empty(10, 4)[Tensor([1, 2, 3], dtype="int")],
    "argmax": lambda: empty(4, 8).argmax(1),
    # an activation over the halves of a product, read by a second product
    "swiglu_down": lambda: (lambda gu: (gu[:, :, 0].clamp(max_=7.0) * (gu[:, :, 0] * 1.702).sigmoid()
                                        * (gu[:, :, 1].clamp(-7.0, 7.0) + 1)) @ empty(32, 64))(
        (empty(7, 64) @ empty(64, 64)).reshape(7, 32, 2)),
    # an exponential of a row expanded to a matrix, read where it is broadcast:
    # stored as the row, it computes 8 exponentials
    "exp_dead_axis": lambda: (empty(8).reshape(1, 8).expand(4, 8).exp()[:, :, None] * empty(8, 3)).sum(1),
    # an exponential read where it is broadcast, and a cheap value of it read
    # the same way, which reads the stored exponential
    "exp_cheap_consumer": lambda: (lambda p: (p[:, None] * empty(8, 4)).sum(0) + ((p + 1)[:, None] * empty(8, 4)).sum(0))(
        empty(8).exp()),
    # gathers at loaded indices, read where they are broadcast or twice, and a
    # masked sum that is no gather
    "gather_broadcast": lambda: empty(8, 32) @ empty(20, 32)[empty(12, dtype=dtypes.int)].T,
    "gather_read_twice": lambda: (lambda g: (g * empty(32)).sum(1) + (g * empty(32)).sum(1))(
        empty(64, 32)[empty(8, dtype=dtypes.int)]),
    "gather_rotary": lambda: (lambda c: (empty(8, 32) * c) @ (empty(8, 32) * c).T)(
        empty(64, 32)[Tensor.arange(8, dtype=dtypes.int) + empty(1, dtype=dtypes.int)]),
    "prefix_sum_broadcast": lambda: (lambda c: (empty(8, 32) * c) @ (empty(8, 32) * c).T)(
        (Tensor.arange(64, dtype=dtypes.int).reshape(1, 64, 1) < empty(8, 1, 1, dtype=dtypes.int))
        .where(empty(1, 64, 32), 0).sum(1)),
    # fusions that the scheduler must keep in one kernel, or split
    "elementwise_three": lambda: empty(256) + empty(256) + empty(256),
    "mulacc": lambda: (empty(256) * empty(256)).sum(),
    "binop_reshape": lambda: (empty(10) + empty(10)).reshape(5, 2) + empty(5, 2),
    "binop_permute": lambda: (empty(2, 5) + empty(2, 5)).permute(1, 0) + empty(5, 2),
    "shared_sum": lambda: (lambda ab: ab + empty(10) + ab + empty(10))(empty(10) + empty(10)),
    "reduce_unary": lambda: -(empty(16).sum().sqrt()),
    "reduce_reshape_binop": lambda: empty(10, 10).sum(0).reshape(10) + empty(10),
    "reduce_permute_binop": lambda: empty(10, 10, 10).sum(0, keepdim=True).permute(2, 1, 0) + empty(10, 10, 1),
    "reduce_permute_nofuse": lambda: empty(32, 32, 32).sum(2).permute(1, 0) + empty(32, 32),
    "permute_through_reshape": lambda: (empty(16, 16) + empty(16, 16)).reshape(4, 4, 4, 4).permute(2, 3, 0, 1),
    "children_dont_push": lambda: (lambda ab: ab.expand(10, 10, 10) + ab.permute(2, 1, 0))(
        empty(10, 10, 1) + empty(10, 10, 1)),
    "shrink_fuse": lambda: (empty(64, 16) * empty(64, 16))[0:1] * empty(1, 16),
    "multistage_reduce": lambda: empty(32, 32, 32).sum(2).relu().sum(1),
    "reduce_shrink": lambda: empty(32, 32).sum(1)[:16] + empty(16),
    "contiguous_add": lambda: (empty(32) + empty(32)).contiguous() + empty(32),
    "reshape_chain": lambda: empty(4, 4).reshape(16).reshape(2, 8) + empty(2, 8),
    # TestSchedule
    "arange_sum": lambda: Tensor.arange(6).reshape(3, 2).sum(axis=1).clone(),
    "permute_arange": lambda: Tensor.arange(6).reshape(6, 1, 1).permute(2, 0, 1).sum(axis=1).clone(),
    "expand_before_cast": lambda: empty(4, 2, 1).permute((1, 0, 2)).cast(dtypes.half).expand((2, 4, 4)) + 2,
    "push_pads_elementwise": lambda: (empty(4, 4).reciprocal() * empty(4, 4)).pad((None, (0, 1),)).sum(),
    "allow_push_permutes": lambda: empty(10, 10, 10).sum(axis=0, keepdim=True).permute(2, 1, 0) + empty(10, 10, 1),
    "div_collapse": lambda: (lambda a, b: (a * b) / b)(empty(4), empty(4)),
    "reduce_same_size": lambda: (lambda s: (s + 2, s + 4, (s + 2) * (s + 4)))(empty(4, 4).sum()),
    "reduce_multiple_paths": lambda: (lambda a: (a.sum().exp2(), a.sum() + a.sum().exp2()))(empty(4, 4)),
    "reduce_ext_reduce_child": lambda: (lambda a, b: (a.sum() + b.sum() + 2, a.sum() + b.sum() + 4))(
        empty(4, 4), empty(4, 4)),
    "reduce_expand_child": lambda: (lambda a, b: (a.sum() + 2, a.sum() + b))(empty(32, 32, 32), empty(1, 16)),
    "reduce_broadcast_not_recomputed": lambda: (lambda a: a - a.mean(axis=0, keepdim=True))(empty(32, 16)),
    "ugly_reduceop_pairing": lambda: (lambda a, b, c: (c * a.sum(-1, keepdim=True)).sum(-1)
                                      + (b * a.sum(-1, keepdim=True)).sum(-1))(empty(4, 32), empty(4, 32), empty(4, 32)),
    "reduce_expand_reduce": lambda: (lambda a: (a + a.sum(-1, keepdim=True)).sum(-1))(empty(4, 32)),
    "multireduce_parallel": lambda: empty(4, 32).sum(axis=-1) + empty(4, 32).sum(axis=-1),
    "std": lambda: empty(4, 32).std(-1),
    "multireduce_diffops_parallel": lambda: empty(4, 32).sum(-1) + empty(4, 32).max(-1),
    "multimatmul": lambda: empty(4, 64) @ empty(64, 8) + empty(4, 64) @ empty(64, 8),
    "multireduce_push_shrink_chase": lambda: (empty(16, 16).sum(1) + empty(16))[:4] * empty(4) + empty(16, 16).sum(1)[:4],
    "multireduce_midreduce_nochase": lambda: (lambda a: (a.sum(0) + a.max(0) + a.max(1) + a.sum(1)) + 2)(empty(16, 16)),
    "partial_fuse": lambda: (lambda a, b: (a.sum() + 2, (a.sum() - b.sum()) * 4))(empty(16, 16), empty(16, 16)),
    "pad_reduce_safe": lambda: (empty(3, 4, 5) + empty(3, 4, 5)).pad(((0, 1), (0, 1), (0, 1)), value=1.0).sum().contiguous(),
    "pad_reduce_unsafe": lambda: empty(3, 4, 5).log2().pad(((0, 1), (0, 1), (0, 1)), value=1.0).sum().contiguous(),
    "shrink_pad_unsafe": lambda: empty(3).exp2().shrink(((0, 1),)).pad(((0, 1),)).contiguous(),
    "base_change_expand_pad": lambda: empty(3, 3).exp2()[:, None, :].pad(((0, 0), (1, 1), (0, 0))) * 2,
    "base_change_pad_expand": lambda: (empty(4, 4) + empty(4, 4)).pad(((1, 1), (1, 1))).cast(dtypes.int).expand((2, 6, 6)) * 4,
    "zero_size_children": lambda: (lambda r: r.reshape(1) * 2 + (r.reshape(1).shrink(((1, 1),)) * 2).pad(((1, 0),)))(
        empty(1, 2).sum(axis=(1,), keepdim=True)),
    "preserve_multistage_reduce": split(lambda: (lambda x: (x - x.max(keepdim=True)).max())(empty(32768))),
    "clone": lambda: empty(4).clone(),
    # stores into storage
    "contiguous": lambda: (empty(4, 4) + 1).contiguous().sum(0),
    "assign": lambda: empty(4, 4).assign(empty(4, 4) + 1),
    "assign_permuted": lambda: empty(4, 4, dtype="int").permute(1, 0).assign(Tensor.arange(16).reshape(4, 4)),
    "assign_double_diamond": assign_double_diamond,
    "setitem": lambda: setitem(slice(2, 4), 1.0),
    "setitem_tensor": lambda: setitem((slice(None), slice(2, 4)), empty(8, 2)),
    # the setitem's mask is a constant staged over the ranges of the storage's
    # axes, which pm_const_buffer_folding folds to the constant
    "setitem_column": lambda: setitem(slice(2, 5), 1.0, shape=(8, 1)),
    "setitem_cube": lambda: setitem(slice(1, 3), 42.0, shape=(4, 4, 4)),
    "custom_kernel": custom_kernel,
    "inline_function": inline_function,
    # symbolic shapes
    "variable_shrink": lambda: Tensor.ones(10)[:Variable("v", 1, 10).bind(3)] * 2,
    "variable_reduce": lambda: empty(10, 4)[:Variable("v", 1, 10).bind(5)].sum(0),
    "variable_two": lambda: (empty(10)[:Variable("v", 1, 10).bind(4)] * 2,
                             empty(10)[:Variable("w", 1, 10).bind(7)] + 1),
    "variable_same": lambda: (lambda v: (empty(10)[:v] * 2, empty(10, 2)[:v].sum(1)))(Variable("v", 1, 10).bind(4)),
    "variable_offset": lambda: (lambda v: empty(10)[v:v + 3] * 2)(Variable("i", 0, 7).bind(2)),
    "variable_staged": lambda: (lambda x: x + x.flip(0))(
        (empty(10)[:Variable("v", 1, 10).bind(3)] + 1).contiguous()),
    "symbolic_kept": symbolic_kept,
    "symbolic_contiguous": lambda: (empty(4, 10)[:, :Variable("v", 1, 10).bind(3)] + 1).contiguous().sum(0),
    "variable_read_twice": lambda: (lambda x: x.sum() + x.max())(empty(4, 10)[:, :Variable("v", 1, 10).bind(3)].exp2()),
    # values whose staging a later pass removes or keeps
    "expand_staged": lambda: (lambda e: e + e.permute(1, 0))(empty(4, 1).exp2().expand(4, 4)),
    "expand_kept": lambda: (lambda e: e + e.permute(1, 0))(
        (empty(4, 1) + empty(4, 1) + empty(4, 1) + empty(4, 1)).exp2().expand(4, 4)),
    "padded_twice": lambda: (lambda p: p + p.flip(0))(empty(4, 4).exp2().pad(((1, 1), (1, 1)))),
    "stage_of_assigned": stage_of_assigned,
    "assigned_read_twice": assigned_read_twice,
    "assigned_contiguous": assigned_contiguous,
    "sum_kept_broadcast": lambda: (lambda s: s + s.permute(1, 0))(empty(4, 4).sum(1, keepdim=True).expand(4, 4)),
    "mean_kept_broadcast": lambda: (lambda x: x - x.mean((0, 2), keepdim=True))(empty(2, 3, 4)),
    "unit_axis_staged": lambda: (lambda x: x + x.flip(1))(empty(1, 4).exp2()),
    "custom_kernel_of_views": custom_kernel_of_views,
    "custom_kernel_permuted": lambda: Tensor.custom_kernel(empty(4, 4), empty(4, 4).permute(1, 0), fxn=add_one)[0],
    "custom_kernel_in_place": lambda: (lambda x: Tensor.custom_kernel(x, x, fxn=add_one)[0])(empty(4, 4)),
    "precompiled_function": precompiled_function,
    "two_outputs": lambda: (lambda x: (x + 1, x * 2))(empty(4, 4)),
    "empty_sum": lambda: empty(0, 4).sum(0) + 1,
    "long_cumsum": lambda: empty(1024).cumsum(),
    "disk_view_to": lambda: empty(64, dtype=dtypes.uint8, device=DISK)[8:24].to("CPU") + 1,
    "disk_bitcast_to": lambda: empty(64, dtype=dtypes.uint8, device=DISK)[8:24].bitcast(dtypes.float32).to("CPU") + 1,
    "disk_store": lambda: empty(16, dtype=dtypes.uint8, device=DISK).assign(empty(16, dtype=dtypes.uint8).to(DISK)),
    # invalid values
    "full_invalid": lambda: Tensor.full((4,), Invalid, dtype=dtypes.float),
    "invalids_read": lambda: Tensor.invalids(8).contiguous() + empty(8),
    "partially_invalid": partially_invalid,
    "invalids_sharded": lambda: Tensor.invalids(8).shard(D2, axis=0).contiguous(),
    # limits on the buffers of a kernel
    "many_inputs": lambda: many_inputs(8),
    "many_inputs_limited": limited(lambda: many_inputs(8), 4),
    "many_matrices_limited": limited(lambda: many_inputs(8, 4, 4), 4),
    "many_cubes_limited": limited(lambda: many_inputs(8, 2, 3, 4), 4),
    "many_sums_limited": limited(lambda: sum((empty(8, 4).sum(1) for _ in range(6)), start=empty(8)), 4),
    "many_sharded_limited": limited(lambda: sum((sharded(8, 4, axis=0) for _ in range(6)), start=sharded(8, 4, axis=0)), 4),
    # several devices
    "copy": lambda: empty(4, 4).to("CPU:1") * 2,
    "shard_add": lambda: sharded(4, 8, axis=0) + 1,
    "shard_sum": lambda: sharded(4, 8, axis=0).sum(0),
    "mesh_add": lambda: mesh(4, 4) + 1,
    "mesh_sum": lambda: mesh(4, 4, 2).sum(2),
    "mesh_to_one": lambda: mesh(4, 4).to("CPU") + 1,
    "shard_of_computed": lambda: (empty(4, 4) + 1).shard(D2, 0) * 2,
    "shard_gather": lambda: sharded(8, 4, axis=1)[Tensor([1, 2], dtype=dtypes.int32).shard(D2)],
    "replicated_reshape": lambda: empty(4, 4).to(D2).reshape(16) + 1,
    "shard_matmul": lambda: sharded(8, 16, axis=0, devices=D4) @ empty(16, 4, device=D4),
}

# Programs whose DEBUG_RANGEIFY printout is recorded.
DEBUG = ["softmax"]


def forked(fn):
    """`fn()`, computed in a child process, so that the generator's state stays
    the one each golden starts from."""
    read, write = os.pipe()
    if (pid := os.fork()) == 0:
        os.close(read)
        with os.fdopen(write, "w") as out: out.write(repr(fn()))
        os._exit(0)
    os.close(write)
    with os.fdopen(read) as out: text = out.read()
    if os.waitpid(pid, 0)[1] != 0: raise RuntimeError("the child failed")
    return eval(text)


def kernel_count(sink):
    return sum(1 for u in sink.toposort(enter_calls=False) if u.op is Ops.CALL and u.src[0].op is Ops.SINK)


def declare(name, program):
    counts = forked(lambda: [kernel_count(out) for _, out in scheduled(program)])
    for n in range(len(counts)):
        stem = name if n == 0 else f"{name}_{n}"
        def given(n=n): return scheduled(program)[n][0]
        def made(n=n): return scheduled(program)[n][1]
        given.__name__, made.__name__ = stem, f"{stem}_kernels"
        graph(given)
        graph(made)
    return [(name if n == 0 else f"{name}_{n}", c) for n, c in enumerate(counts)]


COUNTS = [row for name, program in PROGRAMS.items() for row in declare(name, program)]


@table
def kernel_counts():
    return ["program", "kernels"], COUNTS


def declare_debug(name, program):
    def printed():
        tsink = scheduled(program)[0][0]
        out = io.StringIO()
        DEBUG_RANGEIFY.value = 1
        with contextlib.redirect_stdout(out):
            tinygrad.schedule.rangeify.get_kernel_graph(tsink)
        return out.getvalue().rstrip("\n")
    printed.__name__ = f"{name}_debug"
    text(printed)


for name in DEBUG:
    declare_debug(name, PROGRAMS[name])


# Gathers: tensor graphs built of UOps, since a Tensor program gathers with a
# one-hot sum. Before ranges an INDEX by an integer value of a shape is a
# gather; its index is clamped into range, as rune lowers one. `<name>.golden`
# is what `get_kernel_graph` receives, `<name>_kernels.golden` what it returns.


def param(slot, *shape, dtype=dtypes.float):
    size = 1
    for n in shape: size *= n
    return UOp.param(slot, dtype, size, "CPU").reshape(shape)


def clamp(n, i): return i.maximum(0).minimum(n - 1).cast(dtypes.weakint)


def stores(value):
    out = param(0, *value.shape, dtype=value.dtype)
    return UOp.sink(out.after(out.store(value)))


def rows_x(): return param(1, 8, 4)
def rows_l(): return clamp(8, param(2, 3, dtype=dtypes.int32))


def zip_gather():
    x, i = param(1, 2, 8), param(2, 2, 3, dtype=dtypes.int32)
    row = UOp.arange(2).cast(dtypes.weakint).reshape((2, 1)).expand((2, 3))
    return stores(x.reshape((16,)).index(row * 8 + clamp(8, i)))


def zero_fill():
    i = param(2, 3, dtype=dtypes.int32)
    inside = ((i >= 0) & (i < 8)).reshape((3, 1)).expand((3, 4))
    return stores(inside.where(rows_x().index(rows_l()), 0.0))


def assign_gathered_self():
    xp = UOp.param(1, dtypes.float, 32, "CPU")
    x, i = xp.reshape((8, 4)), param(2, 6, dtype=dtypes.int32)
    l = clamp(8, i).cat(clamp(8, i.shrink(((0, 2),))))
    return UOp.sink(xp.after(x.store(x.index(l))))


GATHERS = {
    "index_rows": lambda: stores(rows_x().index(rows_l())),
    "index_rows_computed": lambda: stores((rows_x() * rows_x() + rows_x()).index(rows_l())),
    "index_read_twice": lambda: (lambda g: stores(g + g))(rows_x().index(rows_l())),
    "index_under_reduce": lambda: stores((rows_x().index(rows_l()) * param(3, 3, 4))._rop(Ops.ADD, (0,))),
    "index_broadcast": lambda: (lambda g: stores(g._rop(Ops.ADD, (1,)).reshape((3, 1)).expand((3, 4)) * g))(rows_x().index(rows_l())),
    "index_view_source": lambda: stores(param(1, 4, 8).permute((1, 0)).index(rows_l())),
    "index_zip": zip_gather,
    "index_of_index": lambda: stores(rows_x().index(clamp(8, param(2, 6, dtype=dtypes.int32).index(clamp(6, param(2, 3, dtype=dtypes.int32)))))),
    "index_zero_fill": zero_fill,
    "index_assign_self": assign_gathered_self,
}


def declare_gather(name, program):
    def given(): return tinygrad.schedule.prepare.prepare_rangeify(program())
    def made(): return tinygrad.schedule.rangeify.get_kernel_graph(given())
    given.__name__, made.__name__ = name, f"{name}_kernels"
    graph(given)
    graph(made)
    return (name, forked(lambda: kernel_count(made())))


GATHER_COUNTS = [declare_gather(name, program) for name, program in GATHERS.items()]


@table
def gather_kernel_counts():
    return ["program", "kernels"], GATHER_COUNTS
