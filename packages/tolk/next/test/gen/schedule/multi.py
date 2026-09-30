"""Goldens of tinygrad/schedule/multi.py: operations on sharded values.

Recorded programs: `<program>.golden` is the graph that `multi_pm` rewrites
when the tensors of a `Tensor` program on several CPU devices are scheduled,
and `<program>_multi.golden` what it makes of it. A program that schedules
several graphs, such as an allreduce compiled as a function of its own, records
the later ones as `<program>_<n>.golden` and `<program>_<n>_multi.golden`.
`<program>_early.golden` is what `multi_pm` makes of the first graph with
LATE_ALLREDUCE=0, which expands each allreduce in place.

Kernels: `<kernel>.golden` is a kernel whose values are sharded across the
threads of a workgroup, and `<kernel>_multi.golden` what `multi_pm` makes of it
before the kernel is compiled.
"""

import importlib
import os

from golden import graph
from tinygrad import Tensor, Variable, dtypes, function
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import ALLREDUCE_CAST, DEV, getenv
from tinygrad.uop.ops import AxisType, KernelInfo, UOp, graph_rewrite
import tinygrad.schedule.multi
import tinygrad.schedule.prepare

DEV.value = "CPU"
D2 = ("CPU:0", "CPU:1")
D4 = tuple(f"CPU:{k}" for k in range(4))
D8 = tuple(f"CPU:{k}" for k in range(8))


def scheduled(program):
    """The (input, output) pairs of each `multi_pm` rewrite that scheduling the
    tensors of `program()` runs, in order."""
    seen, rewrite = [], tinygrad.schedule.prepare.graph_rewrite

    def capture(sink, *args, name=None, **kwargs):
        out = rewrite(sink, *args, name=name, **kwargs)
        if name == "multi_pm": seen.append((sink, out))
        return out

    tinygrad.schedule.prepare.graph_rewrite = capture
    try:
        first, *rest = tensors if isinstance(tensors := program(), tuple) else (tensors,)
        first.linear_with_vars(*rest)
    finally:
        tinygrad.schedule.prepare.graph_rewrite = rewrite
    if not seen: raise RuntimeError("scheduling the tensors runs no multi_pm")
    return seen


def empty(*shape, dtype=dtypes.float, devices=None): return Tensor.empty(*shape, dtype=dtype, device=devices)


def sharded(*shape, axis, devices=D2, dtype=dtypes.float):
    """A tensor of `shape` whose storage is split along `axis` across `devices`."""
    shard = tuple(n // len(devices) if a == axis else n for a, n in enumerate(shape))
    return Tensor(empty(*shard, dtype=dtype, devices=devices).uop.unshard(axis), device=devices)


def sharded_2d(*shape):
    """A tensor of `shape` split along its first two axes across four devices,
    as a grid of two by two."""
    rng = UOp.range(4, -1, AxisType.DEVICE)
    rows, cols = rng // 2, rng % 2
    u = empty(*shape).uop.copy_to_device(D4)._shard(0, rows)._shard(1, cols).unshard((0, 1), (rows, cols))
    return Tensor(u)


def setitem(index, value, devices=D2):
    t = sharded(8, 8, axis=0, devices=devices).contiguous()
    t[index] = value() if callable(value) else value
    return t


def add_two_partitions():
    t = sharded(8, 8, axis=0, devices=D4)
    return t[2:4].pad(((2, 4), None)) + t[6:8].pad(((6, 0), None))


def batchnorm_stats():
    x = sharded(8, 4, 2, 2, axis=0, devices=D4)
    return (x - x.mean((0, 2, 3), keepdim=True)) * (x.var((0, 2, 3), keepdim=True) + 1e-5).rsqrt()


def custom_add_one(C, A):
    C, A = C.flatten(), A.flatten()
    i = UOp.range(A.numel(), 0)
    return C[i].store(A[i] + 1).end(i).sink(arg=KernelInfo(name=f"add_one_{A.numel()}"))


def custom_kernel():
    out = Tensor(empty(2, 4, devices=D2).uop.unshard(0), device=D2)
    return Tensor.custom_kernel(out, sharded(4, 4, axis=0), fxn=custom_add_one)[0] * 2


def inline_function():
    @function
    def f(a): return a * 2 + 1
    return f(sharded(4, 8, axis=0)).sum(1)


def explicit_allreduce():
    t = sharded(8, 4, axis=0, devices=D4)
    return Tensor(UOp.allreduce(t.uop, tinygrad.uop.ops.Ops.ADD, t.device))


def allreduce_cast(cast, dtype=dtypes.half):
    def program():
        # each golden runs in a process of its own, which the setting outlives
        ALLREDUCE_CAST.value = cast
        return sharded(8, 4, axis=0, dtype=dtype).float().sum(0)
    return program


PROGRAMS = {
    # elementwise
    "add": lambda: sharded(4, 8, axis=0) + sharded(4, 8, axis=0),
    "add_scalar": lambda: sharded(4, 8, axis=0) * 2,
    "add_whole": lambda: sharded(4, 8, axis=0) + empty(4, 8, devices=D2),
    "add_broadcast": lambda: sharded(4, 8, axis=0) + empty(8, devices=D2),
    "add_resharded": lambda: sharded(4, 8, axis=0) + sharded(4, 8, axis=1),
    "add_four": lambda: sharded(4, 64, axis=1, devices=D4) + 1,
    "add_replicated_scalar": lambda: sharded(4, axis=0) * Tensor(2.0).to(D2),
    "add_eight": lambda: sharded(16, 8, axis=0, devices=D8) + sharded(16, 8, axis=0, devices=D8),
    "where": lambda: (lambda x: (x > 0).where(x, 0))(sharded(4, 8, axis=1)),
    "cast_half": lambda: sharded(4, 8, axis=0, dtype=dtypes.half).float() * 2,
    "bitcast": lambda: sharded(4, 8, axis=0).bitcast(dtypes.int32) + 1,
    # reductions
    "sum_sharded_axis": lambda: sharded(4, 8, axis=0).sum(0),
    "sum_other_axis": lambda: sharded(4, 8, axis=0).sum(1),
    "sum_all": lambda: sharded(4, 8, axis=1, devices=D4).sum(),
    "max_sharded_axis": lambda: sharded(8, 4, axis=0, devices=D4).max(0),
    "allreduce_cast": allreduce_cast(1),
    "allreduce_no_cast": allreduce_cast(0),
    "allreduce_cast_bfloat16": allreduce_cast(1, dtypes.bfloat16),
    "allreduce_cast_float": allreduce_cast(1, dtypes.float),
    "explicit_allreduce": explicit_allreduce,
    # movements
    "reshape_split": lambda: sharded(4, 8, axis=0).reshape(2, 2, 8) + 1,
    "reshape_inner": lambda: sharded(4, 8, axis=1).reshape(4, 2, 4) + 1,
    "expand": lambda: sharded(4, 1, axis=0).expand(4, 8) + empty(4, 8, devices=D2),
    "permute": lambda: sharded(4, 8, axis=0).permute(1, 0) + 1,
    "pad": lambda: sharded(4, 8, axis=0).pad(((0, 0), (1, 2))) + 1,
    "flip": lambda: sharded(4, 8, axis=0).flip(1) + 1,
    "shrink_other_axis": lambda: sharded(4, 8, axis=0)[:, 2:6] + 1,
    "shrink_one_shard": lambda: sharded(4, 8, axis=0)[2:4] + 1,
    "shrink_element": lambda: sharded(8, 12, axis=1, devices=D4)[5] + 1,
    "shrink_chained": lambda: sharded(10, 8, axis=1)[2:8][1:4] + 1,
    "shrink_rows": lambda: sharded(6, 4, axis=1)[1:4] + 1,
    "reshape_then_shrink": lambda: sharded(8, 6, axis=1).reshape(4, 2, 6)[1] + 1,
    "add_two_partitions": add_two_partitions,
    "variable_shrink": lambda: sharded(4, 8, axis=0)[:, :Variable("v", 1, 8).bind(3)] + 1,
    "stack": lambda: Tensor.stack(sharded(4, 8, axis=0), sharded(4, 8, axis=0)) + 1,
    "cat": lambda: Tensor.cat(sharded(4, 8, axis=0), sharded(4, 8, axis=0), dim=1),
    "repeat": lambda: sharded(4, 8, axis=0).repeat(1, 2) + 1,
    # two sharded axes
    "grid_add": lambda: sharded_2d(4, 4) + 1,
    "grid_sum_all": lambda: sharded_2d(4, 4).sum(),
    "grid_sum_other_axis": lambda: sharded_2d(4, 4, 2).sum(2),
    "grid_matmul": lambda: sharded_2d(4, 4) @ sharded_2d(4, 4),
    "grid_to_one": lambda: sharded_2d(4, 4).to("CPU:0") + 1,
    # matmuls
    "matmul_rows": lambda: sharded(8, 16, axis=0) @ empty(16, 4, devices=D2),
    "matmul_columns": lambda: empty(8, 16, devices=D2) @ sharded(16, 4, axis=1),
    "matmul_contracted": lambda: sharded(8, 16, axis=1) @ sharded(16, 4, axis=0),
    "matmul_resharded": lambda: sharded(8, 16, axis=0) @ sharded(16, 4, axis=0),
    "double_matmul": lambda: sharded(8, 8, axis=0) @ empty(8, 8, devices=D2) @ empty(8, 8, devices=D2),
    # networks
    "softmax": lambda: sharded(4, 8, axis=0).softmax(-1),
    "softmax_sharded_axis": lambda: sharded(4, 8, axis=1).softmax(-1),
    "layernorm": lambda: sharded(4, 8, axis=0, devices=D4).layernorm(),
    "rmsnorm": lambda: (lambda x: x * (x * x).mean(-1, keepdim=True).add(1e-5).rsqrt() * empty(8, devices=D2))(
        sharded(4, 8, axis=0)),
    "attention": lambda: sharded(2, 2, 8, 4, axis=0).scaled_dot_product_attention(
        sharded(2, 2, 8, 4, axis=0), sharded(2, 2, 8, 4, axis=0), is_causal=True),
    "conv": lambda: sharded(2, 2, 8, 8, axis=0).conv2d(empty(3, 2, 3, 3, devices=D2)),
    "embedding": lambda: sharded(10, 4, axis=1)[Tensor([1, 2, 3], dtype="int").to(D2)],
    "arange": lambda: sharded(4, 4, axis=0) + Tensor.arange(16).reshape(4, 4).to(D2),
    "cumsum": lambda: sharded(4, 8, axis=0).cumsum(1),
    "sort": lambda: sharded(4, 8, axis=0).sort(1)[0],
    "interpolate": lambda: sharded(4, 8, 8, axis=0).interpolate((5, 5)),
    "batchnorm_stats": batchnorm_stats,
    # copies
    "shard": lambda: empty(4, 8).shard(D2, axis=0) + 1,
    "replicate": lambda: empty(4, 8).to(D2) + 1,
    "gather": lambda: sharded(4, 8, axis=0).to("CPU:0") + 1,
    "gather_four": lambda: sharded(8, 4, axis=0, devices=D4).to("CPU:1") + 1,
    "select_first": lambda: empty(4, 8, devices=D2).to("CPU:1") + 1,
    "reshard_devices": lambda: sharded(4, 8, axis=0).to(D4[2:]) + 1,
    # stores
    "contiguous": lambda: (sharded(4, 8, axis=0) + 1).contiguous().sum(1),
    "assign": lambda: sharded(4, 8, axis=0).assign(sharded(4, 8, axis=0) + 1),
    "assign_shard": lambda: sharded(4, 8, axis=0).assign(empty(4, 8).shard(D2, axis=0)),
    "setitem_rows": lambda: setitem(slice(2, 4), 1.0),
    "setitem_row": lambda: setitem(5, 1.0),
    "setitem_columns": lambda: setitem((slice(None), slice(2, 4)), 1.0),
    "setitem_stride": lambda: setitem(slice(None, None, 4), 0.0, D4),
    "setitem_replicated_value": lambda: setitem(slice(2, 6), lambda: empty(4, 8).to(D4), D4),
    "setitem_sharded_value": lambda: setitem(slice(None, None, 2), lambda: sharded(4, 8, axis=0, devices=D4), D4),
    "shard_invalids": lambda: Tensor.invalids(8).shard(D2, axis=0).contiguous(),
    "custom_kernel": custom_kernel,
    "inline_function": inline_function,
}

# Programs whose allreduces are also expanded in place (LATE_ALLREDUCE=0).
EARLY = ["sum_sharded_axis", "matmul_contracted", "explicit_allreduce"]


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


def declare(name, program):
    for n in range(forked(lambda: len(scheduled(program)))):
        stem = name if n == 0 else f"{name}_{n}"
        def given(n=n): return scheduled(program)[n][0]
        def made(n=n): return scheduled(program)[n][1]
        given.__name__, made.__name__ = stem, f"{stem}_multi"
        graph(given)
        graph(made)


def declare_early(name, program):
    def early():
        given = scheduled(program)[0][0]
        # multi.py reads the setting, through getenv's cache, when it is imported
        os.environ["LATE_ALLREDUCE"] = "0"
        getenv.cache_clear()
        return graph_rewrite(given, importlib.reload(tinygrad.schedule.multi).multi_pm)
    early.__name__ = f"{name}_early"
    graph(early)


for name, program in PROGRAMS.items():
    declare(name, program)
for name in EARLY:
    declare_early(name, PROGRAMS[name])


# Kernels whose values are sharded across a workgroup's threads.

def threads(n, axis=0): return UOp.range(n, axis, AxisType.LOCAL)
def out(*shape, slot=0): return UOp.param(slot, dtypes.float, shape_size(shape), "CPU").reshape(shape)
def shape_size(shape):
    n = 1
    for s in shape: n *= s
    return n


def fragment_index(pattern):
    # thread ty owns 8 of 64 rows: rows ty*8+i (blocks), ty+8*i (strided)
    ty, ir, j = threads(8), UOp.range(8, 1, AxisType.LOOP), UOp.range(8, 2, AxisType.LOOP)
    frag = UOp.placeholder((8, 8), dtypes.float32, 0, AddrSpace.REG).unshard((0,), (ty,))
    row = ty * 8 + ir if pattern == "blocks" else ty + ir * 8
    C = out(64, 8)
    return C[row, j].store(frag[row, j]).end(j, ir, ty).sink(arg=KernelInfo(name=f"{pattern}_fragment"))


def fragment(shape, axes, rngs, addrspace=AddrSpace.LOCAL):
    return UOp.placeholder(shape, dtypes.float32, 0, addrspace).unshard(axes, rngs)


def alu_scalar():
    ty = threads(8)
    frag = fragment((8,), (0,), (ty,))
    v = frag.after(frag.store(1.5)) * 2.0
    return out(64).store(v).end(ty).sink(arg=KernelInfo(name="alu_scalar", opts_to_apply=()))


def alu_whole():
    ty = threads(8)
    frag = fragment((8,), (0,), (ty,))
    v = frag.after(frag.store(0.0)) + out(64, slot=1)
    return out(64).store(v).end(ty).sink(arg=KernelInfo(name="alu_whole", opts_to_apply=()))


def store_value():
    ty = threads(8)
    frag = fragment((8,), (0,), (ty,))
    v = frag.after(frag.store(0.0)) + 2.5
    return out(64).store(v).end(ty).sink(arg=KernelInfo(name="store_value", opts_to_apply=()))


def store_value_two_axes():
    ty, tx = threads(4, 0), threads(2, 1)
    frag = fragment((2, 1, 1, 2), (1, 2), (ty, tx), AddrSpace.REG)
    v = frag.after(frag.store(0.0)) + out(2, 4, 2, 2, slot=1)
    return out(2, 4, 2, 2).store(v).end(tx, ty).sink(arg=KernelInfo(name="store_value_two_axes", opts_to_apply=()))


def store_load():
    ty = threads(8)
    frag = fragment((8,), (0,), (ty,), AddrSpace.REG)
    return out(64).store(frag.after(frag.store(out(64, slot=1)))).end(ty).sink(
        arg=KernelInfo(name="store_load", opts_to_apply=()))


KERNELS = {
    "fragment_blocks": lambda: fragment_index("blocks"),
    "fragment_strided": lambda: fragment_index("strided"),
    "alu_scalar": alu_scalar,
    "alu_whole": alu_whole,
    "store_value": store_value,
    "store_value_two_axes": store_value_two_axes,
    "store_load": store_load,
}


def declare_kernel(name, kernel):
    def given(): return kernel()
    def made(): return graph_rewrite(kernel(), tinygrad.schedule.multi.multi_pm)
    given.__name__, made.__name__ = name, f"{name}_multi"
    graph(given)
    graph(made)


for name, kernel in KERNELS.items():
    declare_kernel(name, kernel)
