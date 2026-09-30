"""Goldens of tinygrad/schedule/indexing.py: movements as index arithmetic, and
the ranges of a tensor graph.

Recorded programs: `<program>.golden` is the graph that `run_rangeify` receives
when the tensors of a `Tensor` program are scheduled on the CPU, and
`<program>_rangeified.golden` what it returns. A program that schedules several
graphs, such as an allreduce compiled as a function of its own, records the
later ones as `<program>_<n>.golden` and `<program>_<n>_rangeified.golden`.
`<program>_debug.golden` is what `run_rangeify` prints of the first graph with
`debug`.

Movements: `movement_<case>.golden` is the sink of the indices that
`apply_movement_op` gives for one movement of a source, from indices into the
result that are ranges, or expressions of them.
"""

from golden import graph, text
from tinygrad import Tensor, Variable, dtypes
from tinygrad.helpers import DEV, RING
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp
import contextlib
import io
import os
import tinygrad.schedule.indexing
import tinygrad.schedule.rangeify

DEV.value = "CPU"
DEVICES = ("CPU:0", "CPU:1", "CPU:2", "CPU:3")


def scheduled(program):
    """The (input, output) pairs of each `run_rangeify` that scheduling the
    tensors of `program()` runs, in order."""
    seen, run = [], tinygrad.schedule.rangeify.run_rangeify

    def capture(tsink, debug=False):
        out = run(tsink, debug)
        seen.append((tsink, out))
        return out

    tinygrad.schedule.rangeify.run_rangeify = capture
    try:
        first, *rest = tensors if isinstance(tensors := program(), tuple) else (tensors,)
        first.linear_with_vars(*rest)
    finally:
        tinygrad.schedule.rangeify.run_rangeify = run
    if not seen: raise RuntimeError("scheduling the tensors runs no rangeify")
    return seen


def empty(*shape, dtype=dtypes.float): return Tensor.empty(*shape, dtype=dtype)


def assign_double_diamond():
    # each assign reads, through a sum, the buffer the other overwrites
    b0, b1 = empty(16).contiguous(), empty(16).contiguous()
    r0, r1 = (empty(16, 16) - b1.contiguous()).sum(1), (empty(16, 16) - b0.contiguous()).sum(1)
    return b0.assign(r0 * b0), b1.assign(r1 * b1)


def add_one_kernel(B, A):
    A, B = A.flatten(), B.flatten()
    i = UOp.range(A.numel(), 0)
    return B[i].store(A[i] + 1).end(i).sink(arg=KernelInfo(name=f"add_one_{A.numel()}"))


def setitem():
    t = empty(8, 8).contiguous()
    t[2:4] = 1.0
    return t


def shard_sum_ring():
    # each golden runs in a process of its own, which the setting outlives
    RING.value = 2
    return empty(4, 400).shard(DEVICES, axis=0).sum(0)


def custom_kernel():
    out = Tensor.custom_kernel(empty(4, 4), (empty(4, 4) * 2).reshape(16), fxn=add_one_kernel)[0]
    return out + 1


PROGRAMS = {
    # elementwise, reductions and matmuls
    "add": lambda: empty(4, 4) + empty(4, 4),
    "sum": lambda: empty(4, 8).sum(1),
    "sum_all": lambda: empty(4, 8).sum(),
    "matmul": lambda: empty(4, 8) @ empty(8, 3),
    "double_matmul": lambda: empty(16, 16) @ empty(16, 16) @ empty(16, 16),
    "matmul_relu_cat": lambda: Tensor.cat(empty(100, 512), (empty(1, 512) @ empty(512, 512)).relu(), dim=0),
    "einsum": lambda: Tensor.einsum("ij,jk->ik", empty(3, 4), empty(4, 5)),
    "where": lambda: (empty(4, 4) > 0).where(empty(4, 4), 0),
    "cast_half": lambda: empty(4, 4, dtype=dtypes.half).float().sum(),
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
    "rmsnorm": lambda: (lambda x: x * (x * x).mean(-1, keepdim=True).add(1e-5).rsqrt() * empty(8))(empty(2, 8)),
    # one movement each
    "permute": lambda: empty(4, 8).permute(1, 0) + 1,
    "flip": lambda: empty(4, 8).flip(0) * 2,
    "shrink": lambda: empty(8, 8)[2:6, 1:5] + 1,
    "pad": lambda: empty(4, 4).pad(((1, 2), (0, 3))) + 1,
    "reshape": lambda: empty(4, 6).reshape(3, 8) + 1,
    "reshape_split": lambda: empty(24).reshape(2, 3, 4).sum(1),
    "expand": lambda: empty(4, 1).expand(4, 5) + empty(4, 5),
    "outer": lambda: empty(8)[:, None] * empty(8)[None, :],
    # movements in sequence
    "pad_reshape": lambda: empty(3, 5).pad(((0, 1), (0, 3))).reshape(32) + 1,
    "flip_reshape": lambda: empty(4, 6).flip(1).reshape(2, 12).sum(0),
    "pad_reduce": lambda: empty(4, 4).pad(((1, 1), (1, 1))).sum(0),
    "repeat": lambda: empty(2, 3).repeat(3, 2) + 1,
    "roll": lambda: empty(8).roll(3) + 1,
    "interpolate": lambda: empty(1, 1, 4, 4).interpolate((8, 8)),
    # reductions below a broadcast
    "broadcast_reduce": lambda: empty(4, 8) - empty(4, 8).max(1, keepdim=True),
    "softmax": lambda: empty(4, 8).softmax(-1),
    "layernorm": lambda: empty(4, 8).layernorm(),
    "standardize": lambda: (lambda x: (x - x.mean(1, keepdim=True)) / x.std(1, keepdim=True))(empty(4, 8)),
    "diamond": lambda: (lambda a: (a.sum(0, keepdim=True) * a).sum(1) + a.max())(empty(8, 8).exp()),
    "attention": lambda: empty(2, 2, 8, 4).scaled_dot_product_attention(empty(2, 2, 8, 4), empty(2, 2, 8, 4)),
    "argmax": lambda: empty(4, 8).argmax(1),
    # consumers that index a node differently
    "two_consumers": lambda: (lambda x: x.sum(0) + x.sum(1))(empty(4, 4).exp()),
    "two_consumers_permuted": lambda: (lambda x: x + x.permute(1, 0))(empty(4, 4).exp()),
    "shared_view": lambda: (lambda a: a.reshape(16) + a.permute(1, 0).reshape(16))(empty(4, 4) * 2),
    # convolutions and pools
    "conv": lambda: empty(1, 2, 8, 8).conv2d(empty(3, 2, 3, 3)),
    "conv_bn_relu": lambda: empty(1, 2, 8, 8).conv2d(empty(4, 2, 3, 3)).batchnorm(None, None, empty(4),
                                                                                  empty(4).rsqrt()).relu(),
    "maxpool": lambda: empty(1, 2, 8, 8).max_pool2d(),
    "avgpool": lambda: empty(1, 2, 8, 8).avg_pool2d(),
    # stacks, concatenations, and a tensor of a list
    "cat": lambda: Tensor.cat(empty(2, 4), empty(3, 4)),
    "stack": lambda: Tensor.stack(empty(4), empty(4), empty(4)),
    "stack_eight": lambda: Tensor.stack(*[empty(3) for _ in range(8)]),
    "stack_twelve": lambda: Tensor.stack(*[empty(3) for _ in range(12)]),
    "tensor_of_list": lambda: Tensor(list(range(40)), dtype="int") + 1,
    "arange": lambda: Tensor.arange(16).reshape(4, 4) + 1,
    "cumsum": lambda: empty(8).cumsum(),
    "cumsum_rows": lambda: empty(4, 32).cumsum(1),
    "triu": lambda: empty(6, 6).triu(1),
    # gathers, sorts and selections
    "embedding": lambda: empty(10, 4)[Tensor([1, 2, 3], dtype="int")],
    "gather": lambda: Tensor(list(range(8)), dtype="int").gather(0, Tensor([0, 0], dtype="int")),
    "two_gathers": lambda: (lambda f, i: f.gather(0, i) + f.gather(0, i * 2 + 1))(
        Tensor(list(range(8)), dtype="int"), Tensor([0, 0], dtype="int")),
    "sort": lambda: empty(8).sort()[0],
    "topk": lambda: empty(16).topk(3)[0],
    # stores into storage
    "contiguous": lambda: (empty(4, 4) + 1).contiguous().sum(0),
    "assign": lambda: empty(4, 4).assign(empty(4, 4) + 1),
    "assign_permuted": lambda: empty(4, 4, dtype="int").permute(1, 0).assign(Tensor.arange(16).reshape(4, 4)),
    "assign_double_diamond": assign_double_diamond,
    "setitem": setitem,
    "custom_kernel": custom_kernel,
    # symbolic shapes and variables as data
    "variable_shrink": lambda: Tensor.ones(10)[:Variable("v", 1, 10).bind(3)] * 2,
    "variable_offset": lambda: (lambda v: empty(10)[v:v + 3] * 2)(Variable("i", 0, 7).bind(2)),
    "variable_data_and_shape": lambda: (lambda v: Tensor.ones(10)[:v] * Tensor(v.cast(dtypes.float)))(
        Variable("shared_v", 1, 10).bind(3)),
    "variable_stack": lambda: (lambda v: Tensor(UOp.stack(v, v + 1).cast(dtypes.int)) + Tensor.arange(2))(
        Variable("v", 0, 10).bind(3)),
    # several devices
    "copy": lambda: empty(4, 4).to("CPU:1") * 2,
    "bitcast_copy": lambda: empty(4, 4).bitcast(dtypes.int32).to("CPU:1") + 1,
    "shard_add": lambda: empty(4, 64).shard(DEVICES, axis=1) + 1,
    "shard_sum": lambda: empty(4, 64).shard(DEVICES, axis=0).sum(0),
    "shard_sum_ring": shard_sum_ring,
    "shard_matmul": lambda: empty(8, 16).shard(DEVICES, axis=0) @ empty(16, 4).shard(DEVICES, axis=None),
}

# Programs whose DEBUG_RANGEIFY printout is recorded: a pad that a reduction
# realizes, and a node whose consumers index it differently.
DEBUG = ["pad_reduce", "two_consumers", "shared_view", "softmax"]


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
        given.__name__, made.__name__ = stem, f"{stem}_rangeified"
        graph(given)
        graph(made)


def declare_debug(name, program):
    def printed():
        tsink = scheduled(program)[0][0]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            tinygrad.schedule.indexing.run_rangeify(tsink, True)
        return out.getvalue().rstrip("\n")
    printed.__name__ = f"{name}_debug"
    text(printed)


for name, program in PROGRAMS.items():
    declare(name, program)
for name in DEBUG:
    declare_debug(name, PROGRAMS[name])


# Hand-built graphs: `<graph>.golden` and `<graph>_rangeified.golden`, as for
# programs.

def expand_by_range(stored_twice):
    # an expand whose added axis is a range, as a kernel's own loop injects it
    r = UOp.range(4, 7, AxisType.LOOP)
    e = UOp.param(1, dtypes.float, 3, "CPU").exp2()._mop(Ops.EXPAND, (r,))
    out = UOp.param(0, dtypes.float, 12, "CPU")
    value = e + e.flip((1,)) if stored_twice else e
    return UOp.sink(out.after(out.reshape((4, 3)).store(value)))


def mstack_of_values():
    # the shards of a value on two devices, each computed on its device
    shards = [UOp.param(1 + k, dtypes.float, 4, d).exp2() for k, d in enumerate(DEVICES[:2])]
    out = UOp.param(0, dtypes.float, 4, DEVICES[:2])
    return UOp.sink(out.after(out.store(UOp.mstack(*shards))))


def scalar_and_wide_uses():
    # a scalar used by a full reduction and broadcast to a wide product: the
    # two stores reach nodes that differ only in their movements
    x = UOp.param(1, dtypes.float, 32, "CPU").reshape((4, 8))
    recip = UOp.param(2, dtypes.float, 1, "CPU").reshape(()).reciprocal()
    scalar = x._rop(Ops.ADD, (0, 1)) * recip
    wide = x * recip.reshape((1, 1)).expand((4, 8))
    out0, out1 = UOp.param(0, dtypes.float, 1, "CPU"), UOp.param(3, dtypes.float, 32, "CPU")
    return UOp.sink(out0.after(out0.reshape(()).store(scalar)), out1.after(out1.reshape((4, 8)).store(wide)))


GRAPHS = {
    "expand_by_range": lambda: expand_by_range(False),
    "expand_by_range_stored": lambda: expand_by_range(True),
    "mstack_of_values": mstack_of_values,
    "scalar_and_wide_uses": scalar_and_wide_uses,
}


def declare_graph(name, sink):
    def given(): return sink()
    def made(): return tinygrad.schedule.indexing.run_rangeify(sink())
    given.__name__, made.__name__ = name, f"{name}_rangeified"
    graph(given)
    graph(made)


for name, sink in GRAPHS.items():
    declare_graph(name, sink)


# Movements

def rng(n, axis): return UOp.range(n, axis, AxisType.WEAK)
def ranges(*shape): return tuple(rng(n, axis) for axis, n in enumerate(shape))
def moved(op, in_shape, arg, rngs):
    return UOp.sink(*tinygrad.schedule.indexing.apply_movement_op(op, in_shape, arg, rngs))

N = UOp.variable("n", 1, 8)

MOVEMENTS = {
    "shrink": lambda: moved(Ops.SHRINK, (8, 6), ((2, 4), (0, 6)), ranges(4, 6)),
    "permute": lambda: moved(Ops.PERMUTE, (2, 3, 4), (2, 0, 1), ranges(4, 2, 3)),
    "flip": lambda: moved(Ops.FLIP, (4, 5), (True, False), ranges(4, 5)),
    "expand": lambda: moved(Ops.EXPAND, (3,), (2, 4), ranges(2, 4, 3)),
    "expand_symbolic": lambda: moved(Ops.EXPAND, (3,), (N,), (rng(N, 0), rng(3, 1))),
    "pad": lambda: moved(Ops.PAD, (4, 4), ((1, 7), (0, 4)), ranges(7, 4)),
    "pad_end": lambda: moved(Ops.PAD, (4, 4), ((0, 6), (0, 4)), ranges(6, 4)),
    "pad_symbolic": lambda: moved(Ops.PAD, (N, 4), ((1, N + 2), (0, 4)), (rng(N + 2, 0), rng(4, 1))),
    "reshape_flatten": lambda: moved(Ops.RESHAPE, (2, 3), (6,), ranges(6)),
    "reshape_unflatten": lambda: moved(Ops.RESHAPE, (6,), (2, 3), ranges(2, 3)),
    "reshape_regroup": lambda: moved(Ops.RESHAPE, (4, 6), (3, 8), ranges(3, 8)),
    "reshape_unit_axes": lambda: moved(Ops.RESHAPE, (1, 4, 1), (4,), ranges(4)),
    "reshape_symbolic": lambda: moved(Ops.RESHAPE, (N, 4), (N * 4,), (rng(N * 4, 0),)),
    "reshape_of_flipped": lambda: moved(Ops.RESHAPE, (12,), (3, 4), (2 - rng(3, 0), rng(4, 1))),
    "reshape_of_padded": lambda: moved(Ops.RESHAPE, (8,), (2, 4),
                                       ((rng(3, 0) - 1).valid(rng(3, 0) >= 1), rng(4, 1))),
}


def declare_movement(name, case):
    def given(): return case()
    given.__name__ = f"movement_{name}"
    graph(given)


for name, case in MOVEMENTS.items():
    declare_movement(name, case)
