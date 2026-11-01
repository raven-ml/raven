"""Goldens of tinygrad/schedule/__init__.py: schedules of tensor programs.

Recorded programs: `<program>.golden` is the sink that realizing the tensors of
a `Tensor` program on the CPU hands to `create_linear_with_vars`, and
`<program>_linear.golden` the schedule it returns, its buffers placed in
arenas. `<program>_kernels.golden` is a kernel graph that scheduling hands to
`create_schedule`, and `<program>_schedule.golden` what it returns; a program
that schedules several kernel graphs records the later ones as
`<program>_kernels_<n>.golden` and `<program>_schedule_<n>.golden`.
`var_vals.golden` is the value of each variable of each program's schedule.
A kernel graph is the body of a call, and reads the call's scalar arguments as
parameters of their slots: `arguments.golden` is the value of each, by slot,
for each kernel graph.
"""

import os
import sys

from golden import graph, table
from tinygrad import Tensor, Variable, dtypes, function
from tinygrad.dtype import AddrSpace, Invalid
from tinygrad.helpers import DEV
from tinygrad.uop.ops import KernelInfo, Ops, UOp
import tinygrad.schedule
import tinygrad.tensor

DEV.value = "CPU"
DISK = "DISK:/tmp/tolk-schedule"


def scheduled(program):
    """The (big sink, linear, var_vals) of the last schedule that realizing
    the tensors of `program()` makes, and the (input, output, arguments) of
    each `create_schedule` it runs, `arguments` the value of each scalar
    argument of the call whose body the input is, by slot."""
    linears, schedules, calls = [], [], []
    linear_with_vars, schedule = tinygrad.tensor.create_linear_with_vars, tinygrad.schedule.create_schedule
    to_call = tinygrad.schedule.transform_to_call

    def capture_linear(sink):
        linear, var_vals = linear_with_vars(sink)
        linears.append((sink, linear, var_vals))
        return linear, var_vals

    def capture_call(sink):
        calls.append(out := to_call(sink))
        return out

    def capture_schedule(sink):
        out = schedule(sink)
        # the call lower_sink_to_linear lowers; its scalar arguments are bound
        # variables, or parameters of the big sink's call, whose are
        call = sys._getframe(1).f_locals["call"]
        outer = {k: b.arg.val for k, b in enumerate(calls[-1].src[1:]) if b.is_bound_var}
        def value(b):
            if b.is_bound_var: return b.arg.val
            if b.op is Ops.PARAM and b.arg.addrspace is AddrSpace.ALU: return outer[b.arg.slot]
            return None
        arguments = [(k, v) for k, b in enumerate(call.src[1:]) if (v := value(b)) is not None]
        schedules.append((sink, out, arguments))
        return out

    tinygrad.tensor.create_linear_with_vars, tinygrad.schedule.create_schedule = capture_linear, capture_schedule
    tinygrad.schedule.transform_to_call = capture_call
    try:
        first, *rest = tensors if isinstance(tensors := program(), tuple) else (tensors,)
        first.linear_with_vars(*rest)
    finally:
        tinygrad.tensor.create_linear_with_vars, tinygrad.schedule.create_schedule = linear_with_vars, schedule
        tinygrad.schedule.transform_to_call = to_call
    # data given as a Python list is realized by schedules of its own first
    if not linears: raise RuntimeError("scheduling the tensors makes no schedule")
    return linears[-1], schedules


def empty(*shape, dtype=dtypes.float, device=None): return Tensor.empty(*shape, dtype=dtype, device=device)


def add_one(C, A):
    C, A = C.flatten(), A.flatten()
    i = UOp.range(A.numel(), 0)
    return C[i].store(A[i] + 1).end(i).sink(arg=KernelInfo(name=f"add_one_{A.numel()}"))


def add_n(C, n, A):
    C, A = C.flatten(), A.flatten()
    i = UOp.range(A.numel(), 0)
    return C[i].store(A[i] + n).end(i).sink(arg=KernelInfo(name=f"add_n_{A.numel()}"))


def custom_kernel_scalar():
    """A variable passed between two buffers (runtime/test_custom_kernel.py,
    test_scalar_arg_any_position), on CPU:1, which the engine's tests place on
    each GPU."""
    n = Tensor(Variable("n", 0, 100, dtype=dtypes.int32).bind(5), device="CPU:1")
    a = empty(4, dtype=dtypes.int32, device="CPU:1") * 2
    return Tensor.custom_kernel(empty(4, dtype=dtypes.int32, device="CPU:1"), n, a, fxn=add_n)[0] + 1


def add_first_n(n, C, A):
    C, A = C.flatten(), A.flatten()
    i = UOp.range(A.numel(), 0)
    return C[i].store(A[i] + n).end(i).sink(arg=KernelInfo(name=f"add_first_n_{A.numel()}"))


def custom_kernel_scalar_first():
    """The variable in slot 0, before the buffers (test_scalar_arg_any_position
    and test_scalar_arg_first_other_device), on CPU:1."""
    n = Tensor(Variable("n", 0, 100, dtype=dtypes.int32).bind(5), device="CPU:1")
    a = empty(4, dtype=dtypes.int32, device="CPU:1") * 2
    return Tensor.custom_kernel(n, empty(4, dtype=dtypes.int32, device="CPU:1"), a, fxn=add_first_n)[1] + 1


def custom_kernel_scalar_with_other_variable():
    """A kernel of another variable realized beside the custom kernel's
    (test_scalar_arg_with_other_variable), on CPU:1."""
    m = Variable("m", 0, 100, dtype=dtypes.int32).bind(70)
    other = (empty(4, dtype=dtypes.int32, device="CPU:1") + Tensor(m, device="CPU:1")).contiguous()
    return other, custom_kernel_scalar()


def add_ignoring_first(C, A, B, n):
    C, B = C.flatten(), B.flatten()
    k = UOp.variable("k", 0, 100, dtype=dtypes.int32)
    i = UOp.range(C.numel(), 0)
    return C[i].store(B[i] + n + k).end(i).sink(arg=KernelInfo(name=f"ignore_first_var_{C.numel()}"))


def custom_kernel_unused_then_variable():
    """A kernel that ignores its second buffer, takes a variable last and reads
    one of its own, which the run binds (test_unused_arg_then_var), on CPU:1."""
    n = Tensor(Variable("n", 0, 100, dtype=dtypes.int32).bind(5), device="CPU:1")
    srcs = (empty(4, dtype=dtypes.int32, device="CPU:1"), empty(4, dtype=dtypes.int32, device="CPU:1"),
            empty(4, dtype=dtypes.int32, device="CPU:1") * 2, n)
    return Tensor.custom_kernel(*srcs, fxn=add_ignoring_first)[0] + 1


def assign_double_diamond():
    # each assign reads, through a sum, the buffer the other overwrites
    b0, b1 = empty(16).contiguous(), empty(16).contiguous()
    r0, r1 = (empty(16, 16) - b1.contiguous()).sum(1), (empty(16, 16) - b0.contiguous()).sum(1)
    return b0.assign(r0 * b0), b1.assign(r1 * b1)


def read_then_overwrite():
    # a value reads a buffer that an assign then overwrites
    a = empty(8).contiguous()
    b = a * 2
    a.assign(empty(8) + 1)
    return b, a


def read_overwritten_through():
    # a value reads a buffer as it was, and a value computed from the assign
    # that overwrites it
    a = empty(8).contiguous()
    b = a * 2
    a.assign(a + 1)
    return b + (a * 3).contiguous()


def setitem():
    t = empty(8, 8).contiguous()
    t[2:4] = 1.0
    return t


def precompiled_function():
    @function(precompile=True)
    def f(a): return (a * 2 + 1).sum(0)
    return f(empty(4, 4)) + 1


def chained_functions():
    # one body scheduled three times, with call-local storage of its own
    @function(precompile=True)
    def f(x): return (x + 1).clone() + (x + 2).clone()
    return f(f(f(empty(4))))


def precompiled_scalar():
    # a precompiled call of a buffer and a bound variable
    x = empty(3, dtype=dtypes.int32)
    scalar = UOp.param(1, x.dtype, addrspace=AddrSpace.ALU)
    bound = Variable("v", 1, 8, dtype=x.dtype).bind(3)
    out, = UOp.call_with_outputs((x.uop.param_like(0) * scalar,), x.uop, bound, precompile=True)
    return Tensor(out)


def inline_function():
    @function
    def f(a, b): return a * 2 + b
    return f(empty(4, 8), empty(4, 8)).sum(1)


PROGRAMS = {
    "add": lambda: empty(4, 4) + empty(4, 4),
    "sum": lambda: empty(4, 8).sum(1),
    "matmul": lambda: empty(4, 8) @ empty(8, 3),
    "double_matmul": lambda: empty(16, 16) @ empty(16, 16) @ empty(16, 16),
    "softmax": lambda: empty(4, 8).softmax(-1),
    "layernorm": lambda: empty(4, 8).layernorm(),
    "attention": lambda: empty(2, 2, 8, 4).scaled_dot_product_attention(empty(2, 2, 8, 4), empty(2, 2, 8, 4)),
    "conv": lambda: empty(1, 2, 8, 8).conv2d(empty(3, 2, 3, 3)),
    "cat": lambda: Tensor.cat(empty(2, 4), empty(3, 4)),
    # integer divisions on a pad's padding, which reads zeros
    "pad_idiv": lambda: (lambda a, b: (a.div(b, rounding_mode="trunc") + a % b).pad(((1, 1),)).contiguous())(
        empty(10, dtype=dtypes.int32), empty(10, dtype=dtypes.int32)),
    # a divisor bounded away from 0, stored: the padding loads its buffer
    "pad_idiv_stored": lambda: (lambda a, b: a.div(b.maximum(1).contiguous(), rounding_mode="trunc").pad(((1, 1),))
                                .contiguous())(empty(10, dtype=dtypes.int32), empty(10, dtype=dtypes.int32)),
    # a quotient padded twice, the outer pad as wide as the host's upcast
    "pad_idiv_nested": lambda: (lambda a, b: (a.div(b, rounding_mode="trunc").pad(((1, 1),)) + 1).pad(((8, 8),))
                                .contiguous())(empty(38, dtype=dtypes.int32), empty(38, dtype=dtypes.int32)),
    # a quotient shrunk, then padded
    "pad_idiv_shrunk": lambda: (lambda a, b: a.div(b, rounding_mode="trunc")[1:].pad(((1, 1),)).contiguous())(
        empty(11, dtype=dtypes.int32), empty(11, dtype=dtypes.int32)),
    "sort": lambda: empty(8).sort()[0],
    "embedding": lambda: empty(10, 4)[Tensor([1, 2, 3], dtype="int")],
    "split_sum": lambda: empty(65536).sum(),
    "two_outputs": lambda: (lambda x: (x + 1, x * 2))(empty(4, 4)),
    "reduce_multiple_paths": lambda: (lambda a: (a.sum().exp2(), a.sum() + a.sum().exp2()))(empty(4, 4)),
    # stores into storage, and the order of reads and writes
    "contiguous": lambda: (empty(4, 4) + 1).contiguous().sum(0),
    "assign": lambda: empty(4, 4).assign(empty(4, 4) + 1),
    "assign_permuted_self": lambda: (lambda a: a.assign(a.permute(1, 0) + 1))(empty(4, 4).contiguous()),
    # stores whose value reads the destination at other indices, as runtime/test_assign.py's *_from_self
    "assign_flipped_from_self": lambda: (lambda a: a.flip(0).assign(a))(empty(3, dtype=dtypes.int32).contiguous()),
    "add_row_from_self": lambda: (lambda a: a.assign(a + a[0]))(empty(3, 2, dtype=dtypes.int32).contiguous()),
    "roll_from_self": lambda: (lambda a: a.assign(a.roll(1, 0)))(empty(2, 2, dtype=dtypes.int32).contiguous()),
    "assign_double_diamond": assign_double_diamond,
    "read_then_overwrite": read_then_overwrite,
    "read_overwritten_through": read_overwritten_through,
    "setitem": setitem,
    "assign_bitcast": lambda: empty(4, dtype=dtypes.uint32).contiguous().bitcast(dtypes.float).assign(empty(4) + 1),
    "clone": lambda: empty(4).clone(),
    "full_invalid": lambda: Tensor.full((4,), Invalid, dtype=dtypes.float),
    # calls
    "custom_kernel": lambda: Tensor.custom_kernel(empty(4, 4), empty(4, 4) * 2, fxn=add_one)[0] + 1,
    "custom_kernel_scalar": custom_kernel_scalar,
    "custom_kernel_scalar_first": custom_kernel_scalar_first,
    "custom_kernel_scalar_with_other_variable": custom_kernel_scalar_with_other_variable,
    "custom_kernel_unused_then_variable": custom_kernel_unused_then_variable,
    "inline_function": inline_function,
    "precompiled_function": precompiled_function,
    "chained_functions": chained_functions,
    "precompiled_scalar": precompiled_scalar,
    # copies
    "copy": lambda: empty(4, 4).to("CPU:1") * 2,
    "copy_computed": lambda: (empty(4, 4) + 1).to("CPU:1"),
    "copy_view": lambda: empty(8, 8)[2:6].to("CPU:1") + 1,
    "copy_one": lambda: empty(1).to("CPU:1") * 2,
    "disk_to": lambda: empty(64, dtype=dtypes.uint8, device=DISK).to("CPU") + 1,
    "disk_view_to": lambda: empty(64, dtype=dtypes.uint8, device=DISK)[8:24].to("CPU") + 1,
    "disk_store": lambda: empty(16, dtype=dtypes.uint8, device=DISK).assign(empty(16, dtype=dtypes.uint8).to(DISK)),
    # variables
    "variable_shrink": lambda: empty(10)[:Variable("v", 1, 10).bind(3)] * 2,
    "variable_reduce": lambda: empty(10, 4)[:Variable("v", 1, 10).bind(5)].sum(0),
    "variable_two": lambda: (empty(10)[:Variable("v", 1, 10).bind(4)] * 2,
                             empty(10)[:Variable("w", 1, 10).bind(7)] + 1),
    "variable_same": lambda: (lambda v: (empty(10)[:v] * 2, empty(10, 2)[:v].sum(1)))(Variable("v", 1, 10).bind(4)),
    "variable_offset": lambda: (lambda v: empty(10)[v:v + 3] * 2)(Variable("i", 0, 7).bind(2)),
    # a variable that folds away before any kernel reads it
    "variable_unused": lambda: empty(4, dtype=dtypes.int32) + Tensor(Variable("v", 1, 10).bind(3)) * 0,
}


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


def words(pairs): return " ".join(f"{k}={v}" for k, v in pairs) or "-"


def declare(name, program):
    def summary():
        (_, _, var_vals), schedules = scheduled(program)
        return sorted(var_vals.items()), [arguments for _, _, arguments in schedules]
    var_vals, arguments = forked(summary)
    def big(): return scheduled(program)[0][0]
    def linear(): return scheduled(program)[0][1]
    big.__name__, linear.__name__ = name, f"{name}_linear"
    graph(big)
    graph(linear)
    graphs = []
    for n, args in enumerate(arguments):
        suffix = "" if n == 0 else f"_{n}"
        def kernels(n=n): return scheduled(program)[1][n][0]
        def schedule(n=n): return scheduled(program)[1][n][1]
        kernels.__name__, schedule.__name__ = f"{name}_kernels{suffix}", f"{name}_schedule{suffix}"
        graph(kernels)
        graph(schedule)
        graphs.append((kernels.__name__, words(args)))
    return (name, words(var_vals)), graphs


DECLARED = [declare(name, program) for name, program in PROGRAMS.items()]


@table
def var_vals():
    return ["program", "var_vals"], [row for row, _ in DECLARED]


@table
def arguments():
    return ["graph", "arguments"], [row for _, graphs in DECLARED for row in graphs]
