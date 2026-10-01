"""Goldens of tinygrad/engine/realize.py: what the calls of a schedule do, and
the compilation of its kernels.

Each program is a `Tensor` program realized on the CPU: `<program>.golden` is
its schedule, as `create_linear_with_vars` returns it, and
`<program>_compiled.golden` that schedule after `lower_and_compile`, each
kernel compiled for Clang on x86_64. No case compiles: a program's binary is
its source's bytes.

`calls.golden` reads each call of each compiled schedule, by its position in
it, with the schedule's variables: its buffer arguments (`get_call_arg_uops`) and the values of its
program's variables (`get_call_var_uops`), both as the positions of the
call's sources or as values; the buffers it writes and reads
(`get_call_outs_ins`); the buffers it writes and does not read
(`get_call_written_bufs`), as positions among its buffer arguments of the
buffers they view; its name (`get_call_name`, in colour, with the schedule's
variables); and its cost (`estimate_uop`), with the schedule's variables.
"""

from golden import graph, table
from tinygrad import Tensor, Variable, dtypes
from tinygrad.device import Compiler
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import DEV, Context
from tinygrad.uop.ops import KernelInfo, Ops, UOp, sym_infer
import tinygrad.engine.realize as realize
import tinygrad.tensor

DEV.value = "CPU::x86_64,x86-64"
Compiler.compile_cached = lambda self, src: src.encode()
# Workers compile in processes of their own, with tinygrad's compilers.
realize.get_worker_pool = lambda: None


def scheduled(program):
    """The (linear, var_vals) that realizing the tensors of `program()` makes
    last."""
    linears, create = [], tinygrad.tensor.create_linear_with_vars

    def capture(sink):
        linears.append(out := create(sink))
        return out

    tinygrad.tensor.create_linear_with_vars = capture
    try:
        first, *rest = tensors if isinstance(tensors := program(), tuple) else (tensors,)
        first.linear_with_vars(*rest)
    finally:
        tinygrad.tensor.create_linear_with_vars = create
    return linears[-1]


def empty(*shape, dtype=dtypes.float, device=None): return Tensor.empty(*shape, dtype=dtype, device=device)


def sharded(*shape, axis, devices=("CPU:0", "CPU:1")):
    shard = tuple(n // len(devices) if a == axis else n for a, n in enumerate(shape))
    return Tensor(empty(*shard, device=devices).uop.unshard(axis), device=devices)


def add_one(C, A):
    C, A = C.flatten(), A.flatten()
    i = UOp.range(A.numel(), 0)
    return C[i].store(A[i] + 1).end(i).sink(arg=KernelInfo(name=f"add_one_{A.numel()}"))


def precompiled_scalar():
    x = empty(3, dtype=dtypes.int32)
    scalar = UOp.param(1, x.dtype, addrspace=AddrSpace.ALU)
    bound = Variable("v", 1, 8, dtype=x.dtype).bind(3)
    out, = UOp.call_with_outputs((x.uop.param_like(0) * scalar,), x.uop, bound, precompile=True)
    return Tensor(out)


PROGRAMS = {
    "add": lambda: empty(4, 4) + empty(4, 4),
    "sum": lambda: empty(4, 8).sum(1),
    "two_outputs": lambda: (lambda x: (x + 1, x * 2))(empty(4, 4)),
    "assign": lambda: (lambda a: a.assign(a + 1))(empty(4, 4).contiguous()),
    "custom_kernel": lambda: Tensor.custom_kernel(empty(4, 4), empty(4, 4) * 2, fxn=add_one)[0] + 1,
    "precompiled_scalar": precompiled_scalar,
    "copy": lambda: empty(4, 4).to("CPU:1") * 2,
    "copy_view": lambda: empty(8, 8)[2:6].to("CPU:1") + 1,
    "copy_big": lambda: empty(1 << 20, dtype=dtypes.uint8).to("CPU:1") + 1,
    "copy_variable": lambda: empty(10)[:Variable("v", 1, 10).bind(3)].to("CPU:1") + 1,
    "shard_add": lambda: sharded(4, 8, axis=0) + 1,
    "shard_to_one": lambda: sharded(4, 8, axis=0).to("CPU:0") + 1,
    "variable_shrink": lambda: empty(10)[:Variable("v", 1, 10).bind(3)] * 2,
    "variable_two": lambda: (empty(10)[:Variable("v", 1, 10).bind(4)] * 2,
                             empty(10)[:Variable("w", 1, 10).bind(7)] + 1),
}


def compiled(name): return realize.lower_and_compile(scheduled(PROGRAMS[name])[0])


def position(call, u):
    """`u` as the position of a source of `call`, or as its value."""
    return f"%{call.src.index(u)}" if u in call.src else str(u.arg)


def sym(x, var_vals): return str(sym_infer(x, var_vals))


def call_row(program, i, call, var_vals):
    bufs = realize.get_call_arg_uops(call)
    ast = call.body
    variables = realize.get_call_var_uops(call, ast) if ast.op is Ops.PROGRAM else []
    outs, ins = realize.get_call_outs_ins(call)
    storage = [b.storage_base for b in bufs]
    storage = [s.src[0].storage_base if s.op is Ops.MSELECT else s for s in storage]
    written = [storage.index(b) for b in realize.get_call_written_bufs(call)]
    with Context(NO_COLOR=0): name = realize.get_call_name(call, bufs, var_vals)
    e = realize.estimate_uop(call)
    return (program, str(i), " ".join(f"{k}={v}" for k, v in sorted(var_vals.items())) or "-", " ".join(position(call, b) for b in bufs) or "-",
            " ".join(v.arg.name if v.op is Ops.PARAM else str(v.arg) for v in variables) or "-",
            " ".join(map(str, outs)) or "-", " ".join(map(str, ins)) or "-", " ".join(map(str, written)) or "-",
            name, sym(e.ops, var_vals), sym(e.lds, var_vals), sym(e.mem, var_vals))


def declare(name):
    def schedule(): return scheduled(PROGRAMS[name])[0]
    def compiled_schedule(): return compiled(name)
    schedule.__name__, compiled_schedule.__name__ = name, f"{name}_compiled"
    graph(schedule)
    graph(compiled_schedule)


for name in PROGRAMS: declare(name)


@table
def calls():
    rows = []
    for program in PROGRAMS:
        linear, var_vals = scheduled(PROGRAMS[program])
        for i, call in enumerate(realize.lower_and_compile(linear).src):
            rows.append(call_row(program, i, call, var_vals))
    return ["program", "call", "var_vals", "bufs", "vars", "outs", "ins", "written", "name", "ops", "lds", "mem"], rows
