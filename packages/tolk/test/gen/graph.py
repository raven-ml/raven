"""Write a UOp graph in the graph format, which test/README.md specifies.

A graph is its nodes in topological order, one line each:

    <index> <op> <dtype> [<source indices>] [<arg>]

`write(sink)` is the text of the graph under `sink`, `boundary(*tensors)`
is the sink that a tinygrad `Tensor` program hands to the compiler, and
`stage(name, kernel, renderer)` is the sink that compiling a kernel hands to
one pass of the codegen pipeline.
"""

import dataclasses
import enum
import math
import struct

from tinygrad.dtype import DType, InvalidType
from tinygrad.uop.ops import KernelInfo, Ops, UOp


# Values

def string(s, prefix=""):
    """A string or bytes, quoted, with every byte outside printable ASCII escaped."""
    data = s.encode() if isinstance(s, str) else s
    out = []
    for b in data:
        c = chr(b)
        if c in '"\\': out.append("\\" + c)
        elif c == "\n": out.append("\\n")
        elif c == "\t": out.append("\\t")
        elif 0x20 <= b < 0x7f: out.append(c)
        else: out.append(f"\\x{b:02x}")
    return prefix + '"' + "".join(out) + '"'


# A record's fields that are runtime state, which the format leaves out.
RUNTIME = {("ParamArg", "buffer")}


def nan(x):
    """A NaN: `nan` for the positive quiet NaN, else its bits, which a bitcast reveals."""
    bits = struct.unpack("<Q", struct.pack("<d", x))[0]
    return "nan" if bits == 0x7FF8_0000_0000_0000 else f"nan(0x{bits:016x})"


def fields(x):
    """The fields of the record `x` that differ from their default, in order."""
    for f in dataclasses.fields(x):
        if (type(x).__name__, f.name) in RUNTIME: continue
        v = getattr(x, f.name)
        if f.default is not dataclasses.MISSING and v == f.default and type(v) is type(f.default): continue
        if f.default_factory is not dataclasses.MISSING and v == f.default_factory(): continue
        yield f.name, v


def value(x, index):
    """The text of the value `x`, where `index` numbers the nodes."""
    if isinstance(x, UOp): return f"%{index[x]}"
    if x is None or isinstance(x, bool): return repr(x)
    if isinstance(x, InvalidType): return "Invalid"
    if isinstance(x, enum.Enum): return f"{type(x).__name__}.{x.name}"
    if isinstance(x, int): return str(x)
    if isinstance(x, float): return nan(x) if math.isnan(x) else float.__repr__(x)
    if isinstance(x, str): return string(x)
    if isinstance(x, bytes): return string(x, prefix="b")
    if isinstance(x, DType): return repr(x)
    if isinstance(x, tuple):
        items = [value(v, index) for v in x]
        return "(" + items[0] + ",)" if len(items) == 1 else "(" + ", ".join(items) + ")"
    if dataclasses.is_dataclass(x):
        if getattr(x, "grad_fxn", None) is not None: raise ValueError(f"{x!r} holds a function")
        return type(x).__name__ + "(" + ", ".join(f"{k}={value(v, index)}" for k, v in fields(x)) + ")"
    raise TypeError(f"no encoding for {type(x).__name__} value {x!r}")


def uops(x):
    """The UOps inside the value `x`, in the order `value` writes them."""
    if isinstance(x, UOp): yield x
    elif isinstance(x, tuple):
        for v in x: yield from uops(v)
    elif dataclasses.is_dataclass(x) and not isinstance(x, type):
        for _, v in fields(x): yield from uops(v)


# Graphs

def toposort(sink):
    """The nodes under `sink`, each after its sources and the UOps of its arg.

    Sources come first, in order, so that a graph whose args hold no UOp is in
    `sink.toposort()` order."""
    order, stack = {}, [(sink, False)]
    while stack:
        node, visited = stack.pop()
        if node in order: continue
        if visited:
            order[node] = len(order)
            continue
        stack.append((node, True))
        stack.extend((s, False) for s in reversed([*node.src, *uops(node.arg)]))
    return order


def value_map(x, fn):
    """`x` with `fn` applied to each UOp inside it."""
    if isinstance(x, UOp): return fn(x)
    if isinstance(x, tuple): return tuple(value_map(v, fn) for v in x)
    if dataclasses.is_dataclass(x) and not isinstance(x, type) and any(True for _ in uops(x)):
        return dataclasses.replace(x, **{k: value_map(v, fn) for k, v in fields(x)})
    return x


def rewrite(sink, fn):
    """`sink` with each node `u` replaced by `fn(u, src, arg)`, its sources and the
    UOps of its arg rebuilt first."""
    memo = {}
    for u in toposort(sink):
        memo[u] = fn(u, tuple(memo[s] for s in u.src), value_map(u.arg, memo.__getitem__))
    return memo[sink]


def unplaced_kernels(sink):
    """`sink` with each parameter of a kernel body naming no device.

    In a tensor graph a `PARAM` names the device of its storage. A kernel body's
    `PARAM`s are slots its call binds: the call's arguments name the devices, so
    a body's parameters name none, wherever the body sits. A `PARAM` a body
    shares with its call's argument is therefore split, by position: the
    argument keeps its device. A body is what is under a `SINK` holding a
    `KernelInfo`; a copy's `STORE` as a call's body, which moves data across
    devices or to a disk; and what such a `SINK` became, a `PROGRAM` or a
    `LINEAR` holding it. A function's body, a `SINK` or `LINEAR` with no
    `KernelInfo`, is not one: its parameters keep their devices, as does a
    `STORE` body over one device, a generator's stand-in for a copy."""
    def kernel(u):
        return u.op is Ops.SINK and isinstance(u.arg, KernelInfo)

    def copy_body(u):
        devices = {p.arg.device for p in toposort(u) if p.op is Ops.PARAM and p.arg.device is not None}
        return u.op is Ops.STORE and (len(devices) > 1 or any(str(d).startswith("DISK") for d in devices))

    def erase(u, src, arg):
        if u.op in (Ops.PARAM, Ops.REFPARAM) and arg.device is not None:
            arg = dataclasses.replace(arg, device=None)
        return u.replace(src=src, arg=arg)

    def enter(u, src, arg):
        if u.op is Ops.PROGRAM or kernel(u) or (u.op is Ops.LINEAR and any(map(kernel, u.src))):
            return rewrite(u, erase)
        if u.op is Ops.CALL and copy_body(u.src[0]):
            src = (rewrite(u.src[0], erase),) + src[1:]
        return u.replace(src=src, arg=arg)

    return rewrite(sink, enter)


def write(sink):
    """The graph under `sink` in the graph format, with its kernels' parameters
    unplaced (`unplaced_kernels`)."""
    index = toposort(unplaced_kernels(sink))
    lines = []
    for u, i in index.items():
        line = f"{i} {u.op} {u.dtype} [{', '.join(str(index[s]) for s in u.src)}]"
        if u.arg is not None: line += " " + value(u.arg, index)
        lines.append(line)
    return "\n".join(lines) + "\n"


def boundary(*tensors):
    """The sink that realizing `tensors` hands to the compiler: the argument of
    `create_linear_with_vars`, after the `Tensor` layer has placed each result in
    storage."""
    import tinygrad.tensor
    from tinygrad import Tensor

    class Captured(Exception):
        pass

    def capture(sink):
        raise Captured(sink)

    schedule, tinygrad.tensor.create_linear_with_vars = tinygrad.tensor.create_linear_with_vars, capture
    try:
        Tensor.schedule_linear(*tensors)
    except Captured as e:
        return e.args[0]
    finally:
        tinygrad.tensor.create_linear_with_vars = schedule
    raise RuntimeError("realizing the tensors scheduled nothing")


def kernels(*tensors):
    """The kernels that realizing `tensors` compiles, in order."""
    from tinygrad import Tensor
    from tinygrad.uop.ops import Ops

    return [call.src[0] for call in Tensor.schedule_linear(*tensors).src if call.src[0].op is Ops.SINK]


def stage(name, kernel, renderer):
    """The sink that compiling `kernel` for `renderer` hands to the pass `name`
    of the codegen pipeline: a `graph_rewrite` by its name, "apply_opts", or
    "linearize"."""
    import tinygrad.codegen
    from tinygrad.uop.ops import KernelInfo

    if kernel.arg is None: kernel = kernel.replace(arg=KernelInfo())
    full = lambda: tinygrad.codegen.full_rewrite_to_sink(kernel, renderer, optimize=kernel.arg is None or not kernel.arg.optimized)
    if name == "linearize": return full()

    class Captured(Exception):
        pass

    def capture(sink, *args, name=None, **kwargs):
        if name == wanted: raise Captured(sink)
        return rewrite(sink, *args, name=name, **kwargs)

    def capture_opts(sink, *args, **kwargs):
        raise Captured(sink)

    wanted, rewrite, opts = name, tinygrad.codegen.graph_rewrite, tinygrad.codegen.apply_opts
    if name == "apply_opts": tinygrad.codegen.apply_opts = capture_opts
    else: tinygrad.codegen.graph_rewrite = capture
    try:
        full()
    except Captured as e:
        return e.args[0]
    finally:
        tinygrad.codegen.graph_rewrite, tinygrad.codegen.apply_opts = rewrite, opts
    raise RuntimeError(f"compiling the kernel runs no pass {name!r}")
