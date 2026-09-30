"""Write a UOp graph in the graph format, which test/README.md specifies.

A graph is its nodes in topological order, one line each:

    <index> <op> <dtype> [<source indices>] [<arg>] [tag=<tag>]

`write(sink)` is the text of the graph under `sink`, and `boundary(*tensors)`
is the sink that a tinygrad `Tensor` program hands to the compiler.
"""

import dataclasses
import enum

from tinygrad.dtype import DType, InvalidType
from tinygrad.uop.ops import UOp


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
    if isinstance(x, float): return float.__repr__(x)
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


def write(sink):
    """The graph under `sink` in the graph format."""
    index = toposort(sink)
    lines = []
    for u, i in index.items():
        line = f"{i} {u.op} {u.dtype} [{', '.join(str(index[s]) for s in u.src)}]"
        if u.arg is not None: line += " " + value(u.arg, index)
        if u.tag is not None: line += " tag=" + value(u.tag, index)
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
