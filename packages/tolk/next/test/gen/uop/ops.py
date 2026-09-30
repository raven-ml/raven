"""Goldens of tinygrad/uop/ops.py: arithmetic on constants, bounds, data types,
argument reprs and the graphs the constructors build.

A cell that holds a value is its repr: `True`, `-128`, `1.5`, `inf`, `nan`,
`Invalid`, and a list of values is written `[1, 2.5, True]`. A call that raises
is `raises` and the exception's class name. A graph golden is a sink in the
graph format of test/README.md.
"""

import itertools
import math

from golden import graph, table, text
from graph import toposort
from tinygrad.dtype import AddrSpace, Invalid, dtypes
from tinygrad.helpers import Target
from tinygrad.uop import Ops
from tinygrad.uop.ops import (AxisType, BottomUpGate, CallInfo, KernelInfo, ParamArg, ProgramInfo, UOp, axis_colors,
                              axis_letters, axis_to_pos, dtype_from_uop, exec_alu, identity_element)
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.renderer import Estimates
from tinygrad.schedule.indexing import BufferizeOpts


def attempt(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        return f"raises {type(e).__name__}"


def value(x):
    """The cell of a value: floats as their repr, whatever their class."""
    if isinstance(x, str): return x
    if isinstance(x, bool) or x is Invalid: return repr(x)
    if isinstance(x, float): return float.__repr__(x)
    return repr(x)


def values(xs): return "[" + ", ".join(value(x) for x in xs) + "]"


# exec_alu

INT_DTYPES = [dtypes.int8, dtypes.uint8, dtypes.int16, dtypes.uint16, dtypes.int32, dtypes.uint32, dtypes.int64,
              dtypes.uint64, dtypes.weakint]
FLOAT_DTYPES = [dtypes.half, dtypes.bfloat16, dtypes.float, dtypes.double, dtypes.weakfloat]

INT_PAIRS = [(0, 0), (7, 3), (7, -3), (-7, 3), (-7, -3), (-50, 6), (8, 2), (1, 0), (0, 1), (-1, 1), (5, 5),
             (250, 250), (256, 0), (0, -1), (0, -1000), (-100, -100), (-130, 0), (127, 1), (-128, -1),
             (2**31 - 1, 1), (-2**31, -1), (2**63 - 1, 1), (-2**63, -1), (2**64 - 1, 3), (2**100, 2**90), (3, 63)]
FLOAT_PAIRS = [(0.0, 0.0), (1.0, -0.0), (-0.0, 0.0), (1.5, 2.5), (-2.5, 0.5), (7.0, 3.0), (2.0, 0.5), (1e300, 1e300),
               (-1e300, 1e300), (65504.0, 32.0), (math.inf, 1.0), (1.0, math.inf), (-math.inf, math.inf),
               (math.nan, 1.0), (1.0, math.nan), (0.1, 0.2), (3.4028234663852886e38, 2.0)]
BOOL_PAIRS = [(False, False), (False, True), (True, False), (True, True)]
INT_UNARY = [0, 1, -1, 3, -7, 8, 127, -128, 255, 2**31 - 1, 2**63, -2**70]
FLOAT_UNARY = [0.0, -0.0, 0.5, 1.0, -1.0, 2.5, -2.5, 3.0, 4.0, 1e-45, 1e300, 65504.0, 65520.0, 3.4028235677973366e38,
               math.inf, -math.inf, math.nan]

INT_BINARY = [Ops.ADD, Ops.SUB, Ops.MUL, Ops.CDIV, Ops.CMOD, Ops.FLOORDIV, Ops.FLOORMOD, Ops.MAX, Ops.AND, Ops.OR,
              Ops.XOR, Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ]
FLOAT_BINARY = [Ops.ADD, Ops.SUB, Ops.MUL, Ops.FDIV, Ops.MAX, Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ, Ops.POW]
BOOL_BINARY = [Ops.ADD, Ops.MUL, Ops.AND, Ops.OR, Ops.XOR, Ops.MAX, Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ]
FLOAT_UNARY_OPS = [Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.SQRT, Ops.RECIPROCAL, Ops.TRUNC, Ops.NEG]
COMPARISONS = {Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ}


def pow_overflows(x, y):
    """Whether CPython raises OverflowError on x ** y, which tolk.next answers with inf (IEEE)."""
    try:
        pow(x, y)
        return False
    except OverflowError:
        return True
    except (ZeroDivisionError, ValueError):
        return False


def alu_rows():
    rows = []
    def row(op, dt, operands, truncate=True):
        rows.append((str(op), repr(dt), values(operands), repr(truncate),
                     value(attempt(exec_alu, op, dt, list(operands), truncate_output=truncate))))
    for dt in INT_DTYPES:
        for op in INT_BINARY:
            for pair in INT_PAIRS: row(op, dtypes.bool if op in COMPARISONS else dt, pair)
        for op in (Ops.SHL, Ops.SHR):
            for pair in [(1, 0), (1, 7), (1, 31), (-8, 1), (255, 4), (2**64 - 1, 63), (-1, 63), (5, 64)]: row(op, dt, pair)
        for x in INT_UNARY: row(Ops.NEG, dt, (x,))
        row(Ops.MULACC, dt, (3, 4, 5))
        row(Ops.MULACC, dt, (2**40, 2**40, -1))
        row(Ops.WHERE, dt, (True, 2, 4))
        row(Ops.WHERE, dt, (False, 2, 4))
    for dt in FLOAT_DTYPES:
        for op in FLOAT_BINARY:
            for pair in FLOAT_PAIRS:
                if op is Ops.POW and pow_overflows(*pair): continue
                row(op, dtypes.bool if op in COMPARISONS else dt, pair)
        for op in FLOAT_UNARY_OPS:
            for x in FLOAT_UNARY: row(op, dt, (x,))
        for op in (Ops.CDIV, Ops.CMOD, Ops.FLOORDIV, Ops.FLOORMOD):
            for pair in [(7.5, 2.0), (-7.5, 2.0), (7.5, -2.0), (1.0, 0.0), (-0.5, 3.0)]:
                # CPython's cdiv and floordiv by zero give the int 0, which only a
                # float dtype's truncation turns back into a float: the suite
                # states the weak float's 0.0 itself
                if dt == dtypes.weakfloat and op in (Ops.CDIV, Ops.FLOORDIV) and pair[1] == 0: continue
                row(op, dt, pair)
        row(Ops.MULACC, dt, (1.5, 2.0, 0.25))
        row(Ops.WHERE, dt, (True, 2.2, 4.5))
        row(Ops.WHERE, dt, (False, 2.2, 4.5))
    for op in BOOL_BINARY:
        for pair in BOOL_PAIRS: row(op, dtypes.bool, pair)
    row(Ops.NEG, dtypes.bool, (True,))
    for cond in (False, True): row(Ops.WHERE, dtypes.bool, (cond, False, True))
    # a float operation on integer operands, as a weak integer constant folds
    for op in FLOAT_UNARY_OPS:
        for x in (0, 1, 4, 8, -3, 7): row(op, dtypes.float, (x,))
    # integer powers stay integers, and a negative exponent gives a float
    for pair in [(2, 3), (3, 0), (2, -1), (-2, -2), (0, 5)]: row(Ops.POW, dtypes.weakint, pair)
    # a rounding division with a float operand divides as floats
    for op in (Ops.CDIV, Ops.FLOORDIV, Ops.CMOD, Ops.FLOORMOD):
        for pair in [(7, 2.0), (-7.0, 2), (7, -2.5)]: row(op, dtypes.weakfloat, pair)
    for op, pair in [(Ops.POW, (2, 3)), (Ops.POW, (0, 0)), (Ops.POW, (2.0, -1.0)), (Ops.POW, (0.0, -1.0)),
                     (Ops.POW, (-0.0, -1.0)), (Ops.POW, (-8.0, 1 / 3)), (Ops.POW, (-2.0, 3.0)), (Ops.POW, (-2.0, 0.5)),
                     (Ops.POW, (-math.inf, 0.5)), (Ops.POW, (-math.inf, 3.0)), (Ops.POW, (-math.inf, -0.5))]:
        row(op, dtypes.double, pair)
    # comparisons between an integer and a float are exact
    for op in COMPARISONS:
        for pair in [(2**53 + 1, 9007199254740992.0), (9007199254740992.0, 2**53 + 1), (1, 1.0), (True, 1), (0, -0.0)]:
            row(op, dtypes.bool, pair)
    # Invalid poisons every binary operation, whatever the result type
    for op, dt in [(Ops.ADD, dtypes.weakint), (Ops.MUL, dtypes.int32), (Ops.CMPLT, dtypes.bool),
                   (Ops.CMPNE, dtypes.bool), (Ops.MAX, dtypes.float)]:
        row(op, dt, (Invalid, 1))
        row(op, dt, (1, Invalid))
    # a selection of Invalid is Invalid
    row(Ops.WHERE, dtypes.weakint, (True, Invalid, 1))
    row(Ops.WHERE, dtypes.weakint, (False, Invalid, 1))
    # operations exec_alu has no rule for, and operands that do not fit
    for op, dt, operands in [(Ops.THREEFRY, dtypes.uint64, (1, 2)),
                             (Ops.SQRT, dtypes.float, (Invalid,)), (Ops.ADD, dtypes.int32, (1,)),
                             (Ops.WHERE, dtypes.int32, (True, 1))]:
        row(op, dt, operands)
    # without truncation, the exact value
    for op, dt, pair in [(Ops.ADD, dtypes.uint8, (250, 250)), (Ops.MUL, dtypes.int32, (2**31 - 1, 2)),
                         (Ops.NEG, dtypes.uint32, (1,)), (Ops.ADD, dtypes.float, (0.1, 0.2)),
                         (Ops.RECIPROCAL, dtypes.half, (3.0,))]:
        row(op, dt, pair, truncate=False)
    return rows


@table
def exec_alu_values():
    # a comparison's rows repeat across the dtypes of its operands
    return ["op", "dtype", "operands", "truncate", "result"], list(dict.fromkeys(alu_rows()))


# Bounds

RANGES = [(0, 0), (3, 3), (-3, -3), (0, 10), (10, 20), (-20, -10), (-7, 7), (1, 10), (3, 5), (-5, -3)]
BOUND_OPS = [Ops.ADD, Ops.SUB, Ops.MUL, Ops.CDIV, Ops.CMOD, Ops.FLOORDIV, Ops.FLOORMOD, Ops.MAX, Ops.AND, Ops.XOR,
             Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ]


def bounds(u):
    return value(u.vmin), value(u.vmax)


@table
def binary_bounds():
    rows = []
    def row(op, dt, a, b):
        x, y = UOp.variable("a", a[0], a[1], dt), UOp.variable("b", b[0], b[1], dt)
        rows.append((str(op), repr(dt), value(a[0]), value(a[1]), value(b[0]), value(b[1]), *bounds(x.alu(op, y))))
    for dt in (dtypes.weakint, dtypes.int32):
        for op in BOUND_OPS:
            for a in RANGES:
                for b in RANGES: row(op, dt, a, b)
        for op in (Ops.SHL, Ops.SHR):
            for a in [(0, 10), (-8, 8), (3, 3)]:
                for b in [(0, 0), (2, 2), (5, 5), (1, 3)]: row(op, dt, a, b)
        row(Ops.XOR, dt, (3, 7), (-1, -1))
        row(Ops.XOR, dt, (-10, -3), (-1, -1))
        row(Ops.XOR, dt, (0, 10), (-1, 3))
        row(Ops.AND, dt, (10, 20), (48, 48))
        row(Ops.AND, dt, (-100, 100), (511, 511))
        row(Ops.AND, dt, (-100, 100), (-1, -1))
    # a fixed width does not clamp: the interval is the exact one
    for dt in (dtypes.int8, dtypes.uint8, dtypes.uint32):
        for op in (Ops.ADD, Ops.SUB, Ops.MUL):
            for a, b in [((100, 127), (1, 100)), ((0, 255), (1, 1)), ((2, 7), (3, 3))]: row(op, dt, a, b)
    for op in (Ops.OR, Ops.AND, Ops.CMPNE, Ops.CMPLT, Ops.XOR):
        for a in [(False, False), (False, True), (True, True)]:
            for b in [(False, False), (False, True), (True, True)]: row(op, dtypes.bool, a, b)
    for op in (Ops.ADD, Ops.MUL, Ops.MAX, Ops.CMPLT):
        row(op, dtypes.float, (0.0, 1.0), (2.0, 3.0))
    return ["op", "dtype", "a_lo", "a_hi", "b_lo", "b_hi", "vmin", "vmax"], rows


# Loads from a constant table read its bytes as the loaded type
LOAD_TABLES = [(dtypes.bool, b"\x00\x01\x00"), (dtypes.int8, b"\x80\x7f\x05"), (dtypes.uint8, b"\x03\x09\x01"),
               (dtypes.int16, b"\x00\x80\xff\x7f"), (dtypes.uint16, b"\x00\x80\x01\x00"),
               (dtypes.int32, b"\xff\xff\xff\xff\x05\x00\x00\x00"), (dtypes.uint32, b"\xff\xff\xff\xff\x05\x00\x00\x00"),
               (dtypes.int64, b"\xff" * 8 + b"\x07" + b"\x00" * 7), (dtypes.uint64, b"\xff" * 8 + b"\x07" + b"\x00" * 7),
               (dtypes.half, b"\x00\x3c\x00\xc0"), (dtypes.float, b"\x00\x00\xc0\x3f\x00\x00\x20\xc1"),
               (dtypes.double, b"\x00" * 6 + b"\xf8\x3f" + b"\x00" * 7 + b"\xc0")]


@table
def load_bounds():
    rows = []
    for dt, data in LOAD_TABLES:
        table = UOp(Ops.BINARY, arg=data)
        table = table if dt == dtypes.uint8 else table.bitcast(dt)
        u = table.index(UOp.range(len(data) // dt.itemsize, 0)).load()
        rows.append((repr(dt), data.hex(), *bounds(u)))
    return ["dtype", "bytes", "vmin", "vmax"], rows


CAST_SOURCES = [
    (dtypes.weakint, [(5, 10), (-1, 10), (250, 260), (-300, 300), (0, 255), (-255, 0), (200, 300), (0, 16777219)]),
    (dtypes.int32, [(-10, 10), (5, 7), (0, 16777219)]),
    (dtypes.uint8, [(0, 255), (1, 10)]),
    (dtypes.float, [(-4.5, 4.5), (2e9, 3e9), (3e9, 4e9), (-4e9, -3e9), (0.25, 0.75), (-math.inf, math.inf)]),
    (dtypes.bool, [(False, True), (True, True)]),
]
CAST_TARGETS = [dtypes.bool, dtypes.int8, dtypes.uint8, dtypes.int16, dtypes.int32, dtypes.uint32, dtypes.int64,
                dtypes.uint64, dtypes.half, dtypes.bfloat16, dtypes.float, dtypes.double, dtypes.weakint, dtypes.weakfloat]


@table
def cast_bounds():
    rows = []
    for src, ranges in CAST_SOURCES:
        for lo, hi in ranges:
            for dst in CAST_TARGETS:
                if dst == src: continue
                u = UOp.variable("x", lo, hi, src).cast(dst)
                rows.append((repr(src), value(lo), value(hi), repr(dst), *bounds(u)))
    return ["from", "lo", "hi", "to", "vmin", "vmax"], rows


# Data types

DTYPE_OF_DTYPES = [dtypes.bool, dtypes.weakint, dtypes.int8, dtypes.uint8, dtypes.int32, dtypes.uint32, dtypes.int64,
                   dtypes.weakfloat, dtypes.half, dtypes.bfloat16, dtypes.float, dtypes.double]


def operand(dt): return UOp.variable("x", False, True, dt) if dt == dtypes.bool else UOp.variable("x", 0, 1, dt)


@table
def dtypes_of():
    rows = []
    for op in (Ops.ADD, Ops.MAX, Ops.CMPLT, Ops.FDIV, Ops.SHL, Ops.THREEFRY):
        for a in DTYPE_OF_DTYPES:
            for b in DTYPE_OF_DTYPES:
                rows.append((str(op), repr(a), repr(b), value(attempt(lambda: repr(dtype_from_uop(op, (operand(a), operand(b)), None))))))
    for op in (Ops.SQRT, Ops.EXP2, Ops.RECIPROCAL, Ops.NEG, Ops.TRUNC):
        for a in DTYPE_OF_DTYPES:
            rows.append((str(op), repr(a), "-", value(attempt(lambda: repr(dtype_from_uop(op, (operand(a),), None))))))
    # a selection promotes its branches, and needs a boolean condition
    for c in (dtypes.bool, dtypes.weakint):
        for b in DTYPE_OF_DTYPES:
            rows.append(("Ops.WHERE", repr(c), repr(b),
                         value(attempt(lambda: repr(dtype_from_uop(Ops.WHERE, (operand(c), operand(b), operand(dtypes.float)), None))))))
    return ["op", "a", "b", "dtype"], rows


@table
def identities():
    rows = []
    for op in (Ops.ADD, Ops.MUL, Ops.MAX):
        for dt in [dtypes.bool, *INT_DTYPES, *FLOAT_DTYPES]: rows.append((str(op), repr(dt), value(identity_element(op, dt))))
    return ["op", "dtype", "identity"], rows


@table
def axis_types():
    return ["axis_type", "letter", "color", "position"], [
        (repr(a), axis_letters.get(a, "raises KeyError"), axis_colors.get(a, "raises KeyError"),
         value(axis_to_pos.get(a, "raises KeyError"))) for a in AxisType]


# Argument reprs: the text of each argument, by the name the suite builds it under

def reprs_of():
    n = UOp.variable("n", 1, 10)
    return {
        "none": UOp(Ops.NOOP),
        "int": UOp.const(42),
        "negative int": UOp.const(-3),
        "huge int": UOp.const(2**100),
        "float": UOp.const(1.5),
        "negative zero": UOp.const(-0.0),
        "nan": UOp.const(math.nan),
        "inf": UOp.const(math.inf),
        "bool": UOp.const(True),
        "invalid": UOp.invalid(),
        "dtype": UOp.const(1).cast(dtypes.half),
        "range": UOp.range(4, 0, AxisType.REDUCE),
        "range with sub-axes": UOp(Ops.RANGE, src=(UOp.const(4),), arg=(1, 0, AxisType.UPCAST)),
        "negative range": UOp.range(4, -1, AxisType.DEVICE),
        "reduce": UOp(Ops.REDUCE, src=(UOp.const(1.0).expand((4,)),), arg=(Ops.ADD, 1)),
        "special": UOp.special(8, "gidx0"),
        "permute": UOp.param(0, dtypes.float, (2, 3)).permute((1, 0)),
        "flip": UOp.param(0, dtypes.float, (2, 3)).flip(1),
        "device": UOp.param(0, dtypes.float, 4, device="CPU").copy_to_device("CUDA"),
        "devices": UOp.param(0, dtypes.float, 4, device="CPU").copy_to_device(("CPU:0", "CPU:1")),
        "mselect": UOp.param(0, dtypes.float, 4, device=("CPU:0", "CPU:1")).mselect(1),
        "unshard": UOp.param(0, dtypes.float, 4, device=("CPU:0", "CPU:1")).unshard(0),
        "allreduce": UOp.param(0, dtypes.float, 4, device=("CPU:0", "CPU:1")).allreduce(Ops.ADD, ("CPU:0", "CPU:1")),
        "custom": UOp(Ops.CUSTOM, src=(), arg=("barrier();", dtypes.void)),
        "ins": UOp(Ops.INS, src=(), arg=("s_endpgm", dtypes.void)),
        "binary": UOp(Ops.BINARY, arg=b"\x7fELF\x00\n'\""),
        "source": UOp(Ops.SOURCE, arg="int x = 'a';\n"),
        "param": UOp.param(3, dtypes.float, 256),
        "scalar param": UOp.param(1, dtypes.int),
        "variable": n,
        "bound variable": n.bind(3),
        "named local buffer": UOp.placeholder((4,), dtypes.int, 2, addrspace=AddrSpace.LOCAL, tag="buf"),
        "volatile param": UOp.param(0, dtypes.uint32, 4, volatile=True, device="CPU"),
        "alloc": UOp.alloc((4,), dtypes.float, slot=7, device="CPU"),
        "kernel": UOp.sink(arg=KernelInfo()),
        "kernel with opts": UOp.sink(arg=KernelInfo(
            name="kern", applied_opts=(Opt(OptOps.SPLIT, 0, (4, AxisType.UPCAST)), Opt(OptOps.SPLIT, 1, (16, AxisType.LOCAL, True)),
                                       Opt(OptOps.TC, 0, (-1, 2, 1)), Opt(OptOps.PADTO, 1, 32), Opt(OptOps.SWAP, 0, 1)),
            opts_to_apply=(Opt(OptOps.PADTO, 0, 4),), estimates=Estimates(1, 2, 3), beam=2)),
        "bufferize": UOp.const(1.0).bufferize(arg=BufferizeOpts(device="CPU", addrspace=AddrSpace.LOCAL, removable=False)),
        "bufferize without device": UOp.const(1.0).bufferize(arg=BufferizeOpts(device=None)),
        "call": UOp.sink().call(name="f"),
        "call returning": UOp.custom_function("f").call(ret_dtype=dtypes.int, precompile=True),
        "program": UOp(Ops.PROGRAM, src=(UOp.sink(arg=KernelInfo()),), arg=ProgramInfo(
            global_size=(4, 1, 1), local_size=(8, 1, 1), vars=(), globals=(0, 1), outs=(0,), ins=(1,), target=Target("CPU", "CLANG"))),
        "wmma": UOp.wmma(UOp.param(0, dtypes.half, (8,)), UOp.param(1, dtypes.half, (8,)), UOp.param(2, dtypes.float, (4,)),
                         (8, 16, 16), 32, ((((0,), 2), ((1,), 2)), (((2,), 2),), (((3,), 2),))),
        "tagged": UOp.const(1).rtag(("x", 1, True, dtypes.int, ())),
        "bytes tag": UOp.const(2).rtag(b"\x00a'"),
    }


@table
def reprs():
    return ["name", "arg", "tag"], [(name, u.argstr(), repr(u.tag)) for name, u in reprs_of().items()]


# Printed nodes: the calls that build them, with shared nodes named

def printed_nodes():
    forty_two, one, n = UOp.const(42), UOp.const(1), UOp.variable("n", 1, 8)
    return [forty_two, forty_two + UOp.const(3), forty_two + forty_two, (one + one) * (one + one), forty_two.rtag("x"),
            forty_two.rtag(("y", 1)), n, UOp.sink(n + 1, n * 2, arg=KernelInfo()), UOp.range(4, 0, AxisType.REDUCE),
            UOp.const(3, dtypes.int32), UOp.const(1.5), UOp.param(0, dtypes.float, (2, 3))]


@text
def pretty():
    return "\n\n".join(repr(u) for u in printed_nodes()) + "\n"


# Graphs: what the constructors build

def var(name, lo=0, hi=10, dt=dtypes.int32): return UOp.variable(name, lo, hi, dt)
def fvar(name, dt=dtypes.float): return UOp.variable(name, -10.0, 10.0, dt)


@graph
def typed_constants():
    return UOp.sink(UOp.const(3, dtypes.int32), UOp.const(True), UOp.const(1.5), UOp.const(2, dtypes.float),
                    UOp.const(1.5, dtypes.int32), UOp.const(True, dtypes.bool), UOp.const(Invalid, dtypes.float),
                    UOp.const(300, dtypes.char), UOp.cconst(True, dtypes.bool), UOp.const((1, 2.5, True)),
                    UOp.const((1, 2), dtypes.int8))


@graph
def ccast():
    return UOp.sink(UOp.const(3).ccast(dtypes.float), var("a").ccast(dtypes.int64), UOp.const(3, dtypes.int32).ccast(dtypes.int8))


@graph
def subtraction():
    return UOp.sink(var("a") - var("b"), var("a") - 1, 1 - var("a"), fvar("x") - 1.5)


@graph
def negation():
    return UOp.sink(-var("a"), -fvar("x"), -UOp.variable("p", False, True, dtypes.bool), -UOp.const(1, dtypes.uint8))


@graph
def weak_promotion():
    return UOp.sink(var("a") + 1, var("a") + 1.5, fvar("x") + 2, fvar("x", dtypes.half) + var("a"),
                    UOp.const(1) + UOp.const(2.5), var("a", dt=dtypes.int8) + var("b", dt=dtypes.uint8), var("a", dt=dtypes.int8) + 1,
                    UOp.const(1).expand((4,)) + UOp.param(0, dtypes.float, (4,)))


@graph
def division():
    return UOp.sink(var("a") / var("b"), var("a") // var("b"), var("a") % var("b"), fvar("x") / fvar("y"),
                    fvar("x") // fvar("y"), fvar("x") % fvar("y"), var("a").div(var("b"), rounding_mode="trunc"),
                    fvar("x").div(fvar("y"), rounding_mode="trunc"), var("a").fmod(var("b")), fvar("x").fmod(fvar("y")),
                    UOp.variable("p", False, True, dtypes.bool) / var("a"), 8 // var("a"), 9 / fvar("x"))


@graph
def constant_division():
    a, x, p = var("a"), fvar("x"), UOp.variable("p", False, True, dtypes.bool)
    return UOp.sink(a / 3, a // 3, 7 // a, a % 3, a.div(3, rounding_mode="trunc"), a.fmod(3), x // 2.0, x % 2.0,
                    x.div(2.0, rounding_mode="trunc"), x.fmod(2.0), p // True, p % True, p.fmod(True), a // 2.0, a % 2.0)


@graph
def comparisons():
    a, b = var("a"), var("b")
    return UOp.sink(a < b, a > b, a <= b, a >= b, a.ne(b), a.eq(b), a < 3, 3 < a, fvar("x") < 1)


@graph
def bitwise():
    a, b = var("a"), var("b")
    u, p = var("u", dt=dtypes.uint8), UOp.variable("p", False, True, dtypes.bool)
    return UOp.sink(a & b, a | b, a ^ b, ~a, ~u, ~p, a << 2, a >> 1, 1 << a, p & True, p | False, p.logical_not(),
                    a.logical_not())


@graph
def extrema():
    return UOp.sink(var("a").maximum(var("b")), var("a").minimum(var("b")), fvar("x").minimum(fvar("y")),
                    var("u", dt=dtypes.uint8).minimum(3), fvar("x").maximum(0), var("a").minimum(1.5))


@graph
def selection():
    c = var("a") < var("b")
    return UOp.sink(c.where(var("a"), 0), c.where(1.5, fvar("x")), c.where(1, 2), c.where(fvar("x", dtypes.half), fvar("y")))


@graph
def unary():
    a, x = var("a"), fvar("x")
    return UOp.sink(a.sqrt(), x.sqrt(), a.exp2(), x.log2(), x.reciprocal(), a.reciprocal(), x.trunc(), x.floor(),
                    UOp.const(4).sqrt(), x.cast(dtypes.half).exp2())


@graph
def powers():
    return UOp.sink(fvar("x") ** 2, fvar("x") ** fvar("y"), var("a") ** 2, fvar("x").pow(0.5), UOp.const(2.0).pow(var("a")),
                    var("a") ** 0.5)


@graph
def sums_and_products():
    p, q = UOp.variable("p", False, True, dtypes.bool), UOp.variable("q", False, True, dtypes.bool)
    return UOp.sink(var("a").usum(var("b"), var("c")), var("a").uprod(var("b"), 2), p.usum(q), p.uprod(q), var("a").usum())


@graph
def casts():
    a = var("a")
    return UOp.sink(a.cast(dtypes.int32), a.cast(dtypes.float), a.cast(dtypes.float).cast(dtypes.float), a.bitcast(dtypes.uint32),
                    a.bitcast(dtypes.int32), fvar("x").bitcast(dtypes.int32), UOp.const(1).cast(dtypes.bool))


@graph
def bitcasts():
    return UOp.sink(UOp.param(0, dtypes.uint8, 4).bitcast(dtypes.uint16), UOp.param(1, dtypes.uint16, 4).bitcast(dtypes.uint8),
                    UOp.param(2, dtypes.float, 4).bitcast(dtypes.uint32), UOp.const(1).stack(UOp.const(2)))


@graph
def movement():
    p = UOp.param(0, dtypes.float, (2, 3, 4))
    return UOp.sink(p.reshape((6, 4)), p.reshape((-1, 2)), p.reshape((2, 3, 4)), p.permute((2, 0, 1)), p.permute((0, -1, 1)),
                    p.permute((0, 1, 2)), p.flip(0, 2), p.flip(-1), p.shrink(((0, 1), None, (1, 3))),
                    p.shrink((None, None, None)), p.shrink_to((1, None, 2)), p.pad(((1, 0), None, (0, 2))),
                    p.pad_to((None, 5, 6)), p.flatten(), p.flatten(1), p.flatten(0, 1), p.reshape((2, 12)).unflatten(1, (3, 4)))


@graph
def expansion():
    p = UOp.param(0, dtypes.float, (3, 1))
    s = UOp.param(1, dtypes.float, (1, 4, 1))
    return UOp.sink(p.expand((2, 3, 1)), p.expand((3, 5)), p.expand((2, 3, 5)), p.expand((-1, 5)), s.expand((2, 4, 3)),
                    UOp.const(1.0).expand((4, 8)), UOp.const(1.0).expand(()))


@graph
def squeezes():
    p = UOp.param(0, dtypes.float, (1, 3, 1, 2))
    return UOp.sink(p.squeeze(), p.squeeze(0), p.squeeze(2), p.squeeze(1), p.squeeze(-2))


@graph
def stacks():
    a, b = UOp.param(0, dtypes.float, (2, 3)), UOp.param(1, dtypes.float, (2, 3))
    h = UOp.param(2, dtypes.half, (2, 3))
    return UOp.sink(a.stack(b), a.stack(b, dim=1), a.stack(b, dim=-1), a.stack(h), a.stack(b, a), UOp.stack(UOp.invalid(), UOp.const(1.0).cast(dtypes.float)),
                    UOp.const(1).broadcast(3), UOp.const(1).broadcast(1))


@graph
def concatenation():
    a, b = UOp.param(0, dtypes.float, (2, 3)), UOp.param(1, dtypes.float, (2, 3))
    c = UOp.param(2, dtypes.float, (4, 3))
    return UOp.sink(a.cat(b), a.cat(b, dim=1), a.cat(c), a.cat(c, b), a.cat(b, dim=-1))


@graph
def pools():
    p = UOp.param(0, dtypes.float, (2, 5, 6))
    return UOp.sink(p._pool((3,)), p._pool((3,), 2), p._pool((2, 2), (1, 2), (2, 1)), p._pool((5, 6)), p._pool((2,), 3),
                    p.repeat((2, 1, 1)), p.repeat((3, 1, 1, 2)))


@graph
def padding():
    p = UOp.param(0, dtypes.float, (4, 4))
    return UOp.sink(p.pad(((1, 1), (2, 0)), value=0.0), p.pad(((1, 1), None), value=1.5), p.pad(((-1, 2), (0, -2))),
                    p.pad_to((6, None), value=2.0), p.pad(((1, -1), (0, 2)), value=-1.0))


@graph
def reductions():
    p = UOp.param(0, dtypes.float, (2, 3, 4))
    q = UOp.param(1, dtypes.float, (2, 1, 4))
    return UOp.sink(p._rop(Ops.ADD, (0,)), p._rop(Ops.MAX, (2, 0)), p._rop(Ops.MUL, (1,)), q._rop(Ops.ADD, (1,)),
                    q._rop(Ops.ADD, (1, 2)), p._rop(Ops.ADD, ()))


@graph
def constants_like():
    p = UOp.param(0, dtypes.float, (2, 3))
    r = UOp.range(4, 0)
    return UOp.sink(p.const_like(0), p.const_like(True, dtypes.bool), r.const_like(3), UOp.const(1).const_like(2),
                    UOp.stack(UOp.const(1.0).cast(dtypes.half), UOp.const(2.0).cast(dtypes.half)).vconst_like(0))


@graph
def kernel_nodes():
    buf = UOp.param(0, dtypes.float, 16)
    r = UOp.range(16, 0)
    ld = buf.index(r).load()
    red = ld.reduce(r, arg=Ops.ADD)
    gidx = UOp.special(4, "gidx0")
    st = UOp.param(1, dtypes.float, 4).index(gidx).store(red + 1.0, gate=gidx < 2)
    return UOp.sink(st.end(r), st.end(), red.after(st), red.after(), st.barrier(), UOp.loop(3), UOp.range(8, 2, AxisType.UPCAST, dtype=dtypes.int),
                    UOp.const(1).backedge(UOp.loop(1), r < 2), buf.index(UOp.const(3)), UOp.const((1, 2, 3)).index(UOp.const(1)),
                    UOp.const((1, 2, 3)).index(UOp.const(1, dtypes.int)), UOp.group(st, st), UOp.group(st))


@graph
def validity():
    r = UOp.range(10, 0)
    v = r.valid(r < 5)
    return UOp.sink(v, v.get_idx(), v.get_valid(), r.get_idx(), r.get_valid(), UOp.invalid().get_valid(),
                    UOp.stack(v, r).get_idx(), UOp.stack(v, r).get_valid())


@graph
def contraction():
    r0, r1 = UOp.range(2, 0, AxisType.UPCAST), UOp.range(3, 1, AxisType.UPCAST)
    return UOp.sink((r0 * 3 + r1).contract(r0, r1), (r0 + 1).contract(r0))


@graph
def storage():
    return UOp.sink(UOp.param(2, dtypes.float, (2, 3, 4), device="CPU"),
                    UOp.param(3, dtypes.half), UOp.param(4, dtypes.float, (4,), vmin_vmax=(0.0, 1.0), name="w", volatile=True),
                    UOp.placeholder((2, 3), dtypes.weakint, 5), UOp.placeholder((8,), dtypes.float, 6, addrspace=AddrSpace.LOCAL),
                    UOp.placeholder((4,), dtypes.half, 7, addrspace=AddrSpace.REG, tag="acc"),
                    UOp.alloc((), dtypes.int, slot=9),
                    UOp.alloc((4, 2), dtypes.float, slot=10, device=("CPU:0", "CPU:1"), axis=0),
                    UOp.new_buffer("CPU", 4, dtypes.float, 11).view_as((2, 2)))


@graph
def storage_like():
    p = UOp.param(0, dtypes.float, (2, 3), device="CPU")
    v = UOp.variable("n", 1, 8)
    s = UOp.param(1, dtypes.float, (4, 3), device=("CPU:0", "CPU:1")).unshard(0)
    return UOp.sink(p.param_like(5), v.param_like(6), v.bind(3).param_like(7), s.param_like(8), p.placeholder_like(9),
                    p.alloc_like(10), s.alloc_like(11, addrspace=AddrSpace.LOCAL))


@graph
def sets():
    p = UOp.param(0, dtypes.float, (4,))
    r = UOp.range(4, 0)
    q = p.reshape((2, 2))
    return UOp.sink(p.index(r).set(1.0, end=r), q.set(q.const_like(2.0)))


@graph
def shards():
    p = UOp.param(0, dtypes.float, (4, 6), device="CPU")
    m = UOp.param(1, dtypes.float, (4, 6), device=("CPU:0", "CPU:1"))
    return UOp.sink(p.shard(("CPU:0", "CPU:1")), m.unshard(0), m.unshard((1, 0), (UOp.range(2, -1, AxisType.DEVICE), UOp.range(3, -2, AxisType.LOCAL))),
                    m.mselect(0), m.mselect(0).mstack(m.mselect(1)), m.mselect(0).mstack(), m.allreduce(Ops.ADD, ("CPU:0", "CPU:1")),
                    p.copy_to_device("CUDA"), m.copy_to_device(("CUDA:0", "CUDA:1")), m.copy_to_device("CPU", arg=1))


@graph
def calls():
    body = UOp.param(0, dtypes.float, 4).index(UOp.range(4, 0)).store(1.0).end(UOp.range(4, 0)).sink()
    a = UOp.param(0, dtypes.float, 4, device="CPU")
    return UOp.sink(body.call(a), body.call(a, name="fill", precompile=True), UOp.custom_function("f", UOp.const(0, dtypes.uint64)).call(ret_dtype=dtypes.int),
                    UOp.param(0, dtypes.float, 4, device="CPU").store_call(UOp.param(1, dtypes.float, 4, device="CPU")))


@graph
def custom_kernels():
    a, b = UOp.param(0, dtypes.float, 4, device="CPU"), UOp.param(1, dtypes.float, 4, device="CPU")
    return UOp.sink(*UOp.custom_kernel(a, b, fxn=lambda x, y: x.index(UOp.range(4, 0)).store(y.index(UOp.range(4, 0)).load()).end(UOp.range(4, 0)).sink()))


@graph
def variables_and_binding():
    n, m = UOp.variable("n", 1, 8), UOp.variable("m", 0, 4, multiple_of=2)
    e = n.bind(3) * m.bind(2) + n
    unbound, _ = e.unbind_all()
    return UOp.sink(n.bind(3), n.bind(3).unbound(), n.bind(3).rtag("t").unbound(), unbound, *e.variables(),
                    *(UOp.range(4, -1, AxisType.DEVICE) + n).variables())


@graph
def getaddrs():
    p = UOp.param(0, dtypes.float, 4, device="CPU")
    return UOp.sink(p.getaddr(), p.getaddr("CPU:1"), UOp(Ops.BINARY, arg=b"ab").getaddr("CPU"), UOp.const(1).getaddr(),
                    p.after(UOp.sink()).getaddr())


@graph
def instructions():
    x = UOp.param(0, dtypes.float, 4).index(UOp.const(0))
    return UOp.sink(x.ins("v_mov"), x.ins("v_add", src=(x, x), dtype=dtypes.half), x.rtag(1).ins("nop", dtype=dtypes.void, tag=None))


@graph
def wmmas():
    a, b, c = UOp.param(0, dtypes.half, (8,)), UOp.param(1, dtypes.half, (8,)), UOp.param(2, dtypes.float, (4,))
    return UOp.sink(UOp.wmma(a, b, c, (8, 16, 16), 32), UOp.wmma(a, b, c, (8, 16, 16), 32, ((((0,), 2),), (((1,), 2),), (((2,), 2),))))


@graph
def substitution():
    a, b, c = var("a"), var("b"), var("c")
    x = (a + 4) + (a + 5)
    return UOp.sink(x.substitute({a: b}), x.substitute({a: a}), (a + 4).replace(tag=1).substitute({a: c}),
                    fvar("x").sin().sin().substitute({fvar("x").sin(): fvar("x").sqrt()}))


@graph
def hcq_calls():
    from tinygrad.runtime.support.hcq2 import HCQInfo
    b = UOp.new_buffer("AMD", 4, dtypes.float, 3)
    kernel = (("AMD",), "k", Estimates(1, 2, 3), (0, 1), b"key", (0,), ((0,), (0,)))
    info = HCQInfo(("AMD",), kernels=(kernel,), estimates=Estimates(1, 2, 3), nargs=2, table=1, inputs=((b, 0, "GLOBAL"),),
                   slots=(("AMD", 1),), host_deps=(("CPU", "AMD"),), written_bufs=(b,), skip_wait=True)
    return UOp.sink(UOp.custom_function("f").call(b, aux=info), UOp.custom_function("f").call(b, aux=HCQInfo(("AMD",))))


# Storage without a given slot takes the next number of a counter. These graphs
# number theirs from MINTED, in the order the graph lists them, as the suite
# renumbers its own.
MINTED = 1000


def minted(fn):
    def body():
        UOp.unique_num = itertools.count(MINTED)
        sink = fn()
        slots = [u.arg.slot for u in toposort(sink) if u.op is Ops.ALLOC]
        if list(dict.fromkeys(slots)) != list(range(MINTED, MINTED + len(set(slots)))):
            raise RuntimeError(f"{fn.__name__} lists its minted slots out of order: {slots}")
        return sink
    body.__name__ = fn.__name__
    return graph(body)


@minted
def clones():
    p = UOp.param(0, dtypes.float, (2, 3), device="CPU")
    return UOp.sink(p.clone(), p.clone("CUDA"), p.empty_like(), p.empty_like(dtypes.half, "CUDA"),
                    UOp.empty((2, 3), dtype=dtypes.int, device="CPU"))


@minted
def outputs():
    a = UOp.param(0, dtypes.float, 4, device="CPU")
    return UOp.sink(*UOp.call_with_outputs((a + 1.0, a * 2.0), a, name="two"), a.call_with_output(a, precompile=True))


# Symbolic sizes: their shape arguments are simplified by the symbolic rules

@graph
def symbolic_storage():
    n = UOp.variable("n", 1, 8)
    return UOp.sink(UOp.param(0, dtypes.float, (2, n)), UOp.param(1, dtypes.int, n), UOp.alloc((2, n), dtypes.weakfloat, slot=8))


@graph
def symbolic_shards():
    return UOp.sink(UOp.param(0, dtypes.float, (4, 6), device="CPU").shard(("CPU:0", "CPU:1"), axis=1))


@graph
def symbolic_shard_slices():
    p = UOp.param(0, dtypes.float, (4, 6))
    return UOp.sink(p._shard(1, UOp.range(2, -1, AxisType.DEVICE)), p._shard(0, UOp.range(4, 0, AxisType.LOCAL)))


@minted
def symbolic_outputs():
    a = UOp.param(0, dtypes.float, 4, device="CPU")
    n = UOp.variable("n", 1, 4)
    b = UOp.param(1, dtypes.float, (n,), device="CPU")
    formal = UOp.param(0, dtypes.int, vmin_vmax=(1, 8), name="d", addrspace=AddrSpace.ALU)
    return UOp.sink(*UOp.call_with_outputs((b + 1.0,), a, b, output_pos=(0,)),
                    UOp.const(1.0).cast(dtypes.float).expand((formal,)).call_with_output(UOp.const(5)),
                    UOp.empty((2, n), dtype=dtypes.int, device="CPU"))
