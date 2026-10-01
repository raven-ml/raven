"""Goldens of tinygrad/uop/spec.py: the verdict of each specification on
chosen nodes, bounds checking, and type_verify's failures.

A table of nodes comes with a graph golden of the same name and the suffix
`_nodes`: the sources of its sink are the table's nodes, in the order of its
rows. A verdict is what `PatternMatcher.rewrite` returns: `True`, `False`, or
`None` when no rule decides.
"""

import sys
import types
from dataclasses import replace

from golden import graph, listing, table, text
from tinygrad.device import Buffer, MultiBuffer
from tinygrad.dtype import AddrSpace, Invalid, dtypes
from tinygrad.helpers import Context
from tinygrad.uop import Ops
from tinygrad.uop.ops import AxisType, CallInfo, KernelInfo, ParamArg, UOp

# z3 is not a dependency of tolk, which fails the accesses that the bounds
# of their index do not prove, as z3 does when it cannot decide. The stand-in
# gives that verdict for every access tinygrad would hand to z3.
z3_unknown = types.ModuleType("tinygrad.uop.validate")
z3_unknown.validate_index_with_z3 = lambda sz, idx, gate: False
sys.modules["tinygrad.uop.validate"] = z3_unknown

from tinygrad.uop.spec import spec_full, spec_hcq, spec_kernel_graph, spec_program, spec_shared, spec_tensor, type_verify  # noqa: E402

SPECS = [("shared", spec_shared), ("tensor", spec_tensor), ("program", spec_program), ("hcq", spec_hcq),
         ("full", spec_full), ("kernel_graph", spec_kernel_graph)]


def verdict(spec, u):
    ret = spec.rewrite(u)
    assert ret in (True, False, None), f"{ret!r} is not a verdict"
    return ret


class Cases:
    """Named nodes, each a row of a table and a source of a graph's sink."""

    def __init__(self):
        self.names, self.nodes = [], []

    def __call__(self, name, u):
        assert name not in self.names, f"case {name!r} is declared twice"
        self.names.append(name)
        self.nodes.append(u)

    def sink(self): return UOp(Ops.SINK, src=tuple(self.nodes))


# Building blocks

def i32(x): return UOp.const(x, dtypes.int32)
def f32(x): return UOp.const(x, dtypes.float32)
def cvar(name, dt, lo=0, hi=10): return UOp.variable(name, lo, hi, dt)


def fvar(name, dt=dtypes.float32): return UOp.variable(name, -10.0, 10.0, dt)


def flag(name): return UOp.variable(name, False, True, dtypes.bool)


def buf(slot=0, dt=dtypes.float32, size=16): return UOp.param(slot, dt, size)


CPU2 = ("CPU:0", "CPU:1")


def device_range(n=2, dt=dtypes.weakint): return UOp.range(n, -1, AxisType.DEVICE, dtype=dt)


def global_buffer(device="CPU", size=16, dt=dtypes.float32, slot=100):
    return UOp.new_buffer(device, size, dt, num=slot)


def rebuffer(b, **fields):
    """A global BUFFER like `b`, which holds a device buffer, with other fields."""
    src = fields.pop("src", b.src)
    return UOp(Ops.BUFFER, src=src, arg=replace(b.arg, **fields))


def alloc(size=16, dt=dtypes.float32, addrspace=AddrSpace.GLOBAL, device=None, src=None, slot=200):
    return UOp(Ops.ALLOC, src=UOp.device_range_src(device) if src is None else src,
               arg=ParamArg(slot, dt, size, addrspace=addrspace, device=device))


def local(size=8, dt=dtypes.float32, slot=0): return UOp.placeholder((size,), dt, slot=slot, addrspace=AddrSpace.LOCAL)


def register(size=4, dt=dtypes.float32, slot=1): return UOp.placeholder((size,), dt, slot=slot, addrspace=AddrSpace.REG)


def shape(*dims): return UOp(Ops.STACK, src=tuple(UOp.const(d) for d in dims))


# Verdicts of every specification

def verdict_cases():
    case = Cases()
    x, y = buf(0), buf(1)
    ld = x.index(UOp.const(0)).load()
    st = y.index(UOp.const(0)).store(ld)
    r = UOp.range(16, 0)

    # sinks, constants and stacks
    case("a sink of a load", UOp.sink(ld))
    case("a sink with kernel info", UOp.sink(st, arg=KernelInfo(name="k")))
    case("an empty sink", UOp.sink())
    case("a noop", UOp(Ops.NOOP))
    case("an integer constant", UOp.const(3))
    case("a float constant", UOp.const(1.5))
    case("a boolean constant", UOp.const(True))
    case("the invalid constant", UOp.invalid())
    case("a constant with a source", UOp(Ops.CONST, src=(UOp.const(1),), arg=2))
    case("an empty stack", UOp(Ops.STACK))
    case("a stack of int32", UOp(Ops.STACK, src=(i32(1), i32(2))))
    case("a stack of int32 and a weak integer", UOp(Ops.STACK, src=(i32(1), UOp.const(2))))
    case("a stack of int32 and float32", UOp(Ops.STACK, src=(i32(1), f32(2.0))))
    case("a stack of float32 and the invalid constant", UOp(Ops.STACK, src=(f32(1.0), UOp.invalid())))
    case("a stack of a scalar and a vector", UOp(Ops.STACK, src=(f32(1.0), x)))

    # arithmetic
    c = flag("c")
    case("a where over float32", c.where(fvar("a"), fvar("b")))
    case("a where over float32 and a weak float", UOp(Ops.WHERE, src=(c, fvar("a"), UOp.const(2.0))))
    case("a where over int32 and float32", UOp(Ops.WHERE, src=(c, cvar("a", dtypes.int32), fvar("b"))))
    case("a where over float32 and the invalid constant", UOp(Ops.WHERE, src=(c, fvar("a"), UOp.invalid())))
    case("a where over the invalid constant", UOp(Ops.WHERE, src=(UOp.invalid(), fvar("a"), fvar("b"))))
    for op in (Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ):
        name = str(op).removeprefix("Ops.").lower()
        case(f"a {name} of int32", UOp(op, src=(cvar("a", dtypes.int32), cvar("b", dtypes.int32))))
        case(f"a {name} of int32 and float32", UOp(op, src=(cvar("a", dtypes.int32), fvar("b"))))
    case("a cmplt of int32 and a weak integer", UOp(Ops.CMPLT, src=(cvar("a", dtypes.int32), UOp.const(3))))
    case("a cmplt of a weak float and float16", UOp(Ops.CMPLT, src=(UOp.const(3.0), fvar("h", dtypes.half))))
    case("a cmplt of the invalid constant and float32", UOp(Ops.CMPLT, src=(UOp.invalid(), fvar("a"))))
    case("a cmplt of float32 and the invalid constant", UOp(Ops.CMPLT, src=(fvar("a"), UOp.invalid())))
    for op in (Ops.AND, Ops.OR, Ops.XOR):
        name = str(op).removeprefix("Ops.").lower()
        case(f"an {name} of float32", UOp(op, src=(fvar("a"), fvar("b"))))
        case(f"an {name} of int32", UOp(op, src=(cvar("a", dtypes.int32), cvar("b", dtypes.int32))))
    case("an and of weak floats", UOp(Ops.AND, src=(UOp.const(1.0), UOp.const(2.0))))
    case("an and of booleans", UOp(Ops.AND, src=(flag("a"), flag("b"))))
    case("an and of int32 and int8", UOp(Ops.AND, src=(cvar("a", dtypes.int32), cvar("b", dtypes.int8))))
    for op in (Ops.SHL, Ops.SHR):
        name = str(op).removeprefix("Ops.").lower()
        case(f"a {name} of int32 by int32", UOp(op, src=(cvar("a", dtypes.int32), cvar("n", dtypes.int32))))
        case(f"a {name} of int8 by uint32", UOp(op, src=(cvar("a", dtypes.int8), cvar("n", dtypes.uint32))))
        case(f"a {name} of int8 by int32", UOp(op, src=(cvar("a", dtypes.int8), cvar("n", dtypes.int32))))
    case("a shl of int8 by a weak integer", UOp(Ops.SHL, src=(cvar("a", dtypes.int8), UOp.const(2))))
    case("a shl of uint64 by uint16", UOp(Ops.SHL, src=(cvar("a", dtypes.uint64), cvar("n", dtypes.uint16))))
    case("a shl of the invalid constant by int32", UOp(Ops.SHL, src=(UOp.invalid(), cvar("n", dtypes.int32))))
    for op in (Ops.CDIV, Ops.CMOD, Ops.FLOORDIV, Ops.FLOORMOD):
        name = str(op).removeprefix("Ops.").lower()
        case(f"a {name} of int32", UOp(op, src=(cvar("a", dtypes.int32), cvar("b", dtypes.int32))))
        case(f"a {name} of float32", UOp(op, src=(fvar("a"), fvar("b"))))
    case("a cdiv of float32 by the invalid constant", UOp(Ops.CDIV, src=(fvar("a"), UOp.invalid())))
    case("a cmod of the invalid constant by int32", UOp(Ops.CMOD, src=(UOp.invalid(), cvar("b", dtypes.int32))))
    case("a cdiv of int32 by float32", UOp(Ops.CDIV, src=(cvar("a", dtypes.int32), fvar("b"))))
    case("a sum of int32", UOp(Ops.ADD, src=(cvar("a", dtypes.int32), cvar("b", dtypes.int32))))
    case("a sum of int32 and float32", UOp(Ops.ADD, src=(cvar("a", dtypes.int32), fvar("b"))))
    case("a sum of float16 and float32", UOp(Ops.ADD, src=(fvar("h", dtypes.half), fvar("b"))))
    case("a sum of float32 and a weak float", UOp(Ops.ADD, src=(fvar("a"), UOp.const(1.0))))
    case("a sum of int32 and a weak integer", UOp(Ops.ADD, src=(cvar("a", dtypes.int32), UOp.const(1))))
    case("a sum of float32 and the invalid constant", UOp(Ops.ADD, src=(fvar("a"), UOp.invalid())))
    case("a sum of weak integers", UOp(Ops.ADD, src=(cvar("a", dtypes.weakint), cvar("b", dtypes.weakint))))
    case("a product of float64", UOp(Ops.MUL, src=(fvar("d", dtypes.double), fvar("e", dtypes.double))))
    case("a maximum of uint8 and int8", UOp(Ops.MAX, src=(cvar("u", dtypes.uint8), cvar("s", dtypes.int8))))
    case("a subtraction of bfloat16", UOp(Ops.SUB, src=(fvar("a", dtypes.bfloat16), fvar("b", dtypes.bfloat16))))
    case("a float division of int32", UOp(Ops.FDIV, src=(cvar("a", dtypes.int32), cvar("b", dtypes.int32))))
    case("a float division of float32", UOp(Ops.FDIV, src=(fvar("a"), fvar("b"))))
    case("a power of float32", UOp(Ops.POW, src=(fvar("a"), fvar("b"))))
    case("a multiply-add of float32", UOp(Ops.MULACC, src=(fvar("a"), fvar("b"), fvar("c"))))
    case("a multiply-add of float32 and float16", UOp(Ops.MULACC, src=(fvar("a"), fvar("b"), fvar("h", dtypes.half))))
    case("a threefry of uint64", UOp(Ops.THREEFRY, src=(cvar("k", dtypes.uint64), cvar("c", dtypes.uint64))))
    case("a negation of int32", UOp(Ops.NEG, src=(cvar("a", dtypes.int32),)))
    case("a truncation of float32", UOp(Ops.TRUNC, src=(fvar("a"),)))
    for op in (Ops.SIN, Ops.LOG2, Ops.EXP2, Ops.SQRT, Ops.RECIPROCAL):
        name = str(op).removeprefix("Ops.").lower()
        case(f"a {name} of float32", UOp(op, src=(fvar("a"),)))
        case(f"a {name} of int32", UOp(op, src=(cvar("a", dtypes.int32),)))
    case("a sin of the invalid constant", UOp(Ops.SIN, src=(UOp.invalid(),)))
    case("a sin of a weak integer", UOp(Ops.SIN, src=(UOp.const(2),)))

    # casts
    case("a cast of int32 to float32", cvar("a", dtypes.int32).cast(dtypes.float32))
    case("a bitcast of float32 to int32", fvar("a").bitcast(dtypes.int32))
    case("a cast of two sources", UOp(Ops.CAST, src=(fvar("a"), fvar("b")), arg=dtypes.int32))

    # ranges, indexes and loops
    case("a range", r)
    case("a range with a nested axis", UOp(Ops.RANGE, src=(UOp.const(16),), arg=(1, 0, AxisType.REDUCE)))
    case("a global int32 range", UOp.range(i32(16), 1, AxisType.GLOBAL, dtype=dtypes.int32))
    case("a range with extra sources", UOp(Ops.RANGE, src=(UOp.const(4), r), arg=(2, AxisType.LOOP)))
    case("a loop header", UOp.loop(3))
    case("a device range", device_range())
    case("an index by a weak integer", x.index(UOp.const(3)))
    case("an index by int32", x.index(i32(3)))
    case("an index by two integers", UOp(Ops.INDEX, src=(x.reshape((4, 4)), UOp.const(1), UOp.const(2))))
    case("an index by a float", x.index(fvar("f")))
    case("an index by a boolean", x.index(flag("b")))
    case("an index by the invalid constant", x.index(UOp.invalid()))
    case("an index by a guarded range", x.index(r.valid(r < 8)))
    case("an index of a stack", UOp(Ops.INDEX, src=(UOp(Ops.STACK, src=(f32(1.0), f32(2.0))), UOp.const(1))))
    case("an end of a store over a range", x.index(r).store(f32(0.0)).end(r))
    case("an end of a store over two ranges", x.index(r).store(f32(0.0)).end(r, UOp.range(4, 1)))
    case("an end of a store over an int32 range",
         x.index(r).store(f32(0.0)).end(UOp.range(i32(4), 1, AxisType.LOOP, dtype=dtypes.int32)))
    case("an end of a store over a constant", UOp(Ops.END, src=(st, UOp.const(4))))
    case("an end of a store over a launch dimension", UOp(Ops.END, src=(st, UOp.special(4, "gidx0"))))
    case("an end of a store over a loop header", UOp(Ops.END, src=(st, UOp.loop(3))))
    case("an end of a value over a range", UOp(Ops.END, src=(ld, r)))
    case("an end of a store alone", UOp(Ops.END, src=(st,)))
    loop = UOp.loop(5)
    case("a backedge on a boolean", st.backedge(loop, flag("go")))
    case("a backedge on the invalid constant", st.backedge(loop, UOp.invalid()))
    case("a backedge on a vector of booleans", st.backedge(loop, buf(2, dtypes.bool, 4)))
    case("a backedge of a bounded range", UOp(Ops.BACKEDGE, src=(st, r, flag("go"))))
    case("a backedge of a value", ld.backedge(loop, flag("go")))

    # storage and effects
    case("a parameter", x)
    case("a scalar parameter", UOp.param(3, dtypes.int32))
    case("a named parameter on a device", UOp.param(4, dtypes.half, 8, device="CPU", name="w"))
    case("a variable", cvar("n", dtypes.weakint))
    case("an int32 variable", cvar("n", dtypes.int32))
    case("a variable on a device",
         UOp(Ops.PARAM, arg=ParamArg(-1, dtypes.int32, vmin_vmax=(0, 10), name="n", addrspace=AddrSpace.ALU, device="CPU")))
    case("a bound variable", cvar("n", dtypes.int32).bind(3))
    case("a local buffer", local())
    case("a register buffer", register())
    gb = global_buffer()
    case("a global buffer", gb)
    case("a global buffer without a device", rebuffer(gb, device=None))
    case("a global buffer without a size", rebuffer(gb, size=None))
    mb = global_buffer(CPU2, slot=101)
    case("a buffer on two devices", mb)
    case("a buffer on two devices without its range", rebuffer(mb, src=()))
    case("a buffer on two devices over three", rebuffer(mb, src=(device_range(3),)))
    case("a buffer on two devices over a loop range", rebuffer(mb, src=(UOp.range(2, -1, AxisType.LOOP),)))
    case("a buffer on one device with a range", rebuffer(gb, src=(device_range(1),)))
    case("an alu buffer", UOp(Ops.BUFFER, arg=ParamArg(5, dtypes.float32, 4, addrspace=AddrSpace.ALU)))
    case("a global allocation", alloc())
    case("a global allocation on a device", alloc(device="CPU", slot=201))
    case("a global allocation without a size", alloc(size=None, slot=202))
    case("an allocation on two devices", alloc(device=CPU2, slot=203))
    case("an allocation on two devices without its range", alloc(device=CPU2, src=(), slot=204))
    case("a local allocation", alloc(addrspace=AddrSpace.LOCAL, slot=205))
    case("a register allocation", alloc(addrspace=AddrSpace.REG, slot=206))
    case("a local allocation on a device", alloc(addrspace=AddrSpace.LOCAL, device="CPU", slot=207))
    case("an alu allocation", alloc(addrspace=AddrSpace.ALU, slot=208))
    case("binary data", UOp(Ops.BINARY, arg=b"\x7fELF"))
    case("a group of stores", UOp.group(st, x.index(UOp.const(1)).store(f32(1.0))))
    case("a group of a value", UOp(Ops.GROUP, src=(st, ld)))
    case("a parameter after a store", x.after(st))
    case("an index after a store", x.index(UOp.const(0)).after(st))
    case("a reshape after a store", x.reshape((4, 4)).after(st))
    case("a bitcast after a store", x.bitcast(dtypes.int32).after(st))
    case("an allocation after a store", alloc(slot=209).after(st))
    case("a staged value after a store", UOp(Ops.STAGE, src=(ld,)).after(st))
    case("an instruction after a store", UOp(Ops.INS, arg=("s_waitcnt", dtypes.void)).after(st))
    case("an after after a store", x.after(st).after(st))
    case("a constant after a store", UOp.const(1).after(st))
    case("a load after a store", ld.after(st))
    case("custom code", UOp(Ops.CUSTOM, src=(ld,), arg=("{0}+1", dtypes.float32)))
    case("inline custom code", UOp(Ops.CUSTOMI, src=(ld,), arg=("({0}*2)", dtypes.float32)))
    case("a void custom statement", UOp(Ops.CUSTOM, arg=("__syncthreads();", dtypes.void)))
    case("a custom function", UOp.custom_function("fn", x))
    case("a custom function without sources", UOp.custom_function("fn"))
    case("a call of a sink", UOp.sink(st).call(x, y))
    case("a call of a store", st.call(x, y))
    case("a call of a custom function", UOp.custom_function("fn", x).call(x, ret_dtype=dtypes.int32))
    case("a call of a constant", UOp(Ops.CALL, src=(UOp.const(1), x), arg=CallInfo()))
    case("a call of a load", UOp(Ops.CALL, src=(ld,), arg=CallInfo(dtype=dtypes.float32)))
    case("a barrier", UOp(Ops.BARRIER, src=(st,)))
    case("an instruction", UOp(Ops.INS, src=(ld,), arg=("v_mov", dtypes.float32)))
    case("a wmma", UOp.wmma(fvar("a", dtypes.half), fvar("b", dtypes.half), fvar("acc"), (8, 8, 8), 32))

    # loads and stores
    case("a load of an index", ld)
    case("a load of a cast index", x.index(UOp.const(0)).cast(dtypes.int32).load())
    case("a load of a twice cast index", x.index(UOp.const(0)).cast(dtypes.int32).cast(dtypes.int64).load())
    case("a load of a bitcast index", UOp(Ops.LOAD, src=(x.index(UOp.const(0)).bitcast(dtypes.int32),)))
    case("a load of a shrink", UOp(Ops.SHRINK, src=(x, UOp.const(0), UOp.const(4))).load())
    case("a load of a parameter", UOp(Ops.LOAD, src=(x,)))
    gate = flag("g")
    case("a gated load", x.index(UOp.const(0)).load(f32(0.0), gate))
    case("a gated load with a weak alternative", x.index(UOp.const(0)).load(UOp.const(0.0), gate))
    case("a gated load with an int32 alternative", x.index(UOp.const(0)).load(i32(0), gate))
    case("a gated load with the invalid alternative", x.index(UOp.invalid()).load(UOp.invalid(), gate))
    case("a gated load by an int32 gate", x.index(UOp.const(0)).load(f32(0.0), cvar("g", dtypes.int32)))
    case("a store to an index", st)
    case("a store to a cast index", UOp(Ops.STORE, src=(y.index(UOp.const(0)).cast(dtypes.int32), i32(1))))
    case("a store to a shrink", UOp(Ops.SHRINK, src=(y, UOp.const(0), UOp.const(4))).store(f32(1.0)))
    case("a gated store", y.index(UOp.const(0)).store(ld, gate))
    case("a store to a parameter", UOp(Ops.STORE, src=(y, ld)))
    case("a store to an allocation", UOp(Ops.STORE, src=(alloc(slot=210), ld)))
    case("a store to a global buffer", UOp(Ops.STORE, src=(gb, ld)))
    case("a store to a staged value", UOp(Ops.STORE, src=(UOp(Ops.STAGE, src=(ld,)), ld)))
    case("a store to a reshape of a parameter", UOp(Ops.STORE, src=(y.reshape((4, 4)), ld)))
    case("a store to a parameter after a store", UOp(Ops.STORE, src=(y.after(st), ld)))
    case("a store to an index after a store", UOp(Ops.STORE, src=(y.index(UOp.const(0)).after(st), ld)))
    case("a store to a constant", UOp(Ops.STORE, src=(UOp.const(1), ld)))
    case("a store to a load", UOp(Ops.STORE, src=(ld, ld)))

    # tensor graphs
    case("a special over a weak integer", UOp.special(8, "gidx0"))
    case("a special over int32", UOp(Ops.SPECIAL, src=(i32(8),), arg="lidx0"))
    case("a reshape", x.reshape((4, 4)))
    case("an expand", UOp(Ops.EXPAND, src=(x.reshape((1, 16)), shape(4, 16))))
    case("a pad", UOp(Ops.PAD, src=(x, shape(1), shape(1))))
    case("a pad by a shorter end", UOp(Ops.PAD, src=(x.reshape((4, 4)), shape(1, 1), shape(1))))
    case("a shrink", UOp(Ops.SHRINK, src=(x, shape(2), shape(10))))
    case("a permute", x.reshape((4, 4)).permute((1, 0)))
    case("a flip", x.reshape((4, 4)).flip((0,)))
    case("a reduce", UOp(Ops.REDUCE, src=(x,), arg=(Ops.ADD, 1)))
    case("a reduce over ranges", UOp(Ops.REDUCE, src=(ld, r, UOp.range(4, 1)), arg=(Ops.MAX, 0)))
    case("a reduce over an int32 range",
         UOp(Ops.REDUCE, src=(ld, UOp.range(i32(4), 1, AxisType.REDUCE, dtype=dtypes.int32)), arg=(Ops.ADD, 0)))
    case("a reduce over a uint32 range",
         UOp(Ops.REDUCE, src=(ld, UOp.range(UOp.const(4, dtypes.uint32), 1, dtype=dtypes.uint32)), arg=(Ops.ADD, 0)))
    case("a reduce by sine", UOp(Ops.REDUCE, src=(x,), arg=(Ops.SIN, 1)))
    case("a copy", x.copy_to_device("CPU"))
    case("a copy to two devices", x.copy_to_device(CPU2))
    case("a copy to two devices without its range", UOp(Ops.COPY, src=(x,), arg=CPU2))
    case("a copy to a disk", UOp(Ops.COPY, src=(x,), arg="DISK:/tmp/x"))
    sharded = UOp.param(6, dtypes.float32, 16, device=CPU2)
    case("an allreduce", UOp(Ops.ALLREDUCE, src=(sharded,), arg=(Ops.ADD, CPU2)))
    case("an allreduce by sine", UOp(Ops.ALLREDUCE, src=(sharded,), arg=(Ops.SIN, CPU2)))
    case("an allreduce by a bitwise or", UOp(Ops.ALLREDUCE, src=(sharded.bitcast(dtypes.uint32),), arg=(Ops.OR, CPU2)))
    case("an unshard", sharded.unshard(0))
    case("an unshard over a derived range", UOp(Ops.UNSHARD, src=(sharded, device_range() * 2), arg=(0,)))
    case("an unshard missing a range", UOp(Ops.UNSHARD, src=(sharded,), arg=(0,)))
    case("an unshard over an int32 range", UOp(Ops.UNSHARD, src=(sharded, device_range(2, dtypes.int32)), arg=(0,)))
    case("a selection of a shard", sharded.mselect(1))
    case("a selection past the shards", sharded.mselect(2))
    case("a selection on one device", UOp.param(7, dtypes.float32, 16, device="CPU").mselect(0))
    one = [UOp.param(8 + k, dtypes.float32, 16, device=f"CPU:{k}") for k in range(2)]
    case("a stack of shards", UOp(Ops.MSTACK, src=tuple(one)))
    case("a stack of sharded values", UOp(Ops.MSTACK, src=(sharded, sharded)))
    case("a stack of one value without a device", UOp(Ops.MSTACK, src=(x, x)))
    case("a stack of values without a device", UOp(Ops.MSTACK, src=(x, y)))
    case("a detach", UOp(Ops.DETACH, src=(x,)))
    case("a contiguous backward", UOp(Ops.CONTIGUOUS_BACKWARD, src=(x,)))
    case("a staged value", UOp(Ops.STAGE, src=(ld,)))
    case("a staged value over a range", UOp(Ops.STAGE, src=(ld, r)))
    sink, lin, source, binary = UOp.sink(st), UOp(Ops.LINEAR, src=(st,)), UOp(Ops.SOURCE, arg="void k(){}"), \
        UOp(Ops.BINARY, arg=b"\x00")
    case("a linear", lin)
    case("a source", source)
    case("a program of a sink", UOp(Ops.PROGRAM, src=(sink,)))
    case("a program of a sink and a linear", UOp(Ops.PROGRAM, src=(sink, lin)))
    case("a program of a sink, a linear and a source", UOp(Ops.PROGRAM, src=(sink, lin, source)))
    case("a compiled program", UOp(Ops.PROGRAM, src=(sink, lin, source, binary)))
    case("a program of a linear", UOp(Ops.PROGRAM, src=(lin,)))
    case("a program of a parameter", UOp(Ops.PROGRAM, src=(x,)))
    case("a program of a global buffer after a store", UOp(Ops.PROGRAM, src=(gb.after(st),)))

    # programs
    case("a sum under a constant", UOp(Ops.ADD, src=(cvar("a", dtypes.int32), UOp.const(1))))
    case("an int32 constant", i32(1))
    case("a shrink of a parameter by cast constants", UOp(Ops.SHRINK, src=(x, i32(0), i32(4))))
    case("a shrink of a bitcast parameter by cast constants", UOp(Ops.SHRINK, src=(x.bitcast(dtypes.int32), i32(0), i32(4))))
    case("a shrink of a parameter by a variable", UOp(Ops.SHRINK, src=(x, i32(0), cvar("n", dtypes.int32))))
    case("a shrink of a load by cast constants", UOp(Ops.SHRINK, src=(ld, i32(0), i32(4))))
    case("an if on an index", UOp(Ops.IF, src=(gate, x.index(UOp.const(0)))))
    case("an if on a cast", UOp(Ops.IF, src=(gate, ld.cast(dtypes.int32))))
    case("an if on a shrink", UOp(Ops.IF, src=(gate, UOp(Ops.SHRINK, src=(x, i32(0), i32(4))))))
    case("an if on a load", UOp(Ops.IF, src=(gate, ld)))
    case("an if on an int32 gate", UOp(Ops.IF, src=(cvar("g", dtypes.int32), x.index(UOp.const(0)))))
    iff = UOp(Ops.IF, src=(gate, x.index(UOp.const(0))))
    case("an endif", UOp(Ops.ENDIF, src=(iff,)))
    case("an endif of a store", UOp(Ops.ENDIF, src=(st,)))

    # command queues
    case("an address of a global buffer", gb.getaddr())
    case("an address of an allocation", alloc(device="CPU", slot=211).getaddr())
    case("an address of a buffer after a store", UOp(Ops.GETADDR, src=(gb.after(st),), arg="CPU"))
    case("an address of a stack of shards", UOp(Ops.GETADDR, src=(UOp(Ops.MSTACK, src=tuple(one)),), arg="CPU:0"))
    case("an address of a load", UOp(Ops.GETADDR, src=(ld,), arg="CPU"))

    # kernel graphs
    case("a stack of parameters", UOp(Ops.STACK, src=(x, y)))
    case("a stack of a sum", UOp(Ops.STACK, src=(UOp.const(1) + cvar("a", dtypes.weakint),)))
    case("a cast of a constant", UOp.cconst(1, dtypes.int64))
    case("an after of a stack of shards", UOp(Ops.MSTACK, src=tuple(one)).after(st))

    # nodes without the argument their operation takes
    case("binary data without bytes", UOp(Ops.BINARY))
    case("a custom function without a name", UOp(Ops.CUSTOM_FUNCTION, src=(x,)))
    case("a call without call info", UOp(Ops.CALL, src=(UOp.sink(st), x, y)))
    case("a wmma without an argument", UOp(Ops.WMMA, src=(fvar("a", dtypes.half), fvar("b", dtypes.half), fvar("acc"))))
    case("a special without a name", UOp(Ops.SPECIAL, src=(UOp.const(8),)))
    case("a special over int32 without a name", UOp(Ops.SPECIAL, src=(i32(8),)))
    case("a permute without an order", UOp(Ops.PERMUTE, src=(x.reshape((4, 4)),)))
    case("a reduce without an argument", UOp(Ops.REDUCE, src=(x,)))
    case("a copy without a device", UOp(Ops.COPY, src=(x,)))
    case("an allreduce without an argument", UOp(Ops.ALLREDUCE, src=(sharded,)))
    case("an address without a device", UOp(Ops.GETADDR, src=(gb,)))
    return case


VERDICTS = verdict_cases()


@graph
def verdicts_nodes(): return VERDICTS.sink()


@table
def verdicts():
    return ["case", *[name for name, _ in SPECS]], \
        [(name, *[verdict(spec, u) for _, spec in SPECS]) for name, u in zip(VERDICTS.names, VERDICTS.nodes)]


# Bounds checking

def bounds_cases():
    case = Cases()
    x = buf(0, dtypes.int32, 16)

    def load(idx, target=x): return target.index(idx).load()
    def store(idx, target=x): return target.index(idx).store(i32(0))

    r42, v = UOp.range(42, 0, AxisType.GLOBAL), UOp.variable("v", -5, 80)
    case("a load at the first element", load(UOp.const(0)))
    case("a load at the last element", load(UOp.const(15)))
    case("a load one past the last element", load(UOp.const(16)))
    case("a load far past the last element", load(UOp.const(42)))
    case("a load at a negative constant", load(UOp.const(-1)))
    case("a load at an int32 constant", load(i32(15)))
    case("a load over the elements", load(UOp.variable("i", 0, 15)))
    case("a load past the last element", load(UOp.variable("i", 0, 20)))
    case("a load before the first element", load(UOp.variable("i", -5, 10)))
    case("a load over a range as long as the buffer", load(UOp.range(16, 0)))
    case("a load over a longer range", load(UOp.range(17, 0)))
    case("a load over a guarded range", load(r42.valid(r42 < 16)))
    case("a load over a range guarded too loosely", load(r42.valid(r42 < 17)))
    case("a load over a guarded variable", load(v.valid((v >= 0) & (v < 16))))
    case("a load over a variable guarded on one side", load(v.valid(v < 20)))
    case("a load over half a longer range", load(UOp.range(32, 0, AxisType.GLOBAL) // 2))
    case("a load over a range modulo the length", load(r42 % 16))
    case("a load over a range modulo more than the length", load(r42 % 20))
    case("a load over a range masked to the length", load(r42 & 15))
    case("a load over a range shifted into the buffer", load(UOp.range(64, 0, AxisType.GLOBAL) >> 2))
    case("a load over a range shifted past the buffer", load(UOp.range(128, 0, AxisType.GLOBAL) >> 2))
    case("a load over a clamped variable", load(UOp.variable("w", -10, 15).maximum(0)))
    case("a load over a variable clamped too little", load(UOp.variable("w", -10, 20).maximum(0)))
    case("a load at the invalid index", load(UOp.invalid()))
    case("a gated load past the last element", x.index(UOp.variable("i", 0, 20)).load(i32(0), flag("g")))
    case("a gated load within the buffer", x.index(UOp.variable("i", 0, 15)).load(i32(0), flag("g")))
    case("a load of a cast index past the last element", x.index(UOp.const(16)).cast(dtypes.int64).load())
    case("a load through two indexes", UOp(Ops.INDEX, src=(x.reshape((4, 4)), UOp.const(9), UOp.const(9))).load())
    case("a load of a shrink", UOp(Ops.SHRINK, src=(x, UOp.const(20), UOp.const(24))).load())
    case("a load from a reshape over its elements", load(UOp.range(16, 0), x.reshape((4, 4))))
    case("a load from a reshape past its elements", load(UOp.range(17, 0), x.reshape((4, 4))))
    lcl = UOp.placeholder((8,), dtypes.uint32, slot=0, addrspace=AddrSpace.LOCAL)
    lidx = UOp.special(10, "lidx0")
    case("a load from a local buffer past its end", load(lidx, lcl))
    case("a load from a local buffer within its end", load(lidx.valid(lidx < 8), lcl))
    case("a load from a buffer after a store", load(UOp.const(20), x.after(store(UOp.const(0)))))
    case("a store at the last element", store(UOp.const(15)))
    case("a store past the last element", store(UOp.const(16)))
    case("a store over a guarded variable", store(v.valid((v >= 0) & (v < 16))))
    case("a store over a variable guarded on one side", store(v.valid(v < 20)))
    case("a gated store past the last element", x.index(UOp.const(16)).store(i32(0), flag("g")))
    case("a gated store within the buffer", x.index(UOp.const(3)).store(i32(0), flag("g")))
    case("a store to a shrink past the last element",
         UOp(Ops.SHRINK, src=(x, UOp.const(20), UOp.const(24))).store(i32(0)))
    return case


BOUNDS = bounds_cases()


@graph
def bounds_nodes(): return BOUNDS.sink()


@table
def bounds():
    rows = []
    for name, u in zip(BOUNDS.names, BOUNDS.nodes):
        with Context(CHECK_OOB=0): unchecked = verdict(spec_shared, u)
        with Context(CHECK_OOB=1): checked = verdict(spec_shared, u)
        rows.append((name, unchecked, checked))
    return ["case", "unchecked", "checked"], rows


# type_verify

def failure_cases():
    case = Cases()
    x, y = buf(0), buf(1)
    ld = x.index(UOp.const(0)).load()
    bad = UOp(Ops.ADD, src=(cvar("a", dtypes.int32), fvar("b")))
    case("a program whose sum mixes int32 and float32", UOp.sink(y.index(UOp.const(0)).store(bad)))
    case("a graph whose last node fails", UOp(Ops.GROUP, src=(ld,)))
    case("a graph whose first node fails", UOp.sink(UOp(Ops.CONST, src=(UOp.const(1),), arg=2).alu(Ops.ADD, UOp.const(1))))
    case("a reduce by sine", UOp.sink(UOp(Ops.REDUCE, src=(x,), arg=(Ops.SIN, 1))))
    case("a selection past the shards", UOp.sink(UOp.param(6, dtypes.float32, 16, device=("CPU:0", "CPU:1")).mselect(2)))
    case("a copy to a disk", UOp.sink(UOp(Ops.COPY, src=(x,), arg="DISK:/tmp/x")))
    case("a load with an int32 alternative", UOp.sink(x.index(UOp.const(0)).load(i32(0), flag("g"))))
    case("a where with a float constant", UOp.sink(UOp(Ops.WHERE, src=(flag("c"), cvar("a", dtypes.int32), f32(1.5)))))
    case("a copy to two devices without its range", UOp.sink(UOp(Ops.COPY, src=(x,), arg=CPU2)))
    case("an alu buffer", UOp.sink(UOp(Ops.BUFFER, arg=ParamArg(5, dtypes.float32, 4, addrspace=AddrSpace.ALU))))
    case("an and of weak floats", UOp.sink(UOp(Ops.AND, src=(UOp.const(1.0), UOp.const(-0.0)))))
    case("a float constant with a source", UOp.sink(UOp(Ops.CONST, src=(UOp.const(1),), arg=UOp.const(2.5).arg)))
    case("a launch dimension over a weak integer", UOp.sink(UOp.special(8, "gidx0")))
    case("a call whose body fails", UOp.sink(UOp.sink(y.index(UOp.const(0)).store(bad)).call(x, y)))
    return case


FAILURES = failure_cases()


@graph
def failures_nodes(): return FAILURES.sink()


def failure(u, spec, enter_calls=True):
    try:
        type_verify(u, spec, enter_calls=enter_calls)
    except RuntimeError as e:
        return str(e)
    return "None"


@table
def failures():
    rows = [(name, failure(u, spec_tensor), failure(u, spec_program), failure(u, spec_tensor, enter_calls=False))
            for name, u in zip(FAILURES.names, FAILURES.nodes)]
    return ["case", "tensor", "program", "tensor_outside_calls"], rows


# The graph type_verify prints before failing, when DEBUG is 3 or more
@text
def debug_listing():
    return listing(list(UOp.sink(UOp(Ops.AND, src=(fvar("a"), fvar("b")))).toposort()))
