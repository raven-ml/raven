"""Goldens of tinygrad/codegen/decomp/transcendental.py: the graph each
function builds for each float type, the rewrites of its patterns, and the
values its graphs compute.

The input of a function is a scalar parameter, `UOp.param(slot, dtype)`, so
that nothing folds. Values are computed from the graph itself, node by node, as
tinygrad's Python emulator computes them, and written as `float.hex` prints
them.
"""

import fractions
import math
import random
import struct

from golden import graph, table
from tinygrad.codegen.decomp.transcendental import (cody_waite_reduction, exponent_bias, frexp,
                                                    get_transcendental_patterns, payne_hanek_reduction, pow2if,
                                                    rintk, shl, shr, xexp2, xlog2, xpow, xsin)
from tinygrad.dtype import DType, bitcast, dtypes, truncate
from tinygrad.uop.ops import GroupOp, Ops, UOp, exec_alu, graph_rewrite

FLOATS = {"half": dtypes.half, "float": dtypes.float, "double": dtypes.double}
INTS = {"short": (dtypes.short, dtypes.half), "int": (dtypes.int, dtypes.float), "long": (dtypes.long, dtypes.double)}


def x(dtype, slot=0):
    return UOp.param(slot, dtype)


def declare(name, fn):
    fn.__name__ = name
    graph(fn)


# Graphs of each function, for each float type

for name, dt in FLOATS.items():
    declare(f"xsin_{name}", lambda dt=dt: xsin(x(dt)).sink())
    declare(f"xsin_fast_{name}", lambda dt=dt: xsin(x(dt), fast=True).sink())
    declare(f"xsin_switch_over_{name}", lambda dt=dt: xsin(x(dt), switch_over=100.0).sink())
    declare(f"xexp2_{name}", lambda dt=dt: xexp2(x(dt)).sink())
    declare(f"xlog2_{name}", lambda dt=dt: xlog2(x(dt)).sink())
    declare(f"xpow_{name}", lambda dt=dt: xpow(x(dt), x(dt, 1)).sink())
    declare(f"frexp_{name}", lambda dt=dt: UOp.sink(*frexp(x(dt))))
    declare(f"payne_hanek_{name}", lambda dt=dt: UOp.sink(*payne_hanek_reduction(x(dt))))
    declare(f"cody_waite_{name}", lambda dt=dt: UOp.sink(*cody_waite_reduction(x(dt))))
    declare(f"rintk_{name}", lambda dt=dt: rintk(x(dt)).sink())

for name, (it, ft) in INTS.items():
    declare(f"pow2if_{name}", lambda it=it, ft=ft: pow2if(x(it), ft).sink())


@graph
def shifts():
    i, f = x(dtypes.int), x(dtypes.float, 1)
    return UOp.sink(shl(i, 3), shr(i, 3), shl(i, 0), shr(i, 0), shl(f, 2), shr(f, 2))


# Rewrites of the patterns

def transcendentals(dtype):
    d = x(dtype)
    return UOp.sink(d.exp2(), d.log2(), d.sin(), d.sqrt())


def rewrite(dtype, ops, force):
    return graph_rewrite(transcendentals(dtype), get_transcendental_patterns(ops, force))


ALL = (Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.SQRT)
for name, dt in {**FLOATS, "bfloat16": dtypes.bfloat16, "fp8e4m3": dtypes.fp8e4m3,
                 "fp8e5m2fnuz": dtypes.fp8e5m2fnuz}.items():
    declare(f"patterns_none_{name}", lambda dt=dt: rewrite(dt, (), False))
declare("patterns_all_float", lambda: rewrite(dtypes.float, ALL, False))
declare("patterns_all_forced_float", lambda: rewrite(dtypes.float, ALL, True))
declare("patterns_exp2_log2_float", lambda: rewrite(dtypes.float, (Ops.EXP2, Ops.LOG2), False))
declare("patterns_sqrt_bfloat16", lambda: rewrite(dtypes.bfloat16, (Ops.EXP2, Ops.LOG2, Ops.SIN), False))


@table
def exponent_biases():
    rows = []
    for dt in dtypes.all:
        try:
            bias = exponent_bias(dt)
        except Exception as e:
            bias = f"raises {type(e).__name__}"
        rows.append((dt, bias))
    return ["dtype", "exponent_bias"], rows


# Values

def evaluate(root, inputs):
    """The values of `root`'s sources under `inputs`, a value per parameter
    slot, as tinygrad's emulator computes each node."""
    values = {}
    for u in root.toposort():
        src = [values[s] for s in u.src]
        if u.op is Ops.SINK: continue
        if u.op is Ops.CONST: values[u] = u.arg
        elif u.op is Ops.PARAM: values[u] = inputs[u.arg.slot]
        elif u.op is Ops.CAST: values[u] = truncate.get(u.dtype, lambda v: v)(u.dtype.const(src[0]))
        elif u.op is Ops.BITCAST: values[u] = bitcast(src[0], u.src[0].dtype, u.dtype)
        elif u.op in GroupOp.ALU: values[u] = exec_alu(u.op, u.dtype, src)
        else: raise ValueError(f"cannot evaluate {u.op}")
    return [values[s] for s in root.src]


def cell(v):
    return v.hex() if isinstance(v, float) else repr(v)


def special(dt):
    info = {dtypes.half: (6.103515625e-05, 5.960464477539063e-08, 65504.0),
            dtypes.float: (1.1754943508222875e-38, 1.401298464324817e-45, 3.4028234663852886e+38),
            dtypes.double: (2.2250738585072014e-308, 5e-324, 1.7976931348623157e+308)}[dt]
    tiny, sub, big = info
    base = [0.0, 1.0, 0.5, 2.0, 3.0, 0.75, 1.5, 10.0, 100.0, 1000.0, 1e-4, 6e-5, 1e-5, 9e-7, math.pi / 4, math.pi / 2,
            math.pi, 2 * math.pi, 12 * math.pi, 12 * math.pi + 0.1, 12 * math.pi - 0.1, 25.0, 29.999, 30.0, 30.001, 35.0,
            39800.0, 1e5, 1e7 + 0.3, 76806992.6561, 1e8, 1e10, 1e20, 88.7, 127.9, 128.0, 149.0, 150.0, 151.0, 709.5, 1023.9, 1024.0, 1e300,
            11.0, 15.9, 16.0, 22.9, 23.0, 24.0, tiny, sub, big, tiny * 1e10, tiny * 1e20, tiny * 1e30]
    values = base + [-v for v in base] + [math.inf, -math.inf, math.nan]
    return [truncate.get(dt, float)(v) for v in values]


def drawn(dt, n, seed):
    """`n` values of `dt` of every magnitude, drawn from their bits."""
    rng = random.Random(seed)
    fmt, bits = {dtypes.half: ("e", 16), dtypes.float: ("f", 32), dtypes.double: ("d", 64)}[dt]
    ifmt = {16: "H", 32: "I", 64: "Q"}[bits]
    out = []
    while len(out) < n:
        v = struct.unpack(fmt, struct.pack(ifmt, rng.getrandbits(bits)))[0]
        if not math.isnan(v): out.append(v)
    return out


def inputs(dt, seed):
    seen, out = set(), []
    for v in special(dt) + drawn(dt, 150, seed):
        key = struct.pack("d", v) if not math.isnan(v) else b"nan"
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out


UNARY = {"xsin": xsin, "xsin_fast": lambda d: xsin(d, fast=True), "xexp2": xexp2, "xlog2": xlog2}


@table
def values():
    rows = []
    for fname, f in UNARY.items():
        for name, dt in FLOATS.items():
            root = f(x(dt)).sink()
            for i, v in enumerate(inputs(dt, f"{fname}/{name}")):
                if fname == "xsin_fast" and not abs(v) < 30.0: continue
                rows.append((fname, name, cell(v), cell(evaluate(root, {0: v})[0])))
    return ["function", "dtype", "x", "result"], rows


@table
def pow_values():
    rows = []
    for name, dt in FLOATS.items():
        root = xpow(x(dt), x(dt, 1)).sink()
        bases = [0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, -3.0, 10.0, math.inf, -math.inf, math.nan]
        exponents = [0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 3.0, 0.5, -0.5, 2.5, 10.0, math.inf, -math.inf]
        for b in bases:
            for e in exponents:
                try:
                    result = evaluate(root, {0: truncate[dt](b) if dt in truncate else b, 1: e})[0]
                except (ValueError, OverflowError):
                    continue
                rows.append((name, cell(b), cell(e), cell(result)))
    return ["dtype", "base", "exponent", "result"], rows


# The remainders of angles near multiples of pi/2, from pi to 1400 bits

def pi_scaled(bits):
    """pi * 2^bits, by Machin's formula in integers."""
    prec = bits + 64
    def arctan_inv(n):
        x, s, k, sign = (1 << prec) // n, 0, 0, 1
        while x:
            s += sign * (x // (2 * k + 1))
            x //= n * n
            k, sign = k + 1, -sign
        return s
    return (16 * arctan_inv(5) - 4 * arctan_inv(239)) >> 64


PI_BITS = 1400
PI_SCALED = pi_scaled(PI_BITS)


def reduced(x):
    """x = q * pi/2 + r with |r| <= pi/4: r rounded to a double, and q modulo 4."""
    num, den = x.as_integer_ratio()
    # x / (pi/2) = 2 num 2^PI_BITS / (den PI_SCALED), rounded to the nearest integer
    q = (4 * num * 2**PI_BITS + den * PI_SCALED) // (2 * den * PI_SCALED)
    r = fractions.Fraction(num, den) - fractions.Fraction(q * PI_SCALED, 2 * 2**PI_BITS)
    return float(r), q % 4


@table
def payne_hanek_near_multiples():
    rows = []
    integers = [355.0, 103993.0, 104348.0, 208341.0, 312689.0, 833719.0, 1146408.0, 4272943.0]
    for name, dt in (("float", dtypes.float), ("double", dtypes.double)):
        near = [k * math.pi / 2 for k in (10**6, 10**9 + 7, 10**15 + 3, 2.0**60, 2.0**100)] + [1e20, 1e30]
        if dt == dtypes.double: near += [6381956970095103 * 2.0**797, 1e22, 1e300, 1.7e308]
        for v in integers + near:
            v = truncate[dt](v)
            r, q = reduced(v)
            rows.append((name, cell(v), cell(r), q))
    return ["dtype", "x", "r", "quadrant"], rows
