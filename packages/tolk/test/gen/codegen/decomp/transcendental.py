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

import mpmath

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

# The functions are defined in float32 and float64 only.
WIDE = {name: dt for name, dt in FLOATS.items() if dt != dtypes.half}

for name, dt in WIDE.items():
    declare(f"xsin_{name}", lambda dt=dt: xsin(x(dt)).sink())
    declare(f"xsin_fast_{name}", lambda dt=dt: xsin(x(dt), fast=True).sink())
    declare(f"xsin_switch_over_{name}", lambda dt=dt: xsin(x(dt), switch_over=100.0).sink())
    declare(f"xexp2_{name}", lambda dt=dt: xexp2(x(dt)).sink())
    declare(f"xlog2_{name}", lambda dt=dt: xlog2(x(dt)).sink())

for name, dt in FLOATS.items():
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
        # the empty size of a scalar parameter
        elif u.op is Ops.STACK and not u.src: values[u] = ()
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
        for name, dt in WIDE.items():
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

PI_BITS = 1400
# pi * 2^PI_BITS, rounded down
with mpmath.workprec(PI_BITS + 64):
    PI_SCALED = int(mpmath.floor(mpmath.ldexp(mpmath.pi, PI_BITS)))


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
        # float32's largest exponents take the last words of its table
        if dt == dtypes.float: near += [2.0**126 * math.pi / 2, 1e38, 3.4e38]
        if dt == dtypes.double: near += [6381956970095103 * 2.0**797, 1e22, 1e300, 1.7e308]
        for v in integers + near:
            v = truncate[dt](v)
            r, q = reduced(v)
            rows.append((name, cell(v), cell(r), q))
    return ["dtype", "x", "r", "quadrant"], rows


# Correctly rounded exp2, log2 and sin, against which the accuracy of tolk's
# functions is measured: mpmath's values, at a precision doubled until the
# value's error bound rounds to one value of the type (Ziv's strategy), or the
# exact value where it is a power of two or an integer.

# precision, least and greatest exponents
FORMATS = {"half": (11, -14, 15), "bfloat16": (8, -126, 127), "float": (24, -126, 127), "double": (53, -1022, 1023)}


def round_to(v, fmt):
    """v rounded to nearest, ties to even, in fmt: its subnormals below the
    least normal, an infinity from the greatest value and half an ulp."""
    p, emin, emax = fmt
    if v == 0:
        return 0.0
    _, e = mpmath.frexp(v)
    ulp = mpmath.ldexp(1, max(e - 1, emin) - p + 1)
    r = mpmath.nint(v / ulp) * ulp
    if abs(r) >= mpmath.ldexp(1, emax + 1):
        return math.copysign(math.inf, v)
    return math.copysign(float(r), v)


ZIV_START, ZIV_LAST, ZIV_GUARD = 64, 1 << 16, 16


def correctly_rounded(f, x, fmt):
    prec = ZIV_START
    while prec <= ZIV_LAST:
        with mpmath.workprec(prec + ZIV_GUARD):
            v = f(mpmath.mpf(x))
            err = abs(v) * mpmath.ldexp(1, -prec)
            if (lo := round_to(v - err, fmt)) == round_to(v + err, fmt):
                return lo
        prec *= 2
    raise ArithmeticError(f"{x!r} is too near halfway between values")


def cr_exp2(x, fmt):
    p, emin, emax = fmt
    if math.isnan(x) or x == math.inf:
        return x
    # past these, 2^x is above the greatest value or below half the least
    if x > emax + 1:
        return math.inf
    if x < emin - p:
        return 0.0
    if x.is_integer():
        return round_to(mpmath.ldexp(1, int(x)), fmt)
    return correctly_rounded(lambda v: mpmath.exp(v * mpmath.ln2), x, fmt)


def cr_log2(x, fmt):
    if math.isnan(x) or x < 0:
        return math.nan
    if x == 0:
        return -math.inf
    if x == math.inf:
        return x
    m, e = math.frexp(x)
    if m == 0.5:
        return round_to(mpmath.mpf(e - 1), fmt)
    return correctly_rounded(lambda v: mpmath.log(v, 2), x, fmt)


def cr_sin(x, fmt):
    if not math.isfinite(x):
        return math.nan
    if x == 0:
        return x
    return correctly_rounded(mpmath.sin, x, fmt)


def of_bits(name, rng):
    if name == "double":
        return struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0]
    if name == "float":
        return struct.unpack("<f", struct.pack("<I", rng.getrandbits(32)))[0]
    if name == "bfloat16":
        return struct.unpack("<f", struct.pack("<I", rng.getrandbits(16) << 16))[0]
    return struct.unpack("<e", struct.pack("<H", rng.getrandbits(16)))[0]


def accuracy_inputs(fname, name, rng):
    """The inputs of fname in the type name: values from the type's bits (every
    magnitude, the subnormals, the zeros, the infinities and NaN), from the
    range where the result is neither 0, 1 nor infinite, and for the sine near
    multiples of a quarter turn, where it cancels; first those tinygrad's
    fuzzer found."""
    fmt = FORMATS[name]
    p, emin, emax = fmt
    def between(lo, hi): return lambda: round_to(mpmath.mpf(rng.uniform(lo, hi)), fmt)
    def quarter_turns(n): return lambda: round_to(mpmath.mpf(rng.randint(1, n) * math.pi / 2), fmt)
    least_normal = math.ldexp(1.0, emin)
    if fname == "exp2":
        fuzzed, draws = [], [between(emin - p - 1, emax + 2), between(-1.0, 1.0)]
    elif fname == "log2":
        fuzzed = [s * least_normal * scale for scale in (1.0, 1e10, 1e20, 1e30) for s in (1, -1)] + [0.0, 9e-7]
        draws = [between(0.5, 2.0), between(0.0, 1e-30)]
    else:
        fuzzed = [-35.0, -25.0, 25.0, 30.0, 35.0, 0.0, math.pi / 2, 2 * math.pi]
        draws = [between(-30.0, 30.0), quarter_turns(20), quarter_turns(40000 if name == "half" else 0x3fffffff)]
    draws.append(lambda: of_bits(name, rng))
    seen, out = set(), []
    for v in [round_to(mpmath.mpf(v), fmt) for v in fuzzed] + [rng.choice(draws)() for _ in range(ACCURACY_DRAWS)]:
        key = struct.pack("<d", v) if not math.isnan(v) else b"nan"
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out


ACCURACY_DRAWS = 400
REFERENCES = {"exp2": cr_exp2, "log2": cr_log2, "sin": cr_sin}


@table
def accuracy():
    rows = []
    for fname, reference in REFERENCES.items():
        for name, fmt in FORMATS.items():
            rng = random.Random(f"{fname}/{name}")
            for v in accuracy_inputs(fname, name, rng):
                rows.append((fname, name, cell(v), cell(reference(v, fmt))))
    return ["function", "dtype", "x", "result"], rows
