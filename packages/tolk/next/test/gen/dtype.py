"""Goldens of tinygrad/dtype.py: the dtypes, their promotion, their constants and their casts.

A cell that holds a value is its repr: `True`, `-128`, `1.5`, `inf`, `nan`. A
float given as an input bit for bit is its hex (`0x1.8000000000000p+0`). A call
that raises is `raises` and the exception's class name.
"""

import math
import struct

from golden import table
from tinygrad.dtype import (DType, DTypes, Invalid, bitcast, can_lossless_cast, commit_int, dtypes, from_storage_scalar,
                            least_upper_dtype, least_upper_float, strong_dtype, sum_acc_dtype, to_dtype, to_storage_scalar,
                            truncate, weak_dtype)

# Every dtype that takes part in promotion, in priority order.
SCALARS = sorted([*dtypes.all, *dtypes.weaks])
DTYPES = [dtypes.void, *SCALARS]


def attempt(fn, *args):
    try:
        return fn(*args)
    except Exception as e:
        return f"raises {type(e).__name__}"


def f32(bits): return struct.unpack("<f", struct.pack("<I", bits))[0]


INTS = [0, 1, -1, 2, 127, 128, -128, -129, 255, 256, 32767, 32768, -32768, -32769, 65535, 65536, 2**24 + 1,
        2**31 - 1, 2**31, -2**31, -2**31 - 1, 2**32 - 1, 2**32, 2**53 + 1, 0x12345678ABCDEF01, 2**63 - 1, 2**63,
        -2**63, -2**63 - 1, 2**64 - 1, 2**64, 2**70, -2**70, 2**100,
        2**1024 - 2**970 - 1, -(2**1024 - 2**970 - 1), 2**1024 - 2**970, -(2**1024 - 2**970), 2**1100, -2**1100]

FLOATS = [
    0.0, -0.0, 0.5, -0.5, 1.0, 1.1, 1.5, 2.5, -2.5, 3.1, 0.1, 1 / 3, 1e-3, -777.777, 1234.0, 10000.0, -10000.0, 23456.0, 30000.0, -30000.0, 60000.0, -60000.0,
    100000.0, -1e6,
    # half: its largest finite value, the overflow tie, and its smallest subnormal with the ties around it
    65504.0, -65504.0, 65519.999, -65519.999, 65520.0, -65520.0, 1e-8, 2**-24, 2**-25, 2**-25 * (1 + 2**-52), -1.5 * 2**-25, 3e-8, 6.1035e-05,
    # bfloat16: ties to even below and above 1, rounding through float32, and its overflow edge
    f32(0x3F807000), f32(0x3F80C000), f32(0x3F808000), f32(0x3F818000), f32(0x41238000), f32(0xC1468000),
    1.00390625, 1.00390625 * (1 + 2**-52), 1 + 2**-8 + 2**-40, 1.0078125, f32(0x7F7F7FFF), f32(0x7F7F8000), f32(0x7F7FC000), 3.3895313892515355e38, -3.3895313892515355e38,
    3.3895313892515355e38 * 1.00001, -3.3895313892515355e38 * 1.00001, 3.3895313892515355e38 * 2,
    # float32: its largest finite value, overflow, and its smallest subnormal
    3.4028234663852886e38, 3.4028235677973366e38, 1e39, -1e39, 1e-45, 2**-149, 2**-150,
    # 8-bit floats: largest finite values, overflow thresholds, subnormals and their ties
    448.0, 464.0, 464.00000000000006, 480.0, 500.0, -500.0, 240.0, 247.0, 248.0, 57344.0, 57343.0, 61439.0, 61440.0,
    2**-6, 2**-7, 2**-9, 2**-10, 3 * 2**-11, 2**-10 * (1 + 2**-52), 2**-16, 2**-17, 3 * 2**-18, 2**-15, 2**-8, 2**-14,
    1.0625, 1.1875, 1.125, 1.375, 5 * 2**-10, 5 * 2**-11, 5 * 2**-17, 5 * 2**-18,
    # float64 extremes
    3.5e38, 1e300, 1e-40, 1e-320, 1.7976931348623157e308, 5e-324, -5e-324,
    float("inf"), float("-inf"), float("nan"), -float("nan"),
]



def value(x):
    """The cell of an input value: its repr, except that a NaN keeps its sign."""
    return "-nan" if isinstance(x, float) and math.isnan(x) and math.copysign(1, x) < 0 else repr(x)


VALUES = list({value(x): x for x in [True, False, *INTS, *FLOATS]}.values())


@table
def properties():
    columns = ["dtype", "priority", "bitsize", "itemsize", "name", "fmt", "min", "max",
               "is_int", "is_float", "is_unsigned", "is_bool"]
    return columns, [
        (dt, dt.priority, dt.bitsize, dt.itemsize, dt.name, dt.fmt, dt.min, dt.max,
         dtypes.is_int(dt), dtypes.is_float(dt), dtypes.is_unsigned(dt), dtypes.is_bool(dt))
        for dt in DTYPES]


@table
def projections():
    columns = ["dtype", "weak", "strong", "least_upper_float", "sum_acc", "storage_fmt"]
    storage_fmt = lambda dt: 'H' if dt == dtypes.bfloat16 else 'B' if dt in dtypes.fp8s else dt.fmt  # noqa: E731
    return columns, [(dt, weak_dtype(dt), strong_dtype(dt), attempt(least_upper_float, dt), attempt(sum_acc_dtype, dt),
                      storage_fmt(dt)) for dt in DTYPES]


@table
def groups():
    names = ["fp8_ocp", "fp8_fnuz", "fp8s", "floats", "uints", "sints", "ints", "weaks", "all"]
    return ["group", "members"], [(name, " ".join(map(repr, getattr(dtypes, name)))) for name in names]


@table
def names():
    named = sorted(name for name, dt in vars(DTypes).items() if isinstance(dt, DType))
    return ["name", "dtype"], [(name, to_dtype(name)) for name in [*named, "default_float", "default_int"]]


@table
def least_upper():
    return ["a", "b", "least_upper"], [(a, b, least_upper_dtype(a, b)) for a in SCALARS for b in SCALARS]


@table
def least_upper_triples():
    """The triples whose least upper bound is not the fold of pairwise bounds."""
    fold = lambda a, b, c: least_upper_dtype(least_upper_dtype(a, b), c)  # noqa: E731
    return ["a", "b", "c", "least_upper", "folded"], [
        (a, b, c, least_upper_dtype(a, b, c), fold(a, b, c)) for a in SCALARS for b in SCALARS for c in SCALARS
        if least_upper_dtype(a, b, c) != fold(a, b, c)]


@table
def lossless_cast():
    return ["from", "to", "lossless"], [(a, b, can_lossless_cast(a, b)) for a in SCALARS for b in SCALARS]


@table
def finfo():
    return ["dtype", "exponent", "mantissa"], [(dt, *dtypes.finfo(dt)) for dt in dtypes.floats]


# Constants

@table
def of_const():
    return ["value", "dtype"], [(value(x), dtypes.from_py(x)) for x in [True, False, Invalid, 0, 2**64, 0.0, float("nan")]]


@table
def of_consts():
    lists = [[], [True], [False, True], [Invalid], [Invalid, True], [1], [True, 2], [True, 3.0], [2, 3.0], [True, 2, 3.0],
             [1.5], [Invalid, 1.5], [0, 2**31 - 1], [-2**31, 0], [2**31], [-2**31 - 1], [2**63 - 1], [2**63],
             [2**64 - 1], [-1, 2**63], [2**64], [-2**63 - 1], [2**64, 2**64], [True, 2**64], [Invalid, 2**64],
             [Invalid, 1], [False, 2**31]]
    return ["consts", "dtype"], [(repr(xs), attempt(dtypes.from_py, xs)) for xs in lists]


@table
def commit():
    bounds = [(0, 0), (-128, 127), (-129, 0), (0, 255), (0, 2**31 - 1), (-2**31, 0), (0, 2**31), (-2**31 - 1, 0),
              (0, 2**63 - 1), (-2**63, 2**63 - 1), (0, 2**63), (0, 2**64 - 1), (-1, 2**63), (0, 2**64), (-2**63 - 1, 0),
              (-2**63, -2**63), (2**64 - 1, 2**64 - 1), (2**64, 2**64), (-2**63 - 1, -2**63 - 1)]
    defaults = [None, dtypes.int8, dtypes.uint8, dtypes.int16, dtypes.int64]
    return ["lo", "hi", "default_int", "dtype"], [(lo, hi, d, attempt(commit_int, lo, hi, d))
                                                  for lo, hi in bounds for d in defaults]


@table
def const():
    xs = [True, False, 0, 1, -1, 300, -300, 2**53 + 1, 2**64, 2**1100, 0.0, -0.0, 1.5, -1.5, 2.5, 1e-8, 70000.0, 1e39, -1e39,
          float("inf"), float("-inf"), float("nan"), Invalid]
    return ["dtype", "value", "const"], [(dt, value(x), attempt(lambda: str(dt.const(x)))) for dt in DTYPES for x in xs]


@table
def const_repr():
    floats = [0.0, -0.0, 1.0, -1.0, 0.1, 1.5, 100.0, 0.30000000000000004, 12345.678, 1e15, 9007199254740993.0, 9999999999999998.0, 1234567890123456.5, 1e16,
              1.5e16, 123456789012345678.0, 1e22, 1e100, 1e-4, 0.0001234, 1e-5, 2.5e-05, -1e-7, 5e-324,
              1.7976931348623157e308, float("inf"), float("-inf"), float("nan")]
    consts = [(x.hex(), dtypes.float64.const(x)) for x in floats]
    consts += [(repr(x), dtypes.weakint.const(x)) for x in [0, -1, 2**64, -2**799]]
    consts += [(repr(x), dtypes.bool.const(x)) for x in [True, False]] + [("Invalid", Invalid)]
    return ["const", "printed"], [(c, str(x)) for c, x in consts]


# Casts

@table
def truncation():
    return ["dtype", "value", "truncated", "storage"], [
        (dt, value(x), attempt(truncate[dt], x), attempt(to_storage_scalar, x, dt)) for dt in dtypes.all for x in VALUES]


@table
def decode():
    bf16 = [0x0000, 0x8000, 0x0001, 0x007F, 0x0080, 0x3F80, 0x3F81, 0xBF80, 0x4049, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC0,
            0x7F81, 0xFFFF, 0x1_3F80]
    rows = [(dt, b, from_storage_scalar(b, dt)) for dt in dtypes.fp8s for b in range(256)]
    return ["dtype", "storage", "value"], rows + [(dtypes.bfloat16, b, from_storage_scalar(b, dtypes.bfloat16)) for b in bf16]


@table
def reencode():
    """A word bitcast to a float and back, for every NaN word of the 8-bit floats and a few others."""
    word = {1: dtypes.uint8, 2: dtypes.uint16}
    def nan_words(dt):
        return [w for w in range(256) if math.isnan(bitcast(w, dtypes.uint8, dt))]
    words = {dt: sorted({0x00, 0x7E, 0x80, *nan_words(dt)}) for dt in dtypes.fp8s}
    words[dtypes.bfloat16] = [0x0000, 0x3F80, 0x7F80, 0x7FC0, 0x7FC1, 0x7F81, 0xFF81, 0xFFC5, 0x7FFF, 0xFFFF]
    words[dtypes.float16] = [0x0000, 0x3C00, 0x7C00, 0x7E00, 0x7E01, 0x7C01, 0xFC01, 0xFE05, 0x7FFF, 0xFFFF]
    return ["dtype", "word", "reencoded"], [
        (dt, w, bitcast(bitcast(w, word[dt.itemsize], dt), dt, word[dt.itemsize])) for dt, ws in words.items() for w in ws]


def bitcast_values(dt):
    if dt == dtypes.bool: return [False, True, 0.0, -0.0, 1.5, -2.0, float("inf"), float("nan")]
    if dtypes.is_int(dt): return [*sorted({0, 1, dt.max, dt.min, dt.max // 3, -1 if dt.min else 2, dt.max + 1, dt.min - 1}), 1.5]
    return [0.0, -0.0, 1.0, -2.0, 1.5, 0.1, 448.0, 1e5, 1e39, -1e39, float("inf"), float("-inf"), float("nan"), 1]


@table
def bitcasts():
    pairs = [(a, b) for a in dtypes.all for b in dtypes.all if a.itemsize == b.itemsize]
    return ["from", "to", "value", "bitcast"], [(a, b, value(x), attempt(bitcast, x, a, b))
                                                for a, b in pairs for x in bitcast_values(a)]


# Values

OPERANDS = [True, False, 0, 1, -1, 3, 2**53 - 1, 2**53, 2**53 + 1, 2**63, 2**64, 2**1100, -2**1100,
            0.0, -0.0, 0.5, -1.5, 2.0**53, 1e308, -1e308, float("inf"), float("-inf"), float("nan")]


@table
def values():
    """Python's comparisons and arithmetic on every pair of operands."""
    ops = {"lt": lambda a, b: a < b, "le": lambda a, b: a <= b, "eq": lambda a, b: a == b,
           "ne": lambda a, b: a != b, "min": min, "max": max,
           "add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b}
    return ["a", "b", *ops], [(value(a), value(b), *(attempt(op, a, b) for op in ops.values()))
                              for a in OPERANDS for b in OPERANDS]


@table
def negated():
    return ["a", "negated"], [(value(a), -a) for a in OPERANDS]
