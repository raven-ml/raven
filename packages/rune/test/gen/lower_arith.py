"""The kernels tinygrad schedules for the elementwise operations whose
lowering agrees with it: float arithmetic, comparisons, selections, bitwise
operations on integers, conversions between floats and bit reinterpretations.
Each operand is a 4x4 buffer on the CPU."""

from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp

from golden import graph


def operand(dtype=dtypes.float32):
    return Tensor.empty(4, 4, dtype=dtype, device="CPU")


def kernels(t):
    """The kernel sinks of the schedule that realizes `t`, in order."""
    linear, _ = t.linear_with_vars()
    return UOp.sink(*[call.src[0] for call in linear.src])


@graph
def neg(): return kernels(-operand())
@graph
def recip(): return kernels(operand().reciprocal())
@graph
def sqrt(): return kernels(operand().sqrt())
@graph
def trunc(): return kernels(operand().trunc())
@graph
def ceil(): return kernels(operand().ceil())
@graph
def floor(): return kernels(operand().floor())
@graph
def add(): return kernels(operand() + operand())
@graph
def sub(): return kernels(operand() - operand())
@graph
def mul(): return kernels(operand() * operand())
@graph
def equal(): return kernels(operand() == operand())
@graph
def not_equal(): return kernels(operand() != operand())
@graph
def less(): return kernels(operand() < operand())
@graph
def where(): return kernels((operand() < operand()).where(operand(), operand()))
@graph
def bitwise_and(): return kernels(operand(dtypes.int32) & operand(dtypes.int32))
@graph
def bitwise_or(): return kernels(operand(dtypes.int32) | operand(dtypes.int32))
@graph
def bitwise_xor(): return kernels(operand(dtypes.int32) ^ operand(dtypes.int32))
@graph
def cast_to_half(): return kernels(operand().cast(dtypes.float16))
@graph
def cast_to_double(): return kernels(operand().cast(dtypes.float64))
@graph
def bitcast(): return kernels(operand().bitcast(dtypes.int32))


# Correctly rounded double-precision results, from mpmath at 200 bits, of each
# transcendental function over a sweep of its domain and its special values.
# Where the result is not a finite nonzero real, C's function states it.

import math

import mpmath
import numpy as np

from golden import table

mpmath.mp.prec = 200


def sweep(lo, hi, n):
    """Magnitudes from 2^lo to 2^hi, alternating signs, then the special values."""
    xs = []
    for i in range(n):
        e = lo + (hi - lo) * i / n
        m = 2.0 ** e * (1 + (i % 97) / 97)
        xs.append(m if i % 2 == 0 else -m)
    return xs + [0.0, -0.0, math.inf, -math.inf, math.nan]


def between(a, b, n):
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def exact(f, c, *xs):
    """f(xs) correctly rounded, or C's c(xs) where that is not a finite
    nonzero real."""
    with np.errstate(all="ignore"):
        c_value = float(c(*[np.float64(x) for x in xs]))
    if not all(math.isfinite(x) for x in xs) or c_value == 0 or math.isnan(c_value):
        return c_value
    r = f(*[mpmath.mpf(x) for x in xs])
    return math.nan if isinstance(r, mpmath.mpc) else float(r)


def function_table(f, c, *columns):
    names = ["x", "y"][: len(columns)] + ["expected"]
    return names, [[*(x.hex() for x in xs), exact(f, c, *xs).hex()] for xs in zip(*columns)]


EVERYWHERE = sweep(-40, 12, 600) + between(-10, 10, 201)
UNIT = between(-1, 1, 401) + sweep(-40, 1, 200)


def c_pow(x, y):
    return np.power(x, y)


@table
def exp64():
    return function_table(mpmath.exp, np.exp, between(-750, 750, 601) + sweep(-40, 10, 200))
@table
def log64():
    return function_table(mpmath.log, np.log, [abs(x) for x in EVERYWHERE])
@table
def log1p64():
    return function_table(mpmath.log1p, np.log1p, EVERYWHERE + between(-0.999, 0.999, 201) + [-1.0])
@table
def expm1_64():
    return function_table(mpmath.expm1, np.expm1, between(-750, 750, 601) + sweep(-40, 10, 200) + between(-2, 2, 201))
# Beyond the exact parts of pi/2 that rune subtracts (2^12 in float32, 2^22 in
# float64), up to the greatest double: a sweep, the doubles nearest multiples
# of pi/2, whose remainders are tiny, and the double of the smallest remainder
# of all. The results take a precision above the argument's exponent.

def near_quarter_turns():
    with mpmath.workprec(1200):
        return [float(mpmath.mpf(k) * mpmath.pi / 2) for j in range(13, 1020, 17) for k in (2 ** j + 1, 3 * 2 ** j - 1)]


FAR = sweep(12, 1020, 400) + near_quarter_turns() + [6381956970095103 * 2.0 ** 797]


def precise(f):
    def g(x):
        with mpmath.workprec(1200):
            return f(x)
    return g


@table
def sin_far64():
    return function_table(precise(mpmath.sin), np.sin, FAR)
@table
def cos_far64():
    return function_table(precise(mpmath.cos), np.cos, FAR)
@table
def tan_far64():
    return function_table(precise(mpmath.tan), np.tan, FAR)
@table
def sin64():
    return function_table(mpmath.sin, np.sin, EVERYWHERE)
@table
def cos64():
    return function_table(mpmath.cos, np.cos, EVERYWHERE)
@table
def tan64():
    return function_table(mpmath.tan, np.tan, EVERYWHERE)
@table
def asin64():
    return function_table(mpmath.asin, np.arcsin, UNIT)
@table
def acos64():
    return function_table(mpmath.acos, np.arccos, UNIT)
@table
def atan64():
    return function_table(mpmath.atan, np.arctan, EVERYWHERE)
@table
def sinh64():
    return function_table(mpmath.sinh, np.sinh, between(-720, 720, 601) + sweep(-40, 6, 200))
@table
def cosh64():
    return function_table(mpmath.cosh, np.cosh, between(-720, 720, 601) + sweep(-40, 6, 200))
@table
def tanh64():
    return function_table(mpmath.tanh, np.tanh, EVERYWHERE)
@table
def erf64():
    return function_table(mpmath.erf, math.erf, EVERYWHERE)
@table
def atan2_64():
    return function_table(mpmath.atan2, np.arctan2, EVERYWHERE, EVERYWHERE[::-1])
@table
def pow64():
    n = len(EVERYWHERE)
    return function_table(mpmath.power, c_pow, [abs(x) for x in EVERYWHERE], between(-40, 40, n))
@table
def pow_integral64():
    return function_table(mpmath.power, c_pow, between(-3, 3, 1001), [float(round(v)) for v in between(-60, 60, 1001)])

