"""Independent references for nx's special functions, at mpmath's working
precision.

The incomplete gamma ratios are mpmath.gammainc's and the incomplete beta
ratios mpmath.betainc's, each the smaller tail, the other its complement;
e^-|x| I_nu(|x|) is mpmath.besseli's and log Phi far below 0 mpmath.hyperu's.
Where mpmath's hypergeometric series do not converge, at large shapes, a
tail is mpmath.quad of its defining integral, which check() holds to
mpmath's functions at seeded points both reach. A quantile is
mpmath.findroot on its tail from a bracket, and the tail's sign change
across the root's neighbours proves it.

Nothing here follows nx's algorithms: special_tables.py transcribes those
and proves their counts against these functions. correctly_rounded rounds a
reference to a dtype at two precisions that agree, doubling until they do.
"""

import math
import random
import struct

import mpmath
from mpmath import mp, mpf
from mpmath.libmp import NoConvergence


# Formats


class Format:
    def __init__(self, name, short, p, emin, emax):
        self.name, self.short = name, short
        self.p, self.emin, self.emax = p, emin, emax
        self.u = 2.0 ** -p
        self.eps = 2.0 ** (1 - p)
        self.tiny = 2.0 ** (emin - p + 1)
        self.min_normal = 2.0 ** emin
        self.max = (2.0 - 2.0 ** (1 - p)) * 2.0 ** emax


F32 = Format("float32", "f32", 24, -126, 127)
F64 = Format("float64", "f64", 53, -1022, 1023)
FORMATS = [F32, F64]


class Overflow(float):
    """An infinity a finite value rounds to past the format's range, which a
    golden writes as `overflow`: the largest finite float is one ulp from it,
    where an exact infinity is held exactly."""


def round_to(fmt, v):
    """`v`, an mpf, rounded to nearest even in `fmt`, as a Python float."""
    if mpmath.isnan(v):
        return math.nan
    if mpmath.isinf(v):
        return math.inf if v > 0 else -math.inf
    if v == 0:
        return 0.0
    sign = -1.0 if v < 0 else 1.0
    a = abs(v)
    _, exp = mpmath.frexp(a)
    e = int(exp) - 1
    if e > fmt.emax:
        return Overflow(sign * math.inf)
    q = max(e, fmt.emin) - (fmt.p - 1)
    scaled = mpmath.ldexp(a, -q)
    n = int(mpmath.floor(scaled))
    frac = scaled - n
    if frac > 0.5 or (frac == 0.5 and n % 2 == 1):
        n += 1
    r = math.ldexp(n, q) if q > -1100 else float(mpmath.ldexp(n, q))
    if r > fmt.max:
        return Overflow(sign * math.inf)
    return sign * r


def bits(fmt, x):
    if fmt is F32:
        return struct.unpack("<I", struct.pack("<f", x))[0]
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def of_bits(fmt, b):
    if fmt is F32:
        return struct.unpack("<f", struct.pack("<I", b))[0]
    return struct.unpack("<d", struct.pack("<Q", b))[0]


def next_up(fmt, x):
    """The least float of `fmt` above `x`, a float of `fmt`."""
    if math.isnan(x) or x == math.inf:
        return x
    if x == 0:
        return fmt.tiny
    b = bits(fmt, x)
    return of_bits(fmt, b + 1 if x > 0 else b - 1)


def next_down(fmt, x):
    return -next_up(fmt, -x)


def neighbours(fmt, c, k=8):
    """`c` rounded to `fmt` and the `k` floats on each side."""
    x = round_to(fmt, mpf(c))
    ups, downs, u, d = [], [], x, x
    for _ in range(k):
        u, d = next_up(fmt, u), next_down(fmt, d)
        ups.append(u)
        downs.append(d)
    return downs[::-1] + [x] + ups


def correctly_rounded(fmt, f, *args):
    """`f args` rounded to `fmt`, at precisions that agree on the rounding."""
    prec = fmt.p + 40
    while True:
        with mp.workprec(prec):
            r1 = round_to(fmt, f(*[mpf(a) for a in args]))
        with mp.workprec(2 * prec):
            r2 = round_to(fmt, f(*[mpf(a) for a in args]))
        if r1 == r2 or (math.isnan(r1) and math.isnan(r2)):
            return r2
        prec *= 2
        if prec > 20000:
            raise RuntimeError(f"no stable rounding for {f.__name__}{args}")


def hexf(x):
    if isinstance(x, Overflow):
        return "overflow" if x > 0 else "-overflow"
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "infinity" if x > 0 else "-infinity"
    return x.hex()


def scale(v):
    """A positive scale, rounded up to 4 significant digits."""
    if mpmath.isinf(v) or v > 1e300:
        return "infinity"
    s = float(v)
    if s == 0:
        return "0"
    if abs(s) < 1e-300:
        return "1e-300"
    e = int(mpmath.floor(mpmath.log10(abs(mpf(s)))))
    m = math.ceil(abs(s) / 10 ** (e - 3))
    return f"{m}e{e - 3}"


def exact(v):
    """v as an mpf, unrounded: mpf(v) of an mpf rounds it to the working
    precision."""
    return v if isinstance(v, mpmath.mpf) else mpf(v)


def one_minus(x):
    """1 - x exactly, for the complement's argument: rounded, 1 - 1e-30 would
    lose x."""
    return mpmath.fsub(1, x, exact=True)


def sigmoid(z):
    """1 / (1 + e^-z), relative to the working precision on either side."""
    return 1 / (1 + mpmath.exp(-z))


# Quadrature
#
# A tail of the gamma or the beta far past mpmath's series is the integral
# of its density from the argument away from the mean. With the density's
# logarithm at the argument taken out, the integrand e^phi(s), phi(0) = 0,
# is unimodal: it falls off over w, the smaller of the density's width and
# 1/|phi'(0)|, as fast as a Gaussian or, beside a tiny shape, as slowly as
# 1/s, across every scale up to the argument. In v, s = w (e^v - 1), both
# are smooth over a few units of v, which breakpoints at powers of two let
# tanh-sinh resolve.


def log_integral(phi, slope, width, end, bits):
    """log of the integral of e^phi(s) for s from 0 to `end` (infinity where
    None), phi unimodal, phi(0) = 0 and phi'(0) = `slope`, to `bits`
    relatively."""
    w = min(width, 1 / abs(slope)) if slope != 0 else width
    s_of = lambda v: w * mpmath.expm1(v)
    top = mpmath.inf if end is None else mpmath.log1p(end / w)
    points, v = [mpf(0)], mpf(1) / 2
    while v < top:
        points.append(v)
        # Past the mode the rest is below the working precision, and in two
        # more units of v, where s grows sevenfold, below its square: the
        # integral stops there, short of the exponents infinity would ask.
        if end is None and phi(s_of(v)) + v < -2 * mp.prec:
            top = v + 2
            break
        v *= 2
    points.append(top)

    def f(v):
        s = s_of(v)
        # Rounding can carry s past the end, where the density vanishes.
        if end is not None and s >= end:
            return mpf(0)
        return mpmath.exp(phi(s) + v)
    r, err = mpmath.quad(f, points, error=True)
    if not r > 0 or err > r * mpf(2) ** -bits:
        raise RuntimeError(f"quadrature error {err} of {r}")
    return mpmath.log(w * r)


def guard(*terms):
    """Bits that keep a sum of terms of these magnitudes, each then computed
    again at the working precision plus these bits, exact to the working
    precision, absolutely."""
    return int(mpmath.log(sum(abs(t) for t in terms) + 2, 2)) + 24


def gamma_quad(a, x, upper):
    """log Q(a, x) where upper, else log P(a, x): the integral of t^(a - 1)
    e^-t / Gamma(a) from x, in t = x + s, or up to x, in t = x - s."""
    bits = mp.prec + 8
    with mp.workprec(mp.prec + guard((a - 1) * mpmath.log(x), x, mpmath.loggamma(a))):
        lead = (a - 1) * mpmath.log(x) - x - mpmath.loggamma(a)
        width = mpmath.sqrt(a)
        if upper:
            phi = lambda s: (a - 1) * mpmath.log1p(s / x) - s
            l = log_integral(phi, (a - 1) / x - 1, width, None, bits)
        else:
            phi = lambda s: (a - 1) * mpmath.log1p(-s / x) + s
            l = log_integral(phi, 1 - (a - 1) / x, width, x, bits)
        return lead + l


def beta_quad(a, b, x, y, span=0):
    """log I_x(a, b), y = 1 - x: the integral of t^(a - 1) (1 - t)^(b - 1) /
    B(a, b) up to x, in t = x - s, relative to the working precision but
    `span`, the bits that only held x."""
    bits = mp.prec - span + 8
    sizes = ((a - 1) * mpmath.log(x), (b - 1) * mpmath.log(y), mpmath.loggamma(a), mpmath.loggamma(b),
             mpmath.loggamma(a + b))
    with mp.workprec(mp.prec + guard(*sizes)):
        lb = mpmath.loggamma(a) + mpmath.loggamma(b) - mpmath.loggamma(a + b)
        lead = (a - 1) * mpmath.log(x) + (b - 1) * mpmath.log(y) - lb
        width = mpmath.sqrt(a * b / (a + b) ** 3)
        phi = lambda s: (a - 1) * mpmath.log1p(-s / x) + (b - 1) * mpmath.log1p(s / y)
        return lead + log_integral(phi, (b - 1) / y - (a - 1) / x, width, x, bits)


# The incomplete gamma ratios
#
# P(a, x) and Q(a, x) = 1 - P for finite a, x > 0. The tail on x's side of a
# is taken first, the other where it passes 3/4, so that the one computed is
# below 3/4 and the other, log1p of its negation, loses at most two bits,
# exact to the working precision even where it rounds to 1. Past
# GAMMA_QUAD_FROM, mpmath's series cannot converge, and its sum for an
# integer a would take a terms: quadrature alone.

GAMMA_QUAD_FROM = 2.0 ** 40
IGAMMA = {}


def gamma_tail(a, x, upper):
    """Q(a, x) where upper, else P(a, x)."""
    if a > GAMMA_QUAD_FROM:
        return mpmath.exp(gamma_quad(a, x, upper))
    try:
        if upper:
            return mpmath.gammainc(a, x, mpmath.inf, regularized=True)
        return mpmath.gammainc(a, 0, x, regularized=True)
    except NoConvergence:
        return mpmath.exp(gamma_quad(a, x, upper))


def igamma(a, x):
    """(log P(a, x), log Q(a, x)) for finite a, x > 0."""
    a, x = exact(a), exact(x)
    key = (a, x, mp.prec)
    if key not in IGAMMA:
        with mp.workprec(mp.prec + 16):
            upper = x > a
            t = gamma_tail(a, x, upper)
            if t > 0.75:
                upper = not upper
                t = gamma_tail(a, x, upper)
            small, large = mpmath.log(t), mpmath.log1p(-t)
        IGAMMA[key] = (+large, +small) if upper else (+small, +large)
    return IGAMMA[key]


# The incomplete beta ratio
#
# I_x(a, b) and 1 - I_x(a, b) = I_(1-x)(b, a) for finite a, b > 0 and x in
# (0, 1), each from its own argument, so that x near 1 keeps its complement:
# the tail on x's side of the mean first, the other where it passes 3/4.
# mpmath's 2F1 fails to converge, or converges slowly, once both shapes pass
# BETA_QUAD_FROM, and where one does and x passes 0.8, where it transforms
# to 1 - x and cancels terms as large as 1/b for a tiny b: quadrature takes
# over there, and where mpmath fails, for a >= 1, where the integrand is
# bounded at t = 0.

BETA_QUAD_FROM = 2000
BETAINC = {}


def beta_lower(a, b, x, y, span=0):
    """I_x(a, b), y = 1 - x, `span` of the working precision's bits there
    only to hold x."""
    large = max(a, b) > BETA_QUAD_FROM
    if a < 1 or not (min(a, b) > BETA_QUAD_FROM or (large and x > 0.8)):
        try:
            return mpmath.betainc(a, b, 0, x, regularized=True)
        except (NoConvergence, ValueError):
            # mpmath's hypsum raises ValueError where it does not converge.
            pass
    if a < 1:
        raise RuntimeError(f"no incomplete beta at a = {a}, b = {b}, x = {x}")
    return mpmath.exp(beta_quad(a, b, x, y, span))


# mpmath takes a tail's argument u alone, rounds it to the working
# precision and forms 1 - u from it, so a tail from u > 1/2 needs u exact and
# the bits 1 - u spans past the precision. The arguments are made exact
# complements: the smaller as given, the larger 1 minus it, exactly, where
# that spans at most BETA_SPAN bits. A tail is computed directly from u up
# to 1/2, or from the larger argument where it is exact; where the other
# tail is wanted and cannot be computed, it is the complement of this one at
# a precision that keeps its digits.

BETA_SPAN = 4000
EXACT_FROM = mpf(2) ** -BETA_SPAN


def beta_tails(a, b, x, y):
    """(log I_x(a, b), log (1 - I_x(a, b))) with y = 1 - x, both given to the
    working precision."""
    a, b, x, y = exact(a), exact(b), exact(x), exact(y)
    key = (a, b, x, y, mp.prec)
    if key in BETAINC:
        return BETAINC[key]
    complement = min(x, y) >= EXACT_FROM
    if complement:
        x, y = (x, one_minus(x)) if x <= y else (one_minus(y), y)

    def span(upper):
        """The bits the tail's argument needs past the precision, None where
        it cannot be computed directly."""
        u, rest = (y, x) if upper else (x, y)
        if u <= 0.5:
            return 0
        bits = int(-mpmath.log(rest, 2)) + 8
        return bits if complement else None

    def tail(upper, extra):
        bits = span(upper)
        with mp.workprec(mp.prec + extra + bits):
            return beta_lower(b, a, y, x, bits) if upper else beta_lower(a, b, x, y, bits)

    upper = x > a / (a + b)
    if span(upper) is None:
        upper = not upper
    t = tail(upper, 16)
    if t > 0.75 and span(not upper) is not None:
        upper = not upper
        t = tail(upper, 16)
    with mp.workprec(mp.prec + 16):
        small, large = mpmath.log(t), mpmath.log1p(-t)
    extra = 32
    while t > 0.75:
        t = tail(upper, extra)
        with mp.workprec(mp.prec + extra):
            c = 1 - t
            if c > 0 and c >= mpf(2) ** (16 - extra):
                small, large = mpmath.log(t), mpmath.log(c)
                break
        extra *= 2
        if extra > 100000:
            raise RuntimeError(f"no complement at a = {a}, b = {b}, x = {x}")
    BETAINC[key] = (+large, +small) if upper else (+small, +large)
    return BETAINC[key]


def betainc(a, b, x):
    """(log I_x(a, b), log (1 - I_x(a, b))) at x in (0, 1)."""
    return beta_tails(a, b, x, one_minus(x))


def beta_logit(a, b, z):
    """(log I_x(a, b), log (1 - I_x(a, b))) at x = 1 / (1 + e^-z)."""
    return beta_tails(a, b, sigmoid(z), sigmoid(-z))


# Quantiles

QUANTILES = {}


def cached(f):
    """f, its results kept per argument and working precision."""
    def g(*args):
        key = (f.__name__, args, mp.prec)
        if key not in QUANTILES:
            QUANTILES[key] = f(*args)
        return QUANTILES[key]
    g.__name__ = f.__name__
    return g


def root(f, start, rising, width):
    """The root of f, monotone, rising or falling: mpmath.findroot from the
    bracket that steps doubling from `width` at `start` reach, bisected to
    `width`, the scale over which f is near linear, then checked by f's sign
    change across the root's neighbours at a relative 2^-(prec - 8)."""
    f0 = f(start)
    if f0 == 0:
        return start
    direction = -1 if (f0 > 0) == rising else 1
    lo, step = start, width
    while True:
        hi = lo + direction * step
        fh = f(hi)
        if fh == 0:
            return hi
        if (fh > 0) != (f0 > 0):
            break
        lo, f0, step = hi, fh, 2 * step
        if step > width * mpf(2) ** 400:
            raise RuntimeError("no bracket")
    while abs(hi - lo) > max(width, abs(lo) * mpf(2) ** -20):
        mid = (lo + hi) / 2
        fm = f(mid)
        if fm == 0:
            return mid
        if (fm > 0) == (f0 > 0):
            lo, f0 = mid, fm
        else:
            hi = mid
    r = mpmath.findroot(f, (min(lo, hi), max(lo, hi)), solver="anderson", maxsteps=400, verify=False)
    delta = mpf(2) ** (8 - mp.prec) * max(abs(r), 1)
    below, above = f(r - delta), f(r + delta)
    if below != 0 and above != 0 and (below > 0) == (above > 0):
        raise RuntimeError(f"findroot's {r} is no root")
    return r


@cached
def igamma_quantile(a, p, upper):
    """x with P(a, x) = p, or Q(a, x) = p where upper, for a > 0 and p in (0,
    1): the root in log x of the smaller tail's logarithm against log p."""
    a, p = exact(a), exact(p)
    small, tail = (p, upper) if p <= 0.5 else (1 - p, not upper)
    target = mpmath.log(small)

    def f(y):
        with mp.workprec(mp.prec + 32):
            return igamma(a, mpmath.exp(y))[1 if tail else 0] - target
    # log x spreads over about 1 / sqrt(a) about log a.
    return mpmath.exp(root(f, mpmath.log(a), not tail, min(1, 1 / mpmath.sqrt(a))))


@cached
def beta_quantile_logit(a, b, p, upper):
    """z = log (x / (1 - x)) with I_x(a, b) = p, or 1 - I_x(a, b) = p where
    upper, for a, b > 0 and p in (0, 1): the root of the smaller tail's
    logarithm against log p. Equal shapes have their median at z = 0, by
    symmetry, where tiny ones leave the tail too flat for a root to resolve."""
    a, b, p = exact(a), exact(b), exact(p)
    if a == b and p == 0.5:
        return mpf(0)
    small, tail = (p, upper) if p <= 0.5 else (1 - p, not upper)
    target = mpmath.log(small)

    def f(z):
        with mp.workprec(mp.prec + 32):
            return beta_logit(a, b, z)[1 if tail else 0] - target
    # z spreads over about sqrt(1/a + 1/b) about log (a / b).
    return root(f, mpmath.log(a) - mpmath.log(b), not tail, min(1, mpmath.sqrt(1 / a + 1 / b)))


def beta_quantile(a, b, p, upper):
    return sigmoid(beta_quantile_logit(a, b, p, upper))


def ndtri(p):
    """The standard normal's quantile at p in (0, 1)."""
    p = mpf(p)
    if p == 0.5:
        return mpf(0)
    small = min(p, 1 - p)
    target = mpmath.log(small)

    def f(x):
        with mp.workprec(mp.prec + 32):
            return log_ndtr(x) - target
    x = root(f, mpf(-1), True, mpf(1))
    return x if p < 0.5 else -x


# The others


def log_ndtr(x):
    """log Phi(x); below -100, by Phi(x) = e^(-x^2/2) U(1/2, 1/2, x^2/2) / (2
    sqrt pi), where mpmath's erfc no longer reaches (DLMF 7.11.4, 13.6.7)."""
    if x < -100:
        h = x * x / 2
        return -h + mpmath.log(mpmath.hyperu(0.5, 0.5, h) / (2 * mpmath.sqrt(mpmath.pi)))
    if x > 0:
        return mpmath.log1p(-mpmath.ncdf(-x))
    return mpmath.log(mpmath.ncdf(x))


def i0e(x):
    """e^-|x| I_0(|x|)."""
    a = abs(x)
    return mpmath.besseli(0, a) * mpmath.exp(-a)


def i1e(x):
    """e^-|x| I_1(x), odd."""
    a = abs(x)
    r = mpmath.besseli(1, a) * mpmath.exp(-a)
    return -r if x < 0 else r


# Checks


def check():
    """Holds the quadrature to mpmath.gammainc and mpmath.betainc at seeded
    points where both converge, each tail within 2^-100 relatively at 160
    bits: x near a and near the mean, and the beta's large shapes beside
    tiny ones."""
    rng = random.Random("oracle")
    tol = mpf(2) ** -100
    with mp.workprec(160):
        for _ in range(12):
            a = mpf(2) ** mpf(rng.uniform(1, 12))
            x = a * mpmath.exp(mpf(rng.uniform(-0.5, 0.5)))
            upper = x > a
            l = gamma_quad(a, x, upper)
            ref = gamma_tail(a, x, upper)
            assert abs(mpmath.exp(l) / ref - 1) < tol, ("gamma", a, x, l, ref)
        for _ in range(12):
            a = mpf(2) ** mpf(rng.uniform(1, 10))
            b = mpf(2) ** mpf(rng.uniform(1, 10))
            mean = a / (a + b)
            x = mean * mpmath.exp(mpf(rng.uniform(-0.3, 0)))
            ref = mpmath.betainc(a, b, 0, x, regularized=True)
            l = beta_quad(a, b, x, 1 - x)
            assert abs(mpmath.exp(l) / ref - 1) < tol, ("beta", a, b, x, l, ref)
        # A large shape beside a tiny one, x up to 0.8, where mpmath's
        # series converges.
        for _ in range(8):
            a = mpf(2) ** mpf(rng.uniform(11, 14))
            b = mpf(2) ** mpf(rng.uniform(-100, 0))
            x = mpf(rng.uniform(0.3, 0.8))
            ref = mpmath.betainc(a, b, 0, x, regularized=True)
            l = beta_quad(a, b, x, 1 - x)
            assert abs(mpmath.exp(l) / ref - 1) < tol, ("beta", a, b, x, l, ref)
