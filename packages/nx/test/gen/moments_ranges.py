# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Generate the exact moments that nx's moments over ranges are held to.

    uv run packages/nx/test/gen/moments_ranges.py [--check]

Writes `packages/nx/test/golden/moments_ranges.golden`. With --check nothing
is written, and the run fails if the golden would change.

Each case is rows of one float dtype, exactly representable in it, and
ranges over them. For each range the golden holds the exact mean and
population variance of its rows, computed with Fractions and rounded once
to float64: `nan` for a range of no row or one holding a NaN or both
infinities, the infinity a range holding one sign of infinity has as its
mean, and `infinity` for a variance past float64's greatest value. Nothing
here follows nx's algorithm.

The rows are adversarial for a variance: levels far from zero with small
noise (κ = |μ|/σ up to 1e11), level steps, outliers first in a window,
values whose squares or sums of squares pass max_float while the variance
does not, values of mixed scales, infinities and NaNs. The ranges are
fixed and varied windows, both ends, empty and one-row ranges and ranges
past either end.

A golden is lines of `case <name> <dtype> <n> <m>`, then `x`, `lo`, `hi`,
`mean` and `var` lines of space-separated values: floats in hex, bounds as
integers.
"""

import argparse
import difflib
import math
import random
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
GOLDEN = HERE.parent / "golden" / "moments_ranges.golden"


# Formats: precision, least normal exponent, greatest exponent, and whether
# the format has infinities and NaN.


class Format:
    def __init__(self, name, p, emin, emax, inf, nan):
        self.name, self.p, self.emin, self.emax = name, p, emin, emax
        self.inf, self.nan = inf, nan


F64 = Format("float64", 53, -1022, 1023, True, True)
F32 = Format("float32", 24, -126, 127, True, True)
F16 = Format("float16", 11, -14, 15, True, True)
BF16 = Format("bfloat16", 8, -126, 127, True, True)
E4M3 = Format("float8_e4m3fn", 4, -6, 8, False, True)
E5M2 = Format("float8_e5m2", 3, -14, 15, True, True)
E2M1 = Format("float4_e2m1fn", 2, 0, 2, False, False)


def round_to(fmt, v):
    """The finite float `v` rounded to nearest even in `fmt`."""
    if v == 0 or not math.isfinite(v):
        return v
    a = Fraction(abs(v))
    e = math.frexp(abs(v))[1] - 1
    q = max(e, fmt.emin) - (fmt.p - 1)
    scaled = a / Fraction(2) ** q
    n = math.floor(scaled)
    frac = scaled - n
    if frac > Fraction(1, 2) or (frac == Fraction(1, 2) and n % 2 == 1):
        n += 1
    return math.copysign(float(n * Fraction(2) ** q), v)


# Exact moments


def to_float(q):
    """The Fraction `q` rounded to float64, an infinity past its range."""
    try:
        return float(q)
    except OverflowError:
        return math.inf if q > 0 else -math.inf


def moments(rows):
    """The exact mean and population variance of `rows`, as floats."""
    if not rows or any(math.isnan(t) for t in rows):
        return math.nan, math.nan
    infs = {t for t in rows if math.isinf(t)}
    if len(infs) == 2:
        return math.nan, math.nan
    if infs:
        return infs.pop(), math.nan
    k = len(rows)
    qs = [Fraction(t) for t in rows]
    mu = sum(qs) / k
    var = sum((t - mu) ** 2 for t in qs) / k
    return to_float(mu), to_float(var)


# Ranges


def windows(n, w, rng, varied):
    """Windows of up to `w` rows ending at rows near either end, where the
    windows first reach `w` rows, and every seventh elsewhere; lengths drawn
    in [1, w] where varied."""
    out = []
    for i in range(n):
        if i < 8 or abs(i - w) < 3 or i >= n - 3 or i % 7 == 0:
            length = rng.randint(1, w) if varied else w
            out.append((i + 1 - length, i + 1))
    return out


def arbitrary(n, rng, k):
    """Ranges anywhere, from past the start to past the end: empty, of one
    row, nested and overlapping."""
    out = [(-5, n + 5), (0, n), (0, 1), (n - 1, n), (n, n + 3), (-3, 0), (3, 3)]
    for i in (0, 1, n // 2, n - 2, n - 1):
        out.append((i, i + 1))
    for _ in range(k):
        a, b = rng.randint(-3, n + 3), rng.randint(-3, n + 3)
        out.append((min(a, b), max(a, b)) if rng.random() < 0.8 else (a, b))
    return out


# Cases


def float64_cases():
    rng = random.Random(20261108)
    n = 800
    u = lambda: rng.random()

    def ranges(w, k=150):
        return windows(n, w, rng, False) + windows(n, w, rng, True) + arbitrary(n, rng, k)

    cases = []

    def add(name, x, w=365):
        cases.append((name, F64, x, ranges(w)))

    add("level 6e4, noise 1e-3", [6e4 + 1e-3 * u() for _ in range(n)])
    add("MJD 6e4 + 1e-3 a row, noise 1e-6", [6e4 + 1e-3 * i + 1e-6 * u() for i in range(n)])
    add("drift 6e4 + 1e-2 a row, noise 1e-3", [6e4 + 1e-2 * i + 1e-3 * u() for i in range(n)])
    add("1e8 + 1 a row, noise 1e-3", [1e8 + i + 1e-3 * u() for i in range(n)])
    add("level 1e3, a step to 2e3", [1e3 + (1e3 if i >= n // 2 else 0) + 1e-3 * u() for i in range(n)])
    add("outliers first: 1e6 every 97 rows from row 0",
        [(1e6 if i % 97 == 0 else 6e4) + 1e-3 * u() for i in range(n)])
    add("a constant", [6e4 + 0.1] * n)
    add("small integers and signed zeros",
        [rng.choice([-0.0, 0.0, -4.0, -1.0, 1.0, 3.0, 4.0]) for _ in range(n)])
    add("near 2e155, noise 1e152", [2e155 + 1e152 * u() for _ in range(n)])
    # Squared deviations whose sums pass max_float while the variances, at
    # most about 1e308, do not; and windows of 365 whose variance passes it.
    add("uniform in +-1e154, windows of 50", [1e154 * (2 * u() - 1) for _ in range(n)], w=50)
    add("uniform in +-1e154", [1e154 * (2 * u() - 1) for _ in range(n)])
    add("uniform in +-max_float", [sys.float_info.max * (2 * u() - 1) for _ in range(n)])
    # Values near (c + d) 2^s, c in {0, 1.5, -6}, d in [-1, 1], s in {0, 512,
    # 1018} mixed: small values before and after values near max_float.
    scales = [0, 0, 0, 512, 1018]
    add("mixed scales 2^0, 2^512, 2^1018",
        [math.ldexp(rng.choice([0.0, 1.5, -6.0]) + 2 * u() - 1, rng.choice(scales)) for _ in range(n)],
        w=12)
    specials = [math.inf, -math.inf, math.nan, -0.0, 0.0]
    x = [6e4 + 1e-3 * u() for _ in range(n)]
    for i in rng.sample(range(n), 12):
        x[i] = rng.choice(specials[:3])
    x[0], x[n - 1] = math.inf, math.nan
    add("infinities and NaNs among level 6e4", x, w=40)
    # The talon cases of values at large magnitudes, as their own rows.
    small = [
        ("near 2e155", [2e155, 2.0001e155, 2.0002e155, 2.0003e155]),
        ("1e154, -1e154, 0", [1e154, -1e154, 0.0]),
        ("1e308 three times", [1e308, 1e308, 1e308]),
        ("1, 2, 3, 1e308", [1.0, 2.0, 3.0, 1e308]),
        ("1e200, -1e200", [1e200, -1e200]),
        ("1.7e308, -1.7e308, -1.7e308", [1.7e308, -1.7e308, -1.7e308]),
    ]
    for name, x in small:
        k = len(x)
        rs = [(l, h) for l in range(-1, k + 1) for h in range(-1, k + 2)]
        cases.append((name, F64, x, rs))
    # A run's variance past max_float, the range's not.
    x = [1e155, -1e155] + [0.0] * 998
    cases.append(("1e155, -1e155 among zeros", F64, x,
                  [(0, 1000), (0, 500), (0, 2), (1, 3), (0, 3), (2, 900), (-1, 1001)]))
    # Rows past the depth of the top level: 2^11 + 3.
    m = 2051
    x = [6e4 + 1e-2 * i + 1e-3 * u() for i in range(m)]
    cases.append(("2051 rows of a drift, long ranges", F64, x,
                  windows(m, 2000, rng, True)[::7] + arbitrary(m, rng, 100)))
    return cases


def narrow_cases():
    """Rows of each narrower format: a level of 2^(p - 1), whose ulp is 1,
    with noise of a few units, a step, an outlier first, and its specials."""
    rng = random.Random(7)
    cases = []
    for fmt in [F32, F16, BF16, E4M3, E5M2, E2M1]:
        n = 300
        level = 2.0 ** (fmt.p - 1)
        noise = lambda: float(rng.randint(0, max(1, min(7, int(level) - 1))))

        def rows(f):
            return [round_to(fmt, f(i)) for i in range(n)]

        def ranges(w):
            return windows(n, w, rng, False) + windows(n, w, rng, True) + arbitrary(n, rng, 60)

        cases.append((f"{fmt.name}: level 2^(p-1), noise", fmt, rows(lambda i: level + noise()), ranges(50)))
        cases.append((f"{fmt.name}: a step", fmt, rows(lambda i: level * (2 if i >= n // 2 else 1) + noise()),
                      ranges(50)))
        big = 2.0 ** fmt.emax
        cases.append((f"{fmt.name}: outliers first", fmt,
                      rows(lambda i: big if i % 41 == 0 else level + noise()), ranges(20)))
        if fmt.nan:
            x = rows(lambda i: level + noise())
            for i in rng.sample(range(n), 6):
                x[i] = math.nan
            if fmt.inf:
                for i in rng.sample(range(n), 6):
                    x[i] = rng.choice([math.inf, -math.inf])
            cases.append((f"{fmt.name}: specials", fmt, x, ranges(20)))
    return cases


def hexf(v):
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "infinity" if v > 0 else "-infinity"
    return v.hex()


def render(cases):
    lines = []
    for name, fmt, x, rs in cases:
        n = len(x)
        lines.append(f"case {name.replace(' ', '_')} {fmt.name} {n} {len(rs)}")
        lines.append("x " + " ".join(hexf(v) for v in x))
        lines.append("lo " + " ".join(str(l) for l, _ in rs))
        lines.append("hi " + " ".join(str(h) for _, h in rs))
        ms = [moments(x[max(0, min(n, l)):max(0, min(n, h))]) for l, h in rs]
        lines.append("mean " + " ".join(hexf(m) for m, _ in ms))
        lines.append("var " + " ".join(hexf(v) for _, v in ms))
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    text = render(float64_cases() + narrow_cases())
    if args.check:
        old = GOLDEN.read_text() if GOLDEN.exists() else ""
        if old != text:
            diff = difflib.unified_diff(old.splitlines(), text.splitlines(), lineterm="", n=0)
            sys.stdout.write("\n".join(list(diff)[:20]) + "\n")
            sys.exit(f"{GOLDEN} would change")
        return
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(text)


if __name__ == "__main__":
    main()
