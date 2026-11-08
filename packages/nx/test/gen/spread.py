# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Generate the exact variances and standard deviations nx is held to.

    uv run packages/nx/test/gen/spread.py [--check]

Writes `packages/nx/test/golden/spread.golden`. With --check nothing is
written, and the run fails if the golden would change.

Each case is one lane of elements exactly representable in one float dtype,
and a `ddof`. The golden holds, for that lane, the exact variance
`Σ (x - μ)² / (n - ddof)` and its square root, computed with Fractions and
rounded once to float64, and each result's error allowance `tol`, rounded
up. Nothing here follows nx's algorithm.

A result is `infinity` where the exact one is past the dtype's greatest
value by more than the allowance. The allowance is the nx.cpu bound for a
Moments reduction of `n` terms, computed exactly (L + D blocks, u the
carrier's unit roundoff), for the population variance, scaled by
`n / (n - ddof)` with 2u more, and a variance's low end adds
`n / (n - ddof) + 1` of the carrier's least subnormal, where the squares
underflow before the sum. A standard deviation's allowance is that of the
root of the variance's interval without that low-end term, with 2u more,
and one carrier subnormal more.

The lanes are adversarial for a variance: magnitudes at both ends of the
dtype's range, at the band edges of the scales (`hi = 2^432`, `2^28`; the
caps; `2^∓512`, `2^∓64`), near-constant lanes near the top, alternating
signs, one large term among small ones, wide exponent spreads, subnormal
and constant lanes. Sizes straddle a kernel's block.

A golden is lines of `case <name> <dtype> <n> <ddof>`, then `x`, `var` and
`std` lines: `x` the elements in hex, `var` and `std` two hex floats, the
result and its allowance.
"""

import argparse
import difflib
import math
import random
import sys
from fractions import Fraction

from moments_ranges import F16, F32, F64, BF16, HERE, round_to, to_float

GOLDEN = HERE.parent / "golden" / "spread.golden"
SHIFT = 1200

CARRIER_U = {F64: Fraction(1, 2**53), F32: Fraction(1, 2**24), BF16: Fraction(1, 2**24), F16: Fraction(1, 2**24)}
CARRIER_SUB = {F64: Fraction(1, 2**1074), F32: Fraction(1, 2**149), BF16: Fraction(1, 2**149), F16: Fraction(1, 2**149)}


def fsqrt(q):
    """The square root of Fraction `q >= 0`, below by at most 2^-SHIFT."""
    if q == 0:
        return Fraction(0)
    return Fraction(math.isqrt(int(q * 2 ** (2 * SHIFT))), 2**SHIFT)


def up(q):
    """`q >= 0` rounded up to float64, `infinity` past its range."""
    v = to_float(q)
    if math.isinf(v):
        return v
    return math.nextafter(v, math.inf) if v < q else v


def largest(fmt):
    return Fraction(2 ** fmt.p - 1, 2 ** (fmt.p - 1)) * Fraction(2) ** fmt.emax


def spread(fmt, xs, ddof):
    """The exact variance and root of `xs` with their allowances."""
    n, u, sub = len(xs), CARRIER_U[fmt], CARRIER_SUB[fmt]
    qs = [Fraction(t) for t in xs]
    mu = sum(qs) / n
    s2 = sum((t - mu) ** 2 for t in qs)
    pop = s2 / n
    v = s2 / (n - ddof)
    root = fsqrt(pop)

    L = math.ceil(min(n, 1024) / 16) + 3
    D = 0 if n <= 1024 else math.ceil(math.log2(math.ceil(n / 1024)))
    M = sum(abs(t) for t in qs) / n
    E = (L + D) * u * M + u * abs(mu)
    delta = (L + D + 5) * u * pop + 2 * (L + D + 1) * u * E * root + (3 * L + 3 * D + 7) * u * E * E
    ratio = Fraction(n, n - ddof)
    vbound = delta * ratio * (1 + 2 * u) + 2 * u * v
    vtol = vbound + (ratio + 1) * sub

    sigma = fsqrt(v)
    hi = fsqrt(v + vbound) * (1 + 2 * u)
    lo = fsqrt(max(v - vbound, Fraction(0))) * (1 - 2 * u)
    stol = max(hi - sigma, sigma - lo) + sub

    top = largest(fmt)
    z_v = "infinity" if v - vtol > top else to_float(v)
    z_s = "infinity" if sigma - stol > top else to_float(sigma)
    # A finite allowance below the top, so a constant lane's +inf is refused
    # where the bound itself passes the range.
    cap = top / 2
    return z_v, up(min(vtol, cap)), z_s, up(min(stol, cap))


def hexf(v):
    return "infinity" if v == "infinity" or math.isinf(v) else v.hex()


def fmt_case(name, fmt, xs, ddof):
    z_v, tv, z_s, ts = spread(fmt, xs, ddof)
    head = f"case {name} {fmt.name} {len(xs)} {ddof}\n"
    line = "x " + " ".join(t.hex() for t in xs) + "\n"
    return head + line + f"var {hexf(z_v)} {hexf(tv)}\nstd {hexf(z_s)} {hexf(ts)}\n"


# Lanes: elements exactly representable in `fmt`.


def rep(fmt, v):
    """`v` rounded to `fmt`, its nearest finite value past `fmt`'s range."""
    a = round_to(fmt, v)
    if abs(a) > largest(fmt):
        a = math.copysign(float(largest(fmt)), v)
    return a


def lanes(fmt, seed, repeat):
    rng = random.Random(seed)
    p, emin, emax = fmt.p, fmt.emin, fmt.emax
    pow2 = lambda e: rep(fmt, math.ldexp(1.0, e))
    tiny = math.ldexp(1.0, emin - p + 1)
    half = (emax + 1) // 2
    hi = 432 if fmt is F64 else 28
    cap = 500 if fmt is F64 else 60
    top_exps = sorted(
        set(
            e
            for e in [emax, emax - 1, emax - 3, emax - 13, (2 * emax) // 3, cap + 1, cap, cap - 1, hi + 1, hi, hi - 1,
                      half + 1, half, half - 1, half - 8]
            if emin < e <= emax
        ),
        reverse=True,
    )
    low_exps = sorted(
        set(
            e
            for e in [emin - p + 1, emin - p + 2, emin - p // 2, emin, emin + 1, emin + p, -half - 1, -half, -half + 1,
                      -hi - 1, -hi, -hi + 1, (2 * emin) // 3, emin // 2]
            if emin - p + 1 <= e < emax
        )
    )
    sizes = [2, 2, 3, 3, 5, 16, 17, 64]

    def size():
        return rng.choice(sizes)

    def jitter(e):
        r = rng.random()
        return rep(fmt, math.ldexp(1.0 + r, e)) if e < emax else rep(fmt, float(largest(fmt)) * (1 - r / 4))

    out = []

    def add(name, xs):
        n = len(xs)
        choices = [0] + ([1, n - 1, n // 2] if n > 1 else [])
        ddof = rng.choice(choices)
        out.append((name, xs, ddof))

    for e in top_exps + low_exps:
        for _ in range(repeat):
            n = size()
            # Alternating signs at one exponent: the squares reach 2^(2e).
            add(f"alt_2^{e}", [jitter(e) * ((-1) ** i) for i in range(n)])
            # Symmetric: the mean cancels exactly.
            t = jitter(e)
            add(f"sym_2^{e}", [t, -t])
            # Same sign: the sum reaches n 2^e, and the mean overflows at the top.
            add(f"same_2^{e}", [jitter(e) for _ in range(n)])
            # Near-constant: tiny deviations around a level.
            t = pow2(e)
            if e > emin + p and t != math.inf:
                add(f"near_2^{e}", [rep(fmt, t * (1 + rng.randrange(1, 64) * 2.0 ** (-(p // 2) - 1))) for _ in range(n)])
            # Constant, the true variance zero.
            add(f"const_2^{e}", [t] * n)
            # One term of exponent `e` among small ones.
            add(f"one_big_2^{e}", [jitter(e)] + [jitter(emin + p + (e - emin - p) // 3) for _ in range(n - 1)])

    # Random exponents across the whole range in one lane.
    for _ in range(40 * repeat):
        n = rng.randrange(2, 40)
        add("mixed", [rep(fmt, math.ldexp(rng.choice((-1.0, 1.0)) * (1.0 + rng.random()), rng.randrange(emin - p + 1, emax))) for _ in range(n)])

    # Ordinary magnitudes, the scales staying 1.
    for _ in range(10 * repeat):
        n = size()
        add("ordinary", [rep(fmt, rng.uniform(-3.0, 3.0)) for _ in range(n)])

    # Blocks past the kernels' block size.
    for e in [emax, half + 2, -half - 2]:
        add(f"long_alt_2^{e}", [rep(fmt, jitter(e) * ((-1) ** i)) for i in range(1040)])
    add("all_zero", [0.0, 0.0, 0.0])
    add("n_one", [jitter(emax)])
    add("subnormals", [tiny, -tiny, 3 * tiny, -5 * tiny])
    add("largest", [rep(fmt, float(largest(fmt)))] * 3)
    add("largest_opp", [rep(fmt, float(largest(fmt))), -rep(fmt, float(largest(fmt)))])
    return out


def hand(fmt):
    """Hand-picked lanes, of the dtype's range."""
    if fmt is F64:
        return [
            ("apart", [1e154, -1e154, 0.0]),
            ("twice_top", [1e308, 1e308]),
            ("cross_top", [1e200, -1e200]),
            ("alt_top_64", [1e308 * (-1) ** i for i in range(64)]),
            ("near_top_a", [1e300, 1e300 * (1 + 2.0**-40)]),
            ("tiny_sym", [1e-170, -1e-170]),
            ("tiny_three", [1e-160, -1e-160, 3e-160]),
            ("least_sym", [5e-324, -5e-324]),
            ("v_near_low", [2.0**-540 * (-1) ** i for i in range(16)]),
            ("v_mid", [2.0**540 * (1 + k * 2.0**-41) for k in range(4)]),
        ]
    if fmt is F32:
        return [
            ("cross_19", [rep(F32, 1.5e19), rep(F32, -1.5e19), 0.0]),
            ("cross_25", [rep(F32, 1e25), rep(F32, -1e25)]),
            ("tiny_25", [rep(F32, 1e-25), rep(F32, -1e-25)]),
            ("near_top", [rep(F32, 1e37), rep(F32, 1e37 * (1 + 2.0**-20))]),
        ]
    if fmt is BF16:
        return [("cross_19", [rep(BF16, 3e19), rep(BF16, -3e19), 0.0]), ("cross_38", [rep(BF16, 1e38), rep(BF16, -1e38)]),
                ("tiny_30", [rep(BF16, 1e-30), rep(BF16, -1e-30)])]
    return [("cross", [rep(F16, 300.0), rep(F16, -300.0)]), ("cap", [rep(F16, 65504.0), rep(F16, -65504.0)]),
            ("small", [rep(F16, 1e-5), rep(F16, -1e-5)])]


def render():
    out = []
    for fmt, seed, repeat in [(F64, 11, 2), (F32, 12, 2), (BF16, 13, 1), (F16, 14, 1)]:
        rng = random.Random(seed + 1)
        for name, xs in hand(fmt):
            ddof = 0 if len(xs) < 2 else rng.choice([0, 1])
            out.append(fmt_case(name, fmt, xs, ddof))
        for name, xs, ddof in lanes(fmt, seed, repeat):
            out.append(fmt_case(name, fmt, xs, ddof))
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    text = render()
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
