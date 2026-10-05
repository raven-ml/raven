# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath", "sympy"]
# ///
"""Generate the correct-rounding goldens of Unit.ratio.

    uv run contrib/ymir/test/units/gen/ratio.py [--check]

Each line of golden/ratio.golden is

    <dtype> <factor>... = <result>

where the factors multiply to an exact positive number V and the result is V
rounded once to the dtype as IEEE 754 roundTiesToEven does, with the dtype's
subnormals and an exponent unbounded above. A factor is `n`, `n^a` or `n^a/b`
for an integer 1 <= n < 2^62, or `pi^a` or `pi^a/b`. The result is the
rounded value as a hexadecimal float, or `zero`, `subnormal` or `overflow`
when the rounded value is 0, below the least normal value or above the
largest finite value: the three ways the conversion raises.

Rational values round exactly with fractions. Irrational ones are evaluated
by mpmath with a precision that doubles until the value is provably away
from every rounding boundary; an irrational number never meets one.

The cases are each format's overflow, normal and subnormal edges with their
neighbours (exact ties, then the tie times 1 +- 2^-61, times
((2^61 + 1) / 2^61)^(+-1/2) and times pi q / p for convergents p/q of pi),
coefficients near
the 4096-bit bound, pi powers, and units drawn with a fixed seed: rational
exponents, pi powers and roots, scaled by a power of two onto each edge.

Without --check the golden is written; with it, nothing is written and the
run fails if the golden would change.
"""

import argparse
from fractions import Fraction
import math
from pathlib import Path
import random
import sys

import mpmath
from sympy import factorint, isprime, prevprime

HERE = Path(__file__).resolve().parent
GOLDEN = HERE.parent / "golden" / "ratio.golden"

# name: (precision, least normal exponent, largest finite value)
FORMATS = {
    "float64": (53, -1022, Fraction((2**53 - 1) * 2**971)),
    "float32": (24, -126, Fraction((2**24 - 1) * 2**104)),
    "float16": (11, -14, Fraction(65504)),
    "bfloat16": (8, -126, Fraction((2**8 - 1) * 2**120)),
    "float8_e4m3": (4, -6, Fraction(448)),
    "float8_e5m2": (3, -14, Fraction(57344)),
}

K = 2**61
MERSENNE_61 = 2**61 - 1
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 97, 65521, 65537,
          16777213, 16777259, 2147483647, MERSENNE_61, prevprime(2**62)]
assert all(isprime(p) for p in PRIMES)


# Numbers: a factor is (base, num, den), base an int or "pi".

def exponents(factors):
    """The exponent of each prime and of pi in the product of the factors."""
    primes, pi = {}, Fraction(0)
    for base, num, den in factors:
        e = Fraction(num, den)
        if base == "pi":
            pi += e
            continue
        for p, k in factorint(base).items():
            primes[p] = primes.get(p, Fraction(0)) + k * e
    return {p: e for p, e in primes.items() if e != 0}, pi


def log2(factors):
    primes, pi = exponents(factors)
    with mpmath.workprec(128):
        return (sum(e.numerator * mpmath.log(p, 2) / e.denominator for p, e in primes.items())
                + pi.numerator * mpmath.log(mpmath.pi, 2) / pi.denominator)


def floor_log2(v):
    e = v.numerator.bit_length() - v.denominator.bit_length()
    return e - 1 if Fraction(2)**e > v else e


def round_rational(v, prec, emin):
    q = max(floor_log2(v), emin) - (prec - 1)
    m = v / Fraction(2)**q
    n, rest = divmod(m.numerator, m.denominator)
    half = Fraction(rest, m.denominator) - Fraction(1, 2)
    if half > 0 or (half == 0 and n % 2 == 1):
        n += 1
    return Fraction(n) * Fraction(2)**q


def round_irrational(primes, pi, prec, emin):
    width = 256 + max([abs(e.numerator).bit_length() for e in primes.values()]
                      + [abs(pi.numerator).bit_length()])
    while width <= 1 << 16:
        with mpmath.workprec(width):
            log_v = (sum(e.numerator * mpmath.log(p) / e.denominator for p, e in primes.items())
                     + pi.numerator * mpmath.log(mpmath.pi) / pi.denominator)
            v = mpmath.exp(log_v)
            q = max(int(mpmath.floor(log_v / mpmath.log(2))), emin) - (prec - 1)
            m = mpmath.ldexp(v, -q)
            n = int(mpmath.floor(m))
            frac = m - n
            # The value's relative error is below 2^-(width - 64); the margin
            # keeps it well away from the midpoint frac = 1/2.
            if abs(frac - mpmath.mpf(0.5)) > mpmath.ldexp(1, 128 - width):
                return Fraction(n + (1 if frac > 0.5 else 0)) * Fraction(2)**q
        width *= 2
    sys.exit(f"no precision decides the rounding of {primes} pi^{pi}")


def golden(name, factors):
    prec, emin, largest = FORMATS[name]
    emax = floor_log2(largest)
    estimate = log2(factors)
    if estimate > emax + 2:
        return "overflow"
    if estimate < emin - prec - 2:
        return "zero"
    primes, pi = exponents(factors)
    if pi == 0 and all(e.denominator == 1 for e in primes.values()):
        v = Fraction(1)
        for p, e in primes.items():
            v *= Fraction(p)**int(e)
        r = round_rational(v, prec, emin)
        # Python's conversion of a fraction is correctly rounded in float64.
        assert name != "float64" or r < Fraction(2)**emin or r > largest or r == Fraction(float(v))
    else:
        r = round_irrational(primes, pi, prec, emin)
    if r == 0:
        return "zero"
    if r < Fraction(2)**emin:
        return "subnormal"
    if r > largest:
        return "overflow"
    return float(r).hex()


def show(factors):
    def one(base, num, den):
        b = "pi" if base == "pi" else str(base)
        if (num, den) == (1, 1):
            return b
        return f"{b}^{num}" if den == 1 else f"{b}^{num}/{den}"
    return " ".join(one(*f) for f in factors) or "1"


# Cases

def dyadic(v):
    """v = c 2^j as factors, c odd."""
    j = 0
    while v.numerator % 2 == 0:
        v, j = v / 2, j + 1
    while v.denominator % 2 == 0:
        v, j = v * 2, j - 1
    assert v.denominator == 1 and v.numerator < 2**62
    c = int(v)
    return ([(c, 1, 1)] if c != 1 else []) + ([(2, j, 1)] if j else [])


def pi_convergents():
    """Two convergents p/q of pi with q < 2^31, one on each side of pi."""
    with mpmath.workprec(512):
        x, h, k, h0, k0, out = +mpmath.pi, 1, 0, 0, 1, {}
        while True:
            a = int(mpmath.floor(x))
            h, h0, k, k0 = a * h + h0, h, a * k + k0, k
            if k >= 2**31:
                return out[False], out[True]
            out[mpmath.mpf(h) / k > mpmath.pi] = (h, k)
            x = 1 / (x - a)


BELOW, ABOVE = pi_convergents()

# Factors of numbers just above and below 1: 1 +- 2^-61, ((2^61 + 1) / 2^61)^(+-1/2)
# and pi q / p for convergents p/q of pi, within 2^-60 of 1.
NEIGHBOURS = [
    [],
    [(K + 1, 1, 1), (2, -61, 1)],
    [(K - 1, 1, 1), (2, -61, 1)],
    [(K + 1, 1, 2), (2, -61, 2)],
    [(K + 1, -1, 2), (2, 61, 2)],
    [("pi", 1, 1), (BELOW[1], 1, 1), (BELOW[0], -1, 1)],
    [("pi", 1, 1), (ABOVE[1], 1, 1), (ABOVE[0], -1, 1)],
]


def edges(name):
    prec, emin, largest = FORMATS[name]
    emax = floor_log2(largest)
    ulp_max = Fraction(2)**(emax - prec + 1)
    least = Fraction(2)**emin
    tiny = Fraction(2)**(emin - prec + 1)
    ulp_one = Fraction(2)**(1 - prec)
    points = [
        largest, largest - ulp_max, largest + ulp_max / 2, largest + ulp_max,
        Fraction(2)**(emax + 1),
        least, least - tiny / 2, least - tiny, least + tiny, least + tiny / 2,
        least + 3 * tiny / 2,
        tiny, tiny / 2, tiny / 4, 3 * tiny / 2,
        Fraction(1), 1 + ulp_one / 2, 1 + 3 * ulp_one / 2,
    ]
    return [dyadic(x) + n for x in points for n in NEIGHBOURS]


def wide_coefficients():
    """Coefficients near the 4096-bit bound whose quotient is near 1."""
    return [
        [(3, 2584, 1), (2, -4095, 1)],
        [(2, 4095, 1), (3, -2584, 1)],
        [(5, 1764, 1), (2, -4095, 1)],
        [(16777259, 170, 1), (16777213, -170, 1)],
        [(16777213, 170, 1), (16777259, -170, 1)],
        [(10, 1233, 1), (3, -2584, 1)],
    ]


def pi_powers():
    return ([[("pi", r, 1)] for r in (-660, -620, -600, -77, -1, 1, 77, 600, 619, 620)]
            + [[("pi", r, 2)] for r in (-1239, -1, 1, 1239)]
            + [[("pi", 1, 3)], [("pi", -5, 7)], [("pi", 1, 12)]])


def drawn(rng, name):
    """Units of rational exponents and pi, placed on each edge of the format."""
    prec, emin, largest = FORMATS[name]
    emax = floor_log2(largest)
    targets = [emax + 1, emax, emin, emin - 1, emin - prec + 1, emin - prec, 0]
    cases = []
    for target in targets:
        for _ in range(8):
            factors = []
            for _ in range(rng.randint(1, 3)):
                base = rng.choice(PRIMES[:15] + ["pi"])
                den = rng.choice([1, 2, 3, 5, 7, 12])
                num = rng.choice([n for n in range(-9, 10) if n != 0])
                g = math.gcd(num, den)
                factors.append((base, num // g, den // g))
            if not factors:
                factors = [("pi", 1, 1)]
            shift = target - round(float(log2(factors))) + rng.randint(-1, 1)
            cases.append(factors + ([(2, shift, 1)] if shift else []))
    return cases


def lines():
    rng = random.Random(19)
    out, seen = [], set()
    for name in FORMATS:
        cases = edges(name) + wide_coefficients() + pi_powers() + drawn(rng, name)
        for f in cases:
            line = f"{name} {show(f)} = {golden(name, f)}"
            if line not in seen:
                seen.add(line)
                out.append(line)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = "\n".join(lines()) + "\n"
    if args.check:
        if not GOLDEN.exists() or GOLDEN.read_text() != text:
            sys.exit(f"{GOLDEN} would change; run without --check")
        return
    GOLDEN.parent.mkdir(exist_ok=True)
    GOLDEN.write_text(text)


if __name__ == "__main__":
    main()
