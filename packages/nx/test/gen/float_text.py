# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Generate the golden of the text Nx prints for each float.

    uv run packages/nx/test/gen/float_text.py [--check]

The text of a float is the fewest significant digits that round to it at
its dtype, the nearest such decimal when several have that many digits, and
of two as near the one whose last digit is even.
Two oracles state it. numpy's shortest repr (Dragon4, unique=True) covers
float16, float32 and float64. bfloat16 and the float8 formats, which numpy
lacks, take an exact search: the shortest decimal inside the interval of
values that round to the float, in rationals. The search also runs on every
float16 value and power of two, and must agree with numpy there.

The golden holds every positive finite float16, bfloat16, float8 e4m3 and
float8 e5m2 value and every positive power of two of float32 and float64,
under a line naming the dtype, one per line as its bits in hexadecimal and
its text. Without
--check the golden is written; with it, the run fails if the golden
differs.
"""

import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

GOLDEN = Path(__file__).resolve().parent.parent / "golden" / "float_text.txt"


# Formats: [mant] stored fraction bits, [exp] exponent bits, the bias, and
# whether the all-ones exponent holds infinities and NaN (IEEE) or the format
# keeps it for finite values with one NaN code (float8 e4m3).
class Format:
    def __init__(self, name, exp, mant, ieee):
        self.name, self.exp, self.mant, self.ieee = name, exp, mant, ieee
        self.bias = (1 << (exp - 1)) - 1
        self.width = 1 + exp + mant

    def value(self, code):
        e = (code >> self.mant) & ((1 << self.exp) - 1)
        f = code & ((1 << self.mant) - 1)
        if e == 0:
            return Fraction(f, 1 << self.mant) * Fraction(2) ** (1 - self.bias)
        return Fraction((1 << self.mant) + f, 1 << self.mant) * Fraction(2) ** (
            e - self.bias
        )

    def finite(self, code):
        e = (code >> self.mant) & ((1 << self.exp) - 1)
        if self.ieee:
            return e != (1 << self.exp) - 1
        return (code & ((1 << (self.exp + self.mant)) - 1)) != (
            1 << (self.exp + self.mant)
        ) - 1

    def positives(self):
        return [c for c in range(1, 1 << (self.width - 1)) if self.finite(c)]


F16 = Format("float16", 5, 10, True)
BF16 = Format("bfloat16", 8, 7, True)
E4M3 = Format("float8_e4m3", 4, 3, False)
E5M2 = Format("float8_e5m2", 5, 2, True)
F32 = Format("float32", 8, 23, True)
F64 = Format("float64", 11, 52, True)


def interval(fmt, code):
    """The values that round to [code], to nearest with ties to even: its
    bounds and whether they belong. Past the largest finite value, the values
    within half a step of it round to it."""
    x = fmt.value(code)
    prev = fmt.value(code - 1) if code > 1 else Fraction(0)
    nxt = fmt.value(code + 1) if fmt.finite(code + 1) else x + (x - prev)
    even = code % 2 == 0
    return (prev + x) / 2, (x + nxt) / 2, even


def shortest(fmt, code):
    """The digits and the decimal exponent of the first digit of the shortest
    decimal that rounds to [code], the nearest of them on a tie in length."""
    x = fmt.value(code)
    lo, hi, closed = interval(fmt, code)
    inside = (lambda v: lo <= v <= hi) if closed else (lambda v: lo < v < hi)
    k0 = math.floor(math.log10(x))
    for p in range(1, 40):
        found = []
        for k in (k0 - 1, k0, k0 + 1):
            unit = Fraction(10) ** (k - p + 1)
            first = max(math.ceil(lo / unit), 10 ** (p - 1))
            last = min(math.floor(hi / unit), 10**p - 1)
            for m in range(first, last + 1):
                if inside(m * unit):
                    found.append((abs(m * unit - x), m, k))
        if found:
            # The nearest, and of two as near the one whose last digit is even.
            _, m, k = min(found, key=lambda c: (c[0], c[1] % 2))
            return str(m).rstrip("0"), k
    raise SystemExit(f"{fmt.name} {code:x}: no decimal found")


def numpy_shortest(fmt, code):
    kind = {"float16": np.float16, "float32": np.float32, "float64": np.float64}
    ty = kind[fmt.name]
    v = np.frombuffer(code.to_bytes(fmt.width // 8, "little"), dtype=ty)[0]
    s = np.format_float_scientific(v, unique=True, trim="-")
    mantissa, exponent = s.split("e")
    return mantissa.replace(".", "").rstrip("0"), int(exponent)


def layout(digits, exp):
    """The text Nx prints for the digits [d0 d1 ...] times 10^exp at the first
    digit: in full for exponents from -4 to 15, with an exponent beyond."""
    p = len(digits)
    if exp < -4 or exp >= 16:
        fraction = "." + digits[1:] if p > 1 else ""
        return f"{digits[0]}{fraction}e{'-' if exp < 0 else '+'}{abs(exp):02d}"
    if exp >= p - 1:
        return digits + "0" * (exp - p + 1)
    if exp >= 0:
        return digits[: exp + 1] + "." + digits[exp + 1 :]
    return "0." + "0" * (-exp - 1) + digits


def power_codes(fmt):
    """The codes of the positive powers of two of a wide format: the
    subnormals with one bit set and the normals with a zero fraction."""
    subnormal = [1 << i for i in range(fmt.mant)]
    normal = [e << fmt.mant for e in range(1, (1 << fmt.exp) - 1)]
    return subnormal + normal


def rows():
    out = []
    for fmt in (F16, BF16, E4M3, E5M2):
        out.append(fmt.name)
        for code in fmt.positives():
            d = shortest(fmt, code)
            if fmt is F16 and d != numpy_shortest(fmt, code):
                raise SystemExit(f"float16 {code:x}: numpy and the search differ")
            out.append(f"{code:x} {layout(*d)}")
    for fmt in (F32, F64):
        out.append(fmt.name)
        for code in power_codes(fmt):
            d = numpy_shortest(fmt, code)
            if d != shortest(fmt, code):
                raise SystemExit(f"{fmt.name} {code:x}: numpy and the search differ")
            out.append(f"{code:x} {layout(*d)}")
    return "\n".join(out) + "\n"


def main():
    text = rows()
    if "--check" in sys.argv[1:]:
        if GOLDEN.read_text() != text:
            raise SystemExit(f"{GOLDEN} differs from the oracles")
        return
    GOLDEN.write_text(text)


if __name__ == "__main__":
    main()
