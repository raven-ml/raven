# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0", "numpy>=2"]
# ///
"""Generate the goldens of nx.wide: double-word operands and mpmath's exact
results of them.

    uv run packages/nx/test/wide/gen/generate.py [--check]

Without --check, golden/*.golden are written. With --check, nothing is
written and the run fails when a file differs from what it would write.

Every float is written in hexadecimal at float64, which holds every float32
exactly. A row's dtype is `f32` or `f64`; its operands are double words of
that dtype, normalised (`hi` is `hi + lo` rounded to nearest).

- add, sub, mul, div: `dtype xh xl yh yl r1 r2 r3`, the exact result
  `r1 + r2 + r3`, each word the rest rounded to nearest at float64, which
  holds the result to about 159 bits (division is computed to 600).
- floor: `dtype xh xl z1 z2`, the floor as the dtype's normalised pair.
- compare: `dtype xh xl yh yl less equal`, each 0 or 1.
- sum: `dtype n xh0 xl0 ... r1 r2 r3 s`, with `s` the sum of the summands'
  magnitudes rounded up at float64.

Operands are seeded random pairs over a range of exponents and spreads,
pairs that cancel, pairs of equal high words, zero low words, operands at
the edges of the domain (magnitudes near 2^-969 and 2^969 at float64,
2^-102 and 2^102 at float32, with results inside), and the worst cases
the papers on these algorithms give: Muller and Rideau's generic case for
the addition (relative error 3u^2 - 11u^3) and Joldes, Muller and
Popescu's (2.25u^2), each negated for the subtraction; Muller and
Rideau's product (3.997u^2 at float64) and Joldes, Muller and Popescu's
quotient (5.922u^2 at float64) and binary32 product. At float32, a search
over the algorithms emulated adds the inputs of largest error it finds.
"""

import math
import os
import random
import sys

import mpmath
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "golden")
mpmath.mp.prec = 600
ROWS = 500
SEED = 20261005

# Each dtype: its precision, the float type rounding to it, and the exponents
# operands take, which keep every intermediate normal.
DTYPES = {
    "f32": (24, np.float32, 20),
    "f64": (53, np.float64, 200),
}


def normalised(dtype, hi, lo):
    _, ty, _ = DTYPES[dtype]
    return float(ty(hi) + ty(lo)) == hi


def words(x):
    """The exact [x] as three float64 words, each the rest rounded."""
    x = mpmath.mpf(x)
    r1 = float(x)
    r2 = float(x - r1)
    r3 = float(x - r1 - r2)
    return r1, r2, r3


def pair_of(dtype, x):
    """The exact [x] as its normalised pair at [dtype]."""
    x = mpmath.mpf(x)
    p, _, _ = DTYPES[dtype]
    with mpmath.workprec(p):
        hi = float(+x)
    lo = x - hi
    with mpmath.workprec(p):
        lo_r = float(+lo)
    assert lo_r == lo, "not a double word"
    return hi, lo_r


def random_hi(rng, dtype, e):
    p, _, _ = DTYPES[dtype]
    m = rng.getrandbits(p - 1) | (1 << (p - 1))
    return rng.choice([-1.0, 1.0]) * float(mpmath.ldexp(m, e - p + 1))


def random_lo(rng, dtype, hi):
    """A low word for [hi]: zero, or below half its ulp by a random depth."""
    p, _, _ = DTYPES[dtype]
    if hi == 0 or rng.random() < 0.1:
        return 0.0
    _, e = mpmath.frexp(hi)
    depth = rng.choice([1, 1, 2, 3, rng.randint(4, p + 10)])
    m = rng.getrandbits(p - 1) | (1 << (p - 1))
    return rng.choice([-1.0, 1.0]) * float(
        mpmath.ldexp(m, int(e) - 2 * p - depth)
    )


def random_pair(rng, dtype, e=None):
    _, _, emax = DTYPES[dtype]
    while True:
        if e is None:
            e0 = rng.randint(-emax, emax)
        else:
            e0 = e
        hi = random_hi(rng, dtype, e0)
        lo = random_lo(rng, dtype, hi)
        if normalised(dtype, hi, lo):
            return hi, lo


def exact(x):
    return mpmath.mpf(x[0]) + mpmath.mpf(x[1])


def ulp(dtype, x):
    p, _, _ = DTYPES[dtype]
    _, e = mpmath.frexp(x)
    return float(mpmath.ldexp(1, int(e) - p))


# Domain: operands and results zero or of magnitude in [2^-E, 2^E]. Every
# bound holds there; past it an error term leaves the normal range.
DOMAIN = {"f32": 102, "f64": 969}


# The algorithms at float32, emulated to search for inputs near their bounds:
# each operation rounds once, [fma] from its exact value.


def f32(x):
    return float(np.float32(x))


def fma32(a, b, c):
    with mpmath.workprec(24):
        return float(+(mpmath.mpf(a) * mpmath.mpf(b) + mpmath.mpf(c)))


def two_sum32(a, b):
    s = f32(a + b)
    a1 = f32(s - b)
    b1 = f32(s - a1)
    return s, f32(f32(a - a1) + f32(b - b1))


def fast_two_sum32(a, b):
    s = f32(a + b)
    return s, f32(b - f32(s - a))


def add32(x, y):
    sh, sl = two_sum32(x[0], y[0])
    th, tl = two_sum32(x[1], y[1])
    vh, vl = fast_two_sum32(sh, f32(sl + th))
    return fast_two_sum32(vh, f32(tl + vl))


def mul32(x, y):
    ch = f32(x[0] * y[0])
    cl1 = fma32(x[0], y[0], -ch)
    tl1 = fma32(x[0], y[1], f32(x[1] * y[1]))
    cl2 = fma32(x[1], y[0], tl1)
    return fast_two_sum32(ch, f32(cl1 + cl2))


def div32(x, y):
    th = f32(1.0 / y[0])
    rh = fma32(-y[0], th, 1.0)
    eh, el = fast_two_sum32(rh, -f32(y[1] * th))
    ch = f32(eh * th)
    dh, dl = fast_two_sum32(ch, fma32(el, th, fma32(eh, th, -ch)))
    sh, sl = two_sum32(dh, th)
    mh, ml = fast_two_sum32(sh, f32(dl + sl))
    return mul32(x, (mh, ml))


def searched(rng, op, count=3, tries=4000):
    """The [count] float32 operand pairs of the largest relative error of the
    emulated [op] among [tries] drawn near 1, where the bounds are reached."""
    run = {"add": add32, "sub": lambda x, y: add32(x, (-y[0], -y[1])),
           "mul": mul32, "div": div32}[op]
    apply = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b,
             "mul": lambda a, b: a * b, "div": lambda a, b: a / b}[op]
    found = []
    for _ in range(tries):
        x = random_pair(rng, "f32", 0)
        y = random_pair(rng, "f32", rng.choice([-1, 0]))
        if op in ("add", "sub"):
            y = (-y[0], -y[1]) if (op == "add") == (rng.random() < 0.8) else y
        e = apply(exact(x), exact(y))
        if e == 0:
            continue
        z = run(x, y)
        found.append((abs((mpmath.mpf(z[0]) + z[1] - e) / e), x, y))
    found.sort(key=lambda t: t[0], reverse=True)
    return [(x, y) for _, x, y in found[:count]]


def edges(rng, dtype, op):
    """Operands at the domain's edges, with results inside it."""
    e = DOMAIN[dtype]
    h = e // 2
    near = {"add": [(e - 1, e - 1), (1 - e, 1 - e)],
            "sub": [(e - 1, e - 3), (1 - e, 3 - e)],
            "mul": [(h - 1, h - 1), (1 - h, 1 - h), (e - 1, 0), (1 - e, 0)],
            "div": [(e - 1, 0), (1 - e, 0), (0, e - 1), (0, 1 - e),
                    (e - 1, e - 1), (1 - e, 1 - e)]}[op]
    return [(random_pair(rng, dtype, a), random_pair(rng, dtype, b))
            for a, b in near]


def operands(rng, dtype, op):
    """[ROWS] pairs of operands for [op], structured cases first."""
    p, _, emax = DTYPES[dtype]
    u = 2.0 ** -p
    m = 2.0**p
    worst_add = [
        # Muller and Rideau's generic worst case of the addition.
        ((1.0, u - u * u), (-0.5 + u / 2, -(u * u) / 2 + u**3)),
        # Joldes, Muller and Popescu's generic case of the addition, 2.25u^2.
        (
            (m - 1, -(m - 1) * 2.0 ** (-p - 1)),
            (-(m - 5) / 2, -(m - 1) * 2.0 ** (-p - 3)),
        ),
    ]
    # A difference reaches the sum's worst case on the negated operand.
    out = (worst_add if op != "sub"
           else [(x, (-y[0], -y[1])) for x, y in worst_add])
    if dtype == "f64" and op == "mul":
        # Muller and Rideau's product, 3.997u^2.
        out.append(
            (
                (2251799825991851 / 2**51, 9007199203085987 / 2**106),
                (4503599627471459 / 2**52, 4503599627284651 / 2**105),
            )
        )
    if dtype == "f64" and op == "div":
        # Joldes, Muller and Popescu's quotient, 5.922u^2.
        out.append(
            (
                (4528288502329187.0, 1125391118633487 / 2**51),
                (4522593432466394.0, -9006008290016505 / 2**54),
            )
        )
    if dtype == "f32" and op == "mul":
        # Joldes, Muller and Popescu's binary32 product of their Algorithm
        # 11, 4.936u^2 there.
        out.append(
            (
                (8404039.0, -8284843 / 2**24),
                (8409182.0, -4193899 / 2**23),
            )
        )
    if dtype == "f32":
        out += searched(rng, op)
    out += edges(rng, dtype, op)
    while len(out) < ROWS:
        kind = rng.random()
        x = random_pair(rng, dtype)
        _, ex = mpmath.frexp(x[0])
        ex = int(ex) - 1
        if kind < 0.35:
            y = random_pair(rng, dtype)
        elif kind < 0.55:
            y = random_pair(rng, dtype, ex + rng.randint(-3 * p, 3 * p) // 3)
        elif kind < 0.7:
            # Cancelling high words, one ulp apart or equal.
            yh = -x[0] + rng.choice([0.0, ulp(dtype, x[0]), -ulp(dtype, x[0])])
            yl = random_lo(rng, dtype, yh) if yh != 0 else 0.0
            if not normalised(dtype, yh, yl):
                continue
            y = (yh, yl)
        elif kind < 0.8:
            y = (-x[0], -x[1]) if rng.random() < 0.5 else x
        elif kind < 0.9:
            x = (x[0], 0.0)
            y = (random_pair(rng, dtype)[0], 0.0)
        else:
            y = random_pair(rng, dtype, ex)
        if op == "div" and y[0] == 0:
            continue
        if any(abs(v) > 2.0 ** (emax + 2) for v in x + y):
            continue
        out.append((x, y))
    return out


def binary(op):
    def apply(x, y):
        if op == "add":
            return x + y
        if op == "sub":
            return x - y
        if op == "mul":
            return x * y
        return x / y

    lines = [f"# {op}: dtype xh xl yh yl r1 r2 r3"]
    for dtype in DTYPES:
        rng = random.Random(f"{SEED}-{op}-{dtype}")
        for x, y in operands(rng, dtype, op):
            r = words(apply(exact(x), exact(y)))
            lines.append(row(dtype, *x, *y, *r))
    return lines


def floor():
    lines = ["# floor: dtype xh xl z1 z2"]
    for dtype in DTYPES:
        p, _, _ = DTYPES[dtype]
        rng = random.Random(f"{SEED}-floor-{dtype}")
        xs = [
            (1.0, -(2.0**-60)),
            (1.0, 2.0**-60),
            (-1.0, 2.0**-60),
            (-1.0, -(2.0**-60)),
            (0.5, -(2.0 ** (-p - 3))),
            (2.0**p, -0.5),
            (2.0**p, 0.5),
            (2.0 ** (p + 1), -1.0),
            (3.0 * 2.0**p, -0.25),
            (0.0, 0.0),
            (-0.0, 0.0),
        ]
        while len(xs) < ROWS:
            # Integers and their neighbours, and numbers past [2^p], whose
            # high word is an integer and whose low word is not.
            e = rng.randint(-4, p + 8)
            hi = random_hi(rng, dtype, e)
            if rng.random() < 0.5:
                hi = float(mpmath.floor(hi))
                if hi == 0:
                    continue
            lo = random_lo(rng, dtype, hi)
            if normalised(dtype, hi, lo):
                xs.append((hi, lo))
        for x in xs:
            z = pair_of(dtype, mpmath.floor(exact(x)))
            lines.append(row(dtype, *x, *z))
    return lines


def compare():
    lines = ["# compare: dtype xh xl yh yl less equal"]
    for dtype in DTYPES:
        rng = random.Random(f"{SEED}-compare-{dtype}")
        rows = []
        while len(rows) < ROWS:
            x = random_pair(rng, dtype, rng.randint(-4, 4))
            kind = rng.random()
            if kind < 0.3:
                y = x
            elif kind < 0.7:
                yl = random_lo(rng, dtype, x[0])
                if not normalised(dtype, x[0], yl):
                    continue
                y = (x[0], yl)
            else:
                y = random_pair(rng, dtype, rng.randint(-4, 4))
            rows.append((x, y))
        for x, y in rows:
            a, b = exact(x), exact(y)
            lines.append(row(dtype, *x, *y) + f" {int(a < b)} {int(a == b)}")
    return lines


def sums():
    lines = ["# sum: dtype n xh0 xl0 ... r1 r2 r3 s"]
    for dtype in DTYPES:
        rng = random.Random(f"{SEED}-sum-{dtype}")
        for n, count in [(1, 4), (2, 4), (3, 4), (5, 4), (8, 4), (33, 4), (64, 4), (1000, 1)]:
            for _ in range(count):
                e = rng.randint(-10, 10)
                ws = [random_pair(rng, dtype, e + rng.randint(-8, 8)) for _ in range(n)]
                if rng.random() < 0.5:
                    # Cancelling summands.
                    ws = ws + [(-h, -l) for h, l in ws[: n // 2]]
                    ws = ws[:n]
                    rng.shuffle(ws)
                total = sum((exact(w) for w in ws), mpmath.mpf(0))
                s = sum((abs(exact(w)) for w in ws), mpmath.mpf(0))
                s_up = float(s)
                if s_up < s:
                    s_up = math.nextafter(s_up, math.inf)
                fields = [v for w in ws for v in w]
                lines.append(
                    row(dtype, *fields, *words(total), s_up, n=n)
                )
    return lines


def row(dtype, *values, n=None):
    head = [dtype] + ([str(n)] if n is not None else [])
    return " ".join(head + [float(v).hex() for v in values])


GOLDENS = {
    "add": lambda: binary("add"),
    "sub": lambda: binary("sub"),
    "mul": lambda: binary("mul"),
    "div": lambda: binary("div"),
    "floor": floor,
    "compare": compare,
    "sum": sums,
}


def main():
    check = "--check" in sys.argv[1:]
    stale = []
    for name, make in GOLDENS.items():
        text = "\n".join(make()) + "\n"
        path = os.path.join(OUT, f"{name}.golden")
        if check:
            with open(path) as f:
                if f.read() != text:
                    stale.append(path)
        else:
            with open(path, "w") as f:
                f.write(text)
    if stale:
        sys.exit("stale goldens: " + ", ".join(stale))


if __name__ == "__main__":
    main()
