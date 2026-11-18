# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Generate test/plane_areas.dat, Polygons.area's cases.

    uv run contrib/jord/test/gen/plane_areas.py [--check]

Random features of up to 3 rings on a 2.5 m lattice near UTM's numbers
(x, y), a fifth of their ordinates 1 ulp off it, and a hole vertex within
rounding of its shell's edge in metres (the near-tie at the middle of a
slanted edge, and 1 ulp either side of it). Each feature is written with
its exact even-odd area, computed in Fractions by vertical slices between
every vertex and crossing and rounded once, and its number of crossings.

Without --check the file is written; with it, nothing is written and the
run fails if the file would change.
"""

import math
import random
import sys
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "plane_areas.dat"

HEADER = """\
# Polygons.area against exact areas: random features of up to 3 rings on
# a 2.5 m lattice near UTM's numbers (x, y), a fifth of their ordinates
# 1 ulp off it, and a hole vertex within rounding of its shell's edge in
# metres, each with its exact even-odd area rounded once and its
# crossings, written by
#   uv run contrib/jord/test/gen/plane_areas.py
# 'feature AREA CROSSINGS', then its rings 'ring x y x y ...'; numbers are
# Python's repr, which float_of_string reads back exactly.
"""


def ulps(y, k):
    for _ in range(abs(k)):
        y = math.nextafter(y, math.inf if k > 0 else -math.inf)
    return y


def features():
    """Each feature's rings of (x, y) points."""
    random.seed(31)
    res = []
    for x0, y0 in ((452317.25, 5361243.5), (731000.125, 9998765.375), (166021.0, 12.5)):
        for _ in range(10):
            rings = []
            for _ in range(random.randint(1, 3)):
                r = []
                for _ in range(random.randint(3, 8)):
                    x, y = random.randint(0, 40) * 2.5, random.randint(0, 40) * 2.5
                    if random.random() < 0.2:
                        y = ulps(y, random.choice([-1, 1]))
                    r.append((x0 + x, y0 + y))
                rings.append(r)
            res.append(rings)
        # A hole vertex at the shell edge's midpoint, rounded, and 1 ulp off.
        for _ in range(4):
            p = (x0 + 3.0, y0 + random.uniform(10.5, 19.5))
            q = (x0 + 27.0, y0 + random.uniform(10.5, 19.5))
            xm = x0 + 15.0
            ym = p[1] + (xm - p[0]) * (q[1] - p[1]) / (q[0] - p[0])
            for k in (-1, 0, 1):
                yk = ulps(ym, k)
                shell = [p, q, (q[0], y0 + 38.0), (p[0], y0 + 38.0)]
                hole = [(xm, yk), (xm + 3.0, yk + 6.0), (xm - 3.0, yk + 6.0)]
                res.append([shell, hole])
    return res


def edges_of(feature):
    for ring in feature:
        n = len(ring)
        for m in range(n):
            yield ring[m], ring[(m + 1) % n]


def crossing_x(a, b):
    sa = (a[3] - a[1]) / (a[2] - a[0])
    sb = (b[3] - b[1]) / (b[2] - b[0])
    if sa == sb:
        return None
    x = (b[1] - b[0] * sb - a[1] + a[0] * sa) / (sa - sb)
    if max(min(a[0], a[2]), min(b[0], b[2])) < x < min(max(a[0], a[2]), max(b[0], b[2])):
        return x
    return None


def slanted(feature):
    return [(F(a[0]), F(a[1]), F(b[0]), F(b[1]))
            for a, b in edges_of(feature) if a[0] != b[0]]


def exact_area(feature):
    es = slanted(feature)
    if not es:
        return F(0)
    bps = {e[0] for e in es} | {e[2] for e in es}
    for k in range(len(es)):
        for l in range(k + 1, len(es)):
            x = crossing_x(es[k], es[l])
            if x is not None:
                bps.add(x)

    def yat(e, x):
        return e[1] + (x - e[0]) * (e[3] - e[1]) / (e[2] - e[0])

    bps = sorted(bps)
    tot = F(0)
    for xa, xb in zip(bps, bps[1:]):
        xm = (xa + xb) / 2
        act = sorted((e for e in es if min(e[0], e[2]) < xm < max(e[0], e[2])),
                     key=lambda e: yat(e, xm))
        for k, e in enumerate(act):
            h = (yat(e, xa) + yat(e, xb)) / 2 * (xb - xa)
            tot += h if k % 2 == 1 else -h
    return tot


def crossings(feature):
    es = slanted(feature)
    return sum(1 for i in range(len(es)) for j in range(i + 1, len(es))
               if crossing_x(es[i], es[j]) is not None)


def text():
    out = [HEADER]
    for f in features():
        out.append("feature %r %d\n" % (float(exact_area(f)), crossings(f)))
        for r in f:
            out.append("ring " + " ".join("%r %r" % p for p in r) + "\n")
    return "".join(out)


def main():
    t = text()
    if "--check" in sys.argv[1:]:
        if OUT.read_text() != t:
            sys.exit(f"{OUT} would change; run without --check")
    else:
        OUT.write_text(t)


if __name__ == "__main__":
    main()
