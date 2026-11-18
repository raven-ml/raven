# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.5.5", "shapely==2.2.0"]
# ///
"""Generate test/areas.dat, Rasterize.areas' cases: features in pixel points
with each cell's exact even-odd area (Fractions, rounded once to float64),
and real parcels with GEOS's areas of their cells.

    uv run contrib/jord/test/gen/areas.py [--check]

Exact cases ('case NAME ROWS COLS exact'): one feature each, every cell of
its box with its exact area. Four families of degenerate rings: 1a a hole
vertex within rounding of its shell's edge at the edge's midpoint in the
column; 1b a ring through another ring's edge at a vertex rounded onto it;
1c two edges crossing at slope differences 1e-4, 1e-7, 1e-10; 1d a hole
whose vertices are rounded onto, or one ulp inside or outside, a slanted
shell edge. Then holes, islands in lakes and crossing rings.

The GEOS case ('case NAME ROWS COLS geos'): parcels of BreizhCrops'
belle-ile shapefile, the 2017 RPG's parcels of Belle-Ile (SHP below, from
the URL list of github.com/dl4sits/BreizhCrops,
breizhcrops/datasets/urls.py, pinned by its SHA-256), on a 10 m grid turned
2.5 degrees from Lambert-93, pixel points from the file's least corner, 5
parcels with holes and 5 without, up to 300 m across, each feature's region
the symmetric difference of its rings (even-odd), and the cells of its box
whose area GEOS puts strictly inside (0, 1), with GEOS's area and the exact
one; every other cell is 0 or 1 exactly.

Without --check the file is written; with it, nothing is written and the
run fails if the file would change.
"""
import hashlib
import io
import math
import random
import sys
import tarfile
import urllib.request
from fractions import Fraction as F
from pathlib import Path

from shapely.geometry import Polygon, box

OUT = Path(__file__).resolve().parent.parent / "areas.dat"
SHP = "https://breizhcrops.s3.eu-central-1.amazonaws.com/2017/shp/belle-ile.tar.gz"
SHP_SHA256 = "dfafe0689ec835ca03fd899f9571de3cb84c73d56bfc1f5a1481a972d1771983"
_out = []


def emit(s):
    _out.append(s + "\n")


def fetch_shp():
    req = urllib.request.Request(SHP, headers={"User-Agent": "jord-areas"})
    with urllib.request.urlopen(req, timeout=300) as f:
        b = f.read()
    got = hashlib.sha256(b).hexdigest()
    if got != SHP_SHA256:
        sys.exit(f"{SHP}: SHA-256 {got}, pinned {SHP_SHA256}: source changed")
    with tarfile.open(fileobj=io.BytesIO(b)) as t:
        return t.extractfile("belle-ile.shp").read()

# ---------------------------------------------------------------- oracle

def edges_of(feature):
    for ring in feature:
        n = len(ring)
        for m in range(n):
            yield ring[m], ring[(m + 1) % n]


def exact_cell(feature, i, j):
    """The exact even-odd area of the feature in cell (i, j): vertical slices
    between every vertex, crossing and row-line abscissa, in Fractions."""
    es = []
    for (y0, x0), (y1, x1) in edges_of(feature):
        if x0 == x1:
            continue
        X0, Y0, X1, Y1 = F(x0), F(y0), F(x1), F(y1)
        if max(X0, X1) <= j or min(X0, X1) >= j + 1:
            continue
        es.append((X0, Y0, X1, Y1))
    if not es:
        return F(0)
    lo, hi = F(j), F(j + 1)
    bps = {lo, hi}

    def yat(e, x):
        X0, Y0, X1, Y1 = e
        return Y0 + (x - X0) * (Y1 - Y0) / (X1 - X0)

    for e in es:
        X0, Y0, X1, Y1 = e
        for x in (X0, X1):
            if lo < x < hi:
                bps.add(x)
        for yl in (F(i), F(i + 1)):
            if (Y0 - yl) * (Y1 - yl) < 0:
                x = X0 + (yl - Y0) * (X1 - X0) / (Y1 - Y0)
                if lo < x < hi:
                    bps.add(x)
    for k in range(len(es)):
        for l in range(k + 1, len(es)):
            a, b = es[k], es[l]
            sa = (a[3] - a[1]) / (a[2] - a[0])
            sb = (b[3] - b[1]) / (b[2] - b[0])
            if sa == sb:
                continue
            x = (b[1] - b[0] * sb - a[1] + a[0] * sa) / (sa - sb)
            if lo < x < hi:
                bps.add(x)
    bps = sorted(bps)
    tot = F(0)
    for xa, xb in zip(bps, bps[1:]):
        xm = (xa + xb) / 2
        act = [e for e in es if min(e[0], e[2]) < xm < max(e[0], e[2])]
        act.sort(key=lambda e: yat(e, xm))

        def L(x):
            s = F(0)
            for k, e in enumerate(act):
                c = min(F(1), max(F(0), yat(e, x) - i))
                s += c if k % 2 == 1 else -c
            return s

        tot += (L(xa) + L(xb)) / 2 * (xb - xa)
    return tot


def box_of(feature, rows, cols):
    ys = [y for r in feature for (y, _) in r]
    xs = [x for r in feature for (_, x) in r]
    return (max(0, math.floor(min(ys)) - 1), min(rows - 1, math.floor(max(ys)) + 1),
            max(0, math.floor(min(xs)) - 1), min(cols - 1, math.floor(max(xs)) + 1))


def ring_line(ring):
    return "ring " + " ".join("%r %r" % (y, x) for (y, x) in ring)


def exact_case(name, feature, rows, cols):
    emit("case %s %d %d exact" % (name, rows, cols))
    emit("feature")
    for r in feature:
        emit(ring_line(r))
    i0, i1, j0, j1 = box_of(feature, rows, cols)
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            emit("cell %d %d %r" % (i, j, float(exact_cell(feature, i, j))))
    emit("end")

# ---------------------------------------------------------------- families

def yat_f(p, q, x):
    (y0, x0), (y1, x1) = p, q
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


def yat_exact(p, q, x):
    (y0, x0), (y1, x1) = p, q
    return F(y0) + (F(x) - F(x0)) * (F(y1) - F(y0)) / (F(x1) - F(x0))


def family_1a():
    """A hole vertex at the float the shell edge rounds to at x = 1.5, the
    midpoint of the edge's span in column 1, exactly on the hole's side."""
    random.seed(5)
    out = []
    while len(out) < 4:
        y0, y1 = random.uniform(1.05, 1.95), random.uniform(1.05, 1.95)
        P, Q = (y0, 0.3), (y1, 2.7)
        yk = yat_f(P, Q, 1.5)
        ye = yat_exact(P, Q, 1.5)
        if F(yk) > ye:   # hole below the edge (rows grow down): valid
            hole = [(yk, 1.5), (yk + 0.6, 1.8), (yk + 0.6, 1.2)]
            shell = [P, Q, (3.8, 2.7), (3.8, 0.3)]
            out.append([hole, shell] if len(out) % 2 else [shell, hole])
        elif F(yk) < ye:  # hole above: the shell's bottom edge
            shell = [(0.2, 0.3), (0.2, 2.7), Q, P]
            hole = [(yk, 1.5), (yk - 0.6, 1.2), (yk - 0.6, 1.8)]
            out.append([shell, hole])
    return out


def family_1b():
    """A ring through another ring's edge at a vertex rounded onto it."""
    random.seed(7)
    out = []
    while len(out) < 4:
        y0, y1 = random.uniform(1.05, 1.95), random.uniform(1.05, 1.95)
        P, Q = (y0, 0.3), (y1, 2.7)
        vx = random.uniform(1.1, 1.9)
        vy = yat_f(P, Q, vx)
        if yat_exact(P, Q, vx) == F(vy):
            continue
        other = [(0.2, 1.0), (vy, vx), (3.3, 2.0), (3.3, 0.6)]
        shell = [P, Q, (3.8, 2.7), (3.8, 0.3)]
        out.append([shell, other])
    return out


def family_1c():
    """Two rings whose edges cross at slope differences 1e-4, 1e-7, 1e-10."""
    out = []
    for d in (1e-4, 1e-7, 1e-10):
        s0 = 0.37
        r1 = [(1.2, 0.2), (1.2 + s0 * 3.6, 3.8), (3.9, 3.8), (3.9, 0.2)]
        xc = 1.7 + 0.123
        yc = 1.2 + s0 * (xc - 0.2)
        sl = s0 + d
        r2 = [(yc + sl * (0.25 - xc), 0.25), (yc + sl * (3.75 - xc), 3.75),
              (0.1, 3.75), (0.1, 0.25)]
        out.append((d, [r1, r2]))
    return out


def family_1d():
    """A hole sharing part of a slanted shell edge: its two vertices on the
    edge rounded onto it, then one ulp inside the shell, then one outside."""
    out = []
    for side in (0, 1, -1):
        random.seed(9)
        for t in range(4):
            y0, y1 = random.uniform(1.05, 1.95), random.uniform(1.05, 1.95)
            P, Q = (y0, 0.3), (y1, 6.7)
            xa, xb = random.uniform(0.8, 2.9), random.uniform(3.6, 6.1)

            def off(x):
                y = yat_f(P, Q, x)
                if side:
                    while (F(y) - yat_exact(P, Q, x)) * side <= 0:
                        y = math.nextafter(y, side * math.inf)
                return (y, x)

            A, B = off(xa), off(xb)
            hole = [A, B, (B[0] + 2.1, xb - 0.4), (A[0] + 2.3, xa + 0.3)]
            shell = [P, Q, (5.8, 6.7), (5.8, 0.3)]
            out.append(({0: "on", 1: "inside", -1: "outside"}[side], t, [shell, hole]))
    return out

# ---------------------------------------------------------------- holes, islands, crossings

def R(r):
    return r[::-1]


def ring_cases():
    sq = [(0.5, 0), (0.5, 3), (3, 3), (3, 0)]
    yield "square", [sq], 5, 5
    yield "square-reversed", [R(sq)], 5, 5
    shell = [(0.3, 0.2), (0.1, 7.7), (6.9, 7.4), (7.2, 0.6)]
    hole = [(2.2, 2.1), (2.4, 5.3), (5.1, 5.6), (4.9, 1.9)]
    island = [(3.1, 3.0), (3.0, 4.2), (4.3, 4.4), (4.1, 3.2)]
    yield "hole-same-turn", [shell, hole], 9, 9
    yield "hole-opposite-turn", [shell, R(hole)], 9, 9
    for k in (0, 3, 5, 6):
        f = [shell if k & 1 else R(shell), hole if k & 2 else R(hole),
             island if k & 4 else R(island)]
        yield "island-in-lake-%d" % k, f, 9, 9
    sq8 = [(0.0, 0.0), (0.0, 8.0), (8.0, 8.0), (8.0, 0.0)]
    yield "traced-on-borders", [sq8], 10, 10
    yield "hole-at-a-shell-vertex", [sq8, [(0.0, 3.0), (4.0, 6.0), (4.0, 1.0)]], 10, 10
    yield "hole-sharing-a-shell-edge", [[(0.5, 0.5), (0.5, 6.5), (6.5, 6.5), (6.5, 0.5)],
                                        [(0.5, 2.3), (3.7, 2.3), (3.7, 4.6), (0.5, 4.6)]], 8, 8
    yield "squares-sharing-an-edge", [[(0.5, 0.5), (0.5, 3.5), (3.5, 3.5), (3.5, 0.5)],
                                      [(0.5, 3.5), (0.5, 6.5), (3.5, 6.5), (3.5, 3.5)]], 8, 8
    yield "bow-tie", [[(0.3, 0.4), (5.6, 6.3), (0.5, 6.1), (5.4, 0.2)]], 7, 7
    star = [(4 + 3.7 * math.sin(2 * math.pi * k * 2 / 5 + 0.3),
             4 + 3.7 * math.cos(2 * math.pi * k * 2 / 5 + 0.3)) for k in range(5)]
    yield "pentagram", [star], 9, 9
    yield "overlapping-squares", [[(0.5, 0.5), (0.5, 4.5), (4.5, 4.5), (4.5, 0.5)],
                                  [(2.3, 2.3), (2.3, 6.7), (6.7, 6.7), (6.7, 2.3)]], 8, 8
    yield "hole-crossing-its-shell", [[(0.5, 0.5), (0.5, 5.5), (5.5, 5.5), (5.5, 0.5)],
                                      [(3.2, 3.1), (3.3, 7.4), (6.8, 7.1), (6.6, 2.9)]], 9, 9
    comb = [(0.2, 0.3)]
    for k in range(6):
        comb += [(7.6, 0.3 + 2 * k + 0.4), (7.6, 0.3 + 2 * k + 1.1),
                 (1.0 + 0.1 * k, 0.3 + 2 * k + 1.5)]
    comb += [(1.0, 12.7), (0.2, 12.7)]
    yield "comb", [comb], 9, 14
    random.seed(1)
    for t in range(2):
        pts = [(random.uniform(0.2, 9.8), random.uniform(0.2, 9.8)) for _ in range(9)]
        yield "self-crossing-9-gon-%d" % t, [pts], 11, 11
    # The same island in a lake far from the grid's origin, so rows less the
    # feature's least row and columns less the cell's are what keep it exact.
    far = [[(y + 40000, x + 30000) for (y, x) in r] for r in (shell, R(hole), island)]
    yield "island-in-lake-far", far, 40010, 30010

# ---------------------------------------------------------------- GEOS

def read_shp(b):
    import struct
    off = 100
    out = []
    while off < len(b):
        _, clen = struct.unpack('>ii', b[off:off + 8])
        off += 8
        rec = b[off:off + 2 * clen]
        off += 2 * clen
        if struct.unpack('<i', rec[:4])[0] == 0:
            out.append([])
            continue
        nparts, npts = struct.unpack('<ii', rec[36:44])
        parts = list(struct.unpack('<%di' % nparts, rec[44:44 + 4 * nparts])) + [npts]
        p0 = 44 + 4 * nparts
        xy = struct.unpack('<%dd' % (2 * npts), rec[p0:p0 + 16 * npts])
        rings = []
        for a, z in zip(parts, parts[1:]):
            r = [(xy[2 * k], xy[2 * k + 1]) for k in range(a, z)]
            if len(r) > 1 and r[0] == r[-1]:
                r = r[:-1]
            rings.append(r)
        out.append(rings)
    return out


def geos_case(shp):
    feats = [f for f in read_shp(shp) if f]
    th = math.radians(2.5)
    c, s = math.cos(th), math.sin(th)
    X0 = min(x for f in feats for r in f for (x, _) in r) - 100
    Y0 = max(y for f in feats for r in f for (_, y) in r) + 100

    def px(X, Y):
        dx, dy = X - X0, Y - Y0
        return (-(s * dx + c * dy) / 10.0 + 0.37, (c * dx - s * dy) / 10.0 + 0.61)

    # Parcels up to 300 m across, to keep the file small.
    def small(f):
        xs = [x for r in f for (x, _) in r]
        ys = [y for r in f for (_, y) in r]
        return max(xs) - min(xs) < 300 and max(ys) - min(ys) < 300

    random.seed(2)
    multi = [f for f in feats if len(f) > 1 and small(f)]
    single = [f for f in feats if len(f) == 1 and small(f)]
    chosen = random.sample(multi, 5) + random.sample(single, 5)
    rings = [[[px(X, Y) for (X, Y) in r] for r in f] for f in chosen]
    rows = math.ceil(max(y for f in rings for r in f for (y, _) in r)) + 2
    cols = math.ceil(max(x for f in rings for r in f for (_, x) in r)) + 2
    emit("case belle-ile %d %d geos" % (rows, cols))
    worst = [F(0)]
    for f in rings:
        emit("feature")
        region = None
        for r in f:
            emit(ring_line(r))
            p = Polygon([(x, y) for (y, x) in r])
            region = p if region is None else region.symmetric_difference(p)
        i0, i1, j0, j1 = box_of(f, rows, cols)
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                a = region.intersection(box(j, i, j + 1, i + 1)).area
                if 0 < a < 1:
                    e = exact_cell(f, i, j)
                    worst[0] = max(worst[0], abs(F(a) - e))
                    emit("cell %d %d %r %r" % (i, j, a, float(e)))
    emit("end")
    emit("# GEOS within %.3g of the exact areas" % float(worst[0]))


emit("# Rasterize.areas: exact even-odd areas and GEOS's, written by")
emit("#   uv run contrib/jord/test/gen/areas.py")
emit("# (its docstring names the families and the data),")
emit("# with numpy 2.5.5 and shapely 2.2.0 (GEOS). Pixel points are (row, col);")
emit("# numbers are Python's repr, which float_of_string reads back exactly.")
emit("# 'case NAME ROWS COLS exact|geos', then per feature its rings and cells")
emit("# 'cell I J AREA': every cell of its box for exact (Fractions rounded")
emit("# once); for geos the cells strictly inside (0, 1), 'cell I J GEOS EXACT'.")
emit("# belle-ile is BreizhCrops' 2017 RPG parcels of Belle-Ile up to 300 m")
emit("# across, 5 with holes and 5 without, on a 10 m grid turned 2.5 degrees")
emit("# from Lambert-93. Its geometries are the RPG's, the Registre Parcellaire")
emit("# Graphique 2017 of IGN under Licence Ouverte 2.0 (test/README.md).")
for k, f in enumerate(family_1a()):
    exact_case("1a-%d" % k, f, 5, 4)
for k, f in enumerate(family_1b()):
    exact_case("1b-%d" % k, f, 5, 4)
for d, f in family_1c():
    exact_case("1c-%g" % d, f, 5, 4)
for side, t, f in family_1d():
    exact_case("1d-%s-%d" % (side, t), f, 7, 7)
for name, f, rows, cols in ring_cases():
    exact_case(name, f, rows, cols)
geos_case(fetch_shp())


text = "".join(_out)
if "--check" in sys.argv[1:]:
    if OUT.read_text() != text:
        sys.exit(f"{OUT} would change; run without --check")
else:
    OUT.write_text(text)
