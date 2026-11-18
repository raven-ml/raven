# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.4.1"]
# ///
"""Generate test/albers.dat and test/mercator_sinusoidal.dat: sweeps of
Crs's Albers, Pseudo-Mercator and sinusoidal maps against exact maps.

    uv run contrib/jord/test/gen/crs_sweep.py points DIR   # DIR/cases, DIR/*.in
    dune build ./contrib/jord/test/gen/crs_probe.exe
    ./_build/default/contrib/jord/test/gen/crs_probe.exe DIR   # DIR/*.eager
    uv run contrib/jord/test/gen/crs_sweep.py cut DIR [--check]
    uv run contrib/jord/test/gen/crs_sweep.py judge DIR    # prints the errors

The exact maps are mpmath's at 60 digits. Each takes and gives float64
numbers as exact rationals: the forward map of the double (lon, lat) in
degrees, and the inverse of the double (E, N) in metres. Albers is Snyder's
PP 1395 eqs. (14-1)-(14-6) and (3-12) as IOGP GN 7-2 §3.1.3 prints them
(method 9822); its inverse finds phi from q by bisection, so it needs no
series. Pseudo-Mercator is GN 7-2 §3.2.1.2 (method 1024); Sinusoidal is
Snyder's eqs. (30-1), (30-2), (30-6), (30-7).

Inputs and outputs are float64 as hex, one point a line. The forward
inputs are random points (a third uniform in latitude, a third uniform in
sin latitude, a third within 10^-12..1 degree of a pole) and special
points: the poles at many longitudes, the seam, the standard parallels,
the origin. The inverse inputs are the exact images of the forward ones,
rounded to float64. crs_probe.exe maps them through Crs.transform.

cut writes the two files: for each of their cases, its parameters and D,
then forward and inverse rows with exact values at 6 random points, the
poles at the seam and on lon0, the standard parallels, and the 4 points of
the sweep where Crs.transform erred most each way. Those 4 are the code's
of the day of the cut, 2026-11-16: a cut after a change to Crs or to nx's
functions picks the points that code errs most at. On 2026-11-18 the
probe's outputs had moved, and a cut kept mercator_sinusoidal.dat and moved
22 of albers.dat's rows. With --check nothing is written and the run fails
if a file would change.

judge states, for each case and direction, the largest error in the plane
or on the ground (the ground metric is the ellipsoid's: a dlat along the
meridian radius, cos(lat) N dlon along the parallel), and the mixed error:
for the forward map, |computed - f(p)| over 1 + the map's largest scale at
p; for the inverse, |f(computed) - x|, the backward error in the plane,
which implies the mixed form.
"""
import math
import os
import random
import sys
from pathlib import Path

import mpmath as mp

TEST = Path(__file__).resolve().parent.parent

mp.mp.dps = 60
A = mp.mpf(6378137)
FLAT = {"wgs84": 1 / 298.257223563, "grs80": 1 / 298.257222101}  # doubles


def rad(d):
    return mp.mpf(d) * mp.pi / 180


def deg(r):
    return r * 180 / mp.pi


def dlon(lon, lon0):
    """lon - lon0 reduced into [-180, 180), exactly."""
    d = mp.mpf(lon) - mp.mpf(lon0)
    return d - 360 * mp.floor((d + 180) / 360)


class Albers:
    def __init__(self, ell, lat0, lon0, lat1, lat2, fe=0.0, fn=0.0):
        f = mp.mpf(FLAT[ell])
        self.e2 = f * (2 - f); self.e = mp.sqrt(self.e2)
        self.lat0, self.lon0, self.fe, self.fn = lat0, lon0, mp.mpf(fe), mp.mpf(fn)
        p1, p2 = rad(lat1), rad(lat2)
        if lat1 == lat2:
            self.n = mp.sin(p1)
        else:
            self.n = (self.m(p1) ** 2 - self.m(p2) ** 2) / (self.q(p2) - self.q(p1))
        self.C = self.m(p1) ** 2 + self.n * self.q(p1)
        self.rho0 = self.rho(rad(lat0))
        self.qp = self.q(mp.pi / 2)

    def q(self, p):
        s = mp.sin(p); e, e2 = self.e, self.e2
        return (1 - e2) * (s / (1 - e2 * s * s) - mp.log((1 - e * s) / (1 + e * s)) / (2 * e))

    def m(self, p):
        s = mp.sin(p)
        return mp.cos(p) / mp.sqrt(1 - self.e2 * s * s)

    def rho(self, p):
        return A * mp.sqrt(max(self.C - self.n * self.q(p), 0)) / self.n

    def fwd(self, lon, lat):
        return self.fwd_d(dlon(lon, self.lon0), lat)

    def fwd_d(self, d, lat):
        """The image of the point d degrees east of lon0, d unreduced."""
        th = self.n * rad(d)
        r = self.rho(rad(lat))
        return self.fe + r * mp.sin(th), self.fn + self.rho0 - r * mp.cos(th)

    def inv(self, x, y):
        x = mp.mpf(x) - self.fe; y = self.rho0 - (mp.mpf(y) - self.fn)
        sg = 1 if self.n > 0 else -1
        r = sg * mp.sqrt(x * x + y * y)
        th = mp.atan2(sg * x, sg * y)
        lam = th / self.n
        # Within float64's resolution of the image's edge, the edge.
        if abs(lam) > mp.pi * (1 + mp.mpf(1e-12)):
            return None
        lam = max(-mp.pi, min(mp.pi, lam))
        q = (self.C - (r * self.n / A) ** 2) / self.n
        if abs(q) > self.qp * (1 + mp.mpf(1e-12)):
            return None
        q = max(-self.qp, min(self.qp, q))
        lo, hi = -mp.pi / 2, mp.pi / 2
        for _ in range(230):
            mid = (lo + hi) / 2
            if self.q(mid) < q: lo = mid
            else: hi = mid
        return mp.mpf(self.lon0) + deg(lam), deg((lo + hi) / 2)

    def scale(self, lon, lat):
        """(h, k), the scales along the meridian and the parallel."""
        p = rad(lat)
        if abs(lat) == 90:
            return mp.mpf(0), mp.inf
        k = self.rho(p) * self.n / (A * self.m(p))
        return 1 / k, k


class PseudoMercator:
    def __init__(self, lat0=0.0, lon0=0.0, fe=0.0, fn=0.0):
        self.lat0, self.lon0, self.fe, self.fn = lat0, lon0, mp.mpf(fe), mp.mpf(fn)
        self.y0 = A * mp.asinh(mp.tan(rad(lat0)))

    def fwd(self, lon, lat):
        return self.fwd_d(dlon(lon, self.lon0), lat)

    def fwd_d(self, d, lat):
        return (self.fe + A * rad(d),
                self.fn + A * mp.asinh(mp.tan(rad(lat))) - self.y0)

    def inv(self, x, y):
        lam = (mp.mpf(x) - self.fe) / A
        if abs(lam) > mp.pi:
            return None
        phi = mp.atan(mp.sinh((mp.mpf(y) - self.fn + self.y0) / A))
        return mp.mpf(self.lon0) + deg(lam), deg(phi)

    def scale(self, lon, lat):
        s = 1 / mp.cos(rad(lat))
        return s, s


class Sinusoidal:
    def __init__(self, radius=6371007.181, lon0=0.0, fe=0.0, fn=0.0):
        self.R = mp.mpf(radius); self.lon0, self.fe, self.fn = lon0, mp.mpf(fe), mp.mpf(fn)

    def fwd(self, lon, lat):
        return self.fwd_d(dlon(lon, self.lon0), lat)

    def fwd_d(self, d, lat):
        p = rad(lat)
        return self.fe + self.R * rad(d) * mp.cos(p), self.fn + self.R * p

    def inv(self, x, y):
        p = (mp.mpf(y) - self.fn) / self.R
        if abs(p) > mp.pi / 2:
            return None
        c = mp.cos(p)
        if c == 0:
            return mp.mpf(self.lon0), deg(p)
        lam = (mp.mpf(x) - self.fe) / (self.R * c)
        if abs(lam) > mp.pi:
            return None
        return mp.mpf(self.lon0) + deg(lam), deg(p)

    def scale(self, lon, lat):
        """The largest singular value of the map's Jacobian against the
        ground metric of the sphere, and its least."""
        p = rad(lat); lam = rad(dlon(lon, self.lon0))
        a = lam * mp.sin(p)  # dx/dphi / R, the shear; dx/dlam/(R cos) = 1, dy/dphi/R = 1
        # J in (east, north) ground units: [[1, -a], [0, 1]]
        t = mp.sqrt((2 + a * a) ** 2 - 4)
        return mp.sqrt((2 + a * a - t) / 2), mp.sqrt((2 + a * a + t) / 2)


# The sweep

# name, method, ellipsoid, parameters (lat0, lon0, lat1, lat2, fe, fn) for
# Albers; (lat0, lon0, fe, fn) for Pseudo-Mercator; (radius, lon0, fe, fn) for
# Sinusoidal.
CASES = [
    ("5070", "albers", "grs80", (23.0, -96.0, 29.5, 45.5, 0.0, 0.0)),
    ("3577", "albers", "grs80", (0.0, 132.0, -18.0, -36.0, 0.0, 0.0)),
    ("3338", "albers", "grs80", (50.0, -154.0, 55.0, 65.0, 0.0, 0.0)),
    ("small_n", "albers", "wgs84", (0.0, 0.0, 20.0, -10.0, 0.0, 0.0)),
    ("sym_1", "albers", "grs80", (0.0, 0.0, 30.0, -29.0, 0.0, 0.0)),
    ("sym_1e-4", "albers", "grs80", (0.0, 0.0, 30.0, -29.9999, 0.0, 0.0)),
    ("high", "albers", "grs80", (75.0, 0.0, 60.0, 89.0, 0.0, 0.0)),
    ("apex_60", "albers", "grs80", (90.0, 0.0, 60.0, 90.0, 0.0, 0.0)),
    ("apex_89", "albers", "grs80", (90.0, 0.0, 89.0, 90.0, 0.0, 0.0)),
    ("apex_south", "albers", "wgs84", (-90.0, 30.0, -60.0, -90.0, 0.0, 0.0)),
    ("close", "albers", "grs80", (45.0, 0.0, 45.0, 45.000001, 0.0, 0.0)),
    ("equal", "albers", "grs80", (45.0, 10.0, 45.0, 45.0, 1e6, 2e6)),
    ("opposite", "albers", "grs80", (0.0, 0.0, 89.0, -88.9, 0.0, 0.0)),
    ("near_pole_4", "albers", "grs80", (89.0, 0.0, 60.0, 89.9999, 0.0, 0.0)),
    ("near_pole_8", "albers", "grs80", (89.0, 0.0, 60.0, 89.99999999, 0.0, 0.0)),
    ("near_pole_12", "albers", "wgs84", (89.0, 0.0, 10.0, 89.999999999999, 0.0, 0.0)),
    ("near_pole_s", "albers", "wgs84", (-89.0, 0.0, -45.0, -89.99999, 0.0, 0.0)),
    ("3857", "pm", "wgs84", (0.0, 0.0, 0.0)),
    ("pm_off", "pm", "grs80", (150.0, 1e6, -2e6)),
    ("modis", "sinusoidal", "wgs84", (6371007.181, 0.0, 0.0, 0.0)),
    ("sinu_off", "sinusoidal", "wgs84", (6371007.181, -120.0, 1e6, 2e6)),
]


# 120 cones of random parameters, parallels uniform in latitude.
_rng = random.Random(7)
for _k in range(120):
    while True:
        _l1, _l2 = _rng.uniform(-90, 90), _rng.uniform(-90, 90)
        if abs(_l1 + _l2) > 1e-3:
            break
    CASES.append((f"rand{_k}", "albers", _rng.choice(["wgs84", "grs80"]),
                  (_rng.uniform(-90, 90), _rng.uniform(-180, 180), _l1, _l2, 0.0, 0.0)))


def system(name):
    for c in CASES:
        if c[0] == name:
            _, method, ell, p = c
            if method == "albers":
                return Albers(ell, *p)
            if method == "pm":
                return PseudoMercator(0.0, *p)
            return Sinusoidal(*p)
    raise KeyError(name)


def hx(x):
    return float(x).hex()


def write(path, pts):
    with open(path, "w") as f:
        for a, b in pts:
            f.write(f"{hx(a)} {hx(b)}\n")


def read(path):
    out = []
    with open(path) as f:
        for l in f:
            a, b = l.split()
            out.append((float.fromhex(a), float.fromhex(b)))
    return out


def lon_wrap(x):
    return x - 360 * math.floor((x + 180) / 360)


def points(name, rng):
    _, method, _, p = next(c for c in CASES if c[0] == name)
    lon0 = {"albers": p[1] if len(p) > 1 else 0, "pm": p[0], "sinusoidal": p[1] if len(p) > 1 else 0}[method]
    pts = []
    for k in range(400 if name.startswith("rand") else 3000):
        lon = rng.uniform(-180, 180)
        kind = k % 3
        if kind == 0:
            lat = rng.uniform(-90, 90)
        elif kind == 1:
            lat = math.degrees(math.asin(rng.uniform(-1, 1)))
            if method == "pm":
                lat = rng.uniform(80, 88) * rng.choice((-1, 1))
        else:
            lat = math.copysign(90 - 10 ** rng.uniform(-12, 0), rng.choice((-1, 1)))
        pts.append((lon, lat))
    lats = [90.0, -90.0, 89.9999999, -89.9999999, 89.99, -89.99, 0.0, 45.0, -45.0]
    if method == "albers":
        lats += [p[0], p[2], p[3]]
    if method == "pm":
        lats = [88.0, -88.0, 87.9999999999, 85.06, -85.06, 0.0, 45.0, 87.0]
    for lat in lats:
        for lon in (-180.0, -90.0, 0.0, 45.0, 179.99999999, lon0,
                    lon_wrap(lon0 + 180.0), lon_wrap(lon0 + 180.0) + 1e-9,
                    lon_wrap(lon0 - 179.999999999), lon_wrap(lon0 + 40.0)):
            pts.append((lon, lat))
    return pts


def main_points(d):
    os.makedirs(d, exist_ok=True)
    rng = random.Random(20261116)
    with open(os.path.join(d, "cases"), "w") as f:
        for name, method, ell, p in CASES:
            f.write(" ".join([name, method, ell] + [repr(float(x)) for x in p]) + "\n")
    for name, *_ in CASES:
        m = system(name)
        fwd = points(name, rng)
        write(os.path.join(d, name + ".fwd.in"), fwd)
        inv = []
        for lon, lat in fwd:
            x, y = m.fwd(lon, lat)
            inv.append((float(x), float(y)))
        write(os.path.join(d, name + ".inv.in"), inv)
        print(name, len(fwd), "points", flush=True)


def ground(lat, dlat_deg, dlon_deg):
    """Metres on the WGS 84 sphere of radius a, as the module's slack is."""
    r = 6378137.0
    return math.hypot(dlat_deg * math.pi / 180 * r,
                      dlon_deg * math.pi / 180 * r * math.cos(math.radians(lat)))


def band(lat):
    a = abs(lat)
    for lo, hi in ((0, 60), (60, 80), (80, 89), (89, 89.9999), (89.9999, 90.1)):
        if lo <= a < hi:
            return f"|lat| {lo}-{hi}"
    return "?"


def tangent(m):
    """The latitude of least scale between the parallels, by golden section
    on k, in degrees."""
    lo, hi = -90.0, 90.0
    g = (mp.sqrt(5) - 1) / 2
    a, b = mp.mpf(lo) + 1e-9, mp.mpf(hi) - 1e-9
    k = lambda x: m.scale(0, x)[1]
    for _ in range(200):
        c1 = b - g * (b - a); c2 = a + g * (b - a)
        if k(c1) < k(c2): b = c2
        else: a = c1
    return (a + b) / 2


def extent(m, method, p, fin):
    """D: the largest distance of an exact image of the sweep from the
    image of the central meridian at the latitude of least scale."""
    lat_t = tangent(m) if method == "albers" else mp.mpf(0)
    lon0 = {"albers": p[1], "pm": p[0], "sinusoidal": p[1]}[method]
    ox, oy = m.fwd(lon0, float(lat_t))
    D = 0
    for lon, lat in fin:
        if method == "pm" and abs(lat) > 88:
            continue
        x, y = m.fwd(lon, lat)
        D = max(D, mp.sqrt((x - ox) ** 2 + (y - oy) ** 2))
    return float(lat_t), float(D)


def main_judge(d):
    u = 2.0 ** -53
    summary = []
    only = os.environ.get("ONLY")
    for name, method, ell, p in CASES:
        if only and method not in only.split(","):
            continue
        m = system(name)
        if not os.path.exists(os.path.join(d, name + ".fwd.eager")):
            continue
        fin = read(os.path.join(d, name + ".fwd.in"))
        iin = read(os.path.join(d, name + ".inv.in"))
        lat_t, D = extent(m, method, p, fin)
        lon0 = {"albers": p[1], "pm": p[0], "sinusoidal": p[1]}[method]
        print(f"{name}: tangent latitude {lat_t:.9f}, D {D:.6g} m, u D {u * D * 1e9:.3f} nm")
        paths = ["eager"]
        for path in paths:
            fout = read(os.path.join(d, f"{name}.fwd.{path}"))
            iout = read(os.path.join(d, f"{name}.inv.{path}"))
            worst = {}
            def keep(key, v, at):
                if key not in worst or v > worst[key][0]:
                    worst[key] = (v, at)
            nan_f = nan_i = 0
            for (lon, lat), (x, y) in zip(fin, fout):
                if math.isnan(x):
                    nan_f += 1; continue
                ex, ey = m.fwd(lon, lat)
                err = float(mp.sqrt((x - ex) ** 2 + (y - ey) ** 2))
                h, k = m.scale(lon, lat)
                s = float(max(h, k))
                keep("fwd plane", err, (lon, lat))
                keep("fwd mixed", err / (1 + s) if math.isfinite(s) else 0.0, (lon, lat))
            for (lon, lat), (x, y), (olon, olat) in zip(fin, iin, iout):
                if math.isnan(olon):
                    nan_i += 1; continue
                fx, fy = m.fwd(olon, olat)
                back = float(mp.sqrt((fx - x) ** 2 + (fy - y) ** 2))
                # The seam's meridian has an image on each side: take the
                # nearer, of lon0 - 180 and of lon0 + 180.
                d0 = dlon(olon, lon0)
                if abs(float(d0)) > 179.999999:
                    gx, gy = m.fwd_d(d0 - 360 if d0 > 0 else d0 + 360, olat)
                    back = min(back, float(mp.sqrt((gx - x) ** 2 + (gy - y) ** 2)))
                # The mixed form also holds with x itself as the witness
                # where the result lies near x's exact preimage on the ground.
                pre = m.inv(x, y)
                if pre is not None:
                    dl = float(dlon(olon, pre[0]))
                    back = min(back, ground(olat, olat - float(pre[1]), dl))
                keep("inv backward, plane", back, (x, y))
                keep(f"inv backward, plane, {band(lat)}", back, (x, y))
            print(f"{name} {path}: NaN {nan_f} fwd, {nan_i} inv")
            summary.append((name, path, nan_f, nan_i, worst.get("fwd plane", (0,))[0] / (u * D),
                            worst.get("inv backward, plane", (0,))[0] / (u * D), D))
            for key in sorted(worst):
                v, at = worst[key]
                print(f"    {key:38s} {v * 1e9:10.3f} nm  {v / (u * D):7.2f} u D  at {at}")
        sys.stdout.flush()
    for path in ("eager",):
        rows = [r for r in summary if r[1] == path]
        f = max(rows, key=lambda r: r[4]); i = max(rows, key=lambda r: r[5])
        print(f"SUMMARY {path}: worst forward {f[4]:.2f} u D ({f[0]}, D {f[6]:.4g}), worst inverse "
              f"backward {i[5]:.2f} u D ({i[0]}, D {i[6]:.4g}), refusals {sum(r[3] for r in rows)}")


def cut(d, names, method_label):
    """The committed fixture: for each case, its parameters and D, then
    forward rows (lon, lat, E, N, h, k) and inverse rows (E, N, lon, lat, h,
    k) with exact values, h and k the scales along the meridian and the
    parallel at the point. The rows are 6 random points, the poles at the
    seam and on lon0, the standard parallels, and the 4 points of the sweep
    where Crs.transform erred most each way (DIR's eager outputs)."""
    rng = random.Random(11)
    u = 2.0 ** -53
    lines = []
    for name in names:
        _, method, ell, p = next(c for c in CASES if c[0] == name)
        m = system(name)
        fin = read(os.path.join(d, name + ".fwd.in"))
        iin = read(os.path.join(d, name + ".inv.in"))
        fout = read(os.path.join(d, name + ".fwd.eager"))
        iout = read(os.path.join(d, name + ".inv.eager"))
        _, D = extent(m, method, p, fin)
        ferr = []
        for k, ((lon, lat), (x, y)) in enumerate(zip(fin, fout)):
            if math.isnan(x):
                continue
            ex, ey = m.fwd(lon, lat)
            ferr.append((float(mp.sqrt((x - ex) ** 2 + (y - ey) ** 2)), k))
        ierr = []
        for k, ((x, y), (olon, olat)) in enumerate(zip(iin, iout)):
            if math.isnan(olon):
                continue
            pre = m.inv(x, y)
            if pre is None or abs(float(pre[1])) == 90:
                continue
            ierr.append((ground(olat, olat - float(pre[1]), float(dlon(olon, pre[0]))), k))
        inside = [k for k, (lon, lat) in enumerate(fin)
                  if not (method == "pm" and abs(lat) > 88)]
        pick = set(rng.sample(inside, 6))
        pick |= {k for _, k in sorted(ferr)[-4:]} | {k for _, k in sorted(ierr)[-4:]}
        pick |= set(range(len(fin) - 40, len(fin), 7))  # special points
        pick = {k for k in pick if k in set(inside)}
        lines.append(f"case {name} {method} {ell} " + " ".join(repr(float(x)) for x in p) + f" D {D!r}")
        for k in sorted(pick):
            lon, lat = fin[k]
            ex, ey = m.fwd(lon, lat)
            h, kk = m.scale(lon, lat)
            sc = lambda v: "inf" if not mp.isfinite(v) else mp.nstr(v, 17)
            lines.append(f"fwd {lon!r} {lat!r} {mp.nstr(ex, 25)} {mp.nstr(ey, 25)} {sc(h)} {sc(kk)}")
            x, y = iin[k]
            pre = m.inv(x, y)
            if pre is not None:
                h, kk = m.scale(float(pre[0]), float(pre[1]))
                lines.append(f"inv {x!r} {y!r} {mp.nstr(pre[0], 25)} {mp.nstr(pre[1], 25)} {sc(h)} {sc(kk)}")
    return (f"# {method_label} against mpmath at 60 digits, written by\n"
            f"#   uv run contrib/jord/test/gen/crs_sweep.py cut DIR\n"
            f"# (its docstring gives the sweep). Each case line gives\n"
            f"# the parameters and D, the largest distance of the image from the\n"
            f"# tangent point's (Albers) or the origin's; then rows of\n"
            f"#   fwd lon lat E N h k  and  inv E N lon lat h k,\n"
            f"# inputs as float64, exact outputs, and the scales along the meridian\n"
            f"# and the parallel at the point.\n"
            + "\n".join(lines) + "\n")


FILES = [
    ("albers.dat", ["5070", "3577", "high", "apex_60", "apex_south", "opposite",
                    "sym_1e-4", "equal"], "Albers' equal-area conic"),
    ("mercator_sinusoidal.dat", ["3857", "pm_off", "modis", "sinu_off"],
     "Pseudo-Mercator and the sinusoidal projection"),
]


def main_cut(d, check):
    for name, names, label in FILES:
        text = cut(d, names, label)
        out = TEST / name
        if check:
            if out.read_text() != text:
                sys.exit(f"{out} would change; run without --check")
        else:
            out.write_text(text)


if __name__ == "__main__":
    mode, d = sys.argv[1], sys.argv[2]
    if mode == "cut":
        main_cut(d, "--check" in sys.argv[3:])
    else:
        {"points": main_points, "judge": main_judge}[mode](d)
