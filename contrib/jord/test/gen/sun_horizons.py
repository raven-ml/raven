# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Generate test/sun_horizons.dat from JPL Horizons (DE441), and measure
USNO's formula, which Sun computes, against it.

    uv run contrib/jord/test/gen/sun_horizons.py [--check]

It fetches from https://ssd.jpl.nasa.gov/api/horizons.api (CSV, JD UT,
airless):

  - the Sun's geocentric apparent RA/Dec (true equator and equinox of date)
    and distance, every 3 days from 1950-01-01 to 2100-01-01;
  - the Sun's topocentric apparent azimuth and elevation at SITES, every
    STEP minutes from 1972-01-01 (UTC's leap seconds) to 2100-01-01.

Horizons' UT is UTC after 1962 (manual, "Time scales"); its topocentric
angles use JPL's EOP file for UT1-UTC and polar motion. The model below
(lib/sun.ml, line for line) takes UTC as UT1, so the comparison measures
the whole error a caller with a UTC instant sees.

It prints the statistics, for both GMST and GAST (USNO's equation of the
equinoxes), and writes the file: a regular subsample of each series plus
its worst rows, so that a test over the cut reaches each series' maximum.
Horizons' answers for instants after a fetch follow JPL's predicted ΔT and
EOP, so a later fetch can differ in their last digit, and so in which rows
are the worst: a fetch on 2026-11-18 moved 1e-7 degrees on rows after
2026-11-15 and changed 10 of the 766 rows. FETCHED is the date of the fetch
the file holds; --check fails once Horizons' answers have moved.

Without --check the file is written; with it, nothing is written and the
run fails if the file would change.
"""

import json
import math
import sys
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "sun_horizons.dat"
FETCHED = "2026-11-16"

API = "https://ssd.jpl.nasa.gov/api/horizons.api"

# (lon, lat): a site in Europe, the tropics on both sides, high latitudes on
# both sides, one beside the antimeridian, and two polar sites.
SITES = [
    (10.0, 50.0),
    (-60.0, -3.0),
    (140.0, -35.0),
    (-150.0, 65.0),
    (25.0, 70.0),
    (179.5, -17.0),
    (-70.0, -75.0),
    (-20.0, 80.0),
    (30.0, -88.0),
]
STEP = 9973  # minutes: not commensurate with the day, so hours sweep.
GEO_EVERY = 60  # every 60th 3-day row: 180 days apart
TOPO_EVERY = 90  # every 90th daylight row of a site
WORST = 6


def horizons(params):
    q = {"format": "json", "COMMAND": "'10'", "OBJ_DATA": "'NO'",
         "MAKE_EPHEM": "'YES'", "EPHEM_TYPE": "'OBSERVER'",
         "ANG_FORMAT": "'DEG'", "APPARENT": "'AIRLESS'", "EXTRA_PREC": "'YES'",
         "CSV_FORMAT": "'YES'", "CAL_FORMAT": "'JD'", "TIME_TYPE": "'UT'"}
    q.update(params)
    url = API + "?" + urllib.parse.urlencode(q)
    with urllib.request.urlopen(url, timeout=600) as r:
        res = json.load(r)["result"]
    if "$$SOE" not in res:
        raise SystemExit("Horizons: " + res[-2000:])
    head = res.split("$$SOE")[0]
    src = [l for l in head.splitlines() if "Target body name" in l][0]
    body = res.split("$$SOE")[1].split("$$EOE")[0]
    rows = [[f.strip() for f in l.split(",")] for l in body.strip().splitlines()]
    return src.strip(), rows


def posix(jd):
    # JD UT of a whole minute; Horizons prints 9 decimals (86 us).
    return round((float(jd) - 2440587.5) * 86400.0)


# jord's Sun, as lib/sun.ml computes it.
def usno(t):
    d = (t - 946728000.0) / 86400.0
    g = math.radians(357.529 + 0.98560028 * d)
    q = 280.459 + 0.98564736 * d
    lam = math.radians(q + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g))
    e = math.radians(23.439 - 0.00000036 * d)
    ra = math.atan2(math.cos(e) * math.sin(lam), math.cos(lam))
    dec = math.asin(math.sin(e) * math.sin(lam))
    r = 1.00014 - 0.01671 * math.cos(g) - 0.00014 * math.cos(2 * g)
    return d, ra, dec, r


def gmst(d):
    # USNO, "Computing Approximate Sidereal Time", the form in D alone (hours).
    return math.radians(15.0 * (18.697375 + 24.065709824279 * d))


def eqeq(d):
    # USNO, "Computing Approximate Sidereal Time": eqeq = dpsi cos(eps).
    om = math.radians(125.04 - 0.052954 * d)
    l = math.radians(280.47 + 0.98565 * d)
    dpsi = -0.000319 * math.sin(om) - 0.000024 * math.sin(2 * l)  # hours
    eps = math.radians(23.4393 - 0.0000004 * d)
    return math.radians(dpsi * math.cos(eps) * 15.0)


def angles(t, lon, lat, gast):
    d, ra, dec, _ = usno(t)
    th = gmst(d) + (eqeq(d) if gast else 0.0)
    h = th + math.radians(lon) - ra
    p = math.radians(lat)
    east = -math.cos(dec) * math.sin(h)
    north = math.cos(p) * math.sin(dec) - math.sin(p) * math.cos(dec) * math.cos(h)
    up = math.sin(p) * math.sin(dec) + math.cos(p) * math.cos(dec) * math.cos(h)
    z = math.degrees(math.atan2(math.hypot(east, north), up))
    az = math.degrees(math.atan2(east, north)) % 360.0
    return z, az


def separation(ra1, de1, ra2, de2):
    # Vincenty's form of the angle between two directions.
    dra = ra2 - ra1
    num = math.hypot(math.cos(de2) * math.sin(dra),
                     math.cos(de1) * math.sin(de2)
                     - math.sin(de1) * math.cos(de2) * math.cos(dra))
    den = (math.sin(de1) * math.sin(de2)
           + math.cos(de1) * math.cos(de2) * math.cos(dra))
    return math.atan2(num, den)


def stats(name, xs, unit, fmt=".4f"):
    s = sorted(xs)
    n = len(s)
    print(f"{name}: n {n}, max {s[-1]:{fmt}}{unit}, p99 {s[int(0.99 * (n - 1))]:{fmt}}{unit},"
          f" median {s[n // 2]:{fmt}}{unit}")


def wrap180(x):
    return (x + 180.0) % 360.0 - 180.0


def main():
    geo = []
    for a, b in [(1950, 2000), (2000, 2050), (2050, 2100)]:
        src, rows = horizons({"CENTER": "'500@399'", "QUANTITIES": "'2,20'",
                              "START_TIME": f"'{a}-01-01'",
                              "STOP_TIME": f"'{b}-01-01'",
                              "STEP_SIZE": "'3 d'"})
        for r in rows:
            t = posix(r[0])
            if geo and t <= geo[-1][0]:
                continue
            geo.append((t, float(r[3]), float(r[4]), float(r[5])))
    print(src)
    sep, dra, dde, ddist = [], [], [], []
    for t, ra, dec, dist in geo:
        _, ra1, dec1, r1 = usno(float(t))
        sep.append(math.degrees(separation(ra1, dec1, math.radians(ra), math.radians(dec))) * 60)
        dra.append(abs(wrap180(math.degrees(ra1) - ra)) * 60)
        dde.append(abs(math.degrees(dec1) - dec) * 60)
        ddist.append(abs(r1 - dist))
    for lo, hi in [(1950, 2100), (1980, 2040), (1972, 2030)]:
        a = (lo - 1970) * 31556952
        b = (hi - 1970) * 31556952
        idx = [i for i, g in enumerate(geo) if a <= g[0] < b]
        stats(f"RA/Dec separation {lo}-{hi}", [sep[i] for i in idx], "'")
        stats(f"  |dRA| {lo}-{hi}", [dra[i] for i in idx], "'")
        stats(f"  |dDec| {lo}-{hi}", [dde[i] for i in idx], "'")
        stats(f"  |dR| {lo}-{hi}", [ddist[i] for i in idx], " au", ".2e")

    keep = set(range(0, len(geo), GEO_EVERY))
    keep |= set(sorted(range(len(geo)), key=lambda i: -sep[i])[:WORST])
    keep |= set(sorted(range(len(geo)), key=lambda i: -ddist[i])[:WORST])

    topo = []
    for lon, lat in SITES:
        _, rows = horizons({"CENTER": "'coord@399'", "COORD_TYPE": "'GEODETIC'",
                            "SITE_COORD": f"'{lon},{lat},0'", "QUANTITIES": "'4'",
                            "START_TIME": "'1972-01-01'",
                            "STOP_TIME": "'2100-01-01'",
                            "STEP_SIZE": f"'{STEP} min'"})
        for r in rows:
            el = float(r[4])
            if el > 0:
                topo.append((posix(r[0]), lon, lat, float(r[3]), el))
    err = {False: [], True: []}
    for gast in (False, True):
        for t, lon, lat, az, el in topo:
            z, a = angles(float(t), lon, lat, gast)
            err[gast].append((abs(z - (90 - el)), abs(wrap180(a - az)) * math.cos(math.radians(el))))
        name = "GAST" if gast else "GMST"
        stats(f"zenith, {name}, {len(SITES)} sites 1972-2100", [e[0] for e in err[gast]], " deg")
        stats(f"azimuth x cos(el), {name}", [e[1] for e in err[gast]], " deg")
        for lon, lat in SITES:
            idx = [i for i, r in enumerate(topo) if (r[1], r[2]) == (lon, lat)]
            stats(f"  zenith at {lon},{lat}", [err[gast][i][0] for i in idx], " deg")
        a, b = (2015 - 1970) * 31556952, (2027 - 1970) * 31556952
        idx = [i for i, r in enumerate(topo) if (r[1], r[2]) == (10.0, 50.0) and a <= r[0] < b]
        stats(f"  zenith at 10,50 2015-2026, {name}", [err[gast][i][0] for i in idx], " deg")
        lo = [i for i, r in enumerate(topo) if r[0] < (2027 - 1970) * 31556952]
        stats(f"  zenith 1972-2026, {name}", [err[gast][i][0] for i in lo], " deg")
    e = err[False]
    tkeep = set()
    for lon, lat in SITES:
        idx = [i for i, r in enumerate(topo) if (r[1], r[2]) == (lon, lat)]
        tkeep |= set(idx[::TOPO_EVERY])
        tkeep |= set(sorted(idx, key=lambda i: -e[i][0])[:WORST])
        tkeep |= set(sorted(idx, key=lambda i: -e[i][1])[:WORST])

    out = []
    out.append(f"""\
# The Sun from JPL Horizons, {' '.join(src.split(':', 1)[1].split())}, fetched {FETCHED} by
#   uv run contrib/jord/test/gen/sun_horizons.py
# from {API}, airless, TIME_TYPE UT (UTC since 1972).
# g rows: POSIX seconds, the geocentric apparent right ascension and
#   declination (true equator and equinox of date) in degrees, the distance
#   in au; every {GEO_EVERY}th of a 3-day series 1950-2100, and its {WORST} rows
#   farthest from USNO's direction and distance.
# t rows: POSIX seconds, a site's longitude and latitude (WGS 84, height
#   0), the topocentric apparent azimuth and elevation in degrees, Sun
#   above the horizon; every {TOPO_EVERY}th of a {STEP}-minute series 1972-2100
#   per site, and each site's {WORST} rows of largest zenith and azimuth error.
""")
    for i in sorted(keep):
        t, ra, dec, dist = geo[i]
        out.append(f"g {t} {ra:.7f} {dec:.7f} {dist:.9f}\n")
    for i in sorted(tkeep, key=lambda i: (SITES.index(topo[i][1:3]), topo[i][0])):
        t, lon, lat, az, el = topo[i]
        out.append(f"t {t} {lon:g} {lat:g} {az:.6f} {el:.6f}\n")
    text = "".join(out)
    if "--check" in sys.argv[1:]:
        if OUT.read_text() != text:
            sys.exit(f"{OUT} would change; run without --check")
    else:
        OUT.write_text(text)
    print(f"{len(keep)} g rows, {len(tkeep)} t rows")


if __name__ == "__main__":
    main()
