# /// script
# requires-python = ">=3.11"
# dependencies = ["pyerfa==2.0.1.5"]
# ///
"""Generate the Earth's series and the Earth's test references.

    uv run contrib/ymir/test/gen/earth.py [--check]

The sources, each read from its tagged URL and pinned by its SHA-256 below,
are ERFA's, the IAU SOFA library's functions under ERFA's BSD license:

    nut00a.c   eraNut00a, the IAU 2000A nutation (Mathews et al. 2002):
               678 luni-solar and 687 planetary terms.
    s06.c      eraS06, the series of the CIO locator s + XY/2 of the
               IAU 2006/2000A precession-nutation (IERS Conventions 2010,
               Table 5.2d): its polynomial and 66 terms.
    epv00.c    eraEpv00, the Earth's heliocentric and barycentric position
               and velocity, a simplified VSOP2000 (Moisson & Bretagnon
               2001): 1951 terms and the rotation onto DE405's axes.

A source whose SHA-256 is not its pin fails the run with "source changed:
update pins and regenerate".

It writes:

lib/earth_tables.ml
    The three functions' terms as ERFA writes them, with ERFA's copyright
    notice and license, which its terms ask for.

test/support/earth_reference.ml
    At TT instants drawn from 1900 to 2100, and a few from 1600 to 2400,
    with UT1 a drawn offset from TT: the Earth rotation angle and GMST,
    their definitions (IERS Conventions 2010, 5.14 and 5.32) evaluated
    exactly and rounded once, which ERFA's eraEra00 rounds to 4e-13 rad;
    eraEo06a's equation of the origins; eraPnm06a's and eraC2i06a's matrices; eraEpv00's
    heliocentric and barycentric position and velocity at the instant's
    TDB; and for fixed ICRS directions, the GCRS proper direction of
    eraLdsun then eraAb with the parameters eraApcs gives at the geocentre
    from eraEpv00, as eraAtciq applies them, and its CIRS direction through
    eraC2i06a. Directions are drawn near the Sun too, inside eraLdsun's
    limiter and out of it. And UTC instants, a leap second's among them,
    with UT1 - UTC: the Earth rotation angle at eraUtcut1's UT1.

An instant there is a modified Julian date's day and the seconds since its
0h; ERFA reads it as the two-part Julian date (2400000.5 + day,
seconds / 86400). TDB is the fixed point of tdb = tt + dtdb(tdb) at the
geocentre, reached in two steps, as test/gen/times.py reads it.

Without --check both files are written; with it, nothing is written and the
run fails if a file would change.
"""

import argparse
from fractions import Fraction
import hashlib
import math
from pathlib import Path
import random
import re
import sys
import urllib.request
import warnings

import erfa

# eraEpv00 flags a date outside 1900-2100 and computes it as any other: the
# references draw a few such dates on purpose.
warnings.filterwarnings("ignore", message=".*1900-2100.*")
# ERFA flags a UTC year past its release as dubious and computes it as any
# other: the references draw such UTC years on purpose.
warnings.filterwarnings("ignore", message=".*dubious year.*")

HERE = Path(__file__).resolve().parent
TABLES = HERE.parents[1] / "lib" / "earth_tables.ml"
REFERENCE = HERE.parent / "support" / "earth_reference.ml"

BASE = "https://raw.githubusercontent.com/liberfa/erfa/v2.0.1/src/"
NUT00A = BASE + "nut00a.c"
S06 = BASE + "s06.c"
EPV00 = BASE + "epv00.c"

PINS = {
    NUT00A: "a1538ec0277745db974ab76ca4b678a7602249bccecf9474e2b0268f9c2032ef",
    S06: "6ad19dd59c3150d9f065b3f76d254c1cf3adb0cb6bacd6422c1dc6b6f14122ea",
    EPV00: "4568b28026cd83800b4a4f66d44f8f3451d7306f1c3fcf7e4cd6bfeb206dde57",
}

HEADER = """(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)
"""

TABLES_HEADER = """(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  Copyright (C) 2013-2023, NumFOCUS Foundation (ERFA's terms of the
  nutation, the CIO locator and the Earth's ephemeris).
  SPDX-License-Identifier: ISC AND BSD-3-Clause
  ---------------------------------------------------------------------------*)
"""

NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?"


def source(url):
    with urllib.request.urlopen(url) as r:
        data = r.read()
    if hashlib.sha256(data).hexdigest() != PINS[url]:
        sys.exit(f"source changed: {url}: update pins and regenerate")
    return data.decode("latin-1")


def literal(text):
    """An OCaml float literal of a C decimal literal, the same digits."""
    text = text.lstrip("+")
    if text.startswith("-."):
        text = "-0." + text[2:]
    elif text.startswith("."):
        text = "0." + text[1:]
    if "." not in text and "e" not in text:
        text += "."
    return text


def table(text, name):
    """The body of the C array [name], up to its closing brace."""
    body = text.split(name, 1)
    if len(body) != 2:
        sys.exit(f"no table {name}")
    return body[1].split("};", 1)[0]


def rows(body, width):
    """The brace-delimited rows of a table of structs, each [width]
    numbers."""
    out = []
    for row in re.findall(r"\{([^{}]*(?:\{[^{}]*\}[^{}]*)?)\}", body):
        numbers = re.findall(NUMBER, row.replace("{", " ").replace("}", " "))
        if len(numbers) != width:
            sys.exit(f"a row has {len(numbers)} numbers, not {width}: {row!r}")
        out.append(numbers)
    return out


# eraNut00a

def nutation(text):
    lunisolar = rows(table(text, "xls[] = {"), 11)
    planetary = rows(table(text, "xpl[] = {"), 17)
    if len(lunisolar) != 678 or len(planetary) != 687:
        sys.exit(f"nut00a.c has {len(lunisolar)} and {len(planetary)} terms")
    return lunisolar, planetary


# eraS06

def cio(text):
    poly = re.findall(NUMBER, table(text, "sp[] = {").split("*/", 1)[1])
    if len(poly) != 6:
        sys.exit("s06.c's polynomial is not 6 coefficients")
    terms = []
    counts = []
    for power in range(5):
        group = rows(table(text, f"TERM s{power}[] = {{"), 10)
        counts.append(len(group))
        terms += [[str(power)] + row for row in group]
    if counts != [33, 3, 25, 4, 1]:
        sys.exit(f"s06.c's series has {counts} terms by power")
    return poly, terms


# eraEpv00

ROTATION = ["am12", "am13", "am21", "am22", "am23", "am32", "am33"]


def ephemeris(text):
    body = text.split("{", 1)[1]
    terms = []
    for power in range(3):
        for column, (model, axis) in enumerate(
                [(m, a) for m in "es" for a in "xyz"]):
            name = f"{model}{power}{axis}[] = {{"
            numbers = re.findall(NUMBER, table(body, name))
            if len(numbers) % 3:
                sys.exit(f"epv00.c's {name} is not triples")
            for i in range(0, len(numbers), 3):
                terms.append([str(power), str(column)] + numbers[i:i + 3])
    if len(terms) != 1951:
        sys.exit(f"epv00.c has {len(terms)} terms, not 1951")
    values = {}
    for name in ROTATION:
        match = re.search(name + r"\s*=\s*(" + NUMBER + ")", body)
        if not match:
            sys.exit(f"epv00.c has no {name}")
        values[name] = match.group(1)
    v = values
    rotation = ["1.0", v["am12"], v["am13"], v["am21"], v["am22"], v["am23"],
                "0.0", v["am32"], v["am33"]]
    return terms, rotation


# Rendering

def flat_rows(rows_, indent="  "):
    return "\n".join(indent + "; ".join(literal(x) for x in r) + ";"
                     for r in rows_)


def erfa_notice(text):
    """ERFA's copyright notice and license, as each source ends."""
    start = text.rfind("Copyright (C)")
    end = text.rfind("*/")
    lines = [line.removeprefix("**").rstrip()
             for line in text[start:end].splitlines()]
    return "\n".join(("  " + line.strip(" ")) if line.strip() else ""
                     for line in lines).rstrip()


def render_tables(texts):
    nut, s06, epv = texts
    lunisolar, planetary = nutation(nut)
    poly, cio_terms = cio(s06)
    eph, rotation = ephemeris(epv)
    out = [TABLES_HEADER, """
(* Written by test/gen/earth.py; do not edit. Regenerate with
   uv run contrib/ymir/test/gen/earth.py. Each source and its SHA-256:
"""]
    for url in (NUT00A, S06, EPV00):
        out.append(f"     {url}\n       {PINS[url]}\n")
    out.append("""
   The terms are ERFA's eraNut00a's, eraS06's and eraEpv00's, the IAU SOFA
   library's iauNut00a, iauS06 and iauEpv00 as ERFA distributes them under
   the license at the end of this file and in LICENSE-erfa. They are
   derived from ERFA and are neither ERFA nor SOFA software, nor endorsed
   by either. {!Earth} sums the same series, evaluated in nx. *)

[@@@ocamlformat "disable"]

(* The IAU 2000A luni-solar nutation, 678 terms of 11 numbers: the
   multipliers of the Delaunay arguments l, l', F, D, Om, then sp, spt,
   cp, ce, cet, se in 0.1 microarcseconds. The term adds
   (sp + spt t) sin a + cp cos a to the nutation in longitude and
   (ce + cet t) cos a + se sin a to the nutation in obliquity, a the
   argument and t TT in Julian centuries since J2000.0. *)
let lunisolar = [|
""")
    out.append(flat_rows(lunisolar))
    out.append("""
|]

(* The IAU 2000A planetary nutation, 687 terms of 17 numbers: the
   multipliers of l, F, D, Om, the mean longitudes of Mercury to Neptune
   and the general precession in longitude, then sp, cp, se, ce in 0.1
   microarcseconds. The term adds sp sin a + cp cos a to the nutation in
   longitude and se sin a + ce cos a to the nutation in obliquity. *)
let planetary = [|
""")
    out.append(flat_rows(planetary))
    out.append("""
|]

(* The CIO locator: s + XY/2 is [sum_k w_k t^k] arcseconds, w_k the
   polynomial's coefficient k plus the terms of power k. *)
let cio_polynomial = [|
""")
    out.append(flat_rows([poly]))
    out.append("""
|]

(* Its 66 terms of 11 numbers: the power k, the multipliers of l, l', F,
   D, Om, the mean longitudes of Venus and the Earth and the general
   precession in longitude, then s and c in arcseconds. The term adds
   s sin a + c cos a to w_k. *)
let cio = [|
""")
    out.append(flat_rows(cio_terms))
    out.append("""
|]

(* The Earth's ephemeris, 1951 terms of 5 numbers: the power k of T, the
   column (0-2 the Sun-to-Earth x, y, z, 3-5 the barycentre-to-Sun x, y,
   z, on the ecliptic axes of the model), then a in au, b in radians and
   c in radians per Julian year. The term adds a T^k cos (b + c T) to its
   column, T TDB in Julian years since J2000.0. *)
let ephemeris = [|
""")
    out.append(flat_rows(eph))
    out.append("""
|]

(* The rotation from the model's axes onto DE405's, row-major: eraEpv00's
   am12 to am33, found by comparison with DE405 over 1900-2100. *)
let ephemeris_rotation = [|
""")
    out.append(flat_rows([rotation[0:3], rotation[3:6], rotation[6:9]]))
    out.append("\n|]\n\n(* ERFA's notice, from its sources:\n\n")
    out.append(erfa_notice(nut).replace("(*", "( *").replace("*)", "* )"))
    out.append("\n*)\n")
    return "".join(out)


# References

def tdb_of_tt(t1, t2):
    dtr = erfa.dtdb(t1, t2, 0.0, 0.0, 0.0, 0.0)
    dtr = erfa.dtdb(t1, t2 + dtr / 86400.0, 0.0, 0.0, 0.0, 0.0)
    return erfa.tttdb(t1, t2, dtr)


def mjd(y, m, d):
    return int(round(erfa.cal2jd(y, m, d)[1]))


def instants():
    """(TT day, TT seconds, UT1 - TT in seconds)."""
    rng = random.Random(2006)
    out = [(51544, 43200.0, -64.0)]
    first, last = mjd(1900, 1, 1), mjd(2100, 1, 1)
    for _ in range(40):
        out.append((rng.randrange(first, last), round(rng.uniform(0, 86400), 6),
                    round(rng.uniform(-120, 0), 6)))
    for y in (1600, 1750, 2250, 2400):
        out.append((mjd(y, 6, 1), round(rng.uniform(0, 86400), 6),
                    round(rng.uniform(-500, 500), 6)))
    return out


# pi to 50 digits, so that 2 pi times an exact number of turns rounds once.
PI = Fraction("3.14159265358979323846264338327950288419716939937511")


def turns_radians(turns):
    """An exact number of turns, less whole turns, in radians, rounded
    once."""
    return float((turns - math.floor(turns)) * 2 * PI)


def era_exact(day, second, offset):
    """The Earth rotation angle (IERS Conventions 2010, 5.14) at UT1 the TT
    clock reading moved by [offset] seconds, exactly."""
    tu = (day - Fraction(103089, 2)) + (Fraction(second) + Fraction(offset)) / 86400
    return Fraction("0.7790572732640") + Fraction("1.00273781191135448") * tu


def gmst_exact(day, second, offset):
    """IAU 2006 GMST (IERS Conventions 2010, 5.32), exactly, in turns."""
    t = ((day - Fraction(103089, 2)) + Fraction(second) / 86400) / 36525
    cs = ["0.014506", "4612.156534", "1.3915817", "-0.00000044",
          "-0.000029956", "-0.0000000368"]
    poly = sum(Fraction(c) * t ** k for k, c in enumerate(cs))
    return era_exact(day, second, offset) + poly / 1296000


def turns_of_jd(j1, j2):
    """The Earth rotation angle in turns at the UT1 Julian date j1 + j2,
    exactly."""
    tu = Fraction(j1) + Fraction(j2) - 2451545
    return Fraction("0.7790572732640") + Fraction("1.00273781191135448") * tu


def utc_rows():
    """(date, clock, UT1 - UTC, ERA) for UTC instants: in and around the
    leap second at the end of 2016, and drawn from 1972 to 2030. (Before
    1972 eraUtcut1 adds UT1 - UTC to the SI seconds since UTC's 0h, which
    ran apart from UTC's clock by up to 1.3 ms a day.)
    UT1 is eraUtcut1's; ERA its definition at it, exactly, rounded once."""
    rng = random.Random(1972)
    out = [((2016, 12, 31), (23, 59, 59.5), -0.4),
           ((2016, 12, 31), (23, 59, 60.5), -0.4),
           ((2016, 12, 31), (23, 59, 60.99), -0.4),
           ((2017, 1, 1), (0, 0, 0.3), 0.59)]
    for _ in range(12):
        y = rng.randrange(1972, 2030)
        out.append(((y, rng.randrange(1, 13), rng.randrange(1, 29)),
                    (rng.randrange(24), rng.randrange(60),
                     round(rng.uniform(0, 59.999), 6)),
                    round(rng.uniform(-0.9, 0.9), 6)))
    rows_ = []
    for (y, m, d), (h, mi, sec), dut1 in out:
        u1, u2 = erfa.dtf2d("UTC", y, m, d, h, mi, sec)
        t1, t2 = erfa.utcut1(u1, u2, dut1)
        rows_.append(((y, m, d), (h, mi, sec), dut1,
                      turns_radians(turns_of_jd(t1, t2))))
    return rows_


def unit(lon, lat):
    return [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon),
            math.sin(lat)]


def stars(eh):
    """ICRS unit vectors: spread over the sky, and near the Sun as the
    geocentre sees it, -eh, at 0.1, 1, 10 and 100 arcminutes, and 2 and 30
    degrees."""
    rng = random.Random(1990)
    out = [unit(rng.uniform(0, 2 * math.pi), math.asin(rng.uniform(-1, 1)))
           for _ in range(4)]
    sun = [-x for x in eh]
    lon = math.atan2(sun[1], sun[0])
    lat = math.atan2(sun[2], math.hypot(sun[0], sun[1]))
    for arcmin in (0.1, 1.0, 10.0, 100.0, 120.0, 1800.0):
        out.append(unit(lon, lat + math.radians(arcmin / 60.0)))
    return out


def apparent(p, ebpv, ehp):
    """eraAtciq's ldsun then ab, with eraApcs's parameters at the
    geocentre."""
    geocentre = erfa.p2pv([0.0] * 3)
    astrom = erfa.apcs(0.0, 0.0, geocentre, ebpv, ehp)
    pnat = erfa.ldsun(p, astrom["eh"], astrom["em"])
    return list(erfa.ab(pnat, astrom["v"], astrom["em"], astrom["bm1"]))


def references():
    rows_ = []
    for day, second, offset in instants():
        t1, t2 = 2400000.5 + day, second / 86400.0
        b1, b2 = tdb_of_tt(t1, t2)
        era = turns_radians(era_exact(day, second, offset))
        gmst = turns_radians(gmst_exact(day, second, offset))
        eo = float(erfa.eo06a(t1, t2))
        npb = [float(x) for x in erfa.pnm06a(t1, t2).flatten()]
        c2i = erfa.c2i06a(t1, t2)
        pvh, pvb = erfa.epv00(b1, b2)
        ph = [float(x) for x in pvh["p"]]
        vh = [float(x) for x in pvh["v"]]
        pb = [float(x) for x in pvb["p"]]
        vb = [float(x) for x in pvb["v"]]
        em = math.sqrt(sum(x * x for x in ph))
        eh = [x / em for x in ph]
        dirs = []
        for p in stars(eh):
            g = apparent(p, pvb, pvh["p"])
            c = [float(x) for x in c2i @ g]
            dirs.append((p, [float(x) for x in g], c))
        rows_.append(((day, second, offset), era, gmst, eo, npb,
                      [float(x) for x in c2i.flatten()], (ph, vh, pb, vb),
                      dirs))
    return rows_


def ocaml_float(x):
    text = repr(float(x))
    return text if any(c in text for c in ".en") else text + "."


def floats(xs):
    return "[| " + "; ".join(ocaml_float(x) for x in xs) + " |]"


def render_reference():
    out = [HEADER, f"""
(* Written by test/gen/earth.py; do not edit. Values of pyerfa's ERFA
   2.0.1. *)

[@@@ocamlformat "disable"]

(* Per instant: (TT day, TT seconds since 0h, UT1 - TT in seconds); ERA
   and GMST in radians, each its definition evaluated exactly and rounded
   once; eraEo06a's equation of the origins; the NPB and C2I matrices, row-major; the
   Earth's heliocentric position (au) and velocity (au/d), then its
   barycentric ones, at the instant's TDB; and for each ICRS direction,
   its GCRS proper direction and that direction in CIRS. *)
type row = {{
  instant : int * float * float;
  era : float;
  gmst : float;
  eo : float;
  npb : float array;
  c2i : float array;
  heliocentric : float array * float array;
  barycentric : float array * float array;
  directions : (float array * float array * float array) list;
}}

let rows = [
"""]
    for (instant, era, gmst, eo, npb, c2i, (ph, vh, pb, vb),
         dirs) in references():
        day, second, offset = instant
        out.append(f"""  {{ instant = ({day}, {ocaml_float(second)}, {ocaml_float(offset)});
    era = {ocaml_float(era)}; gmst = {ocaml_float(gmst)}; eo = {ocaml_float(eo)};
    npb = {floats(npb)};
    c2i = {floats(c2i)};
    heliocentric = ({floats(ph)}, {floats(vh)});
    barycentric = ({floats(pb)}, {floats(vb)});
    directions = [
""")
        for p, g, c in dirs:
            out.append(f"      ({floats(p)},\n       {floats(g)},\n"
                       f"       {floats(c)});\n")
        out.append("    ] };\n")
    out.append("]\n")
    out.append("""
(* A UTC date and clock, UT1 - UTC in seconds, and the Earth rotation angle
   in radians at eraUtcut1's UT1 of them: its definition evaluated exactly
   and rounded once. *)
let utc = [
""")
    for date, clock, dut1, era in utc_rows():
        (y, m, d), (h, mi, sec) = date, clock
        out.append(f"  (({y}, {m}, {d}), ({h}, {mi}, {ocaml_float(sec)}), "
                   f"{ocaml_float(dut1)}, {ocaml_float(era)});\n")
    out.append("]\n")
    return "".join(out)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    texts = [source(url) for url in (NUT00A, S06, EPV00)]
    files = [(TABLES, render_tables(texts)), (REFERENCE, render_reference())]
    if args.check:
        for path, text in files:
            if not path.exists() or path.read_text() != text:
                sys.exit(f"{path} would change; run without --check")
        return
    for path, text in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


if __name__ == "__main__":
    main()
