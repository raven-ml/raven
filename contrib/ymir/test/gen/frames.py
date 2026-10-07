# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath", "pyerfa", "astropy"]
# ///
"""Generate the fixed frames' orientations and the frames' test references.

    uv run contrib/ymir/test/gen/frames.py [--check]

Each fixed frame's orientation R (v_frame = R . v_icrs) is its standard's
definition evaluated at 60 digits and rounded once to float64. The script
checks that 100 digits round to the same doubles, and that ERFA's 30-digit
Galactic table (eraIcrs2g) rounds to them bit for bit. It writes:

lib/orientation.ml
    The rounded orientations as hexadecimal literals, row-major.

test/support/frames_reference.ml
    - each orientation as a double-double (hi, lo), hi the literal;
    - pyerfa's ecm06 at J2000 TT;
    - ERFA's t_s2c case, whose own expected values are within 1e-12;
    - directions in ICRS with their Galactic and barycentric mean ecliptic
      vectors from astropy;
    - pairs of float64 vectors with their separation and position angle,
      evaluated at 60 digits from the vectors' exact values and rounded once.

Without --check both files are written; with it, nothing is written and the
run fails if a file would change.
"""

import argparse
from fractions import Fraction
from pathlib import Path
import sys

import mpmath
from mpmath import mp, mpf

HERE = Path(__file__).resolve().parent
ORIENTATION = HERE.parent.parent / "lib" / "orientation.ml"
REFERENCE = HERE.parent / "support" / "frames_reference.ml"

HEADER = """(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)
"""

# ERFA's eraIcrs2g table, R_3(-R) R_1(pi/2-Q) R_3(pi/2+P) to 30 digits.
ERFA_GALACTIC = [
    "-0.054875560416215368492398900454", "-0.873437090234885048760383168409",
    "-0.483835015548713226831774175116", "+0.494109427875583673525222371358",
    "-0.444829629960011178146614061616", "+0.746982244497218890527388004556",
    "-0.867666149019004701181616534570", "-0.198076373431201528180486091412",
    "+0.455983776175066922272100478348",
]


# Exact arithmetic at the working precision

def deg(text):
    return mpf(Fraction(text).numerator) / Fraction(text).denominator * mp.pi / 180


def arcsec(text):
    return deg(text) / 3600


def r1(t):
    c, s = mpmath.cos(t), mpmath.sin(t)
    return mpmath.matrix([[1, 0, 0], [0, c, s], [0, -s, c]])


def r2(t):
    c, s = mpmath.cos(t), mpmath.sin(t)
    return mpmath.matrix([[c, 0, -s], [0, 1, 0], [s, 0, c]])


def r3(t):
    c, s = mpmath.cos(t), mpmath.sin(t)
    return mpmath.matrix([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def rv2m(w):
    """eraRv2m: the rotation matrix of the rotation vector w."""
    phi = mpmath.sqrt(sum(x * x for x in w))
    s, c = mpmath.sin(phi), mpmath.cos(phi)
    f = 1 - c
    x, y, z = (v / phi for v in w)
    return mpmath.matrix([
        [x * x * f + c, x * y * f + z * s, x * z * f - y * s],
        [y * x * f - z * s, y * y * f + c, y * z * f + x * s],
        [z * x * f + y * s, z * y * f - x * s, z * z * f + c],
    ])


def galactic():
    # Hipparcos Catalogue, Vol. 1 §1.5.3: the north Galactic pole at ICRS
    # (192.85948, 27.12825) degrees, the ascending node at l = 32.93192.
    return (r3(-deg("32.93192")) * r1(mp.pi / 2 - deg("27.12825"))
            * r3(mp.pi / 2 + deg("192.85948")))


def orientations():
    """Each fixed frame's name, definition and orientation from ICRS."""
    mas = [arcsec(t) / 1000 for t in ("-19.9", "-9.1", "22.9")]
    eps0 = arcsec("84381.406")
    bias = (r1(-eps0) * r3(-arcsec("-0.041775")) * r1(arcsec("84381.412819"))
            * r3(arcsec("-0.052928")))
    g = galactic()
    return [
        ("fk5_j2000",
         "r5h^T, r5h the rotation by (-19.9, -9.1, +22.9) mas: FK5's\n"
         "   orientation from Hipparcos at J2000.0 (Mignard and Froeschle 2000;\n"
         "   eraFk5hip).",
         rv2m(mas).T),
        ("galactic",
         "R3(-32.93192) R1(90 - 27.12825) R3(90 + 192.85948), in degrees: the\n"
         "   Hipparcos Catalogue's definition on ICRS (ESA 1997, Vol. 1 1.5.3;\n"
         "   eraIcrs2g).",
         g),
        ("ecliptic_j2000",
         "R1(eps0) B, eps0 = 84381.406\", B the IAU 2006 frame bias at J2000\n"
         "   from the Fukushima-Williams angles gamma = -0.052928\",\n"
         "   phi = 84381.412819\", psi = -0.041775\" (eraEcm06 at J2000 TT).",
         r1(eps0) * bias),
        ("supergalactic",
         "R3(90) R2(90 - 6.32) R3(47.37) R_galactic, in degrees: the north\n"
         "   supergalactic pole at Galactic (47.37, +6.32), the origin at\n"
         "   l = 137.37 (de Vaucouleurs et al. 1976; Lahav et al. 2000).",
         r3(deg("90")) * r2(deg("90") - deg("6.32")) * r3(deg("47.37")) * g),
    ]


# Rounding

def exact(x):
    """The exact rational value of an mpf."""
    sign, man, exp, _ = mpf(x)._mpf_
    v = Fraction(man) * Fraction(2) ** exp
    return -v if sign else v


def round_once(x):
    """x rounded once to float64: CPython rounds a Fraction correctly."""
    return float(exact(x))


def double_double(x):
    hi = round_once(x)
    return hi, float(exact(x) - Fraction(hi))


def flat(m):
    return [m[i, j] for i in range(3) for j in range(3)]


def evaluated(digits):
    with mp.workdps(digits):
        return [(name, doc, [round_once(x) for x in flat(m)])
                for name, doc, m in orientations()]


def ocaml_float(x):
    if x == 0:
        return "0." if str(x)[0] != "-" else "-0."
    return x.hex().replace("0x1.0000000000000p", "0x1p")


def check_orientations(rounded):
    if rounded != evaluated(100):
        sys.exit("an orientation rounds differently at 60 and 100 digits")
    galactic_row = next(r for n, _, r in rounded if n == "galactic")
    if [float(Fraction(s)) for s in ERFA_GALACTIC] != galactic_row:
        sys.exit("ERFA's Galactic table rounds to other doubles")


def render_orientation(rounded):
    out = [HEADER, """
(* Written by test/gen/frames.py; do not edit. Each value is a fixed frame's
   orientation R from ICRS, v_frame = R . v_icrs, in row-major order: its
   definition evaluated at 60 digits and rounded once to float64. *)

[@@@ocamlformat "disable"]
"""]
    for name, doc, m in rounded:
        out.append(f"\n(* {doc} *)\nlet {name} = [|\n")
        for i in range(3):
            row = "; ".join(ocaml_float(x) for x in m[3 * i:3 * i + 3])
            out.append(f"  {row};\n")
        out.append("|]\n")
    return "".join(out)


# References

def unit_at(lon, lat):
    return [mpmath.cos(lat) * mpmath.cos(lon), mpmath.cos(lat) * mpmath.sin(lon),
            mpmath.sin(lat)]


def offset(lon, lat, angle, bearing):
    """The unit vector at [angle] from (lon, lat) toward [bearing], east of north."""
    a = unit_at(lon, lat)
    n = [-mpmath.sin(lat) * mpmath.cos(lon), -mpmath.sin(lat) * mpmath.sin(lon),
         mpmath.cos(lat)]
    e = [-mpmath.sin(lon), mpmath.cos(lon), 0]
    t = [mpmath.cos(bearing) * n[i] + mpmath.sin(bearing) * e[i] for i in range(3)]
    return [mpmath.cos(angle) * a[i] + mpmath.sin(angle) * t[i] for i in range(3)]


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def wrap(t):
    return t + 2 * mp.pi if t < 0 else t


def measures(a, b):
    """Separation and position angle of the rays through the exact vectors."""
    a = [mpf(x) for x in a]
    b = [mpf(x) for x in b]
    sep = mpmath.atan2(mpmath.sqrt(dot(cross(a, b), cross(a, b))), dot(a, b))
    x, y, z = a
    if x == 0 and y == 0:
        n, e = [-mpmath.sign(z), 0, 0], [0, 1, 0]
    else:
        norm = mpmath.sqrt(dot(a, a))
        n, e = [-x * z, -y * z, x * x + y * y], [-y * norm, x * norm, 0]
    pa = mpmath.atan2(dot(b, e), dot(b, n))
    return round_once(sep), round_once(wrap(pa))


def pairs():
    """Pairs of float64 vectors at stated separations, with their measures."""
    out = []
    uas = arcsec("1") / 1_000_000
    angles = [("1 uas", uas), ("1 mas", uas * 1000), ("pi/2", mp.pi / 2),
              ("pi - 1e-9", mp.pi - mpf("1e-9"))]
    # Generic points, a point on the x axis whose neighbours cross a power of
    # two in their largest component, and points next to each pole.
    places = [("generic", deg("37.5"), deg("-21.25"), deg("63")),
              ("axis", mpf(0), mpf(0), deg("200")),
              ("near the north pole", deg("123"), deg("89.9999"), deg("10")),
              ("near the south pole", deg("-77"), deg("-89.99999"), deg("300"))]
    for pname, lon, lat, bearing in places:
        for aname, angle in angles:
            a = [round_once(x) for x in unit_at(lon, lat)]
            b = [round_once(x) for x in offset(lon, lat, angle, bearing)]
            out.append((f"{aname}, {pname}", a, b, *measures(a, b)))
    for pname, z in (("north pole", 1.0), ("south pole", -1.0)):
        for aname, lon, lat in (("1 mas", deg("30"), (z * mp.pi / 2) - z * uas * 1000),
                                ("pi/2", deg("250"), mpf(0)),
                                ("1 deg", deg("100"), z * deg("89"))):
            a = [0.0, 0.0, z]
            b = [round_once(x) for x in unit_at(lon, lat)]
            out.append((f"{aname} from the {pname}", a, b, *measures(a, b)))
    return out


def astropy_points():
    """ICRS directions in degrees, spread over the sphere."""
    return [(0.0, 0.0), (83.633, 22.0145), (266.405, -28.936), (192.85948, 27.12825),
            (10.68458, 41.26917), (299.868, 40.734), (150.0, -60.0), (1.0, 89.5)]


def astropy_references():
    import astropy.units as u
    from astropy.coordinates import (BarycentricMeanEcliptic, Galactic, ICRS,
                                     SkyCoord)
    gal, ecl = [], []
    for ra, dec in astropy_points():
        c = SkyCoord(ICRS(ra=ra * u.deg, dec=dec * u.deg))
        g = c.transform_to(Galactic()).cartesian
        e = c.transform_to(BarycentricMeanEcliptic(equinox="J2000")).cartesian
        gal.append((ra, dec, [float(g.x), float(g.y), float(g.z)]))
        ecl.append((ra, dec, [float(e.x), float(e.y), float(e.z)]))
    return gal, ecl


def ecm06():
    import erfa
    return [float(x) for x in erfa.ecm06(2451545.0, 0.0).flatten()]


def floats(xs):
    return "[| " + "; ".join(ocaml_float(x) for x in xs) + " |]"


def render_reference():
    out = [HEADER, """
(* Written by test/gen/frames.py; do not edit. *)

[@@@ocamlformat "disable"]

(* Each fixed frame's orientation from ICRS at 60 digits, row-major, as
   double-doubles (hi, lo): hi is the frame's literal. *)
let orientations = [
"""]
    with mp.workdps(60):
        for name, _, m in orientations():
            dd = [double_double(x) for x in flat(m)]
            entries = "; ".join(f"({ocaml_float(h)}, {ocaml_float(l)})" for h, l in dd)
            out.append(f'  ("{name}", [| {entries} |]);\n')
        out.append("]\n")
        out.append("\n(* pyerfa's ecm06 at J2000 TT, row-major. *)\n")
        out.append(f"let ecm06_j2000 = {floats(ecm06())}\n")
        gal, ecl = astropy_references()
        for name, rows, what in (("astropy_galactic", gal, "Galactic"),
                                 ("astropy_ecliptic", ecl,
                                  "barycentric mean ecliptic at J2000")):
            out.append(f"\n(* ICRS (ra, dec) in degrees and astropy's {what} unit vector. *)\n")
            out.append(f"let {name} = [\n")
            for ra, dec, v in rows:
                out.append(f"  ({ocaml_float(ra)}, {ocaml_float(dec)}, {floats(v)});\n")
            out.append("]\n")
        out.append("\n(* ERFA's t_s2c at lon 3.0123, lat -0.999, rounded once. *)\n")
        lon, lat = mpf(3.0123), mpf(-0.999)
        out.append(f"let s2c = {floats([round_once(x) for x in unit_at(lon, lat)])}\n")
        out.append("""
(* A label, two float64 vectors a and b, and the separation and position
   angle of b from a, from the vectors' exact values, rounded once. *)
let pairs = [
""")
        for label, a, b, sep, pa in pairs():
            out.append(f'  ("{label}", {floats(a)}, {floats(b)}, {ocaml_float(sep)}, '
                       f'{ocaml_float(pa)});\n')
        out.append("]\n")
    return "".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rounded = evaluated(60)
    check_orientations(rounded)
    files = [(ORIENTATION, render_orientation(rounded)), (REFERENCE, render_reference())]
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
