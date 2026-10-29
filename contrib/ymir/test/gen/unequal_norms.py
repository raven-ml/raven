# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Generate separations and position angles between rows of unequal norms.

    uv run contrib/ymir/test/gen/unequal_norms.py [--check]

A direction is a float64 vector read as the ray it spans, of any norm. Each
pair here is a unit row a and a row b of norm k at a stated separation from
it, both rounded once; their separation and b's position angle from a are
evaluated at 60 digits from the rows' exact values, with frames.py's
measures, and rounded once. It writes:

test/support/unequal_norms_reference.ml

Without --check the file is written; with it, nothing is written and the run
fails if the file would change.
"""

import argparse
from pathlib import Path
import sys

from mpmath import mp, mpf

from frames import (HEADER, arcsec, deg, floats, measures, ocaml_float, offset,
                    round_once, unit_at)

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / "support" / "unequal_norms_reference.ml"


def pairs():
    """Unit rows a and rows b of norm k at stated separations, with their measures."""
    out = []
    uas = arcsec("1") / 1_000_000
    angles = [("1 uas", uas), ("1 mas", uas * 1000), ("1 arcsec", arcsec("1")),
              ("pi/2", mp.pi / 2), ("pi - 1e-9", mp.pi - mpf("1e-9"))]
    places = [("generic", deg("37.5"), deg("-21.25"), deg("63")),
              ("axis", mpf(0), mpf(0), deg("200"))]
    # Norms within a factor sqrt 2, and 3, past the factor 2 a power of two
    # brings a row within.
    norms = [("1.3", mpf("1.3")), ("0.8", mpf("0.8")), ("3", mpf(3))]
    for pname, lon, lat, bearing in places:
        for aname, angle in angles:
            for kname, k in norms:
                a = [round_once(x) for x in unit_at(lon, lat)]
                b = [round_once(k * x) for x in offset(lon, lat, angle, bearing)]
                sep, pa = measures(a, b)
                out.append((f"{aname}, {pname}, norm {kname}", a, b, sep, pa))
    return out


def render():
    out = [HEADER, """
(* Written by test/gen/unequal_norms.py; do not edit. *)

[@@@ocamlformat "disable"]

(* A label, a unit row a, a row b of another norm, and their separation and
   b's position angle from a, from the rows' exact values at 60 digits,
   rounded once. *)
let pairs = [
"""]
    with mp.workdps(60):
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
    text = render()
    if args.check:
        if not REFERENCE.exists() or REFERENCE.read_text() != text:
            sys.exit(f"{REFERENCE} would change; run without --check")
        return
    REFERENCE.write_text(text)


if __name__ == "__main__":
    main()
