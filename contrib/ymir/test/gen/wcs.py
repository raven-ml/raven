# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "astropy"]
# ///
"""Generate the FITS WCS readers' test references.

    uv run contrib/ymir/test/gen/wcs.py [--check]

It writes test/support/wcs_reference.ml: headers whose celestial
descriptions vary each keyword the reader resolves (CD, PC with CDELT,
CROTA2, latitude first, an alternate description, CUNIT in arcseconds,
LONPOLE and the native reference point stated or defaulted, and each frame
it reads), with pixel coordinates and WCSLIB's world coordinates for them
(astropy.wcs, the core WCS with origin 0).

Without --check the file is written; with it, nothing is written and the
run fails if the file would change.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / "support" / "wcs_reference.ml"


def ocaml_float(x):
    x = float(x)
    if x == 0.0:
        return "0." if np.copysign(1.0, x) > 0 else "-0."
    return x.hex().replace("0x1.0000000000000p", "0x1p")


def floats(xs):
    return "[| " + "; ".join(ocaml_float(x) for x in xs) + " |]"


def header(cards):
    """A header of 80-byte records from (keyword, value) pairs, in order."""
    h = fits.Header()
    h["NAXIS"] = 2
    for k, v in cards:
        h.append(fits.Card(k, v))
    return h


# The WCS of the NIRCam F200W i2d mosaic of SMACS 0723 (jw02736), as its SCI
# header states it.
NIRCAM = [
    ("RADESYS", "ICRS"),
    ("WCSAXES", 2),
    ("CRPIX1", 5099.44382803652),
    ("CRPIX2", 2373.7753424908956),
    ("CRVAL1", 110.75544256521349),
    ("CRVAL2", -73.46776600616062),
    ("CTYPE1", "RA---TAN"),
    ("CTYPE2", "DEC--TAN"),
    ("CUNIT1", "deg"),
    ("CUNIT2", "deg"),
    ("CDELT1", 8.67445987394292e-06),
    ("CDELT2", 8.67445987394292e-06),
    ("PC1_1", 0.815256589635877),
    ("PC1_2", 0.5790998990288975),
    ("PC2_1", 0.5790998990288975),
    ("PC2_2", -0.815256589635877),
]

# name, frame, alternate, cards, pixel extent
CASES = [
    ("NIRCam i2d", "icrs", " ", NIRCAM, (4762, 10276)),
    (
        "CD, FK5 at J2000, LONPOLE stated",
        "fk5_j2000",
        " ",
        [
            ("CTYPE1", "RA---TAN"),
            ("CTYPE2", "DEC--TAN"),
            ("CRPIX1", 512.5),
            ("CRPIX2", 300.25),
            ("CD1_1", -2.0e-4),
            ("CD1_2", 3.0e-5),
            ("CD2_1", 2.5e-5),
            ("CD2_2", 2.0e-4),
            ("CRVAL1", 266.4),
            ("CRVAL2", -29.0),
            ("LONPOLE", 180.0),
            ("RADESYS", "FK5"),
            ("EQUINOX", 2000.0),
        ],
        (600, 1024),
    ),
    (
        "CROTA2, Galactic",
        "galactic",
        " ",
        [
            ("CTYPE1", "GLON-TAN"),
            ("CTYPE2", "GLAT-TAN"),
            ("CRPIX1", 100.0),
            ("CRPIX2", 120.5),
            ("CDELT1", -0.01),
            ("CDELT2", 0.012),
            ("CROTA2", 30.0),
            ("CRVAL1", 45.0),
            ("CRVAL2", 10.0),
        ],
        (240, 200),
    ),
    (
        "latitude first, ARC, CD, ICRS by default",
        "icrs",
        " ",
        [
            ("CTYPE1", "DEC--ARC"),
            ("CTYPE2", "RA---ARC"),
            ("CRPIX1", 80.0),
            ("CRPIX2", 100.0),
            ("CD1_1", 0.0),
            ("CD1_2", 0.5),
            ("CD2_1", -0.5),
            ("CD2_2", 0.0),
            ("CRVAL1", -29.0),
            ("CRVAL2", 266.4),
        ],
        (200, 160),
    ),
    (
        "alternate A, J2000 ecliptic",
        "ecliptic_j2000",
        "A",
        [
            ("CTYPE1", "RA---TAN"),
            ("CTYPE2", "DEC--TAN"),
            ("CRPIX1", 1.0),
            ("CRPIX2", 1.0),
            ("CDELT1", 1.0),
            ("CDELT2", 1.0),
            ("CTYPE1A", "ELON-TAN"),
            ("CTYPE2A", "ELAT-TAN"),
            ("CRPIX1A", 64.5),
            ("CRPIX2A", 32.5),
            ("CDELT1A", -0.002),
            ("CDELT2A", 0.002),
            ("PC1_2A", 0.1),
            ("CRVAL1A", 150.0),
            ("CRVAL2A", 60.0),
            ("RADESYSA", "ICRS"),
        ],
        (64, 128),
    ),
    (
        "CUNIT arcsec, PV1_1 and LONPOLE by default",
        "supergalactic",
        " ",
        [
            ("CTYPE1", "SLON-TAN"),
            ("CTYPE2", "SLAT-TAN"),
            ("CUNIT1", "arcsec"),
            ("CUNIT2", "arcsec"),
            ("CRPIX1", 50.0),
            ("CRPIX2", 40.0),
            ("CDELT1", -3.6),
            ("CDELT2", 3.6),
            ("CRVAL1", 36000.0),
            ("CRVAL2", 72000.0),
            ("PV1_1", 30.0),
        ],
        (80, 100),
    ),
    (
        "LONPOLE 150, ARC near the pole",
        "icrs",
        " ",
        [
            ("CTYPE1", "RA---ARC"),
            ("CTYPE2", "DEC--ARC"),
            ("CRPIX1", 30.0),
            ("CRPIX2", 30.0),
            ("CDELT1", -1.0),
            ("CDELT2", 1.0),
            ("CRVAL1", 10.0),
            ("CRVAL2", 80.0),
            ("LONPOLE", 150.0),
            ("LATPOLE", 75.0),
            ("RADESYS", "ICRS"),
        ],
        (60, 60),
    ),
    (
        "PV1_3 and PV1_4 for LONPOLE and LATPOLE",
        "icrs",
        " ",
        [
            ("CTYPE1", "RA---ARC"),
            ("CTYPE2", "DEC--ARC"),
            ("CRPIX1", 30.0),
            ("CRPIX2", 30.0),
            ("CDELT1", -1.0),
            ("CDELT2", 1.0),
            ("CRVAL1", 10.0),
            ("CRVAL2", 80.0),
            ("PV1_3", 150.0),
            ("PV1_4", 75.0),
        ],
        (60, 60),
    ),
]


def case(name, frame, alt, cards, extent):
    h = header(cards)
    key = None if alt == " " else alt
    w = WCS(h, key=alt)
    rows, cols = extent
    r = np.linspace(-0.5, rows - 0.5, 7)
    c = np.linspace(-0.5, cols - 0.5, 9)
    rr, cc = np.meshgrid(r, c, indexing="ij")
    pixels = np.stack([rr.ravel(), cc.ravel()], axis=-1)
    pixels = np.concatenate([pixels, pixels[:20] + [0.37, -0.81]])
    world = np.stack(w.wcs_pix2world(pixels[:, 1], pixels[:, 0], 0), axis=-1)
    if w.wcs.lat == 0:
        world = world[:, ::-1]
    # astropy.wcs gives CUNIT arcsec worlds in degrees.
    text = "".join(c.image for c in h.cards)
    return {
        "name": name,
        "frame": frame,
        "alt": alt,
        "header": text,
        "pixels": pixels,
        "world": world,
        "key": key,
    }


def render():
    out = [
        "(*---------------------------------------------------------------------------",
        "  Copyright (c) 2026 The Raven authors. All rights reserved.",
        "  SPDX-License-Identifier: ISC",
        "  ---------------------------------------------------------------------------*)",
        "",
        "(* Written by test/gen/wcs.py; do not edit. *)",
        "",
        '[@@@ocamlformat "disable"]',
        "",
        "type case = {",
        "  name : string;",
        "  frame : string;  (** The frame's {!Ymir.Frame.name}. *)",
        "  alt : char;",
        "  header : string;  (** 80-byte records, without END. *)",
        "  pixels : float array;  (** [(row, column)] pairs, 0-based. *)",
        "  world : float array;  (** WCSLIB's [(lon, lat)] in degrees for each. *)",
        "}",
        "",
        "let cases = [",
    ]
    for c in (case(*x) for x in CASES):
        out.append("  {")
        out.append(f'    name = "{c["name"]}";')
        out.append(f'    frame = "{c["frame"]}";')
        out.append(f"    alt = '{c['alt']}';")
        out.append("    header =")
        records = [c["header"][i : i + 80] for i in range(0, len(c["header"]), 80)]
        out.append("      " + "\n      ^ ".join(f'"{r}"' for r in records) + ";")
        out.append(f"    pixels = {floats(c['pixels'].ravel())};")
        out.append(f"    world = {floats(c['world'].ravel())};")
        out.append("  };")
    out.append("]")
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser()
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
