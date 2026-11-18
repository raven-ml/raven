# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.5.4", "astropy==8.0.2"]
# ///
"""Generate the FITS WCS readers' test references.

    uv run contrib/ymir/test/gen/wcs.py [--check]

It writes test/support/wcs_reference.ml: headers whose celestial
descriptions vary each keyword the reader resolves (CD, PC with CDELT,
CROTA2, latitude first, an alternate description, CUNIT in arcseconds,
LONPOLE and the native reference point stated or defaulted, and each frame
it reads, each projection with its PV terms, SIP with and without AP and
BP, TPV, and TAN with PV terms as SCAMP writes TPV), with pixel
coordinates and WCSLIB's world coordinates for them (astropy.wcs's
all_pix2world, origin 0, which applies SIP).

It also writes headers of one, three and four axes: a celestial pair with
linear spectral and Stokes axes, the pair's axes apart, a PC or CD element
between the pair and a spectral axis, the pair's axes apart and coupled,
and a spectrum alone, each with the header of its celestial axes alone,
renumbered, where it has them.

Without --check the file is written; with it, nothing is written and the
run fails if the file would change.
"""

import argparse
from pathlib import Path
import re
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


def sip_cards(order, coeffs, prefix):
    cards = [(f"{prefix}_ORDER", order)]
    cards += [(f"{prefix}_{p}_{q}", v) for (p, q), v in coeffs.items()]
    return cards


SIP = (
    sip_cards(2, {(2, 0): 1.3e-6, (1, 1): -2.1e-6, (0, 2): 6.0e-7}, "A")
    + sip_cards(3, {(2, 0): -8.0e-7, (0, 2): 1.9e-6, (3, 0): 2.0e-10, (1, 2): -1.1e-10}, "B")
)
SIP_INVERSE = sip_cards(2, {(1, 0): -1.0e-4, (0, 1): 2.0e-5, (2, 0): -1.3e-6}, "AP") + sip_cards(
    2, {(0, 0): 1.0e-6, (2, 0): 8.0e-7, (0, 2): -1.9e-6}, "BP"
)
DISTORTED_BASE = [
    ("CRPIX1", 1024.5),
    ("CRPIX2", 1000.25),
    ("CD1_1", -1.2e-5),
    ("CD1_2", 4.0e-6),
    ("CD2_1", 4.1e-6),
    ("CD2_2", 1.2e-5),
    ("CRVAL1", 150.1),
    ("CRVAL2", 2.2),
    ("RADESYS", "ICRS"),
]
TPV_TERMS = [
    ("PV1_0", 2.0e-6),
    ("PV1_1", 1.0002),
    ("PV1_2", -3.0e-4),
    ("PV1_4", 2.0e-3),
    ("PV1_7", -4.0e-2),
    ("PV1_11", 1.0e-2),
    ("PV2_1", 0.9998),
    ("PV2_2", 1.0e-4),
    ("PV2_5", -1.0e-3),
    ("PV2_10", 3.0e-2),
    ("PV2_39", 1.0e-1),
]


def projection(code, pv, crval=(30.0, 40.0), extra=()):
    return [
        ("CTYPE1", f"RA---{code}"),
        ("CTYPE2", f"DEC--{code}"),
        ("CRPIX1", 50.5),
        ("CRPIX2", 40.25),
        ("CDELT1", -0.6),
        ("CDELT2", 0.6),
        ("CRVAL1", crval[0]),
        ("CRVAL2", crval[1]),
    ] + [(f"PV2_{m}", v) for m, v in pv] + list(extra)


PROJECTION_CASES = [
    ("AZP", "icrs", " ", projection("AZP", [(1, 2.0), (2, 30.0)]), (80, 100)),
    ("SZP", "icrs", " ", projection("SZP", [(1, 2.0), (2, 180.0), (3, 60.0)]), (80, 100)),
    ("STG", "icrs", " ", projection("STG", []), (80, 100)),
    ("SIN, slant", "icrs", " ", projection("SIN", [(1, 0.2), (2, -0.1)]), (80, 100)),
    ("ZPN", "icrs", " ", projection("ZPN", [(1, 1.0), (3, -0.25), (0, 0.0)]), (80, 100)),
    ("ZEA", "icrs", " ", projection("ZEA", []), (80, 100)),
    ("AIR", "icrs", " ", projection("AIR", [(1, 45.0)]), (80, 100)),
    ("CYP", "icrs", " ", projection("CYP", [(1, 0.5), (2, 0.8)], (100.0, 0.0)), (80, 100)),
    ("CEA", "icrs", " ", projection("CEA", [(1, 0.5)], (100.0, 10.0)), (80, 100)),
    (
        "CAR, native reference stated, LATPOLE south",
        "icrs",
        " ",
        projection("CAR", [], (10.0, 20.0), [("PV1_1", 0.0), ("PV1_2", 0.0), ("LONPOLE", 60.0), ("LATPOLE", -90.0)]),
        (80, 100),
    ),
    ("MER", "icrs", " ", projection("MER", [], (45.0, -20.0)), (80, 100)),
    ("TAN at theta0 45", "icrs", " ", projection("TAN", [], (80.0, 20.0), [("PV1_2", 45.0)]), (80, 100)),
    (
        "SIP",
        "icrs",
        " ",
        [("CTYPE1", "RA---TAN-SIP"), ("CTYPE2", "DEC--TAN-SIP")] + DISTORTED_BASE + SIP,
        (2048, 2048),
    ),
    (
        "SIP with AP and BP",
        "icrs",
        " ",
        [("CTYPE1", "RA---TAN-SIP"), ("CTYPE2", "DEC--TAN-SIP")] + DISTORTED_BASE + SIP + SIP_INVERSE,
        (2048, 2048),
    ),
    (
        "TPV",
        "icrs",
        " ",
        [("CTYPE1", "RA---TPV"), ("CTYPE2", "DEC--TPV")] + DISTORTED_BASE + TPV_TERMS,
        (2048, 2048),
    ),
    (
        "TAN with PV terms",
        "icrs",
        " ",
        [("CTYPE1", "RA---TAN"), ("CTYPE2", "DEC--TAN")] + DISTORTED_BASE + TPV_TERMS,
        (2048, 2048),
    ),
]
CASES += PROJECTION_CASES


def case(name, frame, alt, cards, extent):
    h = header(cards)
    key = None if alt == " " else alt
    reference = h.copy()
    # WCSLIB reads TAN's PV terms as the projection's; SCAMP means TPV.
    if name == "TAN with PV terms":
        reference["CTYPE1"] = "RA---TPV"
        reference["CTYPE2"] = "DEC--TPV"
    w = WCS(reference, key=alt)
    rows, cols = extent
    r = np.linspace(-0.5, rows - 0.5, 7)
    c = np.linspace(-0.5, cols - 0.5, 9)
    rr, cc = np.meshgrid(r, c, indexing="ij")
    pixels = np.stack([rr.ravel(), cc.ravel()], axis=-1)
    pixels = np.concatenate([pixels, pixels[:20] + [0.37, -0.81]])
    world = np.stack(w.all_pix2world(pixels[:, 1], pixels[:, 0], 0), axis=-1)
    keep = np.all(np.isfinite(world), axis=-1)
    pixels, world = pixels[keep], world[keep]
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


# name, frame, cards, tensor shape. Each header's linear axes are Paper
# III's linear spectral types or STOKES.
CUBES = [
    (
        "MUSE-like cube, CD with zero cross terms, AWAV in Angstrom",
        "fk5_j2000",
        [
            ("CTYPE1", "RA---TAN"),
            ("CTYPE2", "DEC--TAN"),
            ("CTYPE3", "AWAV"),
            ("CUNIT3", "Angstrom"),
            ("CRPIX1", 160.5),
            ("CRPIX2", 155.25),
            ("CRPIX3", 1.0),
            ("CD1_1", -5.55555555555556e-05),
            ("CD1_2", 2.0e-07),
            ("CD2_1", 1.5e-07),
            ("CD2_2", 5.55555555555556e-05),
            ("CD3_3", 1.25),
            ("CD1_3", 0.0),
            ("CD2_3", 0.0),
            ("CD3_1", 0.0),
            ("CD3_2", 0.0),
            ("CRVAL1", 150.1163213),
            ("CRVAL2", 2.2009078),
            ("CRVAL3", 4749.8759765625),
            ("RADESYS", "FK5"),
            ("EQUINOX", 2000.0),
        ],
        (7, 6, 5),
    ),
    (
        "radio cube, SIN, FREQ and STOKES",
        "icrs",
        [
            ("WCSAXES", 4),
            ("CTYPE1", "RA---SIN"),
            ("CTYPE2", "DEC--SIN"),
            ("CTYPE3", "FREQ"),
            ("CTYPE4", "STOKES"),
            ("CRPIX1", 129.0),
            ("CRPIX2", 129.0),
            ("CRPIX3", 1.0),
            ("CRPIX4", 1.0),
            ("CDELT1", -0.000555555555555556),
            ("CDELT2", 0.000555555555555556),
            ("CDELT3", 244140.625),
            ("CDELT4", 1.0),
            ("CUNIT3", "Hz"),
            ("CRVAL1", 83.8221),
            ("CRVAL2", -5.3911),
            ("CRVAL3", 1.420405752e09),
            ("CRVAL4", 1.0),
            ("RESTFRQ", 1.420405752e09),
            ("SPECSYS", "LSRK"),
        ],
        (2, 6, 5, 4),
    ),
    (
        "RA, FREQ, DEC: the pair's axes apart",
        "icrs",
        [
            ("CTYPE1", "RA---TAN"),
            ("CTYPE2", "FREQ"),
            ("CTYPE3", "DEC--TAN"),
            ("CUNIT2", "GHz"),
            ("CRPIX1", 10.5),
            ("CRPIX2", 3.0),
            ("CRPIX3", 8.25),
            ("CDELT1", -0.001),
            ("CDELT2", 0.002),
            ("CDELT3", 0.001),
            ("PC1_1", 0.9),
            ("PC1_3", 0.1),
            ("PC3_1", -0.1),
            ("PC3_3", 0.9),
            ("CRVAL1", 210.5),
            ("CRVAL2", 115.27),
            ("CRVAL3", -30.25),
        ],
        (6, 4, 7),
    ),
    (
        "PC3_1 couples VRAD with RA",
        "icrs",
        [
            ("CTYPE1", "RA---TAN"),
            ("CTYPE2", "DEC--TAN"),
            ("CTYPE3", "VRAD"),
            ("CUNIT3", "km/s"),
            ("CRPIX1", 20.0),
            ("CRPIX2", 15.0),
            ("CRPIX3", 4.0),
            ("CDELT1", -0.0002),
            ("CDELT2", 0.0002),
            ("CDELT3", 2.5),
            ("PC3_1", 0.004),
            ("CRVAL1", 45.25),
            ("CRVAL2", 60.75),
            ("CRVAL3", -120.0),
            ("RESTFRQ", 2.30538e11),
        ],
        (8, 9, 10),
    ),
    (
        "RA, FREQ, DEC with PC2_1 coupling FREQ with RA",
        "icrs",
        [
            ("CTYPE1", "RA---TAN"),
            ("CTYPE2", "FREQ"),
            ("CTYPE3", "DEC--TAN"),
            ("CRPIX1", 10.5),
            ("CRPIX2", 3.0),
            ("CRPIX3", 8.25),
            ("CDELT1", -0.001),
            ("CDELT2", 2.0e6),
            ("CDELT3", 0.001),
            ("PC1_3", 0.1),
            ("PC3_1", -0.1),
            ("PC2_1", 0.03),
            ("CRVAL1", 210.5),
            ("CRVAL2", 1.1527e11),
            ("CRVAL3", -30.25),
        ],
        (6, 4, 7),
    ),
    (
        "CD3_1 couples AWAV with RA",
        "icrs",
        [
            ("CTYPE1", "RA---TAN"),
            ("CTYPE2", "DEC--TAN"),
            ("CTYPE3", "AWAV"),
            ("CUNIT3", "Angstrom"),
            ("CRPIX1", 16.5),
            ("CRPIX2", 15.25),
            ("CRPIX3", 1.0),
            ("CD1_1", -5.5e-05),
            ("CD2_2", 5.5e-05),
            ("CD3_3", 1.25),
            ("CD3_1", 0.5),
            ("CRVAL1", 150.1163213),
            ("CRVAL2", 2.2009078),
            ("CRVAL3", 4749.875),
        ],
        (5, 6, 7),
    ),
    (
        "a spectrum, WAVE in nm",
        "",
        [
            ("CTYPE1", "WAVE"),
            ("CUNIT1", "nm"),
            ("CRPIX1", 1.0),
            ("CRVAL1", 360.0),
            ("CDELT1", 0.1),
        ],
        (11,),
    ),
]


AXIS_KEY = re.compile(r"^(CTYPE|CUNIT|CRPIX|CRVAL|CDELT)(\d+)$")
MATRIX_KEY = re.compile(r"^(PC|CD)(\d+)_(\d+)$")


def celestial_alone(cards, axes):
    """The cards of [axes] alone, renumbered 1, 2, ..., with the others'
    dropped and keywords of no axis kept."""
    number = {a: k + 1 for k, a in enumerate(axes)}
    out = [("NAXIS", len(axes))]
    for k, v in cards:
        if k == "WCSAXES":
            out.append((k, len(axes)))
            continue
        m = AXIS_KEY.match(k)
        if m:
            a = int(m.group(2))
            if a in number:
                out.append((f"{m.group(1)}{number[a]}", v))
            continue
        m = MATRIX_KEY.match(k)
        if m:
            i, j = int(m.group(2)), int(m.group(3))
            if i in number and j in number:
                out.append((f"{m.group(1)}{number[i]}_{number[j]}", v))
            continue
        out.append((k, v))
    return out


def cube(name, frame, cards, shape):
    n = len(shape)
    h = fits.Header()
    h["NAXIS"] = n
    for k in range(n):
        h[f"NAXIS{k + 1}"] = shape[n - 1 - k]
    for k, v in cards:
        h.append(fits.Card(k, v))
    w = WCS(h)
    w.wcs.set()
    rng = np.random.default_rng(len(name))
    edges = [np.linspace(-0.5, s - 0.5, 3) for s in shape]
    grid = np.stack(np.meshgrid(*edges, indexing="ij"), axis=-1).reshape(-1, n)
    inside = rng.uniform(-0.5, 1, size=(9, n)) * (np.array(shape) - 0.5)
    pixels = np.concatenate([grid, inside])
    world = np.stack(
        w.all_pix2world(*[pixels[:, n - 1 - k] for k in range(n)], 0), axis=-1
    )
    celestial = [a + 1 for a in (w.wcs.lng, w.wcs.lat) if a >= 0]
    linear = [a for a in range(n) if a + 1 not in celestial]
    order = ([w.wcs.lng, w.wcs.lat] if celestial else []) + linear
    alone = (
        "".join(
            c.image
            for c in fits.Header(
                [fits.Card(k, v) for k, v in celestial_alone(cards, sorted(celestial))]
            ).cards
        )
        if celestial
        else ""
    )
    text = "".join(c.image for c in h.cards)
    return {
        "name": name,
        "frame": frame,
        "header": text,
        "alone": alone,
        "shape": shape,
        "celestial": sorted(celestial),
        "units": [w.wcs.cunit[a].to_string("fits") for a in linear],
        "pixels": pixels,
        "world": world[:, order],
    }


def record_lines(text):
    rs = [text[i : i + 80] for i in range(0, len(text), 80)]
    return "      " + "\n      ^ ".join(f'"{r}"' for r in rs)


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
    out += [
        "",
        "type cube = {",
        "  name : string;",
        "  cube_frame : string;  (** The frame's {!Ymir.Frame.name}, if any. *)",
        "  header : string;  (** 80-byte records, without END. *)",
        "  alone : string;  (** The celestial axes' cards alone, renumbered. *)",
        "  shape : int array;  (** The data's shape, in tensor order. *)",
        "  celestial : int array;  (** The celestial pair's FITS axes. *)",
        "  units : string array;  (** WCSLIB's CUNIT of each linear axis. *)",
        "  pixels : float array;  (** 0-based, in tensor order. *)",
        "  world : float array;",
        "      (** WCSLIB's [(lon, lat)] in degrees, then each linear axis's",
        "          value in its unit, in FITS order. *)",
        "}",
        "",
        "let cubes = [",
    ]
    for c in (cube(*x) for x in CUBES):
        out.append("  {")
        out.append(f'    name = "{c["name"]}";')
        out.append(f'    cube_frame = "{c["frame"]}";')
        out.append("    header =")
        out.append(record_lines(c["header"]) + ";")
        out.append("    alone =")
        out.append((record_lines(c["alone"]) if c["alone"] else '      ""') + ";")
        out.append("    shape = [| " + "; ".join(str(x) for x in c["shape"]) + " |];")
        out.append("    celestial = [| " + "; ".join(str(x) for x in c["celestial"]) + " |];")
        out.append("    units = [| " + "; ".join(f'"{u}"' for u in c["units"]) + " |];")
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
