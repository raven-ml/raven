# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "astropy", "photutils"]
# ///
"""Generate the grids' test references.

    uv run contrib/ymir/test/gen/grids.py [--check]

It writes test/support/grids_reference.ml:

- for a TAN and an ARC header, pixel coordinates with WCSLIB's world
  coordinates (astropy.wcs, the core WCS with origin 0), and world
  coordinates with WCSLIB's pixel coordinates;
- an image of values from an integer formula, with photutils' exact sums over pixel
  circles and annuli at positions that put their edges on every kind of
  cell crossing;
- a TAN image of 0.031 arcsecond pixels, with photutils' exact sums over
  sky circles and annuli.

photutils places a sky aperture in pixels through the local scale at its
position; at an arcsecond from the tangent point that differs from the
exact cap by about 1e-11 relative.

Without --check the file is written; with it, nothing is written and the
run fails if the file would change.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from photutils.aperture import (
    CircularAnnulus,
    CircularAperture,
    SkyCircularAnnulus,
    SkyCircularAperture,
    aperture_photometry,
)

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / "support" / "grids_reference.ml"


def ocaml_float(x):
    x = float(x)
    if x == 0.0:
        return "0." if np.copysign(1.0, x) > 0 else "-0."
    return x.hex().replace("0x1.0000000000000p", "0x1p")


def floats(xs):
    return "[| " + "; ".join(ocaml_float(x) for x in xs) + " |]"


def header(code, crpix, cd, crval):
    w = WCS(naxis=2)
    w.wcs.ctype = [f"RA---{code}", f"DEC--{code}"]
    w.wcs.crpix = crpix
    w.wcs.cd = cd
    w.wcs.crval = crval
    w.wcs.cunit = ["deg", "deg"]
    w.wcs.radesys = "ICRS"
    w.wcs.set()
    return w


# WCSLIB

WCS_CASES = [
    # code, CRPIX (1-based), CD (deg), CRVAL (deg), pixel step
    ("TAN", [512.5, 300.25], [[-2.0e-4, 3.0e-5], [2.5e-5, 2.0e-4]], [110.8375, -73.4537], 97),
    ("ARC", [100.0, 80.0], [[-0.5, 0.0], [0.0, 0.5]], [266.4, -29.0], 41),
]


def wcslib_case(code, crpix, cd, crval, step):
    w = header(code, crpix, cd, crval)
    rows, cols = np.meshgrid(np.arange(-50, 1100, step), np.arange(-40, 700, step), indexing="ij")
    if code == "ARC":
        rows, cols = np.meshgrid(np.arange(-150, 360, step), np.arange(-150, 330, step), indexing="ij")
    pixels = np.stack([rows.ravel(), cols.ravel()], axis=-1).astype(float)
    # Fractional positions as well as integers.
    pixels = np.concatenate([pixels, pixels[: len(pixels) // 2] + [0.37, -0.81]])
    ra, dec = w.wcs_pix2world(pixels[:, 1], pixels[:, 0], 0)
    if code == "ARC":
        r = np.hypot(*(np.asarray(cd) @ (pixels[:, ::-1] + 1 - np.asarray(crpix)).T))
        keep = r < 179.0
        pixels, ra, dec = pixels[keep], ra[keep], dec[keep]
    # World to pixel: points of the sky around the reference.
    sky_ra = crval[0] + np.linspace(-3, 3, 13) * (1 if code == "TAN" else 20)
    sky_dec = crval[1] + np.linspace(-2, 2, 13) * (1 if code == "TAN" else 15)
    sky_ra, sky_dec = np.meshgrid(sky_ra, sky_dec)
    x, y = w.wcs_world2pix(sky_ra.ravel(), sky_dec.ravel(), 0)
    return {
        "code": code,
        "crpix": crpix,
        "cd": np.asarray(cd).ravel(),
        "crval": crval,
        "pixels": pixels,
        "world": np.stack([ra, dec], axis=-1),
        "sky": np.stack([sky_ra.ravel(), sky_dec.ravel()], axis=-1),
        "sky_pixels": np.stack([y, x], axis=-1),
    }


# photutils, pixel apertures

IMAGE_SHAPE = (48, 40)

# (row, column, radius) of each circle; an annulus adds an inner radius.
CIRCLES = [
    (20.0, 18.0, 6.0),
    (20.5, 18.5, 6.0),
    (19.3, 21.7, 7.25),
    (24.0, 20.0, 0.3),
    (24.5, 20.5, 0.5),
    (23.1, 17.9, 1.0),
    (25.0, 19.0, 11.999),
    (22.123, 19.987, 9.5),
]
ANNULI = [(20.0, 18.0, 3.0, 8.0), (21.4, 19.6, 4.5, 10.25), (24.0, 20.0, 0.2, 1.7)]


def image(shape, seed):
    """The test images: [((7919 i + 104729 j + 13 i j + seed) mod 1009) / 64],
    exact in float64 and float32, which the tests compute the same way."""
    i, j = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    return ((7919 * i + 104729 * j + 13 * i * j + seed) % 1009) / 64.0


def pixel_apertures():
    data = image(IMAGE_SHAPE, 0)
    circles = []
    for row, col, r in CIRCLES:
        t = aperture_photometry(data, CircularAperture((col, row), r), method="exact")
        circles.append((row, col, r, float(t["aperture_sum"][0])))
    annuli = []
    for row, col, r_in, r_out in ANNULI:
        t = aperture_photometry(data, CircularAnnulus((col, row), r_in, r_out), method="exact")
        annuli.append((row, col, r_in, r_out, float(t["aperture_sum"][0])))
    return data, circles, annuli


# photutils, sky apertures on a TAN image

SKY_SHAPE = (96, 96)
SKY_PIXEL = 0.031 / 3600.0
SKY_CRPIX = [48.5, 47.25]
SKY_CRVAL = [110.8375, -73.4537]
# (east, north) offsets in arcseconds from CRVAL, and radii in arcseconds.
SKY_CIRCLES = [(0.0, 0.0, 0.5), (0.21, -0.13, 0.5), (-0.3, 0.27, 0.8), (0.05, 0.4, 0.25)]
SKY_ANNULI = [(0.0, 0.0, 0.6, 1.2), (-0.11, 0.07, 0.5, 1.0)]


def sky_wcs():
    angle = np.deg2rad(12.0)
    c, s = np.cos(angle), np.sin(angle)
    cd = SKY_PIXEL * np.array([[-c, s], [s, c]])
    return header("TAN", SKY_CRPIX, cd, SKY_CRVAL), cd


def offset_to_sky(east, north):
    centre = SkyCoord(SKY_CRVAL[0] * u.deg, SKY_CRVAL[1] * u.deg)
    sep = np.hypot(east, north) * u.arcsec
    pa = np.arctan2(east, north) * u.rad
    return centre.directional_offset_by(pa, sep)


def sky_apertures():
    data = image(SKY_SHAPE, 73)
    w, cd = sky_wcs()
    circles = []
    for east, north, r in SKY_CIRCLES:
        pos = offset_to_sky(east, north)
        t = aperture_photometry(data, SkyCircularAperture(pos, r * u.arcsec), wcs=w, method="exact")
        circles.append((east, north, r, float(t["aperture_sum"][0])))
    annuli = []
    for east, north, r_in, r_out in SKY_ANNULI:
        pos = offset_to_sky(east, north)
        ap = SkyCircularAnnulus(pos, r_in * u.arcsec, r_out * u.arcsec)
        t = aperture_photometry(data, ap, wcs=w, method="exact")
        annuli.append((east, north, r_in, r_out, float(t["aperture_sum"][0])))
    return data, cd, circles, annuli


# Rendering


def render():
    out = [
        "(*---------------------------------------------------------------------------",
        "  Copyright (c) 2026 The Raven authors. All rights reserved.",
        "  SPDX-License-Identifier: ISC",
        "  ---------------------------------------------------------------------------*)",
        "",
        "(* Written by test/gen/grids.py; do not edit. *)",
        "",
        '[@@@ocamlformat "disable"]',
        "",
        "type wcs = {",
        "  code : string;",
        "  crpix : float array;  (** FITS order, 1-based. *)",
        "  cd : float array;  (** Row-major, degrees. *)",
        "  crval : float array;  (** Degrees. *)",
        "  pixels : float array;  (** [(row, column)] pairs, 0-based. *)",
        "  world : float array;  (** WCSLIB's [(ra, dec)] in degrees for each pixel. *)",
        "  sky : float array;  (** [(ra, dec)] pairs in degrees. *)",
        "  sky_pixels : float array;  (** WCSLIB's [(row, column)] for each. *)",
        "}",
        "",
        "let wcs = [",
    ]
    for case in (wcslib_case(*c) for c in WCS_CASES):
        out.append("  {")
        out.append(f'    code = "{case["code"]}";')
        for key in ("crpix", "cd", "crval"):
            out.append(f"    {key} = {floats(case[key])};")
        for key in ("pixels", "world", "sky", "sky_pixels"):
            out.append(f"    {key} = {floats(np.asarray(case[key]).ravel())};")
        out.append("  };")
    out.append("]")
    out.append("")

    data, circles, annuli = pixel_apertures()
    out.append(f"let image_shape = [| {IMAGE_SHAPE[0]}; {IMAGE_SHAPE[1]} |]")
    out.append("")
    out.append("(* (row, column, radius, photutils' exact sum). *)")
    out.append("let pixel_circles = [")
    for row, col, r, s in circles:
        out.append(f"  ({ocaml_float(row)}, {ocaml_float(col)}, {ocaml_float(r)}, {ocaml_float(s)});")
    out.append("]")
    out.append("")
    out.append("(* (row, column, inner, outer, photutils' exact sum). *)")
    out.append("let pixel_annuli = [")
    for row, col, r_in, r_out, s in annuli:
        out.append(
            f"  ({ocaml_float(row)}, {ocaml_float(col)}, {ocaml_float(r_in)}, {ocaml_float(r_out)}, {ocaml_float(s)});"
        )
    out.append("]")
    out.append("")

    data, cd, circles, annuli = sky_apertures()
    out.append(f"let sky_shape = [| {SKY_SHAPE[0]}; {SKY_SHAPE[1]} |]")
    out.append(f"let sky_crpix = {floats(SKY_CRPIX)}")
    out.append(f"let sky_cd = {floats(cd.ravel())}")
    out.append(f"let sky_crval = {floats(SKY_CRVAL)}")
    out.append("")
    out.append("(* (east, north, radius) in arcseconds, and photutils' exact sum. *)")
    out.append("let sky_circles = [")
    for east, north, r, s in circles:
        out.append(f"  ({ocaml_float(east)}, {ocaml_float(north)}, {ocaml_float(r)}, {ocaml_float(s)});")
    out.append("]")
    out.append("")
    out.append("(* (east, north, inner, outer) in arcseconds, and photutils' exact sum. *)")
    out.append("let sky_annuli = [")
    for east, north, r_in, r_out, s in annuli:
        out.append(
            f"  ({ocaml_float(east)}, {ocaml_float(north)}, {ocaml_float(r_in)}, {ocaml_float(r_out)}, {ocaml_float(s)});"
        )
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
