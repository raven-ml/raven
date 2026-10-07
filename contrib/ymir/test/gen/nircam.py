# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "astropy", "photutils==3.0.0"]
# ///
"""Generate the NIRCam aperture-photometry references.

    uv run contrib/ymir/test/gen/nircam.py [--check] [MOSAIC]

MOSAIC is the JWST NIRCam F200W i2d mosaic of SMACS 0723,
jw02736-o001_t001_nircam_clear-f200w_i2d.fits (1.76 GB, public on MAST). By
default it is ~/.cache/ymir/<that name>, downloaded there when absent and
checked against its SHA-256.

It writes:

- test/golden/nircam.fits: for each of five targets, the 104x104 SCI and
  ERR cutouts around it (EXTVER 1 to 5), each SCI header the mosaic's with
  NAXISn and CRPIXn moved to the cutout;
- test/support/nircam_reference.ml: for the five targets and for a grid of
  apertures over the whole mosaic, photutils' method='exact' sums of a sky
  circle of 0.5 arcsec and a sky annulus of 1 to 1.5 arcsec on the full
  mosaic, times PIXAR_SR, with non-finite pixels masked; the sum of
  |data * w * PIXAR_SR| that scales the tolerance; and the variance
  sum (w * PIXAR_SR)^2 * ERR^2 over photutils' exact weights.

photutils places a sky aperture in pixels as a circle through the local
scale at its position. Within 3 arcmin of the tangent point that differs
from the exact cap by at most 0.4e-6 relative.

Without --check the files are written; with it, nothing is written and the
run fails if a file would change.
"""

import argparse
import hashlib
import os
from pathlib import Path
import sys
import urllib.request

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from photutils.aperture import SkyCircularAnnulus, SkyCircularAperture, aperture_photometry

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / "support" / "nircam_reference.ml"
GOLDEN = HERE.parent / "golden" / "nircam.fits"

NAME = "jw02736-o001_t001_nircam_clear-f200w_i2d.fits"
URL = f"https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:JWST/product/{NAME}"
SHA256 = "867e16826de13eedac74ce678d911e6506877e7e87732c681c83234a24e9c0a9"
DEFAULT = Path.home() / ".cache" / "ymir" / NAME

STAMP = 104
RADIUS = 0.5
INNER, OUTER = 1.0, 1.5

# (ra, dec) in degrees: the target of the grids' Guide, three bright
# sources, and one whose annulus reaches past the mosaic's coverage.
TARGETS = [
    (110.8375, -73.4537),
    (110.796215, -73.466024),
    (110.857999, -73.464247),
    (110.754009, -73.484710),
    (110.773003, -73.466389),
]

# The whole-mosaic grid of aperture centres, in pixels (row, column).
GRID_ROWS, GRID_COLUMNS = 8, 25


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def mosaic(path):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(".part")
        print(f"downloading {URL} to {path}", file=sys.stderr)
        urllib.request.urlretrieve(URL, part)
        os.replace(part, path)
    digest = sha256(path)
    if digest != SHA256:
        sys.exit(f"{path}: SHA-256 {digest}, expected {SHA256}")
    return path


def ocaml_float(x):
    x = float(x)
    if x == 0.0:
        return "0." if np.copysign(1.0, x) > 0 else "-0."
    return x.hex().replace("0x1.0000000000000p", "0x1p")


def measure(sci, err, w, pixar, ra, dec):
    """photutils' sums for the circle and the annulus about (ra, dec)."""
    finite = np.isfinite(sci)
    data = np.where(finite, sci, 0.0)
    pos = SkyCoord(ra * u.deg, dec * u.deg)
    out = []
    for ap in (
        SkyCircularAperture(pos, RADIUS * u.arcsec),
        SkyCircularAnnulus(pos, INNER * u.arcsec, OUTER * u.arcsec),
    ):
        total = float(aperture_photometry(sci, ap, wcs=w, method="exact", mask=~finite)["aperture_sum"][0])
        mask = ap.to_pixel(w).to_mask(method="exact")
        weight = mask.data
        d = mask.cutout(data, fill_value=0.0)
        e = mask.cutout(np.where(finite, err, 0.0), fill_value=0.0)
        scale = float(np.sum(np.abs(d) * weight))
        variance = float(np.sum((weight * pixar) ** 2 * e.astype(np.float64) ** 2))
        out.append((total * pixar, scale * pixar, variance))
    return out


def cutout_start(w, ra, dec):
    """The first cell of the 104x104 block centred on the cell holding
    (ra, dec), its extra cell on the high side."""
    x, y = w.world_to_pixel_values(ra, dec)
    row, col = int(np.floor(y + 0.5)), int(np.floor(x + 0.5))
    return row - STAMP // 2 + 1, col - STAMP // 2 + 1


def golden(f, w):
    sci_h = f["SCI"].header
    err_h = f["ERR"].header
    hdus = [fits.PrimaryHDU()]
    for k, (ra, dec) in enumerate(TARGETS, start=1):
        r0, c0 = cutout_start(w, ra, dec)
        window = (slice(r0, r0 + STAMP), slice(c0, c0 + STAMP))
        h = sci_h.copy()
        h["CRPIX1"] = sci_h["CRPIX1"] - c0
        h["CRPIX2"] = sci_h["CRPIX2"] - r0
        h["EXTVER"] = k
        e = err_h.copy()
        e["EXTVER"] = k
        hdus.append(fits.ImageHDU(f["SCI"].data[window], header=h, name="SCI", ver=k))
        hdus.append(fits.ImageHDU(f["ERR"].data[window], header=e, name="ERR", ver=k))
    return fits.HDUList(hdus)


def render(sci, err, w, pixar):
    out = [
        "(*---------------------------------------------------------------------------",
        "  Copyright (c) 2026 The Raven authors. All rights reserved.",
        "  SPDX-License-Identifier: ISC",
        "  ---------------------------------------------------------------------------*)",
        "",
        "(* Written by test/gen/nircam.py; do not edit. *)",
        "",
        '[@@@ocamlformat "disable"]',
        "",
        "type sum = {",
        "  sum : float;  (** photutils' sum times PIXAR_SR, MJy. *)",
        "  scale : float;  (** The sum of |data * w * PIXAR_SR|, MJy. *)",
        "  variance : float;  (** The sum of (w * PIXAR_SR)^2 * ERR^2, MJy^2. *)",
        "}",
        "",
        "type aperture = {",
        "  ra : float;  (** Degrees. *)",
        "  dec : float;",
        "  circle : sum;",
        "  annulus : sum;",
        "}",
        "",
        f"let radius = {ocaml_float(RADIUS)}",
        f"let inner = {ocaml_float(INNER)}",
        f"let outer = {ocaml_float(OUTER)}",
        f"let stamp = {STAMP}",
        f'let mosaic = "{NAME}"',
        f'let sha256 = "{SHA256}"',
        "",
    ]

    def aperture(ra, dec):
        (cs, cscale, cv), (asum, ascale, av) = measure(sci, err, w, pixar, ra, dec)
        sums = lambda s, sc, v: f"{{ sum = {ocaml_float(s)}; scale = {ocaml_float(sc)}; variance = {ocaml_float(v)} }}"
        return (
            f"  {{ ra = {ocaml_float(ra)}; dec = {ocaml_float(dec)};\n"
            f"    circle = {sums(cs, cscale, cv)};\n"
            f"    annulus = {sums(asum, ascale, av)} }};"
        )

    out.append("(* The five targets of golden/nircam.fits, EXTVER 1 to 5. *)")
    out.append("let targets = [")
    for ra, dec in TARGETS:
        out.append(aperture(ra, dec))
    out.append("]")
    out.append("")
    rows, cols = sci.shape
    out.append(f"(* A {GRID_ROWS}x{GRID_COLUMNS} grid of apertures over the whole mosaic. *)")
    out.append("let grid = [")
    for i in range(GRID_ROWS):
        for j in range(GRID_COLUMNS):
            y = 60 + (rows - 120) * i / (GRID_ROWS - 1)
            x = 60 + (cols - 120) * j / (GRID_COLUMNS - 1)
            ra, dec = (float(v) for v in w.pixel_to_world_values(x, y))
            ra, dec = round(ra, 7), round(dec, 7)
            out.append(aperture(ra, dec))
    out.append("]")
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("mosaic", nargs="?", default=DEFAULT)
    args = parser.parse_args()
    with fits.open(mosaic(args.mosaic), memmap=True) as f:
        w = WCS(f["SCI"].header)
        pixar = f["SCI"].header["PIXAR_SR"]
        sci = f["SCI"].data.astype(np.float64)
        err = f["ERR"].data
        text = render(sci, err, w, pixar)
        hdul = golden(f, w)
        if args.check:
            if not REFERENCE.exists() or REFERENCE.read_text() != text:
                sys.exit(f"{REFERENCE} would change; run without --check")
            with fits.open(GOLDEN) as g:
                for a, b in zip(g, hdul):
                    if a.header != b.header or not np.array_equal(a.data, b.data, equal_nan=True):
                        sys.exit(f"{GOLDEN} would change; run without --check")
            return
        REFERENCE.write_text(text)
        hdul.writeto(GOLDEN, overwrite=True)
    with fits.open(GOLDEN) as g:
        for k in range(1, len(TARGETS) + 1):
            # The cutout's CRPIX must read back as the mosaic's less an integer.
            h = g["SCI", k].header
            r0, c0 = cutout_start(WCS(fits.getheader(args.mosaic, "SCI")), *TARGETS[k - 1])
            assert h["CRPIX1"] + c0 == fits.getheader(args.mosaic, "SCI")["CRPIX1"]
            assert h["CRPIX2"] + r0 == fits.getheader(args.mosaic, "SCI")["CRPIX2"]


if __name__ == "__main__":
    main()
