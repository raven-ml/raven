# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "astropy", "photutils", "regions"]
# ///
"""Generate the grids' test references.

    uv run contrib/ymir/test/gen/grids.py [--check]

It writes test/support/grids_reference.ml:

- for a TAN and an ARC header, pixel coordinates with WCSLIB's world
  coordinates (astropy.wcs, the core WCS with origin 0), and world
  coordinates with WCSLIB's pixel coordinates;
- for each projection, with parameters, native reference points and
  poles that reach every branch of the pole's solution, the same both
  ways, and the points WCSLIB leaves outside each domain;
- for a SIP header, with and without AP and BP, and a TPV header, pixel
  coordinates with astropy's all_pix2world and world coordinates with its
  all_world2pix;
- an image of values from an integer formula, with photutils' exact sums over pixel
  circles, annuli and ellipses at positions that put their edges on every
  kind of cell crossing, and astropy regions' exact sums over polygons;
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
from astropy.wcs import WCS, Sip
from regions import PixCoord, PolygonPixelRegion
from photutils.aperture import (
    CircularAnnulus,
    CircularAperture,
    EllipticalAperture,
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


# Projections

# Each code's parameters in ymir's order: the FITS index of the first, and
# the defaults.
PARAMETERS = {
    "AZP": (1, [0.0, 0.0]),
    "SZP": (1, [0.0, 0.0, 90.0]),
    "TAN": (1, []),
    "STG": (1, []),
    "SIN": (1, [0.0, 0.0]),
    "ARC": (1, []),
    "ZPN": (0, [0.0] * 30),
    "ZEA": (1, []),
    "AIR": (1, [90.0]),
    "CYP": (1, [1.0, 1.0]),
    "CEA": (1, [1.0]),
    "CAR": (1, []),
    "MER": (1, []),
}
CYLINDRICAL = {"CYP", "CEA", "CAR", "MER"}

# name, code, {PV2_m: value}, CRVAL, plane extent (deg), sky extent (deg),
# and stated (phi0, theta0), LONPOLE, LATPOLE (None when absent).
PROJECTIONS = [
    ("AZP tilted", "AZP", {1: 2.0, 2: 30.0}, [30.0, 40.0], 80.0, (60.0, 50.0), None, None, None),
    ("AZP inside", "AZP", {1: 0.5}, [200.0, -10.0], 160.0, (120.0, 80.0), None, None, None),
    ("SZP", "SZP", {1: 2.0, 2: 180.0, 3: 60.0}, [150.0, 60.0], 80.0, (150.0, 25.0), None, None, None),
    ("TAN at theta0 45", "TAN", {}, [80.0, 20.0], 40.0, (50.0, 40.0), (0.0, 45.0), None, None),
    ("STG", "STG", {}, [10.0, -70.0], 300.0, (180.0, 20.0), None, None, None),
    ("SIN", "SIN", {}, [300.0, 5.0], 70.0, (100.0, 80.0), None, None, None),
    ("SIN slant", "SIN", {1: 0.2, 2: -0.1}, [45.0, 45.0], 70.0, (100.0, 44.0), None, None, None),
    ("ARC beyond the pole", "ARC", {}, [0.0, 85.0], 220.0, (180.0, 5.0), None, 150.0, None),
    ("ZPN cubic", "ZPN", {0: 0.0, 1: 1.0, 2: 0.0, 3: -0.25}, [120.0, -45.0], 60.0, (90.0, 44.0), None, None, None),
    ("ZPN quintic", "ZPN", {1: 1.0, 3: 0.05, 5: -0.01}, [10.0, 10.0], 150.0, (90.0, 79.0), None, None, None),
    ("ZEA", "ZEA", {}, [240.0, -30.0], 150.0, (180.0, 60.0), None, None, None),
    ("AIR", "AIR", {}, [60.0, 70.0], 200.0, (180.0, 19.0), None, None, None),
    ("AIR at 45", "AIR", {1: 45.0}, [60.0, -20.0], 200.0, (180.0, 69.0), None, None, None),
    ("CYP", "CYP", {}, [0.0, 0.0], 230.0, (200.0, 85.0), None, None, None),
    ("CYP narrow", "CYP", {1: 0.5, 2: 0.8}, [100.0, 0.0], 200.0, (200.0, 85.0), None, None, None),
    ("CEA", "CEA", {1: 0.5}, [330.0, 0.0], 230.0, (200.0, 85.0), None, None, None),
    ("CAR oblique", "CAR", {}, [120.0, 30.0], 200.0, (200.0, 59.0), None, None, None),
    ("CAR, LONPOLE 60", "CAR", {}, [10.0, 20.0], 200.0, (200.0, 69.0), None, 60.0, None),
    ("CAR, LONPOLE 60, south pole", "CAR", {}, [10.0, 20.0], 200.0, (200.0, 69.0), None, 60.0, -90.0),
    ("MER", "MER", {}, [45.0, -20.0], 200.0, (200.0, 69.0), None, None, None),
    ("CEA at the pole", "CEA", {}, [10.0, 90.0], 200.0, (200.0, 40.0), None, None, None),
]

PLANE_STEPS = 23
SKY_STEPS = 17


def projection_case(name, code, pv_lat, crval, extent, sky_extent, native, lonpole, latpole):
    first, defaults = PARAMETERS[code]
    pv = list(defaults)
    for m, value in pv_lat.items():
        pv[m - first] = value
    w = WCS(naxis=2)
    w.wcs.ctype = [f"RA---{code}", f"DEC--{code}"]
    crpix = [40.5, 31.25]
    scale = 0.75
    angle = np.deg2rad(20.0)
    cd = scale * np.array([[-np.cos(angle), np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    w.wcs.crpix = crpix
    w.wcs.cd = cd
    w.wcs.crval = crval
    w.wcs.cunit = ["deg", "deg"]
    w.wcs.radesys = "ICRS"
    cards = [(2, m, v) for m, v in pv_lat.items()]
    theta0_default = 0.0 if code in CYLINDRICAL else 90.0
    phi0, theta0 = native if native is not None else (0.0, theta0_default)
    if native is not None:
        cards += [(1, 1, phi0), (1, 2, theta0)]
    w.wcs.set_pv(cards)
    if lonpole is not None:
        w.wcs.lonpole = lonpole
    if latpole is not None:
        w.wcs.latpole = latpole
    w.wcs.set()
    if lonpole is None:
        lonpole = (180.0 if crval[1] < theta0 else 0.0) + phi0
    if latpole is None:
        latpole = 90.0
    # Pixels whose intermediate coordinates fill [-extent, extent]².
    g = np.linspace(-extent, extent, PLANE_STEPS) + 0.123
    gx, gy = np.meshgrid(g, g, indexing="ij")
    plane = np.stack([gx.ravel(), gy.ravel()], axis=-1)
    fits_pixels = plane @ np.linalg.inv(cd).T + np.asarray(crpix) - 1.0
    pixels = fits_pixels[:, ::-1]
    lon, lat = w.wcs_pix2world(fits_pixels[:, 0], fits_pixels[:, 1], 0)
    inside = np.isfinite(lon) & np.isfinite(lat)
    # Directions about CRVAL.
    sl = crval[0] + np.linspace(-sky_extent[0], sky_extent[0], SKY_STEPS) + 0.371
    sb = np.clip(crval[1] + np.linspace(-sky_extent[1], sky_extent[1], SKY_STEPS) + 0.173, -89.9, 89.9)
    sl, sb = np.meshgrid(sl, sb, indexing="ij")
    sky = np.stack([sl.ravel(), sb.ravel()], axis=-1)
    x, y = w.wcs_world2pix(sky[:, 0], sky[:, 1], 0)
    sky_inside = np.isfinite(x) & np.isfinite(y)
    return {
        "name": name,
        "code": code,
        "pv": pv,
        "native": [phi0, theta0],
        "crpix": crpix,
        "cd": cd.ravel(),
        "crval": crval,
        "lonpole": lonpole,
        "latpole": latpole,
        "pixels": pixels[inside],
        "world": np.stack([lon, lat], axis=-1)[inside],
        "outside": pixels[~inside],
        "sky": sky[sky_inside],
        "sky_pixels": np.stack([y, x], axis=-1)[sky_inside],
        "sky_outside": sky[~sky_inside],
    }


# Distortions

DISTORTED_SHAPE = (2048, 2048)
DISTORTED_CRPIX = [1024.5, 1020.25]
DISTORTED_CRVAL = [150.1, 2.2]

# SIP coefficients: A_p_q and B_p_q, order 3, a few pixels at the corners.
SIP_A = {(2, 0): 2.1e-6, (1, 1): -1.4e-6, (0, 2): 8.2e-7, (3, 0): 3.1e-10, (2, 1): -1.7e-10, (0, 3): 6.0e-11}
SIP_B = {(2, 0): -9.5e-7, (1, 1): 1.9e-6, (0, 2): -2.6e-6, (1, 2): 2.2e-10, (3, 0): -8.0e-11, (0, 3): 1.4e-10}
SIP_ORDER = 3
SIP_INVERSE_ORDER = 4

# TPV coefficients on each axis, degrees: PV1_k and PV2_k.
TPV = {
    1: {0: 1.2e-5, 1: 1.0003, 2: -2.1e-4, 4: 3.0e-4, 5: -1.1e-4, 6: 2.2e-4, 7: -5.0e-3, 9: 2.0e-3, 11: 1.5e-3, 17: 4.0e-4, 23: -2.0e-4, 39: 1.0e-5},
    2: {0: -8.0e-6, 1: 0.9996, 2: 1.4e-4, 4: -2.0e-4, 5: 1.7e-4, 6: 1.0e-4, 8: 3.0e-3, 10: -4.0e-3, 11: -1.0e-3, 24: 2.0e-4, 31: -1.0e-4},
}


def sip_matrix(coeffs, n):
    m = np.zeros((n + 1, n + 1))
    for (p, q), v in coeffs.items():
        m[p, q] = v
    return m


def distorted_wcs(kind, seeded):
    w = WCS(naxis=2)
    w.wcs.crpix = DISTORTED_CRPIX
    angle = np.deg2rad(-35.0)
    scale = 0.05 / 3600.0
    w.wcs.cd = scale * np.array([[-np.cos(angle), np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    w.wcs.crval = DISTORTED_CRVAL
    w.wcs.cunit = ["deg", "deg"]
    w.wcs.radesys = "ICRS"
    ap = bp = None
    if kind == "SIP":
        w.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
        a = sip_matrix(SIP_A, SIP_ORDER)
        b = sip_matrix(SIP_B, SIP_ORDER)
        if seeded:
            ap, bp = fit_inverse(a, b)
        w.sip = Sip(a, b, ap, bp, DISTORTED_CRPIX)
    else:
        w.wcs.ctype = ["RA---TPV", "DEC--TPV"]
        w.wcs.set_pv([(i, k, v) for i in (1, 2) for k, v in TPV[i].items()])
    w.wcs.set()
    return w, ap, bp


def fit_inverse(a, b):
    """AP and BP by least squares over the image, as pipelines fit them."""
    g = np.linspace(-1100, 1100, 45)
    u, v = (x.ravel() for x in np.meshgrid(g, g, indexing="ij"))

    def poly(m, u, v):
        return sum(m[p, q] * u**p * v**q for p in range(m.shape[0]) for q in range(m.shape[1]))

    x = u + poly(a, u, v)
    y = v + poly(b, u, v)
    terms = [(p, q) for p in range(SIP_INVERSE_ORDER + 1) for q in range(SIP_INVERSE_ORDER + 1 - p)]
    design = np.stack([x**p * y**q for p, q in terms], axis=-1)
    ap = np.zeros((SIP_INVERSE_ORDER + 1,) * 2)
    bp = np.zeros((SIP_INVERSE_ORDER + 1,) * 2)
    for target, m in ((u - x, ap), (v - y, bp)):
        coef, *_ = np.linalg.lstsq(design, target, rcond=None)
        for (p, q), c in zip(terms, coef):
            m[p, q] = c
    return ap, bp


def distorted_case(name, kind, seeded):
    w, ap, bp = distorted_wcs(kind, seeded)
    rows = np.linspace(-0.5, DISTORTED_SHAPE[0] - 0.5, 11) + 0.31
    cols = np.linspace(-0.5, DISTORTED_SHAPE[1] - 0.5, 13) - 0.17
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    pixels = np.stack([rr.ravel(), cc.ravel()], axis=-1)
    lon, lat = w.all_pix2world(pixels[:, 1], pixels[:, 0], 0)
    # Directions over the image, from other pixels.
    sr = np.linspace(0, DISTORTED_SHAPE[0], 9) + 0.77
    sc = np.linspace(0, DISTORTED_SHAPE[1], 9) - 0.41
    sr, sc = np.meshgrid(sr, sc, indexing="ij")
    slon, slat = w.all_pix2world(sc.ravel(), sr.ravel(), 0)
    x, y = w.all_world2pix(slon, slat, 0, tolerance=1e-12, maxiter=100)
    case = {
        "name": name,
        "kind": kind,
        "crpix": DISTORTED_CRPIX,
        "cd": w.wcs.cd.ravel(),
        "crval": DISTORTED_CRVAL,
        "pixels": pixels,
        "world": np.stack([lon, lat], axis=-1),
        "sky": np.stack([slon, slat], axis=-1),
        "sky_pixels": np.stack([y, x], axis=-1),
    }
    if kind == "SIP":
        case["a"] = w.sip.a.ravel()
        case["b"] = w.sip.b.ravel()
        case["ap"] = None if ap is None else ap.ravel()
        case["bp"] = None if bp is None else bp.ravel()
        case["pv"] = []
    else:
        pv = np.zeros((2, 40))
        pv[0, 1] = pv[1, 1] = 1.0
        for i in (1, 2):
            for k, v in TPV[i].items():
                pv[i - 1, k] = v
        case["pv"] = pv.ravel()
        case["a"] = case["b"] = []
        case["ap"] = case["bp"] = None
    return case


DISTORTED = [("SIP", "SIP", False), ("SIP with AP and BP", "SIP", True), ("TPV", "TPV", False)]


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


# (row, column, a, b, angle in degrees from the column axis toward the row
# axis, photutils' theta) of each ellipse.
ELLIPSES = [
    (20.0, 18.0, 6.0, 3.0, 0.0),
    (20.5, 18.5, 6.0, 3.0, 90.0),
    (19.3, 21.7, 7.25, 2.1, 33.0),
    (24.0, 20.0, 0.9, 0.3, -71.5),
    (22.123, 19.987, 9.5, 9.5, 12.0),
    (23.1, 17.9, 11.0, 0.5, 125.0),
]

# Vertices (row, column) of each polygon: convex, concave, clockwise, with
# edges along cell edges and through cell corners.
POLYGONS = [
    [(10.0, 10.0), (10.0, 20.0), (20.0, 20.0), (20.0, 10.0)],
    [(10.3, 11.7), (25.1, 14.2), (21.9, 30.6), (12.2, 26.0)],
    [(12.0, 12.0), (30.0, 15.5), (18.0, 19.0), (28.0, 31.0), (14.0, 27.5)],
    [(14.0, 27.5), (28.0, 31.0), (18.0, 19.0), (30.0, 15.5), (12.0, 12.0)],
    [(20.5, 20.5), (21.5, 20.5), (21.5, 21.5)],
    [(9.5, 9.5), (9.5, 19.5), (19.5, 19.5), (19.5, 9.5)],
    [(5.25, 5.75), (40.5, 9.5), (33.0, 36.0), (21.0, 22.0), (8.0, 33.5), (15.0, 18.0)],
]


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
    ellipses = []
    for row, col, a, b, angle in ELLIPSES:
        ap = EllipticalAperture((col, row), a, b, theta=np.deg2rad(angle))
        t = aperture_photometry(data, ap, method="exact")
        ellipses.append((row, col, a, b, angle, float(t["aperture_sum"][0])))
    polygons = []
    for vertices in POLYGONS:
        rows, cols = np.array(vertices).T
        region = PolygonPixelRegion(PixCoord(x=cols, y=rows))
        weights = region.to_mask(mode="exact").to_image(IMAGE_SHAPE)
        polygons.append((vertices, float(np.sum(data * weights))))
    return data, circles, annuli, ellipses, polygons


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

    out.append("type projection = {")
    out.append("  name : string;")
    out.append("  code : string;")
    out.append("  pv : float array;  (** ymir's order, defaults filled. *)")
    out.append("  native : float array;  (** (phi0, theta0), degrees. *)")
    out.append("  crpix : float array;")
    out.append("  cd : float array;")
    out.append("  crval : float array;")
    out.append("  lonpole : float;")
    out.append("  latpole : float;")
    out.append("  pixels : float array;")
    out.append("  world : float array;")
    out.append("  outside : float array;  (** Pixels WCSLIB does not map. *)")
    out.append("  sky : float array;")
    out.append("  sky_pixels : float array;")
    out.append("  sky_outside : float array;  (** Directions WCSLIB does not map. *)")
    out.append("}")
    out.append("")
    out.append("let projections = [")
    for c in (projection_case(*p) for p in PROJECTIONS):
        out.append("  {")
        out.append(f'    name = "{c["name"]}";')
        out.append(f'    code = "{c["code"]}";')
        for key in ("pv", "native", "crpix", "cd", "crval"):
            out.append(f"    {key} = {floats(c[key])};")
        for key in ("lonpole", "latpole"):
            out.append(f"    {key} = {ocaml_float(c[key])};")
        for key in ("pixels", "world", "outside", "sky", "sky_pixels", "sky_outside"):
            out.append(f"    {key} = {floats(np.asarray(c[key]).ravel())};")
        out.append("  };")
    out.append("]")
    out.append("")

    out.append("type distorted = {")
    out.append("  name : string;")
    out.append("  kind : string;  (** SIP or TPV. *)")
    out.append("  crpix : float array;")
    out.append("  cd : float array;")
    out.append("  crval : float array;")
    out.append("  a : float array;  (** SIP's A, row-major [(order + 1)²]. *)")
    out.append("  b : float array;")
    out.append("  ap : float array option;")
    out.append("  bp : float array option;")
    out.append("  pv : float array;  (** TPV's [[2; 40]], row-major. *)")
    out.append("  pixels : float array;")
    out.append("  world : float array;  (** all_pix2world's. *)")
    out.append("  sky : float array;")
    out.append("  sky_pixels : float array;  (** all_world2pix's. *)")
    out.append("}")
    out.append("")
    out.append("let distorted = [")
    for c in (distorted_case(*d) for d in DISTORTED):
        out.append("  {")
        out.append(f'    name = "{c["name"]}";')
        out.append(f'    kind = "{c["kind"]}";')
        for key in ("crpix", "cd", "crval", "a", "b"):
            out.append(f"    {key} = {floats(c[key])};")
        for key in ("ap", "bp"):
            v = "None" if c[key] is None else f"Some {floats(c[key])}"
            out.append(f"    {key} = {v};")
        out.append(f"    pv = {floats(c['pv'])};")
        for key in ("pixels", "world", "sky", "sky_pixels"):
            out.append(f"    {key} = {floats(np.asarray(c[key]).ravel())};")
        out.append("  };")
    out.append("]")
    out.append("")

    data, circles, annuli, ellipses, polygons = pixel_apertures()
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
    out.append("(* (row, column, a, b, angle in degrees, photutils' exact sum). *)")
    out.append("let pixel_ellipses = [")
    for row, col, a, b, angle, total in ellipses:
        out.append(
            f"  ({ocaml_float(row)}, {ocaml_float(col)}, {ocaml_float(a)}, {ocaml_float(b)}, {ocaml_float(angle)}, {ocaml_float(total)});"
        )
    out.append("]")
    out.append("")
    out.append("(* Vertices as (row, column) pairs, and regions' exact sum. *)")
    out.append("let pixel_polygons = [")
    for vertices, total in polygons:
        out.append(f"  ({floats(np.array(vertices).ravel())}, {ocaml_float(total)});")
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
