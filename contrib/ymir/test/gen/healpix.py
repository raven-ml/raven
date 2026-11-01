# /// script
# requires-python = ">=3.11"
# dependencies = ["healpy==1.18.1", "numpy==2.5.4", "scipy==1.16.3"]
# ///
"""Generate the HEALPix test references from healpy.

    uv run contrib/ymir/test/gen/healpix.py [--check]

healpy 1.18.1 wraps the HEALPix C++ library, the reference implementation
of Gorski et al. (2005). It writes test/support/healpix_reference.ml:

    - directions with the pixels that may hold them, at nside from 1 to
      2^29: drawn uniformly, at the poles, on the equator, on |z| = 2/3, near
      the poles, on pixel edges and corners, and unnormalised. A direction
      within 2e-14 radians of an edge lists every pixel healpy gives the
      direction moved by 2e-14 radians in eight ways, healpy's own pixel
      first; another lists healpy's pixel alone;
    - pixel centres (pix2vec), corners (boundaries, step 1) and neighbours
      (get_all_neighbours), each with the pixel's RING and NESTED indices:
      every pixel at small nside, and drawn ones, the first and last of each
      region and the corners of each base pixel at large nside;
    - coefficients a_lm drawn in [-1, 1) (real at m = 0), healpy's map of
      them (alm2map) and healpy's analysis of that map (map2alm, unweighted)
      with 0 and 3 iterations; and a map drawn in [-1, 1) with its analysis;
    - rho, the largest error of map2alm(alm2map(a)) relative to a, without
      iterations, over every a at lmax, for lmax = nside and 2 nside at
      nside 4 to 64. Errors are measured in the map's L2 norm on the sphere,
      sum_l (|a_l0|^2 + 2 sum_(m>0) |a_lm|^2). The error is a - A S a = Q a
      with Q = I - A S self-adjoint in that norm, so rho is the largest
      |eigenvalue| of Q (ARPACK, through scipy's eigsh), and after k
      iterations, Q^(k+1), the largest error is rho^(k+1). Each is written to
      4 digits.

Without --check the file is written; with it, nothing is written and the
run fails if the file would change.
"""

import argparse
import os
import sys
from pathlib import Path

# One thread, so that each run sums in one order.
os.environ["OMP_NUM_THREADS"] = "1"

import healpy as hp
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / "support" / "healpix_reference.ml"

HEADER = """(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)
"""

NSIDES = [1, 2, 4, 8, 16, 1024, 2**20, 2**28, 2**29]
SMALL = [1, 2, 4]
LARGE = [1024, 2**29]
NUDGE = 2e-14


def unit(v):
    v = v / np.max(np.abs(v))
    return v / np.linalg.norm(v)


def nudged(v):
    """v moved by NUDGE radians in eight directions of its tangent plane."""
    v = unit(v)
    a = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = unit(np.cross(v, a))
    e2 = np.cross(v, e1)
    out = []
    for k in range(8):
        t = 2 * np.pi * k / 8
        out.append(unit(v + NUDGE * (np.cos(t) * e1 + np.sin(t) * e2)))
    return out


def candidates(nside, v, nest):
    """healpy's pixel of v's direction, then those of the nudged ones."""
    u = unit(v)
    own = int(hp.vec2pix(nside, u[0], u[1], u[2], nest=nest))
    others = {int(hp.vec2pix(nside, *u, nest=nest)) for u in nudged(v)}
    return [own] + sorted(others - {own})


def directions(nside, rng):
    out = [rng.normal(size=3) for _ in range(24)]
    out += [np.array(v, dtype=float) for v in
            [(0, 0, 1), (0, 0, -1), (1, 0, 0), (0, 1, 0), (-1, 0, 0),
             (0, -1, 0), (1, 1, 0), (1, -1, 0), (1e-9, 2e-9, 1),
             (-3e-15, 1e-15, -1), (1e-300, 0, 1)]]
    for _ in range(3):
        a = rng.uniform(0, 2 * np.pi)
        out.append(np.array([np.cos(a), np.sin(a), 0.0]))
    for z in (2 / 3, -2 / 3):
        for _ in range(2):
            a = rng.uniform(0, 2 * np.pi)
            r = np.sqrt((1 - z) * (1 + z))
            out.append(np.array([r * np.cos(a), r * np.sin(a), z]))
    npix = hp.nside2npix(nside)
    for p in rng.integers(0, npix, 4):
        b = hp.boundaries(nside, int(p), step=2)
        out += [b[:, k].copy() for k in range(b.shape[1])]
    out += [3.7 * rng.normal(size=3), 1e-200 * rng.normal(size=3),
            1e200 * rng.normal(size=3)]
    return out


def regions(nside):
    npix = hp.nside2npix(nside)
    ncap = 2 * nside * (nside - 1)
    return [0, 1, 2, 3, ncap - 1, ncap, ncap + 1, npix // 2, npix - ncap - 1,
            npix - ncap, npix - 4, npix - 1]


def spread(v):
    return sum(((v >> i) & 1) << (2 * i) for i in range(30))


def face_corners(nside):
    s = spread(nside - 1)
    return [f * nside * nside + o for f in range(12)
            for o in (0, s, 2 * s, 3 * s)]


def pixel_sets(nside, rng):
    """(RING pixel, NESTED pixel) pairs to tabulate at nside."""
    npix = hp.nside2npix(nside)
    if nside in SMALL:
        ring = list(range(npix))
    else:
        nested = [p for p in face_corners(nside) if p < npix]
        ring = regions(nside) + [int(p) for p in rng.integers(0, npix, 12)]
        ring += [int(hp.nest2ring(nside, p)) for p in nested]
    ring = [p for p in dict.fromkeys(ring) if 0 <= p < npix]
    return [(p, int(hp.ring2nest(nside, p))) for p in ring]


# Rendering

def ocaml_float(x):
    text = repr(float(x))
    if text == "inf" or text == "nan":
        sys.exit("non-finite reference")
    return text if any(c in text for c in ".en") else text + "."


def ints(xs):
    return "[" + "; ".join(f"{int(x)}L" for x in xs) + "]"


def floats(xs):
    return "[|" + "; ".join(ocaml_float(x) for x in xs) + "|]"


def vec(v):
    return "(" + ", ".join(ocaml_float(x) for x in v) + ")"


def dense(alm, lmax):
    """healpy's packed m-major coefficients as [l][m] rows, re and im."""
    re = np.zeros((lmax + 1, lmax + 1))
    im = np.zeros((lmax + 1, lmax + 1))
    for m in range(lmax + 1):
        for l in range(m, lmax + 1):
            a = alm[hp.Alm.getidx(lmax, l, m)]
            re[l, m], im[l, m] = a.real, a.imag
    return re.ravel(), im.ravel()


def drawn_alm(lmax, rng):
    n = hp.Alm.getsize(lmax)
    a = rng.uniform(-1, 1, n) + 1j * rng.uniform(-1, 1, n)
    _, m = hp.Alm.getlm(lmax)
    a[m == 0] = a[m == 0].real
    return a


def quadrature_error(nside, lmax):
    """rho: the largest |eigenvalue| of I - map2alm o alm2map at lmax, in
    the map's L2 norm, which weighs a_lm at m > 0 twice."""
    _, m = hp.Alm.getlm(lmax)
    pos = m > 0
    w = np.where(pos, np.sqrt(2.0), 1.0)
    n0 = len(m)
    size = n0 + int(pos.sum())

    def alm_of(x):
        a = x[:n0].astype(complex)
        a[pos] += 1j * x[n0:]
        return a / w

    def x_of(a):
        a = a * w
        return np.concatenate([a.real, a.imag[pos]])

    def q(x):
        a = alm_of(x)
        b = hp.map2alm(hp.alm2map(a, nside, lmax=lmax), lmax=lmax, iter=0)
        return x - x_of(b)

    op = LinearOperator((size, size), matvec=q, dtype=float)
    vals = eigsh(op, k=1, which="LM", v0=np.ones(size), tol=1e-10,
                 return_eigenvectors=False)
    return float(abs(vals[0]))


def render():
    rng = np.random.default_rng(2005)
    out = [HEADER, """
(* Written by test/gen/healpix.py from healpy 1.18.1; do not edit. *)

[@@@ocamlformat "disable"]

(* A vector and the RING and NESTED pixels that may hold its direction at
   nside: healpy's first, then those within 2e-14 radians. *)
let pixels = [
"""]
    for nside in NSIDES:
        for v in directions(nside, rng):
            ring = candidates(nside, v, False)
            nest = candidates(nside, v, True)
            out.append(f"  ({nside}, {vec(v)}, {ints(ring)}, {ints(nest)});\n")
    out.append("""]

(* A pixel's RING and NESTED indices at nside, its centre (pix2vec), its
   corners north, west, south and east (boundaries), and its neighbours in
   each scheme (get_all_neighbours). *)
let cells = [
""")
    for nside in SMALL + LARGE:
        for ring, nest in pixel_sets(nside, rng):
            c = hp.pix2vec(nside, ring)
            b = hp.boundaries(nside, ring, step=1)
            corners = "[|" + "; ".join(vec(b[:, k]) for k in range(4)) + "|]"
            nr = hp.get_all_neighbours(nside, ring)
            nn = hp.get_all_neighbours(nside, nest, nest=True)
            out.append(f"  ({nside}, {ring}L, {nest}L, {vec(c)},\n"
                       f"   {corners},\n   {ints(nr)}, {ints(nn)});\n")
    out.append("""]

(* At nside and lmax: coefficients [l][m] flattened, real then imaginary
   parts; healpy's RING map of them; and healpy's coefficients of that map
   with 0 and with 3 iterations. *)
let harmonics = [
""")
    for nside, lmax in [(1, 2), (2, 4), (4, 8), (8, 16), (4, 11)]:
        a = drawn_alm(lmax, rng)
        f = hp.alm2map(a, nside, lmax=lmax)
        rows = []
        for alm in [a, hp.map2alm(f, lmax=lmax, iter=0),
                    hp.map2alm(f, lmax=lmax, iter=3)]:
            re, im = dense(alm, lmax)
            rows.append(f"({floats(re)},\n    {floats(im)})")
        out.append(f"  ({nside}, {lmax},\n   {rows[0]},\n   {floats(f)},\n"
                   f"   {rows[1]},\n   {rows[2]});\n")
    out.append("""]

(* A map drawn in [-1, 1) at nside, in RING order, and healpy's coefficients
   of it at lmax with 0 and with 3 iterations. *)
let analyses = [
""")
    for nside, lmax in [(2, 7), (4, 12)]:
        f = rng.uniform(-1, 1, hp.nside2npix(nside))
        rows = []
        for it in (0, 3):
            re, im = dense(hp.map2alm(f, lmax=lmax, iter=it), lmax)
            rows.append(f"({floats(re)},\n    {floats(im)})")
        out.append(f"  ({nside}, {lmax}, {floats(f)},\n"
                   f"   {rows[0]},\n   {rows[1]});\n")
    out.append("""]

(* At nside and lmax, rho: the largest error of an analysis without
   iterations of a map band-limited at lmax, relative to its coefficients,
   in the map's L2 norm, to 4 digits. After k iterations it is
   rho^(k + 1). *)
let quadrature = [
""")
    for nside in (4, 8, 16, 32, 64):
        for lmax in (nside, 2 * nside):
            rho = float(f"{quadrature_error(nside, lmax):.4g}")
            out.append(f"  ({nside}, {lmax}, {ocaml_float(rho)});\n")
    out.append("]\n")
    return "".join(out)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if hp.__version__ != "1.18.1":
        sys.exit(f"healpy {hp.__version__}, not the pinned 1.18.1")
    text = render()
    if args.check:
        if not REFERENCE.exists() or REFERENCE.read_text() != text:
            sys.exit(f"{REFERENCE} would change; run without --check")
        return
    REFERENCE.write_text(text)


if __name__ == "__main__":
    main()
