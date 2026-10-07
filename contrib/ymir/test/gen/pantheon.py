# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy"]
# ///
"""Generate the Pantheon+ fit references.

    uv run contrib/ymir/test/gen/pantheon.py [--check] [DIR]

DIR holds the Pantheon+ data release's distances and their STAT+SYS
covariance (Scolnic et al. 2022, Brout et al. 2022), ~/.cache/ymir by
default, downloaded there when absent and checked against their SHA-256.

It writes test/support/pantheon_reference.ml: the files' names, URLs and
BLAKE2b-256 digests, and for each of the four fits of test/pantheon its
maximum-likelihood parameters, their Fisher errors and the chi^2, found by
scipy's Levenberg-Marquardt with D_M integrated by 400-node Gauss-Legendre in
z, which is exact to rounding for a universe without radiation.

The fits:

- sn, sn_flat: the supernovae with z_HD > 0.01, m_b_corr against
  5 log10 ((1 + z_HEL) D_M(z_HD) / 10 pc) + offset at H0 = 67.66, with
  (Omega_m, Omega_Lambda, offset) or, flat, (Omega_m, offset);
- shoes, shoes_flat: those and the Cepheid calibrators, whose distance
  modulus is CEPH_DIST, against m_b_corr - M, with (Omega_m, Omega_Lambda,
  H0, M) or, flat, (Omega_m, H0, M).

Without --check the file is written; with it, nothing is written and the
run fails if the file would change.
"""

import argparse
import hashlib
import os
from pathlib import Path
import sys
import urllib.request

import numpy as np
from scipy import linalg, optimize

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / "support" / "pantheon_reference.ml"
DEFAULT = Path.home() / ".cache" / "ymir"

# The data release at its last commit, 2022-12-21.
RELEASE = (
    "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/"
    "c447f0fea703fcd0fff57de5000947b5ca81286b/Pantheon%2B_Data/"
    "4_DISTANCES_AND_COVAR/"
)
DISTANCES = "Pantheon+SH0ES.dat"
COVARIANCE = "Pantheon+SH0ES_STAT+SYS.cov"
SHA256 = {
    DISTANCES: "1cb0fc379ef066afdc2ffd1857681cc478024570d8a3eba284fb645775198cf8",
    COVARIANCE: "abf806d966485e64afdb359c87bffc0ecc00d05eff0a31ced66f247385df0fdc",
}

C_KM_S = 299792.458
H0 = 67.66  # Planck 2018's, which the offset absorbs
NODES, WEIGHTS = np.polynomial.legendre.leggauss(400)


def url(name):
    return RELEASE + name.replace("+", "%2B")


def fetch(directory, name):
    path = Path(directory) / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(".part")
        print(f"downloading {url(name)} to {path}", file=sys.stderr)
        urllib.request.urlretrieve(url(name), part)
        os.replace(part, path)
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256[name]:
        sys.exit(f"{path}: SHA-256 {digest}, expected {SHA256[name]}")
    return path, hashlib.blake2b(data, digest_size=32).hexdigest()


def transverse(omega_m, omega_l, z):
    """D_M in units of c/H0."""
    omega_k = 1.0 - omega_m - omega_l
    t = (NODES[:, None] + 1.0) / 2.0 * z[None, :]
    e = np.sqrt(omega_m * (1 + t) ** 3 + omega_k * (1 + t) ** 2 + omega_l)
    chi = (WEIGHTS[:, None] / e).sum(0) * z / 2.0
    if omega_k == 0.0:
        return chi
    s = np.sqrt(abs(omega_k))
    return np.sinh(s * chi) / s if omega_k > 0 else np.sin(s * chi) / s


def modulus(omega_m, omega_l, h0, z_hd, z_hel):
    d_l = (1 + z_hel) * transverse(omega_m, omega_l, z_hd) * C_KM_S / h0
    return 5.0 * np.log10(d_l) + 25.0


def fit(chol, residual, start):
    def whitened(p):
        return linalg.solve_triangular(chol, residual(p), lower=True)

    r = optimize.least_squares(
        whitened, start, method="lm", jac="3-point", xtol=1e-15, ftol=1e-15, gtol=1e-15
    )
    j = r.jac
    sigma = np.sqrt(np.diag(np.linalg.inv(j.T @ j)))
    return r.x, sigma, float(whitened(r.x) @ whitened(r.x))


def fits(directory):
    dat, dat_blake = fetch(directory, DISTANCES)
    cov, cov_blake = fetch(directory, COVARIANCE)
    rows = np.genfromtxt(dat, names=True, dtype=None, encoding=None)
    lines = cov.read_text().split()
    n = int(lines[0])
    c = np.array(lines[1:], dtype=float).reshape(n, n)
    z_hd, z_hel, m_b = rows["zHD"], rows["zHEL"], rows["m_b_corr"]
    calibrator = rows["IS_CALIBRATOR"] == 1

    def sample(mask):
        return np.linalg.cholesky(c[np.ix_(mask, mask)]), mask

    sn_chol, sn = sample(z_hd > 0.01)
    shoes_chol, shoes = sample((z_hd > 0.01) | calibrator)

    def sn_residual(omega_m, omega_l, offset):
        return m_b[sn] - modulus(omega_m, omega_l, H0, z_hd[sn], z_hel[sn]) - offset

    def shoes_residual(omega_m, omega_l, h0, m):
        mu = modulus(omega_m, omega_l, h0, z_hd[shoes], z_hel[shoes])
        mu = np.where(calibrator[shoes], rows["CEPH_DIST"][shoes], mu)
        return m_b[shoes] - m - mu

    results = {
        "sn": fit(sn_chol, lambda p: sn_residual(*p), [0.3, 0.7, -19.4]),
        "sn_flat": fit(sn_chol, lambda p: sn_residual(p[0], 1 - p[0], p[1]), [0.3, -19.4]),
        "shoes": fit(shoes_chol, lambda p: shoes_residual(*p), [0.3, 0.7, 73.0, -19.25]),
        "shoes_flat": fit(
            shoes_chol, lambda p: shoes_residual(p[0], 1 - p[0], p[1], p[2]), [0.3, 73.0, -19.25]
        ),
    }
    return (dat_blake, cov_blake, int(sn.sum()), int(shoes.sum())), results


def ocaml_float(x):
    x = float(x)
    return x.hex().replace("0x1.0000000000000p", "0x1p")


def ocaml_floats(xs):
    return "[| " + "; ".join(ocaml_float(x) for x in xs) + " |]"


def render(files, results):
    dat_blake, cov_blake, n_sn, n_shoes = files
    out = [
        "(*---------------------------------------------------------------------------",
        "  Copyright (c) 2026 The Raven authors. All rights reserved.",
        "  SPDX-License-Identifier: ISC",
        "  ---------------------------------------------------------------------------*)",
        "",
        "(* Written by test/gen/pantheon.py; do not edit. *)",
        "",
        '[@@@ocamlformat "disable"]',
        "",
        "(* scipy's maximum-likelihood parameters, their Fisher errors and chi^2. *)",
        "type fit = { theta : float array; sigma : float array; chi2 : float }",
        "",
        f'let distances = "{DISTANCES}"',
        f'let distances_url = "{url(DISTANCES)}"',
        f'let distances_blake2b = "{dat_blake}"',
        f'let covariance = "{COVARIANCE}"',
        f'let covariance_url = "{url(COVARIANCE)}"',
        f'let covariance_blake2b = "{cov_blake}"',
        f"let n_sn = {n_sn}",
        f"let n_shoes = {n_shoes}",
    ]
    for name, (theta, sigma, chi2) in results.items():
        out += [
            "",
            f"let {name} =",
            f"  {{ theta = {ocaml_floats(theta)};",
            f"    sigma = {ocaml_floats(sigma)};",
            f"    chi2 = {ocaml_float(chi2)} }}",
        ]
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("dir", nargs="?", default=DEFAULT)
    args = parser.parse_args()
    files, results = fits(args.dir)
    text = render(files, results)
    for name, (theta, sigma, chi2) in results.items():
        print(name, theta, sigma, chi2, file=sys.stderr)
    if args.check:
        if not REFERENCE.exists() or REFERENCE.read_text() != text:
            sys.exit(f"{REFERENCE} would change; run without --check")
        return
    REFERENCE.write_text(text)


if __name__ == "__main__":
    main()
