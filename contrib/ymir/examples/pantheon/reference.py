# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.2.1", "scipy==1.15.1"]
# ///
"""The Pantheon+ fit's likelihood maximum, independently of raven.

    uv run contrib/ymir/examples/pantheon/reference.py DIR

DIR holds the Pantheon+ release's Pantheon+SH0ES.dat and
Pantheon+SH0ES_STAT+SYS.cov. The 1590 supernovae with z_HD > 0.01 are fitted
for (Omega_m, Omega_Lambda, offset) and for flat (Omega_m, offset), with no
radiation, distances by adaptive quadrature to 1e-12, chi^2 with the
STAT+SYS covariance, and scipy's least squares on the whitened residuals.
Prints each parameter's value and its standard deviation from (J^T J)^-1:
the numbers pantheon.ml holds as the likelihood's maximum.
"""

import sys

import numpy as np
from scipy import integrate, linalg, optimize

C_KM_S = 299792.458


def read(path):
    d = np.genfromtxt(f"{path}/Pantheon+SH0ES.dat", names=True, dtype=None, encoding=None)
    with open(f"{path}/Pantheon+SH0ES_STAT+SYS.cov") as f:
        n = int(f.readline())
        cov = np.loadtxt(f).reshape(n, n)
    keep = d["zHD"] > 0.01
    chol = np.linalg.cholesky(cov[np.ix_(keep, keep)])
    return d["zHD"][keep], d["zHEL"][keep], d["m_b_corr"][keep], chol


def transverse(om, ol, zs):
    """D_M / D_H at each of zs."""
    ok = 1 - om - ol
    inv_e = lambda x: 1 / np.sqrt(om * (1 + x) ** 3 + ok * (1 + x) ** 2 + ol)
    order = np.argsort(zs)
    edges = np.concatenate([[0.0], zs[order]])
    steps = [integrate.quad(inv_e, a, b, epsabs=0, epsrel=1e-12)[0] for a, b in zip(edges[:-1], edges[1:])]
    chi = np.empty_like(zs)
    chi[order] = np.cumsum(steps)
    if ok > 0:
        return np.sinh(np.sqrt(ok) * chi) / np.sqrt(ok)
    if ok < 0:
        return np.sin(np.sqrt(-ok) * chi) / np.sqrt(-ok)
    return chi


def main():
    z_hd, z_hel, m_b, chol = read(sys.argv[1])

    def residuals(om, ol, offset):
        mu = 5 * np.log10((1 + z_hel) * transverse(om, ol, z_hd) * C_KM_S / 70.0) + 25
        return linalg.solve_triangular(chol, m_b - mu - offset, lower=True)

    fits = [
        ("LambdaCDM", lambda p: residuals(*p), [0.3, 0.7, -19.3], ["Omega_m", "Omega_Lambda"]),
        ("flat LambdaCDM", lambda p: residuals(p[0], 1 - p[0], p[1]), [0.3, -19.3], ["Omega_m"]),
    ]
    for label, f, start, names in fits:
        r = optimize.least_squares(f, start, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        sigma = np.sqrt(np.diag(np.linalg.inv(r.jac.T @ r.jac)))
        print(f"{label}: chi2 {2 * r.cost:.4f}")
        for i, name in enumerate(names):
            print(f"  {name} {r.x[i]:.8f} +- {sigma[i]:.8f}")


if __name__ == "__main__":
    main()
