# golden/bessel.npz for test_designs.ml: the poles of the Bessel prototype of
# each order, -3 dB at 1 rad/s, at float64 from a 3n + 60 digit reference: the
# zeros of y_n polished by mpmath Newton on y_n's exact coefficients from
# SciPy's approximations, poles 1/z, then w3 solving |H(i w)|^2 = 1/2 at that
# precision. Also SciPy's own norm='mag' sections at orders it can design.
# Run from this directory: uv run --locked bessel.py ../golden/bessel.npz
import sys, numpy as np, mpmath as mp
from scipy.signal import bessel
from scipy.signal._filter_design import _campos_zeros

orders = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 25, 50, 100, 200]

def zeros_float(n):
    if n == 1: return np.array([-1 + 0j])
    if n == 2: return np.array([(-3 + 1j * 3**.5) / 6, (-3 - 1j * 3**.5) / 6])
    z = _campos_zeros(n).astype(complex)
    for _ in range(60):
        d = z[:, None] - z[None, :]; np.fill_diagonal(d, np.inf); inv = 1 / d
        F = inv.sum(1) + (z + 1) / z**2
        J = inv**2; np.fill_diagonal(J, -(inv**2).sum(1) - 1 / z**2 - 2 / z**3)
        s = np.linalg.solve(J, F); z = z - s
        if np.max(np.abs(s)) < 1e-15 * np.max(np.abs(z)): break
    return z

out = {}
for n in orders:
    mp.mp.dps = 3 * n + 60
    a = [mp.factorial(n + k) / (mp.mpf(2)**k * mp.factorial(k) * mp.factorial(n - k)) for k in range(n + 1)]
    y = lambda x: mp.polyval(a[::-1], x)
    dy = lambda x: mp.polyval([a[k] * k for k in range(n, 0, -1)], x)
    zs = []
    for z0 in zeros_float(n):
        z = mp.mpc(z0.real, z0.imag)
        for _ in range(200):
            s = y(z) / dy(z); z -= s
            if abs(s) < mp.mpf(10) ** (-mp.mp.dps + 10) * abs(z): break
        zs.append(z)
    assert abs(sum(zs) + 1) < mp.mpf(10) ** -40
    assert abs(sum(1 / z for z in zs) + mp.mpf(n * (n + 1)) / 2) < mp.mpf(10) ** -30
    for i in range(n):
        for j in range(i):
            assert abs(zs[i] - zs[j]) > mp.mpf(10) ** -10
    p = [1 / z for z in zs]
    f = lambda w: sum(mp.log(abs(pk)) - mp.log(abs(1j * w - pk)) for pk in p) + mp.log(2) / 2
    w3 = mp.findroot(f, mp.sqrt((2 * n - 1) * mp.log(2)))
    pn = np.array([complex(pk / w3) for pk in p])
    out[f"{n}.poles"] = pn
    out[f"{n}.w3"] = np.array([float(w3)])
    print(n, float(w3), flush=True)
for n in (1, 2, 3, 4, 5, 8, 10, 16, 25):
    out[f"{n}.sos"] = bessel(n, 0.1, fs=1.0, norm='mag', output='sos')
np.savez(sys.argv[1], **out)
