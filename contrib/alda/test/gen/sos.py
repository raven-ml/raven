# Goldens for Alda.sos: single sections against an exact-enough truth with the
# rounding bound's weights, and cascades against scipy.signal.sosfilt.
# Run from this directory: uv run --locked sos.py ../golden/sos.npz
#
# Keys, single section case <i> (float32, and float64 for even <i>):
#   sec.<i>.row     float64 [6]   raw row b0 b1 b2 a0 a1 a2 (a0 != 1)
#   sec.<i>.<dt>.x  <dt> [n]      the input at the dtype
#   sec.<i>.<dt>.y  float64 [n]   DF-I in long double on the row as the
#                                 evaluation uses it at <dt> (divided by a0
#                                 in <dt>, rounded) and on x
#   sec.<i>.<dt>.w  float64 [n]   (|h| * m')_n, h the impulse response of that
#                                 row, m'_n = sum_i |b_i x_{n-i}|
#                                 + max(|a1|, |p| + |q|) |y_{n-1}| + |a2| |y_{n-2}|
# Cascade case <j>: cas.<j>.sos float64 [S; 6] (scipy design), cas.<j>.x
# float64 [n], cas.<j>.y float64 [n] = sosfilt of the rows divided by a0.
import math
import sys

import numpy as np
import scipy
from scipy import signal

out = sys.argv[1]
rng = np.random.default_rng(20261119)
R = 48000.0
LD = np.longdouble
assert np.finfo(LD).nmant >= 63, "long double must be x87 extended"


def cookbook(kind, f, q, g=0.0):
    A = 10 ** (g / 40)
    w = 2 * math.pi * f / R
    al = math.sin(w) / (2 * q)
    c = math.cos(w)
    if kind == "peak":
        return [1 + al * A, -2 * c, 1 - al * A, 1 + al / A, -2 * c, 1 - al / A]
    if kind == "notch":
        return [1, -2 * c, 1, 1 + al, -2 * c, 1 - al]
    if kind == "low":
        return [(1 - c) / 2, 1 - c, (1 - c) / 2, 1 + al, -2 * c, 1 - al]
    raise ValueError(kind)


def normalised(row, dt):
    r = np.asarray(row, np.float64).astype(dt)
    return [dt(v / r[3]) for v in r]


def df1(row, x):
    b0, b1, b2, _, a1, a2 = [LD(v) for v in row]
    x1 = x2 = y1 = y2 = LD(0)
    y = np.empty(len(x), LD)
    for n, xn in enumerate(np.asarray(x, LD)):
        yn = b0 * xn + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2, x1, y2, y1 = x1, xn, y1, yn
        y[n] = yn
    return y


def weights(row, x, y):
    b0, b1, b2, _, a1, a2 = [float(v) for v in row]
    d = complex(a1 * a1 - 4 * a2)
    p = (-a1 - (1 if a1 >= 0 else -1) * d ** 0.5) / 2
    q = 0 if p == 0 else a2 / p
    c1 = max(abs(a1), abs(p) + abs(q))
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    xm1 = np.concatenate([[0.0], x[:-1]])
    xm2 = np.concatenate([[0.0, 0.0], x[:-2]])
    ym1 = np.concatenate([[0.0], y[:-1]])
    ym2 = np.concatenate([[0.0, 0.0], y[:-2]])
    m = abs(b0 * x) + abs(b1 * xm1) + abs(b2 * xm2) + c1 * abs(ym1) + abs(a2) * abs(ym2)
    imp = np.zeros(len(x))
    imp[0] = 1
    h = np.abs(signal.lfilter([1.0], [1.0, a1, a2], imp))
    return signal.fftconvolve(h, m)[: len(x)]


n = 4096 + 600
t = np.arange(n)
sections = [
    cookbook("peak", 20, 10, 12),
    cookbook("low", 1000, 0.5),
    cookbook("low", 100, 0.7071),
    cookbook("peak", 8000, 2, 6),
    cookbook("notch", 60, 30),
    cookbook("peak", 12000, 1000, 12),
    # real poles near +-(1 - 1e-3), a1 = 0 up to rounding
    [1.0, 0.0, -1.0, 2.0, 0.0, -2 * (1 - 1e-3) ** 2],
    # an exact double pole at 15/16
    [0.25, 0.5, 0.25, 1.0, -1.875, 0.87890625],
]
inputs = [
    lambda: rng.standard_normal(n),
    lambda: np.ones(n),
    lambda: np.sin(2 * math.pi * t / 50),
    lambda: np.sin(2 * math.pi * (1 + (R / 2 - 1) * t / (2 * n)) * t / R),
]
arrays = {}
for i, row in enumerate(sections):
    x = inputs[i % len(inputs)]()
    arrays[f"sec.{i}.row"] = np.asarray(row, np.float64)
    for name, dt in (("f32", np.float32), ("f64", np.float64))[: 1 if i % 2 else 2]:
        xd = x.astype(dt)
        rn = normalised(row, dt)
        y = df1(rn, xd)
        arrays[f"sec.{i}.{name}.x"] = xd
        arrays[f"sec.{i}.{name}.y"] = y.astype(np.float64)
        arrays[f"sec.{i}.{name}.w"] = weights(rn, xd, y)

cascades = [
    signal.butter(8, 1000, fs=R, output="sos"),
    signal.ellip(6, 0.5, 60, [300, 3000], btype="bandpass", fs=R, output="sos"),
    signal.cheby1(5, 1, 5000, btype="highpass", fs=R, output="sos"),
    np.array([cookbook("peak", f, q, g) for f, q, g in
              [(100, 1, 6), (1000, 2, -4), (6000, 0.7, 3)]]),
]
m = 3000
for j, sos in enumerate(cascades):
    x = rng.standard_normal(m)
    arrays[f"cas.{j}.sos"] = np.asarray(sos, np.float64)
    arrays[f"cas.{j}.x"] = x
    sos = np.asarray(sos, np.float64)
    arrays[f"cas.{j}.y"] = signal.sosfilt(sos / sos[:, 3:4], x)
np.savez(out, **arrays)
print("scipy", scipy.__version__, len(arrays), "arrays")
