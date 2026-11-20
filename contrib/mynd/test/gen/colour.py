# Goldens for Mynd's colour functions, in float64.
# - HSV: scikit-image's rgb2hsv and hsv2rgb and torchvision's _rgb_to_hsv
#   and _hsv_to_rgb, which compute Smith's hexcone with hue in turns, as
#   Mynd does, on colours in [0, 1]^3.
# - Lab: exact values from CSS Color 4's rational matrices and CIE's exact
#   epsilon and kappa, rounded once to float64. scikit-image and kornia use
#   six-digit matrices and the rounded 0.008856 and 7.787, so they are not
#   goldens for it.
# Run from this directory: uv run --locked colour.py ../golden/colour.npz
import sys
from decimal import Decimal, getcontext
from fractions import Fraction as F

import numpy as np
import skimage
import torch
import torchvision
from skimage.color import hsv2rgb, rgb2hsv
from torchvision.transforms.v2.functional import _color

getcontext().prec = 60
rng = np.random.default_rng(11)
out = {}


def colours(n):
    """Colours in [0, 1]^3: free, two tied for the greatest, two tied for
    the least, greys and black, primaries and secondaries."""
    x = rng.random((n, 3))
    q = n // 6
    for k in range(q, 2 * q):
        hi, lo = sorted(rng.random(2), reverse=True)
        x[k] = rng.permutation([hi, hi, lo])
    for k in range(2 * q, 3 * q):
        hi, lo = sorted(rng.random(2), reverse=True)
        x[k] = rng.permutation([hi, lo, lo])
    for k in range(3 * q, 3 * q + 8):
        x[k] = rng.random()
    x[3 * q + 8] = 0.0
    corners = [[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1]]
    x[3 * q + 9 : 3 * q + 15] = corners
    return x


x = colours(240)
out["hsv.x"] = x
out["hsv.skimage"] = rgb2hsv(x)
xt = torch.tensor(x.T.copy()).reshape(3, -1, 1)
out["hsv.torchvision"] = _color._rgb_to_hsv(xt).reshape(3, -1).T.numpy()

# HSV forms: free, hues on the sextants' edges, and saturation 0 and 1.
y = rng.random((240, 3))
y[:60, 0] = rng.integers(0, 6, 60) / 6
y[60:80, 1] = 0.0
y[80:100, 1] = 1.0
out["rgb.y"] = y
out["rgb.skimage"] = hsv2rgb(y)
yt = torch.tensor(y.T.copy()).reshape(3, -1, 1)
out["rgb.torchvision"] = _color._hsv_to_rgb(yt).reshape(3, -1).T.numpy()

# Lab, exactly.
M = [
    [F(506752, 1228815), F(87881, 245763), F(12673, 70218)],
    [F(87098, 409605), F(175762, 245763), F(12673, 175545)],
    [F(7918, 409605), F(87881, 737289), F(1001167, 1053270)],
]
M_INV = [
    [F(12831, 3959), F(-329, 214), F(-1974, 3959)],
    [F(-851781, 878810), F(1648619, 878810), F(36519, 878810)],
    [F(705, 12673), F(-2585, 12673), F(705, 667)],
]
WHITE = [sum(row) for row in M]
EPS, KAPPA = F(216, 24389), F(24389, 27)


def cbrt(t):
    d = Decimal(t.numerator) / Decimal(t.denominator)
    return d ** (Decimal(1) / Decimal(3))


def dec(t):
    return Decimal(t.numerator) / Decimal(t.denominator)


def to_lab(c):
    c = [F(float(v)) for v in c]
    w = [sum(m * v for m, v in zip(row, c)) / white for row, white in zip(M, WHITE)]
    f = [cbrt(t) if t > EPS else dec((KAPPA * t + 16) / 116) for t in w]
    return [float(116 * f[1] - 16), float(500 * (f[0] - f[1])), float(200 * (f[1] - f[2]))]


def to_linear(lab):
    l, a, b = (F(float(v)) for v in lab)
    fy = (l + 16) / 116
    fs = [fy + a / 500, fy, fy - b / 200]
    w = [u**3 if u > F(6, 29) else (116 * u - 16) / KAPPA for u in fs]
    xyz = [t * white for t, white in zip(w, WHITE)]
    return [float(sum(m * v for m, v in zip(row, xyz))) for row in M_INV]


lx = colours(120)
lx[100:] = rng.uniform(-0.25, 1.25, (20, 3))
lx[90:100] = rng.uniform(0, 0.02, (10, 3))  # near black, where Y < epsilon
out["lab.x"] = lx
out["lab.y"] = np.array([to_lab(c) for c in lx])
ly = np.column_stack(
    [rng.uniform(0, 100, 120), rng.uniform(-100, 100, 120), rng.uniform(-100, 100, 120)]
)
ly[:20, 0] = rng.uniform(0, 8, 20)  # the linear piece of L
out["linear.y"] = ly
out["linear.x"] = np.array([to_linear(c) for c in ly])

np.savez(sys.argv[1], **out)
print(skimage.__version__, torchvision.__version__, len(out), "arrays")
