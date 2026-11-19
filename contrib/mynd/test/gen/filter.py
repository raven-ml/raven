# Goldens for Mynd's filters, in float64 but for OpenCV's 8-bit median.
# Images are [h, w, c]; a window's tap k / 2 is the pixel itself, which is
# scipy.ndimage's origin 0 and OpenCV's anchor (-1, -1) for every extent.
# Edge rules: Zero is scipy's constant 0 and OpenCV's BORDER_CONSTANT,
# Replicate nearest and BORDER_REPLICATE, Symmetric_whole mirror and
# BORDER_REFLECT_101, Symmetric_half reflect and BORDER_REFLECT, Periodic
# wrap and BORDER_WRAP.
# - correlate: scipy.ndimage.correlate and cv2.filter2D.
# - gaussian_blur: scipy.ndimage.gaussian_filter with its radius, and
#   cv2.GaussianBlur of extent 2 radius + 1, both the sampled Gaussian
#   divided by its sum.
# - median_blur: scipy.ndimage.rank_filter at rank (n - 1) / 2, the lower
#   median (median_filter takes rank n / 2, the upper one for even n), and
#   cv2.medianBlur on uint8 under BORDER_REPLICATE.
# Run from this directory: uv run --locked filter.py ../golden/filter.npz
import sys

import cv2
import numpy as np
import scipy
from scipy import ndimage

rng = np.random.default_rng(13)
out = {}

EDGES = {
    "zero": ("constant", cv2.BORDER_CONSTANT),
    "replicate": ("nearest", cv2.BORDER_REPLICATE),
    "symmetric_whole": ("mirror", cv2.BORDER_REFLECT_101),
    "symmetric_half": ("reflect", cv2.BORDER_REFLECT),
    "periodic": ("wrap", cv2.BORDER_WRAP),
}

x = rng.standard_normal((7, 9, 2))
out["x"] = x
weights = {
    "3x3": rng.standard_normal((3, 3)),
    "4x4": rng.standard_normal((4, 4)),
    "2x5": rng.standard_normal((2, 5)),
    "1x6": rng.standard_normal((1, 6)),
    "11x13": rng.standard_normal((11, 13)),  # larger than the image
}
for name, k in weights.items():
    out[f"k.{name}"] = k
    for edge, (mode, _) in EDGES.items():
        y = ndimage.correlate(x, k[:, :, None], mode=mode, cval=0.0)
        out[f"correlate.scipy.{name}.{edge}"] = y
# OpenCV's filter2D at the small extents, where it sums directly; it does
# not take BORDER_WRAP.
for name in ("3x3", "4x4", "2x5"):
    for edge, (_, border) in EDGES.items():
        if edge == "periodic":
            continue
        y = cv2.filter2D(x, -1, weights[name], anchor=(-1, -1), borderType=border)
        out[f"correlate.opencv.{name}.{edge}"] = y

for sigma, radius in ((0.5, 1), (1.3, 4), (2.0, 6), (0.8, 7)):
    tag = f"{sigma}.{radius}"
    for edge, (mode, border) in EDGES.items():
        y = ndimage.gaussian_filter(x, sigma, mode=mode, cval=0.0, radius=radius, axes=(0, 1))
        out[f"gaussian.scipy.{tag}.{edge}"] = y
        if edge == "periodic":
            continue
        k = 2 * radius + 1
        y = cv2.GaussianBlur(x, (k, k), sigma, sigmaY=sigma, borderType=border)
        out[f"gaussian.opencv.{tag}.{edge}"] = y

# Values with ties, ±0 and infinities; scipy's rank filter orders by value
# alone, so no NaN.
m = rng.integers(-3, 4, (8, 10, 2)) / 2.0
m[0, 0, 0] = np.inf
m[3, 4, 1] = -np.inf
m[2, 2, 0] = -0.0
out["m"] = m
for kh, kw in ((3, 3), (2, 2), (2, 3), (4, 4), (5, 1)):
    n = kh * kw
    for edge, (mode, _) in EDGES.items():
        if edge == "zero":
            y = ndimage.rank_filter(m, (n - 1) // 2, size=(kh, kw, 1), mode=mode, cval=0.0)
        else:
            y = ndimage.rank_filter(m, (n - 1) // 2, size=(kh, kw, 1), mode=mode)
        out[f"median.scipy.{kh}x{kw}.{edge}"] = y
u8 = rng.integers(0, 256, (9, 11, 3), dtype=np.uint8)
out["u8"] = u8
for k in (3, 5):
    out[f"median.opencv.{k}"] = cv2.medianBlur(u8, k)

np.savez(sys.argv[1], **out)
print(scipy.__version__, cv2.__version__, len(out), "arrays")
