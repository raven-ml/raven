# Goldens for Mynd's gradients and their maxima, in float64. Images are
# [h, w, c]; x is along columns (axis 1), y along rows (axis 0).
# - gradient: scipy.ndimage.sobel and cv2.Sobel at aperture 3 are 8 times
#   the `Sobel gradient, cv2.Scharr 32 times the `Scharr one; numpy's
#   gradient, one-sided at the border, is the `Central gradient under
#   Antisymmetric_whole. Edge rules as in filter.py: Zero is scipy's
#   constant 0 and BORDER_CONSTANT, Replicate nearest and
#   BORDER_REPLICATE, Symmetric_whole mirror and BORDER_REFLECT_101,
#   Symmetric_half reflect and BORDER_REFLECT, Periodic wrap; OpenCV's
#   derivatives take no BORDER_WRAP.
# - gradient_maxima: scikit-image's bilinear non-maximum suppression, the
#   step of its canny, over smooth random gradients. Its mask leaves out
#   the border ring, whose neighbours it would read outside the image.
# - Canny's edges: scikit-image's canny at its defaults (thresholds 0.1
#   and 0.2 of raw Sobel strengths, a blur under constant 0 divided by the
#   blur of ones, Sobel under reflect, the border ring dropped) on a disc,
#   a turned square and a ramp under noise.
# Run from this directory: uv run --locked edges.py ../golden/edges.npz
import sys

import cv2
import numpy as np
import scipy
import skimage
from skimage.feature import canny
from scipy import ndimage
from skimage.feature._canny_cy import _nonmaximum_suppression_bilinear

rng = np.random.default_rng(29)
out = {}

EDGES = {
    "zero": ("constant", cv2.BORDER_CONSTANT),
    "replicate": ("nearest", cv2.BORDER_REPLICATE),
    "symmetric_whole": ("mirror", cv2.BORDER_REFLECT_101),
    "symmetric_half": ("reflect", cv2.BORDER_REFLECT),
    "periodic": ("wrap", None),
}

x = rng.standard_normal((7, 9, 2))
out["x"] = x


# scipy's sobel smooths along every other axis, so each channel is its own.
def sobel(axis, mode):
    return np.stack(
        [ndimage.sobel(x[:, :, c], axis=axis, mode=mode, cval=0.0) for c in range(2)],
        axis=-1,
    )


for edge, (mode, border) in EDGES.items():
    out[f"sobel.scipy.{edge}.x"] = sobel(1, mode)
    out[f"sobel.scipy.{edge}.y"] = sobel(0, mode)
    if border is None:
        continue
    out[f"sobel.opencv.{edge}.x"] = cv2.Sobel(x, cv2.CV_64F, 1, 0, ksize=3, borderType=border)
    out[f"sobel.opencv.{edge}.y"] = cv2.Sobel(x, cv2.CV_64F, 0, 1, ksize=3, borderType=border)
    out[f"scharr.opencv.{edge}.x"] = cv2.Scharr(x, cv2.CV_64F, 1, 0, borderType=border)
    out[f"scharr.opencv.{edge}.y"] = cv2.Scharr(x, cv2.CV_64F, 0, 1, borderType=border)
out["central.numpy.x"] = np.gradient(x, axis=1)
out["central.numpy.y"] = np.gradient(x, axis=0)

# Smooth gradients, each component a blur of noise, so that no strength
# ties another within rounding.
h, w = 24, 31
gx = ndimage.gaussian_filter(rng.standard_normal((h, w)), 1.5)
gy = ndimage.gaussian_filter(rng.standard_normal((h, w)), 1.5)
mask = np.zeros((h, w), np.uint8)
mask[1:-1, 1:-1] = 1
kept = _nonmaximum_suppression_bilinear(
    np.ascontiguousarray(gy), np.ascontiguousarray(gx), np.hypot(gx, gy), mask, 0.0
)
out["maxima.gx"] = gx
out["maxima.gy"] = gy
out["maxima.skimage"] = np.asarray(kept) > 0

i, j = np.mgrid[0:48, 0:64] + 0.5
disc = (i - 20) ** 2 + (j - 18) ** 2 < 11**2
u, v = j - 44, i - 26
a, b = np.cos(0.5) * u + np.sin(0.5) * v, np.cos(0.5) * v - np.sin(0.5) * u
square = (np.abs(a) < 9) & (np.abs(b) < 9)
scene = 0.2 + 0.5 * disc + 0.3 * square + 0.004 * j + 0.05 * rng.standard_normal((48, 64))
out["canny.x"] = scene
for sigma in (1, 2):
    out[f"canny.{sigma}"] = canny(scene, sigma=sigma)

np.savez(sys.argv[1], **out)
print(scipy.__version__, cv2.__version__, skimage.__version__, len(out), "arrays")
