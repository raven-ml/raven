# Goldens for Mynd's morphology and reconstruction. Images are [h, w, c];
# footprints [kh, kw] booleans; masks [b, h, w].
# - erode, dilate: cv2.erode and cv2.dilate at their default anchor and
#   border, which leave out the taps outside the image; scikit-image's
#   erosion and dilation under mode="ignore", which do too; scipy's
#   grey_dilation under a constant -inf, the identity of its maximum.
#   Each library takes its own footprint: the test converts it.
# - reconstruct: scipy's binary_fill_holes, whose holes are the background
#   components that touch no border pixel, under a cross (4-connected
#   background) and a square (8-connected).
# Run from this directory: uv run --locked morphology.py ../golden/morphology.npz
import sys

import cv2
import numpy as np
import scipy
import skimage
from scipy import ndimage
from skimage import morphology

rng = np.random.default_rng(31)
out = {}

f32 = rng.standard_normal((9, 11, 2)).astype(np.float32)
u8 = rng.integers(0, 256, (9, 11, 3), dtype=np.uint8)
out["f32"] = f32
out["u8"] = u8

asym = np.array([[0, 1, 1], [0, 1, 0], [1, 1, 0]], bool)
FOOTPRINTS = {
    "rect3x3": np.ones((3, 3), bool),
    "rect2x2": np.ones((2, 2), bool),
    "rect4x4": np.ones((4, 4), bool),
    "rect1x4": np.ones((1, 4), bool),
    "cross5": cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)) > 0,
    "ellipse6": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6)) > 0,
    "asym3x3": asym,
    "asym4x3": rng.random((4, 3)) < 0.5,
    "nocentre3x5": np.array([[1, 0, 0, 0, 1], [0, 1, 0, 0, 0], [0, 0, 0, 1, 1]], bool),
}


def channels(f, x):
    return np.stack([f(x[:, :, c]) for c in range(x.shape[2])], axis=-1)


for name, fp in FOOTPRINTS.items():
    out[f"fp.{name}"] = fp
    k = fp.astype(np.uint8)
    for tag, x in (("f32", f32), ("u8", u8)):
        out[f"opencv.erode.{name}.{tag}"] = cv2.erode(x, k)
        out[f"opencv.dilate.{name}.{tag}"] = cv2.dilate(x, k)
        out[f"skimage.erosion.{name}.{tag}"] = channels(
            lambda c: morphology.erosion(c, fp, mode="ignore"), x
        )
        out[f"skimage.dilation.{name}.{tag}"] = channels(
            lambda c: morphology.dilation(c, fp, mode="ignore"), x
        )
    out[f"scipy.dilation.{name}.f32"] = channels(
        lambda c: ndimage.grey_dilation(c, footprint=fp, mode="constant", cval=-np.inf),
        f32,
    )

masks = rng.random((5, 13, 17)) < 0.55
masks[4] = False
masks[4, 3:9, 4:12] = True
masks[4, 5:7, 6:10] = False  # a ring, whose inside is a hole either way
out["masks"] = masks
cross = ndimage.generate_binary_structure(2, 1)
square = ndimage.generate_binary_structure(2, 2)
out["holes.four"] = np.stack([ndimage.binary_fill_holes(m, structure=cross) for m in masks])
out["holes.eight"] = np.stack([ndimage.binary_fill_holes(m, structure=square) for m in masks])

np.savez(sys.argv[1], **out)
print(scipy.__version__, cv2.__version__, skimage.__version__, len(out), "arrays")
