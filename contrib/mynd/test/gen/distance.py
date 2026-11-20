# /// script
# requires-python = "==3.13.*"
# dependencies = ["numpy==2.5.5", "scipy==1.17.1"]
# ///
"""scipy's Euclidean distance transform, as goldens for Mynd.Mask.distance
and Mynd.Mask.closest (contrib/mynd/test/test_mask.ml).

Run from this directory, with the script's own lock:

    uv run --locked --script distance.py ../golden/distance.npz

For each shape HxW, under the keys "HxW.":

- m: masks [b, h, w], each with a true pixel: densities from 0.01 to 0.5,
  lattices whose pixels tie for the nearest, one true pixel;
- d: distance_transform_edt(~m) in float64, the distance from each pixel
  to the nearest true pixel of its mask;
- i: the row-major index of the true pixel scipy's return_indices names;
- unique: whether that pixel is the only true pixel at the least squared
  distance, by a brute force over the true pixels in integers.
"""

import sys

import numpy as np
from scipy import ndimage

rng = np.random.default_rng(57)
SHAPES = [(9, 13), (32, 32), (1, 40), (40, 1), (64, 17)]


def masks(h, w):
    out = []
    for p in (0.01, 0.05, 0.2, 0.5):
        m = rng.random((h, w)) < p
        m[rng.integers(h), rng.integers(w)] = True
        out.append(m)
    for k in (2, 3):
        m = np.zeros((h, w), bool)
        m[::k, ::k] = True
        out.append(m)
    m = np.zeros((h, w), bool)
    m[h // 2, w // 2] = True
    out.append(m)
    return np.stack(out)


def unique(m):
    h, w = m.shape
    pts = np.argwhere(m)
    ii, jj = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    d = (ii[..., None] - pts[:, 0]) ** 2 + (jj[..., None] - pts[:, 1]) ** 2
    return (d == d.min(-1, keepdims=True)).sum(-1) == 1


out = {}
for h, w in SHAPES:
    ms = masks(h, w)
    ds, idx, un = [], [], []
    for m in ms:
        d, (ri, ci) = ndimage.distance_transform_edt(~m, return_indices=True)
        ds.append(d)
        idx.append((ri * w + ci).astype(np.int64))
        un.append(unique(m))
    key = f"{h}x{w}"
    out[f"{key}.m"] = ms
    out[f"{key}.d"] = np.stack(ds).astype(np.float64)
    out[f"{key}.i"] = np.stack(idx)
    out[f"{key}.unique"] = np.stack(un)

np.savez_compressed(sys.argv[1], **out)
