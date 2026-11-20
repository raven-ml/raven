# torchvision's box_iou and pycocotools' RLE codec, area, boxes and mask IoU,
# as goldens for Mynd.Box, Mynd.Mask and Mynd.Rle.
# Run from this directory: uv run --locked annot.py ../golden/annot.npz
import sys
import numpy as np
import torch
import torchvision
from pycocotools import mask as cm

rng = np.random.default_rng(11)
out = {}

# Boxes in float32: random, nested, touching, identical.
def boxes(n):
    xy = rng.uniform(0, 50, (n, 2)); wh = rng.uniform(0.5, 30, (n, 2))
    return np.concatenate([xy, xy + wh], 1)
a = boxes(7); b = boxes(5)
a[5] = [10, 10, 20, 20]; b[3] = [20, 10, 30, 20]   # touching: IoU 0
a[6] = [12, 14, 15.5, 17]; b[4] = a[6]             # identical: IoU 1
a = a.astype(np.float32); b = b.astype(np.float32)
out["box.a"], out["box.b"] = a, b
out["box.iou"] = torchvision.ops.box_iou(torch.tensor(a), torch.tensor(b)).numpy()

# Masks, h x w, column-major runs.
def runs(m):
    f = m.flatten(order="F").astype(np.int64)
    edges = np.flatnonzero(np.diff(f)) + 1
    starts = np.concatenate([[0], edges]); ends = np.concatenate([edges, [f.size]])
    counts = list(ends - starts)
    if f.size and f[0] == 1: counts = [0] + counts
    return np.array(counts, np.int64)

h, w = 13, 17
ms = []
for _ in range(3):
    m = np.zeros((h, w), np.uint8)
    for _ in range(3):
        y0, x0 = rng.integers(0, h - 2), rng.integers(0, w - 2)
        m[y0:y0 + rng.integers(1, 6), x0:x0 + rng.integers(1, 8)] = 1
    ms.append(m)
ms.append(np.zeros((h, w), np.uint8))               # empty
ms.append(np.ones((h, w), np.uint8))                # full: first run empty
m = np.zeros((h, w), np.uint8); m[0, 0] = 1; m[-1, -1] = 1; ms.append(m)
big = np.zeros((100, 120), np.uint8)                # long runs, both signs
for _ in range(6):
    y0, x0 = rng.integers(0, 90), rng.integers(0, 100)
    big[y0:y0 + rng.integers(2, 40), x0:x0 + rng.integers(2, 60)] = 1
out["m"] = np.stack(ms)
rles = cm.encode(np.asfortranarray(np.stack(ms, -1)))
for i, r in enumerate(rles):
    out[f"rle.{i}.s"] = np.frombuffer(r["counts"], np.uint8)
    out[f"rle.{i}.counts"] = runs(ms[i])
rb = cm.encode(np.asfortranarray(big))
out["big"] = big
out["big.s"] = np.frombuffer(rb["counts"], np.uint8)
out["big.counts"] = runs(big)
out["area"] = cm.area(rles).astype(np.int64)
out["bbox"] = cm.toBbox(rles)
out["iou"] = cm.iou(rles, rles, [0] * len(rles))
np.savez(sys.argv[1], **out)
print(torchvision.__version__, len(out), "arrays")
