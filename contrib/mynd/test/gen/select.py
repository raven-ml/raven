# scipy's connected-component labels and torchvision's NMS, as goldens for
# Mynd.components and Mynd.nms. Labels are each component's least row-major
# index, -1 off the mask.
# Run from this directory: uv run --locked select.py ../golden/select.npz
import sys
import numpy as np
import scipy
from scipy import ndimage
import torch
import torchvision

rng = np.random.default_rng(5)
out = {}

def least(m, structure):
    lab, k = ndimage.label(m, structure=structure)
    idx = np.arange(m.size).reshape(m.shape)
    mins = np.asarray(ndimage.minimum(idx, lab, index=np.arange(1, k + 1)), np.int64)
    res = np.full(m.shape, -1, np.int64)
    res[m] = mins[lab[m] - 1]
    return res

ms = np.stack([rng.random((24, 31)) < 0.55, rng.random((24, 31)) < 0.4])
out["cc.m"] = ms
for name, st in (("four", ndimage.generate_binary_structure(2, 1)),
                 ("eight", np.ones((3, 3), int))):
    out["cc." + name] = np.stack([least(m, st) for m in ms])

# Detector-like boxes: jittered copies around a few objects, scores that
# fall with the distance to the object, some equal.
objs = rng.uniform(0, 200, (12, 2)); size = rng.uniform(10, 60, (12, 2))
n = 400
o = rng.integers(0, 12, n)
c = objs[o] + rng.normal(0, 4, (n, 2)); s = size[o] * rng.uniform(0.7, 1.3, (n, 2))
boxes = np.concatenate([c - s / 2, c + s / 2], 1).astype(np.float32)
d = np.linalg.norm(c - objs[o], axis=1)
scores = (np.exp(-d / 8) + rng.uniform(0, 0.05, n)).astype(np.float32)
scores[10:20] = scores[0]                           # ties
classes = rng.integers(0, 3, n)
out["nms.boxes"], out["nms.scores"], out["nms.classes"] = boxes, scores, classes
bt, st = torch.tensor(boxes), torch.tensor(scores)
for t in (0.3, 0.5, 0.7, 0.85):
    keep = np.zeros(n, bool); keep[torchvision.ops.nms(bt, st, t).numpy()] = True
    out[f"nms.{t}"] = keep
    # Classes by masking: torchvision's per-class path, nms on each class.
    keep = np.zeros(n, bool)
    for k in range(3):
        idx = np.flatnonzero(classes == k)
        keep[idx[torchvision.ops.nms(bt[idx], st[idx], t).numpy()]] = True
    out[f"nms.classes.{t}"] = keep
np.savez(sys.argv[1], **out)
print(scipy.__version__, torchvision.__version__, len(out), "arrays")
