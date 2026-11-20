# Goldens for Mynd.Sample.resize and Mynd.Sample.resize_weights: torch's
# antialiased interpolate in float32 and float64 along the width of a one-row
# image, at the size pairs of SAM 2's and RT-DETR's inputs (torch 2.9's
# antialiased path reads a one-column image as constant), and Pillow's 8-bit
# resize of L, RGB and RGBA images (RGBA through Pillow's premultiplied
# RGBa).
# Run from this directory: uv run --locked resize.py ../golden/resize.npz
#
# Keys: torch.<n>.<n'>.x float32 [n]; torch.<n>.<n'>.<mode>.<f32|f64> [n'],
# mode bilinear or bicubic, the float64 one of the same input.
# pil.<i>.size int64 [4] (h, w, h', w'); pil.<i>.<mode>.x uint8 [h; w; c],
# mode L, RGB or RGBA; pil.<i>.<mode>.<filter> uint8 [h'; w'; c], filter
# bilinear, bicubic or lanczos.
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

out = sys.argv[1]
rng = np.random.default_rng(20261119)
arrays = {}
for n, n2 in [(427, 640), (480, 640), (480, 1024), (640, 1024), (1500, 1024), (2000, 640), (333, 640)]:
    x = rng.random(n).astype(np.float32)
    arrays[f"torch.{n}.{n2}.x"] = x
    for mode in ("bilinear", "bicubic"):
        for name, dt in (("f32", torch.float32), ("f64", torch.float64)):
            xt = torch.tensor(x, dtype=dt).reshape(1, 1, 1, n)
            y = F.interpolate(xt, size=(1, n2), mode=mode, align_corners=False, antialias=True)
            arrays[f"torch.{n}.{n2}.{mode}.{name}"] = y.reshape(n2).numpy()

filters = {"bilinear": Image.Resampling.BILINEAR, "bicubic": Image.Resampling.BICUBIC,
           "lanczos": Image.Resampling.LANCZOS}
sizes = [(23, 37, 9, 61), (40, 17, 13, 16), (31, 29, 31, 7), (12, 50, 30, 50), (64, 48, 17, 23)]
for i, (h, w, h2, w2) in enumerate(sizes):
    arrays[f"pil.{i}.size"] = np.array([h, w, h2, w2], np.int64)
    for mode, c in (("L", 1), ("RGB", 3), ("RGBA", 4)):
        x = rng.integers(0, 256, (h, w, c), dtype=np.uint8)
        if mode == "RGBA":
            # Opaque and transparent pixels, which Pillow passes through.
            x[::5, :, 3] = 255
            x[1::7, :, 3] = 0
        arrays[f"pil.{i}.{mode}.x"] = x
        img = Image.fromarray(x[:, :, 0] if c == 1 else x, mode)
        for name, f in filters.items():
            y = np.asarray(img.resize((w2, h2), f))
            arrays[f"pil.{i}.{mode}.{name}"] = y.reshape(h2, w2, c)
np.savez(out, **arrays)
print("torch", torch.__version__, "pillow", Image.__version__, len(arrays), "arrays")
