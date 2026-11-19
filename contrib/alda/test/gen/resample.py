# Goldens for Alda.resample: torchaudio.functional.resample at its defaults
# (Hann-windowed sinc, width 6, rolloff 0.99, zeros past the ends), and
# torch.nn.functional.interpolate(antialias=True) bilinear and bicubic.
# Run from this directory: uv run --locked resample.py ../golden/resample.npz
#
# Keys: ta.<i>.rates int64 [2] (from, to); ta.<i>.<dt>.x <dt> [n];
# ta.<i>.<dt>.y <dt> [m], <dt> f32 and f64 (the kernel at the signal's dtype).
# img.<j>.x float64 [h; w]; img.<j>.<mode>.y float64 [h'; w'], mode bilinear
# or bicubic, align_corners False.
import math
import sys

import numpy as np
import torch
import torchaudio
from torchaudio import functional as AF

out = sys.argv[1]
rng = np.random.default_rng(20261121)
arrays = {}
for i, (src, dst) in enumerate([(48000, 16000), (44100, 16000), (16000, 48000), (22050, 16000)]):
    n = src // 10
    t = np.arange(n) / src
    x = 0.3 * rng.standard_normal(n) + 0.5 * np.sin(2 * math.pi * 440 * t) + 0.2 * np.sin(2 * math.pi * 6900 * t)
    arrays[f"ta.{i}.rates"] = np.array([src, dst], np.int64)
    for name, dt in (("f32", torch.float32), ("f64", torch.float64)):
        xt = torch.tensor(x, dtype=dt)
        arrays[f"ta.{i}.{name}.x"] = xt.numpy()
        arrays[f"ta.{i}.{name}.y"] = AF.resample(xt, src, dst).numpy()

for j, (h, w, h2, w2) in enumerate([(40, 60, 13, 17), (12, 9, 31, 20), (25, 25, 50, 8), (16, 16, 16, 16)]):
    x = rng.standard_normal((h, w))
    arrays[f"img.{j}.x"] = x
    for mode in ("bilinear", "bicubic"):
        y = torch.nn.functional.interpolate(
            torch.tensor(x)[None, None], size=(h2, w2), mode=mode, align_corners=False, antialias=True
        )[0, 0]
        arrays[f"img.{j}.{mode}.y"] = y.numpy()
np.savez(out, **arrays)
print("torch", torch.__version__, "torchaudio", torchaudio.__version__, len(arrays), "arrays")
