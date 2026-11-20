# torch's grid_sample (align_corners=False) and antialiased interpolate in
# float64, as goldens for Mynd.Sample.at and Mynd.Sample.resize. Positions
# are in pixels, centre of pixel j at j + 1/2: the grid holds g = 2u/w - 1.
# The image is 4 x 8, so on positions that are multiples of 1/16 torch's
# unnormalisation ((g + 1) w - 1) / 2 is exact and its kinks fall where
# ours do.
# Run from this directory: uv run --locked sample.py ../golden/sample.npz
import sys
import numpy as np
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(7)
H, W, C, B, Q = 4, 8, 3, 2, 48
out = {}
x = rng.standard_normal((B, H, W, C))
# Positions: centres, cell edges and other multiples of 1/16, up to three
# pixels outside the image on every side.
u = rng.integers(-48, (W + 3) * 16, size=(B, Q), endpoint=True) / 16.0
v = rng.integers(-48, (H + 3) * 16, size=(B, Q), endpoint=True) / 16.0
u[:, :8] = np.arange(8) + 0.5        # centres of row 1
v[:, :8] = 1.5
u[:, 8:12] = [0.0, 1.0, 7.0, 8.0]    # cell edges
v[:, 8:12] = [0.0, 2.0, 4.0, 3.0]
p = np.stack([u, v], -1)
r = rng.standard_normal((B, Q, C))
out["x"], out["p"], out["r"] = x, p, r
for mode in ("bilinear", "bicubic"):
    for pad in ("zeros", "border", "reflection"):
        xt = torch.tensor(x, requires_grad=True)
        pt = torch.tensor(p, requires_grad=True)
        scale = torch.tensor([2.0 / W, 2.0 / H])
        g = (pt * scale - 1.0)[:, None]                  # [B, 1, Q, 2]
        y = F.grid_sample(xt.permute(0, 3, 1, 2), g, mode=mode,
                          padding_mode=pad, align_corners=False)
        y = y[:, :, 0].permute(0, 2, 1)                  # [B, Q, C]
        (y * torch.tensor(r)).sum().backward()
        k = f"{mode}.{pad}"
        out[k + ".y"] = y.detach().numpy()
        out[k + ".gx"] = xt.grad.numpy()
        out[k + ".gp"] = pt.grad.numpy()
# Antialiased resizes: columns and rows each resampled on cells,
# renormalised at the ends.
z = rng.standard_normal((1, 9, 11, 2))
out["rz.x"] = z
for mode in ("bilinear", "bicubic"):
    for (h2, w2) in ((4, 5), (13, 17), (6, 23), (9, 4)):
        y = F.interpolate(torch.tensor(z).permute(0, 3, 1, 2), size=(h2, w2),
                          mode=mode, antialias=True, align_corners=False)
        out[f"rz.{mode}.{h2}x{w2}"] = y.permute(0, 2, 3, 1).numpy()
np.savez(sys.argv[1], **out)
print(torch.__version__, len(out), "arrays")
