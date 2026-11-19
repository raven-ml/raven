# Goldens for Alda.istft: torch.istft under Zero edges, and the least-squares
# inverse by numpy's pseudo-inverse on small grids.
# Run from this directory: uv run --locked istft.py ../golden/istft.npz
#
# Keys, for each case <c>:
#   <c>.z     complex128 [..., frames, bins], spectra in alda's layout
#   <c>.win   float64 [window]
#   <c>.x     float64 [..., n], the inverse
#   <c>.grid  int64 [6]: edge (0 Zero, 1 Replicate, 2 Symmetric_whole,
#             3 Symmetric_half), lo, hi, window, hop, n
# Cases torch.* come from torch.istft; ls.* from pinv(A) @ irfft(z), A the
# operator from the signal to the windowed frames of its extension.
import sys

import numpy as np
import torch

out = sys.argv[1]
rng = np.random.default_rng(1118)
arrays = {}
MODES = {0: "constant", 1: "edge", 2: "reflect", 3: "symmetric"}


def count(n, lo, hi, w, h):
    d = n + lo + hi - w
    return 0 if d < 0 else d // h + 1


def spectra(lead, m, w):
    b = w // 2 + 1
    return rng.standard_normal(lead + (m, b)) + 1j * rng.standard_normal(lead + (m, b))


def put(name, edge, lo, hi, w, h, n, z, win, x):
    arrays[f"{name}.z"] = z.astype(np.complex128)
    arrays[f"{name}.win"] = win.astype(np.float64)
    arrays[f"{name}.x"] = x.astype(np.float64)
    arrays[f"{name}.grid"] = np.array([edge, lo, hi, w, h, n], dtype=np.int64)


# torch.istft, center False (ends (0, 0)) and True (ends (w/2, w/2)), Zero.
for name, w, h, n, center, win in (
    ("torch.plain", 16, 5, 46, False, torch.hamming_window(16, periodic=False, dtype=torch.float64)),
    ("torch.centred", 16, 4, 40, True, torch.hann_window(16, periodic=True, dtype=torch.float64)),
    ("torch.whisper", 400, 160, 1600, True, torch.hann_window(400, periodic=True, dtype=torch.float64)),
):
    lo = hi = w // 2 if center else 0
    m = count(n, lo, hi, w, h)
    z = spectra((2,), m, w)
    x = torch.istft(torch.from_numpy(z).transpose(-1, -2), w, h, window=win,
                    center=center, length=n, return_complex=False).numpy()
    assert x.shape == (2, n)
    put(name, 0, lo, hi, w, h, n, z, win.numpy(), x)


def extension(edge, lo, hi, n):
    eye = np.eye(n)
    return np.stack([np.pad(eye[:, i], (lo, hi), mode=MODES[edge]) for i in range(n)], axis=1)


def least_squares(edge, lo, hi, w, h, n, win, z):
    e = extension(edge, lo, hi, n)
    m = z.shape[-2]
    a = np.concatenate([win[:, None] * e[k * h:k * h + w] for k in range(m)])
    y = np.fft.irfft(z, w, axis=-1).reshape(m * w)
    return np.linalg.pinv(a) @ y


i = 0
for edge in (0, 1, 2, 3):
    for lo, hi, w, h, n, wname in ((0, 0, 8, 3, 30, "hann"), (4, 4, 8, 3, 30, "hann"),
                                   (3, 6, 8, 2, 30, "random"), (9, 2, 6, 4, 25, "random"),
                                   (6, 7, 8, 3, 5, "random"), (2, 5, 4, 6, 17, "random")):
        if wname == "hann":
            win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(w) / w)
        else:
            win = rng.uniform(0.1, 1.0, w)
        m = count(n, lo, hi, w, h)
        z = spectra((), m, w)
        put(f"ls.{i}", edge, lo, hi, w, h, n, z, win, least_squares(edge, lo, hi, w, h, n, win, z))
        i += 1
np.savez(out, **arrays)
print("torch", torch.__version__, "numpy", np.__version__, len(arrays), "arrays")
