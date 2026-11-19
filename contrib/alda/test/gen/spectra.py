# Goldens for Alda.psd and Alda.power_spectrum: scipy.signal.periodogram, whose
# scaling is _spectral_py.py's (density 1 / (fs sum win^2), spectrum
# 1 / (sum win)^2, one-sided bins but 0 and an even n's n / 2 doubled).
# Run from this directory: uv run --locked spectra.py ../golden/spectra.npz
# Keys, case <i>: <i>.x float64 [w], <i>.win float64 [w], <i>.params float64
# [2] (rate, n), <i>.density and <i>.spectrum float64 [n // 2 + 1].
import sys

import numpy as np
import scipy
from scipy.signal import periodogram, windows

out = sys.argv[1]
rng = np.random.default_rng(5)
arrays = {}
cases = [
    (16, 16, 1.0, "hann"), (16, 17, 8000.0, "hann"), (15, 15, 44100.0, "hamming"),
    (15, 32, 22050.5, "blackman"), (400, 512, 16000.0, "hann"), (1, 1, 3.0, "boxcar"),
    (2, 2, 2.0, "boxcar"), (7, 9, 0.25, "random"),
]
for i, (w, n, rate, kind) in enumerate(cases):
    x = rng.standard_normal(w)
    win = rng.uniform(0.1, 1, w) if kind == "random" else windows.get_window(kind, w, fftbins=True)
    arrays[f"{i}.x"] = x
    arrays[f"{i}.win"] = win.astype(np.float64)
    arrays[f"{i}.params"] = np.array([rate, n], dtype=np.float64)
    for scaling in ("density", "spectrum"):
        f, p = periodogram(x, fs=rate, window=win, nfft=n, detrend=False,
                           return_onesided=True, scaling=scaling)
        assert p.shape == (n // 2 + 1,)
        arrays[f"{i}.{scaling}"] = p
np.savez(out, **arrays)
print("scipy", scipy.__version__, len(arrays), "arrays")
