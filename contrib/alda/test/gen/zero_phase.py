# Goldens for Alda.zero_phase: scipy.signal.sosfiltfilt with its defaults
# (odd extension, pad 3 (2 S + 1 - z), z the lesser of the counts of rows
# with b2 zero and with a2 zero, steady starts) and, where z = 1 (odd
# orders), with padlen = 6 S.
# Run from this directory: uv run --locked zero_phase.py ../golden/zero_phase.npz
#
# Keys, case <i>: <i>.sos float64 [S; 6]; <i>.x float64 [n]; <i>.y float64 [n]
# the default; <i>.y6 float64 [n] with padlen = 6 S (odd orders only).
import math
import sys

import numpy as np
import scipy
from scipy import signal

out = sys.argv[1]
rng = np.random.default_rng(20261120)
arrays = {}
designs = [
    signal.butter(8, 1000, fs=48000, output="sos"),
    signal.ellip(4, 0.5, 60, [300, 3000], btype="bandpass", fs=16000, output="sos"),
    signal.cheby2(6, 50, 0.2, output="sos"),
    signal.butter(5, 2000, fs=48000, output="sos"),
    signal.butter(3, 0.01, btype="highpass", output="sos"),
    signal.ellip(7, 1, 70, 0.3, output="sos"),
]
n = 1500
t = np.arange(n)
for i, sos in enumerate(designs):
    x = rng.standard_normal(n) + 2 * np.sin(2 * math.pi * t / 97) + 1.5
    arrays[f"{i}.sos"] = sos
    arrays[f"{i}.x"] = x
    arrays[f"{i}.y"] = signal.sosfiltfilt(sos, x)
    s = sos.shape[0]
    if min(np.sum(sos[:, 2] == 0), np.sum(sos[:, 5] == 0)) > 0:
        arrays[f"{i}.y6"] = signal.sosfiltfilt(sos, x, padlen=6 * s)
np.savez(out, **arrays)
print("scipy", scipy.__version__, len(arrays), "arrays")
