# Goldens for Alda.Iir: scipy.signal's butter, cheby1, cheby2 and ellip with
# output='sos', and bilinear_zpk then zpk2sos for an analog filter.
# Run from this directory: uv run --locked designs.py ../golden/designs.npz
#
# Keys, case <i>: <i>.spec float64 [8] = (family, order, band, f1, f2, rate,
# ripple, reject), family 0 butter, 1 cheby1, 2 cheby2, 3 ellip; band 0 low,
# 1 high, 2 band pass, 3 band stop (f2 unused for 0, 1); <i>.sos float64
# [S; 6]; <i>.w float64 [128] angular frequencies; <i>.h complex128 [128] the
# response of <i>.sos there (sosfreqz).
# Analog case: a.zeros, a.poles complex128, a.params float64 [2] = (gain,
# rate), a.sos float64 [S; 6]: A-weighting (IEC 61672-1) at 48 kHz.
import math
import sys

import numpy as np
import scipy
from scipy import signal

out = sys.argv[1]
arrays = {}
bands = ["lowpass", "highpass", "bandpass", "bandstop"]
cases = [
    # family, order, band, f1, f2, rate, ripple, reject
    (0, 1, 0, 1000, 0, 48000, 0, 0),
    (0, 4, 0, 1000, 0, 48000, 0, 0),
    (0, 8, 0, 100, 0, 48000, 0, 0),
    (0, 5, 1, 300, 0, 16000, 0, 0),
    (0, 3, 2, 300, 3400, 8000, 0, 0),
    (0, 4, 3, 50, 70, 1000, 0, 0),
    (1, 4, 0, 2000, 0, 44100, 1, 0),
    (1, 5, 1, 1000, 0, 44100, 0.5, 0),
    (1, 3, 2, 500, 1500, 8000, 3, 0),
    (2, 6, 0, 3000, 0, 16000, 0, 60),
    (2, 5, 3, 1000, 2000, 16000, 0, 40),
    (2, 7, 1, 100, 0, 1000, 0, 80),
    (3, 2, 0, 0.1, 0, 1, 1, 20),
    (3, 3, 0, 0.2, 0, 1, 0.01, 100),
    (3, 4, 1, 4000, 0, 48000, 1, 60),
    (3, 5, 0, 0.3, 0, 2, 3, 40),
    (3, 8, 2, 1000, 3000, 16000, 0.5, 80),
    (3, 12, 0, 0.2, 0, 1, 0.1, 120),
    (3, 10, 0, 0.15, 0, 1, 0.001, 150),
    (3, 16, 0, 0.25, 0, 1, 0.1, 200),
    (3, 6, 3, 0.1, 0.2, 1, 0.5, 70),
    (3, 1, 0, 0.1, 0, 1, 1, 30),
]
for i, (fam, order, band, f1, f2, rate, rp, rs) in enumerate(cases):
    wn = f1 if band < 2 else [f1, f2]
    kw = dict(btype=bands[band], fs=rate, output="sos")
    if fam == 0:
        sos = signal.butter(order, wn, **kw)
    elif fam == 1:
        sos = signal.cheby1(order, rp, wn, **kw)
    elif fam == 2:
        sos = signal.cheby2(order, rs, wn, **kw)
    else:
        sos = signal.ellip(order, rp, rs, wn, **kw)
    w, h = signal.sosfreqz(sos, worN=128)
    arrays[f"{i}.spec"] = np.array([fam, order, band, f1, f2, rate, rp, rs], np.float64)
    arrays[f"{i}.sos"] = sos
    arrays[f"{i}.w"] = w
    arrays[f"{i}.h"] = h

# A-weighting, IEC 61672-1 Annex E: poles at 20.598997, 107.65265,
# 737.86223 and 12194.217 Hz (the outer two double), four zeros at 0, gain
# normalised to 0 dB at 1 kHz by A1000 = 1.9997 dB.
f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
zeros = np.zeros(4, complex)
poles = -2 * math.pi * np.array([f1, f1, f2, f3, f4, f4], complex)
gain = (2 * math.pi * f4) ** 2 * 10 ** (1.9997 / 20)
rate = 48000.0
z, p, k = signal.bilinear_zpk(zeros, poles, gain, rate)
arrays["a.zeros"] = zeros
arrays["a.poles"] = poles
arrays["a.params"] = np.array([gain, rate])
arrays["a.sos"] = signal.zpk2sos(z, p, k)
np.savez(out, **arrays)
print("scipy", scipy.__version__, len(arrays), "arrays")
