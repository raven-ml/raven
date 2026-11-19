# Goldens for Whisper's upstream resampling: ffmpeg's swr at its defaults
# (filter_size 32, Kaiser beta 9, cutoff 0.97, exact phases), through PyAV on
# float samples, mono, as `ffmpeg -ar 16000` resamples before its s16 output.
# Run from this directory: uv run --locked swr.py ../golden/swr.npz
#
# Keys, case <i>: <i>.rate int64 [1] the input rate; <i>.x float32 [n];
# <i>.y float32 [m] swr's 16 kHz output, flushed.
import math
import sys

import av
import numpy as np

out = sys.argv[1]
rng = np.random.default_rng(20261122)
arrays = {}
for i, rate in enumerate([48000, 44100, 22050]):
    n = rate // 4
    t = np.arange(n) / rate
    x = 0.25 * rng.standard_normal(n) + 0.4 * np.sin(2 * math.pi * 300 * t)
    x = x.astype(np.float32)
    fr = av.AudioFrame.from_ndarray(x[None, :], format="flt", layout="mono")
    fr.sample_rate = rate
    rs = av.AudioResampler(format="flt", layout="mono", rate=16000)
    outs = rs.resample(fr) + rs.resample(None)
    arrays[f"{i}.rate"] = np.array([rate], np.int64)
    arrays[f"{i}.x"] = x
    arrays[f"{i}.y"] = np.concatenate([o.to_ndarray()[0] for o in outs]).astype(np.float32)
np.savez(out, **arrays)
print("av", av.__version__, len(arrays), "arrays")
