# Goldens for Alda.Mel: librosa.filters.mel at float64, and Whisper's
# mel_filters.npz (librosa's float32 banks, which openai-whisper ships).
# Run from this directory: uv run --locked banks.py ../golden/banks.npz
# Keys: librosa.<i>.bank float64 [bins, bands]; librosa.<i>.params float64
# [6] (scale 0 Slaney 1 Htk, norm 0 Unit_peak 1 Unit_area, rate, n, low,
# high); whisper.<b> float32 [201, b] for b = 80, 128.
import os
import sys

import librosa
import numpy as np
import whisper

out = sys.argv[1]
arrays = {}
cases = [
    # (htk, norm, sr, n_fft, n_mels, fmin, fmax)
    (False, "slaney", 16000, 400, 80, 0.0, 8000.0),
    (False, "slaney", 16000, 400, 128, 0.0, 8000.0),
    (False, "slaney", 22050, 2048, 128, 0.0, 11025.0),
    (False, "slaney", 8000, 256, 40, 20.0, 3800.0),
    (False, "slaney", 44100.0, 1024, 64, 50.0, 22050.0),
    (True, None, 16000, 512, 40, 0.0, 8000.0),
    (True, None, 16000, 400, 23, 20.0, 7600.0),
    (True, None, 22050, 1024, 64, 133.33, 6855.6),
]
for i, (htk, norm, sr, n_fft, n_mels, fmin, fmax) in enumerate(cases):
    b = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax,
                            htk=htk, norm=norm, dtype=np.float64)
    assert b.shape == (n_mels, n_fft // 2 + 1)
    assert (b.max(axis=1) > 0).all(), "empty band"
    arrays[f"librosa.{i}.bank"] = np.ascontiguousarray(b.T)
    arrays[f"librosa.{i}.params"] = np.array(
        [1.0 if htk else 0.0, 1.0 if norm == "slaney" else 0.0, sr, n_fft, fmin, fmax])
f = np.load(os.path.join(os.path.dirname(whisper.__file__), "assets", "mel_filters.npz"))
for b in (80, 128):
    m = f[f"mel_{b}"]
    assert m.dtype == np.float32 and m.shape == (b, 201)
    arrays[f"whisper.{b}"] = np.ascontiguousarray(m.T)
np.savez_compressed(out, **arrays)
print("librosa", librosa.__version__, len(arrays), "arrays")
