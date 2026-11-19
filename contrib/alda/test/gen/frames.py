# Goldens for Alda.Frames: torch.stft's centred reflect framing, and Kaldi's
# frame extraction (feature-window.cc FirstSampleOfFrame, NumFrames and
# ExtractWindow, kaldi-asr/kaldi master, ported below without dither or
# windowing).
# Run from this directory: uv run --locked frames.py ../golden/frames.npz
#
# Keys:
#   whisper.signal             float32 [6000], the signal whose prefixes are framed
#   whisper.counts             int64 [L, 2], rows (n, frames of torch.stft) for
#                              n = 201 .. 2400
#   whisper.padded.<n>         float32, F.pad(x[:n], (200, 200), "reflect"),
#                              the signal torch.stft frames, for a selection of n
#   kaldi.signal               float64 [500], small integers
#   kaldi.<snip>.<w>.<h>.<n>   float64 [frames, w], Kaldi's frames of
#                              kaldi.signal[:n]
import sys

import numpy as np
import torch
import torch.nn.functional as F

out = sys.argv[1]
arrays = {}

# torch.stft(center=True, pad_mode="reflect"), n_fft 400, hop 160: Whisper.
rng = np.random.default_rng(20261118)
x = rng.standard_normal(6000).astype(np.float32)
arrays["whisper.signal"] = x
counts = []
for n in range(201, 2401):
    t = torch.from_numpy(x[:n])
    s = torch.stft(t, 400, 160, window=torch.hann_window(400), center=True,
                   pad_mode="reflect", return_complex=True)
    counts.append((n, s.shape[-1]))
arrays["whisper.counts"] = np.array(counts, dtype=np.int64)
for n in list(range(201, 221)) + [399, 400, 401, 559, 560, 561, 1000, 1601, 4097, 6000]:
    t = torch.from_numpy(x[:n])
    p = F.pad(t[None, None, :], (200, 200), mode="reflect")[0, 0].numpy()
    # torch.stft frames this signal: frame m is p[160 m : 160 m + 400].
    assert p.dtype == np.float32 and p.shape == (n + 400,)
    arrays[f"whisper.padded.{n}"] = p


# Kaldi, feature-window.cc.
def first_sample(f, w, h, snip):
    if snip:
        return f * h
    return h * f + h // 2 - w // 2


def num_frames(n, w, h, snip):
    if snip:
        return 0 if n < w else 1 + (n - w) // h
    return (n + h // 2) // h  # flush = true


def extract(wave, f, w, h, snip):
    start = first_sample(f, w, h, snip)
    out = np.empty(w)
    for s in range(w):
        i = s + start
        while i < 0 or i >= len(wave):
            if i < 0:
                i = -i - 1
            else:
                i = 2 * len(wave) - 1 - i
        out[s] = wave[i]
    return out


k = rng.integers(-1000, 1000, 500).astype(np.float64)
arrays["kaldi.signal"] = k
for snip in (True, False):
    for w, h, ns in ((25, 10, list(range(1, 61)) + [101, 499]), (400, 160, [1, 7, 199, 400, 401, 499])):
        for n in ns:
            m = num_frames(n, w, h, snip)
            fr = np.array([extract(k[:n], f, w, h, snip) for f in range(m)]).reshape(m, w)
            arrays[f"kaldi.{'snip' if snip else 'nosnip'}.{w}.{h}.{n}"] = fr
np.savez(out, **arrays)
print("torch", torch.__version__, len(arrays), "arrays")
