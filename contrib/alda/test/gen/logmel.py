# golden/whisper.npz and librispeech/*.npy for test_whisper.ml, Whisper's
# log-mel frontend composed from alda.
# Run from this directory, with LIBRISPEECH_DIR holding LibriSpeech's
# test-clean and test-other (https://www.openslr.org/12):
#   uv run --locked logmel.py LIBRISPEECH_DIR ../librispeech ../golden/whisper.npz
#
# Writes AUDIO_OUT_DIR/<id>.npy, the utterance's int16 samples (soundfile's
# FLAC decode), and OUT.npz with, for each utterance <id> and b in 80, 128:
#   <id>.<b>.ref64     float32 [k + 1, b], [frames; bands]: a float64
#                      reference of upstream's log-mel, rounded to float32
#                      (at most 6e-8 off, for a test within 1e-4)
# Frames k to 2999 are equal to frame k, which is kept once: they frame the
# 30 s pad's zeros, at the max - 8 floor. It asserts that upstream's
# whisper.log_mel_spectrogram of pad_or_trim(audio) on CPU float32 lies
# within 2e-4 - 1e-4 of the reference, so that a test within 1e-4 of the
# reference is within 2e-4 of upstream.
import os
import sys

import numpy as np
import soundfile as sf
import torch
import whisper
from whisper.audio import HOP_LENGTH, N_FFT

UTTERANCES = [
    ("test-clean", "121-121726-0002"),
    ("test-clean", "61-70968-0000"),
    ("test-other", "367-130732-0001"),
]

root, audio_out, out = sys.argv[1:4]
torch.set_num_threads(2)
FILTERS = np.load(os.path.join(os.path.dirname(whisper.__file__), "assets", "mel_filters.npz"))


def ref64(audio, n_mels, padding):
    # torch.stft(center=True, reflect) with periodic Hann, drop last frame,
    # power, Slaney filters (the float32 values from the npz), log10, max-8.
    a = np.pad(audio.astype(np.float64), (0, padding))
    a = np.pad(a, (N_FFT // 2, N_FFT // 2), mode="reflect")
    n = np.arange(N_FFT)
    w = 0.5 - 0.5 * np.cos(2 * np.pi * n / N_FFT)
    T = 1 + (len(a) - N_FFT) // HOP_LENGTH
    idx = np.arange(N_FFT)[None, :] + HOP_LENGTH * np.arange(T)[:, None]
    spec = np.fft.rfft(a[idx] * w, axis=1)[:-1]
    power = (spec.real ** 2 + spec.imag ** 2).T
    mel = FILTERS[f"mel_{n_mels}"].astype(np.float64) @ power
    ls = np.log10(np.maximum(mel, 1e-10))
    ls = np.maximum(ls, ls.max() - 8.0)
    return (ls + 4.0) / 4.0


arrays = {}
for set_, id_ in UTTERANCES:
    spk, chap, _ = id_.split("-")
    pcm, sr = sf.read(os.path.join(root, set_, spk, chap, id_ + ".flac"), dtype="int16")
    assert sr == 16000 and pcm.ndim == 1
    np.save(os.path.join(audio_out, id_ + ".npy"), pcm)
    audio = whisper.pad_or_trim(pcm.astype(np.float32) / 32768.0)
    for b in (80, 128):
        up = whisper.log_mel_spectrogram(torch.from_numpy(audio), b).numpy().T
        f64 = ref64(audio, b, 0).T
        assert up.dtype == np.float32
        assert up.shape == (3000, b) and f64.shape == (3000, b)
        # The first frame from which every frame equals the floor frame.
        k = 3000 - 1
        while k > 0 and (up[k - 1] == up[-1]).all() and (f64[k - 1] == f64[-1]).all():
            k -= 1
        assert (up[k:] == up[k]).all() and (f64[k:] == f64[k]).all()
        gap = np.abs(up.astype(np.float64) - f64).max()
        assert gap < 2e-4 - 1e-4, (id_, b, gap)
        arrays[f"{id_}.{b}.ref64"] = np.ascontiguousarray(f64[: k + 1].astype(np.float32))
        print(id_, b, "frames kept", k + 1, "samples", len(pcm), "upstream within", gap)
np.savez_compressed(out, **arrays)
