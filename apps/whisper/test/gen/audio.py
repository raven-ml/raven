# /// script
# requires-python = "==3.13.*"
# dependencies = [
#   "openai-whisper @ git+https://github.com/openai/whisper.git@86098128c0b4f24f0e2aa2994de830614b474227",
#   "torch==2.15.0",
#   "numpy==2.5.5",
#   "tiktoken==0.14.0",
#   "soundfile==0.14.0",
#   # ffmpeg 7.0.2, static, for upstream's load_audio.
#   "imageio-ffmpeg==0.6.0",
# ]
#
# [tool.uv]
# # whisper declares triton on Linux x86-64 for its CUDA kernels; the
# # generators run on the CPU.
# override-dependencies = ["triton; sys_platform == 'never'"]
# sources = { torch = { index = "pytorch-cpu" } }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Audio goldens for test_audio, written to OUT:
- audio.npz: upstream's scaled log-mel of the LibriSpeech file FLAC
  (7902-96594-0025) at 80 and 128 bands, with load_audio on ffmpeg, on its
  short path log_mel_spectrogram(pad_or_trim(audio), n) ("short") and its
  transcribe path log_mel_spectrogram(audio, n, padding=N_SAMPLES)
  ("whole"), each as upstream's float32 (up32) and a float64 reference of
  the same samples (ref64), [frames; n], cut after the first frame of the
  trailing run of equal floor frames (audio_common.trim).
  Also a seeded clip ("clip", float32) and, for each length in "lengths",
  upstream's log_mel_spectrogram(clip[:length], 80) and its reference
  (len<L>.up32, len<L>.ref64), uncut, or -1 in "frames" where upstream
  raises (torch's reflect pad needs more than 200 samples), with the
  reference alone from 160 samples on.
- int8.flac: an 8-bit FLAC file (libsndfile's PCM_S8 through soundfile) of
  the 256 values -128..127 in order, at 16 kHz, one channel.

    uv run apps/whisper/test/gen/audio.py FLAC OUT

FLAC is apps/whisper/test/librispeech/7902-96594-0025.flac and OUT
apps/whisper/test/upstream.
"""
import os, sys

import numpy as np
import soundfile as sf
import torch
import whisper
from whisper.audio import N_SAMPLES

from audio_common import put_ffmpeg_on_path, ref64, s16_of, trim

# The fixtures were recorded on two threads.
torch.set_num_threads(2)
flac, out = sys.argv[1:3]
print(put_ffmpeg_on_path())
audio = whisper.load_audio(flac)
x, sr = sf.read(flac, dtype="int16")
assert sr == 16000 and x.ndim == 1
assert np.array_equal(s16_of(audio), x), "ffmpeg and soundfile disagree"
print("samples", len(audio))
arrays = {}


def mel(a, n, padding=0):
    return whisper.log_mel_spectrogram(torch.from_numpy(a), n, padding=padding).numpy().T


short = whisper.pad_or_trim(audio)
for n in (80, 128):
    for path, up32, r64 in (
        ("short", mel(short, n), ref64(short, n)),
        ("whole", mel(audio, n, N_SAMPLES), ref64(audio, n, N_SAMPLES)),
    ):
        frames = len(up32)
        up32, r64 = trim(up32, r64)
        print(n, path, "frames", frames, "kept", len(up32),
              "max |up32 - ref64|", np.abs(up32 - r64).max())
        arrays[f"{path}.mel{n}.up32"] = np.ascontiguousarray(up32, dtype=np.float32)
        arrays[f"{path}.mel{n}.ref64"] = np.ascontiguousarray(r64, dtype=np.float64)

lengths = [1, 159, 160, 200, 201, 319, 320, 399, 400, 401, 1000, 4007]
clip = (np.random.default_rng(3).standard_normal(max(lengths)) * 0.1).astype(np.float32)
frames = []
for L in lengths:
    try:
        up32 = mel(clip[:L], 80)
    except RuntimeError as e:
        print("length", L, "upstream raises:", str(e).splitlines()[0])
        frames.append(-1)
        # numpy's reflect mirrors as often as the pad needs, so the
        # reference has the frames upstream cannot compute.
        if L >= 160:
            r64 = ref64(clip[:L], 80)
            assert len(r64) == L // 160, (L, r64.shape)
            arrays[f"len{L}.ref64"] = np.ascontiguousarray(r64, dtype=np.float64)
        continue
    r64 = ref64(clip[:L], 80)
    assert len(up32) == len(r64) == L // 160, (L, up32.shape, r64.shape)
    print("length", L, "frames", len(up32), "max |up32 - ref64|",
          np.abs(up32 - r64).max() if len(up32) else 0.0)
    frames.append(len(up32))
    arrays[f"len{L}.up32"] = np.ascontiguousarray(up32, dtype=np.float32)
    arrays[f"len{L}.ref64"] = np.ascontiguousarray(r64, dtype=np.float64)
arrays["clip"] = clip
arrays["lengths"] = np.array(lengths, dtype=np.int64)
arrays["frames"] = np.array(frames, dtype=np.int64)
np.savez(os.path.join(out, "audio.npz"), **arrays)

v = np.arange(-128, 128, dtype=np.int16)
path = os.path.join(out, "int8.flac")
sf.write(path, v * 256, 16000, subtype="PCM_S8", format="FLAC")
back, sr = sf.read(path, dtype="int16")
assert sr == 16000 and np.array_equal(back, v * 256), "int8.flac"
print("int8.flac", sf.info(path).subtype)
