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
"""Resampling goldens for test_audio, written to OUT: files of other rates and
channel counts, and upstream's load_audio of each (openai/whisper 86098128,
whisper/audio.py: ffmpeg -ac 1 -ar 16000 to s16le), on imageio-ffmpeg's
static ffmpeg.
- stereo44.wav: 44.1 kHz, 2 channels, 16 bits, 0.25 s (plain PCM).
- mono48.flac: 48 kHz, 1 channel, 16 bits, 0.25 s.
- mono8.wav: 8 kHz, 1 channel, 16 bits, 0.1 s (upsampling).
- stereo16.wav: 16 kHz, 2 channels, 16 bits, 0.1 s (downmix alone).
- surround48.flac: 48 kHz, 6 channels (FLAC's 5.1 order), 24 bits, 0.02 s.
- three22.wav: 22.05 kHz, 3 channels, plain PCM without a channel mask,
  16 bits, 0.05 s (ffmpeg guesses 2.1).
- mono43650.wav, mono47530.wav, mono44101.wav: 1 channel, 16 bits, 0.1 s;
  stereo96001.wav: 2 channels, 16 bits, 0.1 s.
- resample.npz: "<file>" the int16 samples load_audio gives for each file;
  "short.x" a 44.1 kHz int16 clip and "short.<L>" load_audio of a mono WAV
  of its first L samples, for each L in "short.lengths".

    uv run apps/whisper/test/gen/resample.py OUT

OUT is apps/whisper/test/upstream.
"""
import os, sys, wave

import numpy as np
import soundfile as sf
import whisper

from audio_common import put_ffmpeg_on_path, s16_of

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
print(put_ffmpeg_on_path())
rng = np.random.default_rng(5)


def signal(rate, seconds, channels, peak):
    # Tones across the band, some past 8 kHz where the filter rejects, and
    # noise, each channel its own; at most [peak] of full scale.
    n = int(round(rate * seconds))
    t = np.arange(n) / rate
    cols = []
    for c in range(channels):
        f = [220.0 * (c + 1), 3100.0 + 700.0 * c, 7300.0, 11000.0 + 900.0 * c]
        f = [v for v in f if v < rate / 2]
        x = sum(np.sin(2 * np.pi * v * t + c) for v in f) / len(f)
        x = 0.7 * x + 0.3 * rng.uniform(-1, 1, n)
        cols.append(peak * x)
    return np.stack(cols, axis=1)


def to_int(x, bits):
    return np.round(x * 2 ** (bits - 1)).astype(np.int32)


def write_plain_wav(path, rate, x16):
    # Python's wave writes WAVE_FORMAT_PCM, which states no channel mask.
    with wave.open(path, "wb") as w:
        w.setnchannels(x16.shape[1])
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(x16.astype("<i2").tobytes())


arrays = {}


def record(name, path):
    a = s16_of(whisper.load_audio(path))
    info = sf.info(path)
    n = info.frames
    want = -(-n * 16000 // info.samplerate)
    print(f"{name}: {info.samplerate} Hz, {info.channels} ch, {info.subtype}, "
          f"{n} rows -> {len(a)} samples (ceil(n 16000 / rate) = {want})")
    arrays[name] = a


cases = [
    ("stereo44.wav", 44100, 0.25, 2, 16),
    ("mono48.flac", 48000, 0.25, 1, 16),
    ("mono8.wav", 8000, 0.1, 1, 16),
    ("stereo16.wav", 16000, 0.1, 2, 16),
    ("surround48.flac", 48000, 0.02, 6, 24),
    ("three22.wav", 22050, 0.05, 3, 16),
]
for name, rate, seconds, channels, bits in cases:
    path = os.path.join(out, name)
    x = to_int(signal(rate, seconds, channels, 0.8), bits)
    if name == "three22.wav":
        write_plain_wav(path, rate, x)
    elif name.endswith(".flac"):
        sf.write(path, x.astype(np.int16) if bits == 16 else x << 8, rate,
                 subtype="PCM_16" if bits == 16 else "PCM_24", format="FLAC")
    else:
        sf.write(path, x.astype(np.int16), rate, subtype="PCM_16", format="WAV")
    back, r = sf.read(path, dtype="int32", always_2d=True)
    assert r == rate and np.array_equal(back >> (32 - bits), x), name
    record(name, path)

lengths = [1, 2, 3, 10, 45, 46, 47, 91, 92, 93, 200, 441, 1000]
clip = to_int(signal(44100, max(lengths) / 44100, 1, 0.8), 16)[:, 0].astype(np.int16)
arrays["short.x"] = clip
arrays["short.lengths"] = np.array(lengths, dtype=np.int64)
for L in lengths:
    path = os.path.join(out, "tmp-short.wav")
    sf.write(path, clip[:L], 44100, subtype="PCM_16", format="WAV")
    a = s16_of(whisper.load_audio(path))
    print(f"short {L} -> {len(a)} (ceil = {-(-L * 16000 // 44100)})")
    arrays[f"short.{L}"] = a
    os.remove(path)

# Rates drawn after the clip, so the files above keep their samples: 43650
# and 47530 Hz, where swr's tap count (its float order from the raw rates)
# differs from the exact 32 / f; 44101 and 96001 Hz, whose ratio to 16 kHz
# reduces to more than 1024 phases, where swr interpolates between 1024.
for name, rate, seconds, channels, bits in [
    ("mono43650.wav", 43650, 0.1, 1, 16),
    ("mono47530.wav", 47530, 0.1, 1, 16),
    ("mono44101.wav", 44101, 0.1, 1, 16),
    ("stereo96001.wav", 96001, 0.1, 2, 16),
]:
    path = os.path.join(out, name)
    x = to_int(signal(rate, seconds, channels, 0.8), bits)
    sf.write(path, x.astype(np.int16), rate, subtype="PCM_16", format="WAV")
    record(name, path)

np.savez(os.path.join(out, "resample.npz"), **arrays)
for f in sorted(os.listdir(out)):
    print(f, os.path.getsize(os.path.join(out, f)))
