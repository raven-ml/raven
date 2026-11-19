"""What audio.py and resample.py share: upstream's load_audio on a real
ffmpeg, and the float64 reference of the scaled log-mel.
"""
import os, subprocess, tempfile

import imageio_ffmpeg
import numpy as np
import whisper
from whisper.audio import HOP_LENGTH, N_FFT

ASSETS = os.path.join(os.path.dirname(whisper.__file__), "assets")
FILTERS = np.load(os.path.join(ASSETS, "mel_filters.npz"))


def put_ffmpeg_on_path():
    # whisper.load_audio runs `ffmpeg` from PATH: link imageio-ffmpeg's
    # static build under that name, whatever ffmpeg the machine has.
    # Returns its version line.
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    d = tempfile.mkdtemp(prefix="ffmpeg-")
    os.symlink(exe, os.path.join(d, "ffmpeg"))
    os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
    out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
    return out.stdout.splitlines()[0]


def s16_of(audio):
    # load_audio's float32 is int16 / 32768, so the samples come back exactly.
    s = (audio.astype(np.float64) * 32768.0).astype(np.int16)
    assert np.array_equal(s.astype(np.float32) / 32768.0, audio)
    return s


def ref64(audio, n_mels, padding=0):
    # The float64 reference of log_mel_spectrogram(audio,
    # n_mels, padding): `padding` zeros appended, torch.stft(center=True,
    # reflect) with a periodic Hann, the last frame dropped, power, the npz's
    # Slaney filters widened, log10 above 1e-10, the max - 8 floor,
    # (x + 4) / 4. Frames first: [frames; n].
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
    return ((ls + 4.0) / 4.0).T


def trim(*xs):
    # The frames up to the first of the trailing run of equal frames (the
    # floor frames past the audio), the same cut for every array: a reader
    # repeats the last kept frame up to 3000.
    k = 0
    for x in xs:
        j = len(x) - 1
        while j > 0 and np.array_equal(x[j - 1], x[-1]):
            j -= 1
        k = max(k, j)
    return [x[: k + 1] for x in xs]
