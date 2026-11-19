# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "openai-whisper @ git+https://github.com/openai/whisper.git@86098128c0b4f24f0e2aa2994de830614b474227",
#   "torch==2.15.0",
#   "numpy==2.3.5",
#   "soundfile==0.13.1",
#   "jiwer==4.0.0",
# ]
# ///
"""openai-whisper's side of PERFORMANCE.md: bench_whisper.ml's rows, by
upstream's own transcribe at the commit the tests compare with.

    uv run apps/whisper/bench/openai_whisper.py cuda|cpu [--fp16] [MODEL...]

It reads fetch.py's files: the same checkpoints raven reads, and the
recording. transcribe runs at upstream's defaults in English, as raven's
rows do: greedy with the fallback ladder, conditioned on the text before,
timestamps on. The rows, each in milliseconds and times real time:
- minute/MODEL: the recording's first minute, the median of 5 calls after 1;
- recording/MODEL: the whole recording, once, its text written beside it
  for wer.py;
- first-transcript/MODEL: a new process from its start to the first
  minute's text, loading the model included, the median of 3 after 1;
and memory/MODEL, the device memory in use after the minute rows (CUDA), or
the process's peak resident memory (CPU); and encoder/MODEL, the encoder
alone on the clip's first window, the median of 5 calls after 1. Models
default to tiny, small and large-v3-turbo. --fp16 runs float16, upstream's
default on a GPU; the rows compare at float32.
"""
import os, resource, statistics, subprocess, sys, time
import numpy as np
import soundfile
import torch
import whisper

DIR = os.path.join(
    os.environ.get("RAVEN_CACHE_ROOT")
    or os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "raven"),
    "whisper")
MODELS = ["tiny", "small", "large-v3-turbo"]
CLIP = 60 * 16000


def audio():
    x, rate = soundfile.read(os.path.join(DIR, "recording.wav"), dtype="float32")
    assert rate == 16000
    return x


# Upstream keeps float32 weights and casts them to float16 at each call
# under fp16=True.
def load(name, device):
    return whisper.load_model(os.path.join(DIR, name + ".pt"), device=device)


def transcribe(m, x, fp16):
    return whisper.transcribe(m, x, language="en", fp16=fp16, verbose=None)["text"]


def row(name, seconds, audio_seconds, n):
    m = statistics.median(seconds)
    print(f"{name:<28} {m * 1e3:10.1f} ms  {audio_seconds / m:7.1f}x real time  "
          f"({min(seconds) * 1e3:.1f}-{max(seconds) * 1e3:.1f} ms, {n} calls)", flush=True)


# The device memory in use, every process's, on an otherwise empty GPU; on the
# host the process's peak resident memory, which the models, run smallest
# first, raise in turn.
def memory(name, device):
    if device == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"{'memory/' + name:<28} {(total - free) / 1e9:10.2f} GB in use", flush=True)
    else:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        print(f"{'memory/' + name:<28} {rss / 1e9:10.2f} GB peak resident", flush=True)


def timed(f):
    t0 = time.perf_counter(); f(); return time.perf_counter() - t0


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    fp16 = "--fp16" in sys.argv
    if args[:1] == ["first"]:  # A first transcript's process.
        _, name, device = args
        m = load(name, device)
        transcribe(m, audio()[:CLIP], fp16)
        return
    device, models = args[0], args[1:] or MODELS
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    print(f"openai-whisper 86098128, torch {torch.__version__}, {device}, "
          f"{'float16' if fp16 else 'float32'}, {torch.get_num_threads()} threads", flush=True)
    x = audio()
    clip, seconds = x[:CLIP], len(x) / 16000
    for name in models:
        m = load(name, device)
        transcribe(m, clip, fp16)
        row(f"minute/{name}", [timed(lambda: transcribe(m, clip, fp16)) for _ in range(5)], 60, 5)
        memory(name, device)
        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(clip), m.dims.n_mels, device=device)
        mel = mel[None].to(torch.float16 if fp16 else torch.float32)
        encode = torch.no_grad()(lambda: m.embed_audio(mel).sum().item())
        encode()
        row(f"encoder/{name}", [timed(encode) for _ in range(5)], 30, 5)
        text = []
        row(f"recording/{name}", [timed(lambda: text.append(transcribe(m, x, fp16)))], seconds, 1)
        precision = "-fp16" if fp16 else ""
        with open(os.path.join(DIR, f"openai-whisper-{name}-{device}{precision}.txt"), "w") as f:
            f.write(text[0])
        del m
        if device == "cuda":
            torch.cuda.empty_cache()
        first = [timed(lambda: subprocess.run(
            [sys.executable, __file__, "first", name, device] + (["--fp16"] if fp16 else []),
            check=True)) for _ in range(4)][1:]
        row(f"first-transcript/{name}", first, 60, 3)


main()
