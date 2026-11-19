# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "faster-whisper==1.2.1",
#   "ctranslate2==4.8.5",
#   "nvidia-cublas-cu12==12.9.1.4",
#   "nvidia-cudnn-cu12==9.14.0.64",
#   "nvidia-ml-py==13.580.82",
#   "numpy==2.3.5",
#   "soundfile==0.13.1",
# ]
# ///
"""faster-whisper's side of PERFORMANCE.md: bench_whisper.ml's rows, by
CTranslate2's conversion of each release (Systran/faster-whisper-MODEL on
Hugging Face).

    uv run apps/whisper/bench/fasterwhisper.py cuda|cpu [--fp16] [MODEL...]

It reads fetch.py's recording. transcribe runs in English with one beam and
one draw (beam_size=1, best_of=1), as raven's rows decode, and its own
defaults otherwise: the fallback ladder, conditioned on the text before,
timestamps on, no voice activity filter. The rows, each in milliseconds and
times real time:
- minute/MODEL: the recording's first minute, the median of 5 calls after 1;
- recording/MODEL: the whole recording, once, its text written beside it
  for wer.py;
- first-transcript/MODEL: a new process from its start to the first
  minute's text, loading the model included, the median of 3 after 1;
and memory/MODEL, the device memory in use after the minute rows (CUDA, by
NVML), or the process's peak resident memory (CPU). Models default to tiny,
small, large-v3-turbo and large-v3. compute_type is float32, or float16 with
--fp16, faster-whisper's default on a GPU.
"""
import glob, os, resource, statistics, subprocess, sys, time

# ctranslate2 loads cuBLAS and cuDNN by name: the wheels' directories go on the
# loader's path before it starts.
if "FASTER_WHISPER_LIBS" not in os.environ:
    import nvidia.cublas, nvidia.cudnn
    libs = [os.path.join(p, "lib") for m in (nvidia.cublas, nvidia.cudnn) for p in m.__path__]
    env = dict(os.environ, FASTER_WHISPER_LIBS="1",
               LD_LIBRARY_PATH=":".join(libs + [os.environ.get("LD_LIBRARY_PATH", "")]))
    os.execve(sys.executable, [sys.executable] + sys.argv, env)

import soundfile
from faster_whisper import WhisperModel

DIR = os.path.join(
    os.environ.get("RAVEN_CACHE_ROOT")
    or os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "raven"),
    "whisper")
MODELS = ["tiny", "small", "large-v3-turbo", "large-v3"]
CLIP = 60 * 16000
THREADS = int(os.environ.get("THREADS", "4"))


def audio():
    x, rate = soundfile.read(os.path.join(DIR, "recording.wav"), dtype="float32")
    assert rate == 16000
    return x


def load(name, device, fp16):
    return WhisperModel(name, device=device, cpu_threads=THREADS,
                        compute_type="float16" if fp16 else "float32")


def transcribe(m, x):
    segments, _ = m.transcribe(x, language="en", beam_size=1, best_of=1, vad_filter=False)
    return "".join(s.text for s in segments)


def row(name, seconds, audio_seconds, n):
    m = statistics.median(seconds)
    print(f"{name:<28} {m * 1e3:10.1f} ms  {audio_seconds / m:7.1f}x real time  "
          f"({min(seconds) * 1e3:.1f}-{max(seconds) * 1e3:.1f} ms, {n} calls)", flush=True)


# The device memory in use, every process's, on an otherwise empty GPU; on the
# host the process's peak resident memory, which the models, run smallest
# first, raise in turn.
def memory(name, device):
    if device == "cuda":
        import pynvml
        pynvml.nvmlInit()
        info = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0))
        print(f"{'memory/' + name:<28} {info.used / 1e9:10.2f} GB in use", flush=True)
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
        transcribe(load(name, device, fp16), audio()[:CLIP])
        return
    device, models = args[0], args[1:] or MODELS
    import ctranslate2, faster_whisper
    print(f"faster-whisper {faster_whisper.__version__}, ctranslate2 {ctranslate2.__version__}, "
          f"{device}, {'float16' if fp16 else 'float32'}, {THREADS} threads", flush=True)
    x = audio()
    clip, seconds = x[:CLIP], len(x) / 16000
    for name in models:
        m = load(name, device, fp16)
        transcribe(m, clip)
        row(f"minute/{name}", [timed(lambda: transcribe(m, clip)) for _ in range(5)], 60, 5)
        memory(name, device)
        text = []
        row(f"recording/{name}", [timed(lambda: text.append(transcribe(m, x)))], seconds, 1)
        with open(os.path.join(DIR, f"faster-whisper-{name}-{device}.txt"), "w") as f:
            f.write(text[0])
        del m
        first = [timed(lambda: subprocess.run(
            [sys.executable, __file__, "first", name, device] + (["--fp16"] if fp16 else []),
            check=True)) for _ in range(4)][1:]
        row(f"first-transcript/{name}", first, 60, 3)


main()
