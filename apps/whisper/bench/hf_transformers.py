# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "torch==2.15.0",
#   "transformers==5.23.0",
#   "accelerate==1.15.0",
#   "numpy==2.3.5",
#   "soundfile==0.13.1",
# ]
# ///
"""HF transformers' side of PERFORMANCE.md: bench_whisper.ml's rows, by
WhisperForConditionalGeneration on each release's Hugging Face repository
(openai/whisper-MODEL).

    uv run apps/whisper/bench/hf_transformers.py cuda|cpu [--fp16] [MODEL...]

It reads fetch.py's recording. generate runs in English as transformers
documents long-form transcription with upstream's fallback: the
temperatures 0 to 1 by 0.2, the thresholds of compression ratio 2.4 (zlib),
log probability -1 and no speech 0.6, conditioned on the text before,
timestamps on, greedy. The rows, each in milliseconds and times real time:
- minute/MODEL: the recording's first minute, the median of 5 calls after 1;
- recording/MODEL: the whole recording, once, its text written beside it
  for wer.py;
- first-transcript/MODEL: a new process from its start to the first
  minute's text, loading the model included, the median of 3 after 1;
and memory/MODEL, the device memory in use after the minute rows (CUDA), or
the process's peak resident memory (CPU). Models default to tiny, small
and large-v3-turbo. The model runs at float32, or float16 with
--fp16; transformers' attention is SDPA, its default.
"""
import os, resource, statistics, subprocess, sys, time
import soundfile
import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration

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


def load(name, device, fp16):
    repo = "openai/whisper-" + name
    m = WhisperForConditionalGeneration.from_pretrained(
        repo, dtype=torch.float16 if fp16 else torch.float32).to(device).eval()
    return AutoProcessor.from_pretrained(repo), m


@torch.no_grad()
def transcribe(pm, x):
    p, m = pm
    f = p(x, sampling_rate=16000, return_tensors="pt", truncation=False, padding="longest",
          return_attention_mask=True)
    if f.input_features.shape[-1] < 3000:  # Short-form: one padded window.
        f = p(x, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
    f = f.to(m.device, m.dtype)
    ids = m.generate(**f, language="en", task="transcribe", return_timestamps=True,
                     condition_on_prev_tokens=True, temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                     compression_ratio_threshold=2.4, logprob_threshold=-1.0,
                     no_speech_threshold=0.6)
    return p.batch_decode(ids, skip_special_tokens=True)[0]


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
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    if args[:1] == ["first"]:  # A first transcript's process.
        _, name, device = args
        transcribe(load(name, device, fp16), audio()[:CLIP])
        return
    device, models = args[0], args[1:] or MODELS
    import transformers
    print(f"transformers {transformers.__version__}, torch {torch.__version__}, {device}, "
          f"{'float16' if fp16 else 'float32'}, {torch.get_num_threads()} threads", flush=True)
    x = audio()
    clip, seconds = x[:CLIP], len(x) / 16000
    for name in models:
        m = load(name, device, fp16)
        transcribe(m, clip)
        row(f"minute/{name}", [timed(lambda: transcribe(m, clip)) for _ in range(5)], 60, 5)
        memory(name, device)
        text = []
        row(f"recording/{name}", [timed(lambda: text.append(transcribe(m, x)))], seconds, 1)
        precision = "-fp16" if fp16 else ""
        with open(os.path.join(DIR, f"hf-transformers-{name}-{device}{precision}.txt"), "w") as f:
            f.write(text[0])
        del m
        if device == "cuda":
            torch.cuda.empty_cache()
        first = [timed(lambda: subprocess.run(
            [sys.executable, __file__, "first", name, device] + (["--fp16"] if fp16 else []),
            check=True)) for _ in range(4)][1:]
        row(f"first-transcript/{name}", first, 60, 3)


main()
