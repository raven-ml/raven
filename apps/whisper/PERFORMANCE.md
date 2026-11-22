# Whisper's performance

On an NVIDIA RTX 5000 Ada at float32, raven transcribes speech faster than
openai-whisper, faster-whisper and transformers with each model: 408 times
real time with tiny, 124 with small and 89 with large-v3-turbo, against
faster-whisper's 341, 107 and 73 and openai-whisper's 202, 68 and 71.

## Transcribing on the GPU

Times real time: seconds of speech transcribed per second, the median of
three rounds and their range. Each system transcribes the first minute of a
LibriSpeech chapter, model loaded, in English, greedy with Whisper's fallback
to higher temperatures, timestamps on.

| | Precision | tiny | small | large-v3-turbo |
|---|---|---|---|---|
| raven | float32 | 408 (405–410) | 124 (124–124) | 89 (89–90) |
| openai-whisper | float32 | 202 (200–202) | 68 (68–68) | 71 (70–71) |
| faster-whisper | float32 | 341 (340–341) | 107 (107–107) | 73 (73–73) |
| transformers | float32 | 146 (145–147) | 57 (57–57) | 47 (47–47) |
| openai-whisper | float16 | 177 (177–178) | 60 (60–60) | 92 (91–92) |
| faster-whisper | float16 | 441 (395–444) | 163 (163–163) | 147 (147–149) |
| transformers | float16 | 164 (163–164) | 87 (86–87) | 112 (112–112) |
| whisper.cpp | float16 | 205 (205–206) | 113 (113–113) | 123 (122–123) |
| raven | float16 | not measured: `main.exe` loads float32 | | |

float16 is each reference's default on a GPU, and whisper.cpp's files hold
float16 weights; the float32 rows compare like with like. The whole 8-minute
chapter transcribes within 15% of the minute's rate at float32 on every
system (raven: 469, 133 and 90 times real time), and within 26% at float16.

## Transcribing on the CPU

Times real time on two cores of the host, the whole 8-minute chapter, the
median of three rounds and their range.

| | Precision | tiny | small |
|---|---|---|---|
| raven, eager | float32 | 42 (40–42) | 7.0 (7.0–7.1) |
| openai-whisper | float32 | 45 (43–46) | 7.4 (7.3–7.6) |
| faster-whisper | float32 | 48 (41–49) | 7.3 (7.0–7.5) |
| transformers | float32 | 25 (25–25) | 4.6 (3.9–4.7) |
| whisper.cpp | float16 | 22 (18–28) | 4.0 (2.5–4.1) |

raven runs the host's eager kernels, one call each; its encoder takes 1.1
to 1.2 times PyTorch's on a window (193 against 158 ms with tiny, 1.58
against 1.43 s with small). Other users' jobs shared the host's cores at
times, which the lower ends of the ranges show.

## Where the time goes

One minute with large-v3-turbo on the GPU, in milliseconds: the kernels of
its three encoder calls, those of its decoding, and the host's time with no
kernel running.

| | Encoder | Decoder | Host | Minute |
|---|---|---|---|---|
| raven | 327 | 365 | none | 672 |
| openai-whisper | 325 | 427 | 93 | 845 |

raven's encoder is level with PyTorch's: 109 against 110 ms a window (5
against 2 with tiny, 19 against 18 with small). PyTorch's convolution stem
runs on TF32 tensor cores, cuDNN's default, so that part of its encoder is
not IEEE float32. raven's decoder kernels take 0.85 times PyTorch's, and
its decoding loop is one compiled program (`Rune.iterate`), so the GPU
never waits on the host; upstream's Python loop leaves it idle 93 ms a
minute. raven's kernel sum is 20 ms above the minute's wall time: `DEBUG=2`
submits each kernel on its own.

## Starting and memory

A new process from its start to the first minute's transcript with
large-v3-turbo, the model read from a warm page cache, and the device memory
in use after the minute rows, the CUDA context included.

| | First transcript | Cold kernel cache | Device memory |
|---|---|---|---|
| raven | 2.76 s (2.72–2.79) | 7.09 s (7.03–7.20) | 4.11 GB |
| openai-whisper | 5.16 s (4.98–5.21) | | 5.58 GB |
| faster-whisper | 3.30 s (3.25–3.34) | | 6.06 GB |
| transformers | 4.99 s (4.93–4.99) | | 4.11 GB |
| whisper.cpp, float16 | 1.11 s (1.09–1.11) | | 1.86 GB of buffers stated, without the context |

The minute itself takes 0.67 s of raven's first transcript. With tiny,
raven's first transcript takes 1.03 s warm and 4.65 s cold, with small 1.95
and 10.4 s. The references ship their kernels compiled; raven compiles its
own on first use and caches them on disk, so the cold column is raven's
alone.

## How it was measured

- **Machine.** fudge: Intel Core Ultra 5 235, NVIDIA RTX 5000 Ada (sm_89,
  32 GB), driver 615.71.09, CUDA 13.4, Debian 13. Every process ran on two
  cores (`taskset`), the references with two threads. Other users' jobs
  shared the host, never the GPU: its process list was empty before and
  after every GPU run.
- **Recording.** Chapter 122797 of LibriSpeech's test-clean (speaker 672,
  8 min 16 s, CC BY 4.0), its utterances in order; the minute rows read its
  first minute, three windows, the third of 2 s.
- **Versions.** raven at main 50116b53ea. openai-whisper at 86098128 and
  transformers 5.23.0 on torch 2.15.0+cu132; faster-whisper 1.2.1 on
  CTranslate2 4.8.5; whisper.cpp v1.9.5, built with CUDA 13.4 for sm_89, on
  ggerganov/whisper.cpp's files at 5359861c. raven and openai-whisper read
  OpenAI's released checkpoints, the others their conversions of them.
- **Same work.** Every system decodes in English with one beam and one draw,
  Whisper's temperature fallback, its thresholds and timestamps, and without
  voice activity detection. torch's float32 products run at full precision,
  its default.
- **Statistic.** Three rounds, 2026-11-19 and 20, each running every system
  in turn; raven's and openai-whisper's float32 GPU rows, and raven's CPU
  rows, come from three later rounds at raven's commit. raven's minute rows
  are thumper's medians (5 samples), the others the scripts' (5 calls after
  1); first transcripts are the median of 3 processes after 1. The tables
  give the median of the three rounds and their range. The breakdown sums
  the kernel times of one encoder call and one minute, from raven's
  `DEBUG=2` and from `torch.profiler`, copies left out; the host's share is
  the minute's wall time less its kernels.
- **Not measured.** large-v3: on this chapter every implementation repeats
  itself on different windows and falls back there, so its rows time
  different work, and large-v3-turbo runs the same encoder. Also beam
  search, batched transcription, raven at float16, and AMD and Apple GPUs.

```bash
uv run apps/whisper/bench/fetch.py                # recording, checkpoints
WHISPER_DEVICE=cuda dune build @apps/whisper/bench/bench  # raven's minute rows
apps/whisper/bench/raven.sh cuda                  # chapter, first transcript, kernels
uv run apps/whisper/bench/openai_whisper.py cuda  # --fp16 for float16, cpu for the host
uv run apps/whisper/bench/fasterwhisper.py cuda
uv run apps/whisper/bench/hf_transformers.py cuda
apps/whisper/bench/whisper_cpp.sh cuda
```
