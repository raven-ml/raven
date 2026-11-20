# whisper

OpenAI's Whisper on raven: speech recognition, and translation into
English, by the released checkpoints. A recording is read as Whisper hears
it, one channel at 16 kHz, and transcribed 30 seconds at a time: log-mel
features, the encoder, then greedy, sampled or beam decoding with upstream's
fallback to higher temperatures. The transcript is written as text,
subtitles (VTT, SRT), tab-separated values or JSON, as upstream's `whisper`
command writes them.

On an NVIDIA RTX 5000 Ada at float32, raven transcribes speech at 374 times
real time with tiny, 120 with small and 65 with large-v3-turbo, against
openai-whisper's 202, 68 and 71 and faster-whisper's 341, 107 and 73.
[PERFORMANCE.md](PERFORMANCE.md) has every number and how to reproduce it.

```bash
dune exec apps/whisper/bin/main.exe -- --model tiny.en.pt --tokenizer ranks talk.flac
```

prints each segment as its window is decided,

```
[00:00.000 --> 00:02.000]  No, wait.
```

and writes `talk.txt`, `talk.vtt`, `talk.srt`, `talk.tsv` and `talk.json`
(`-o DIR` writes them elsewhere, `-f FORMAT` one format). The audio is WAV or
FLAC, at any rate, of up to 8 channels. The options keep upstream's names and
defaults; `main.exe --help` lists them. `AGENTS.md` lists where the port
departs from upstream, and why.

| File | |
|---|---|
| `lib/audio.mli` | a WAV or FLAC file's samples at 16 kHz, one channel; the log-mel spectrogram |
| `lib/tokenizer.mli` | the rank files' encodings and Whisper's special tokens |
| `lib/model.mli` | dimensions, parameters, checkpoints, the encoder and the decoder |
| `lib/decoding.mli` | greedy, sampled and beam decoding of a window; language detection |
| `lib/transcript.mli` | a recording of any length, window by window, with the fallback ladder |
| `lib/writer.mli` | text, VTT, SRT, TSV and JSON, as upstream writes them |
| `lib/english.mli`, `lib/wer.mli` | the English text normaliser of Whisper's evaluation; word error rates |
| `bin/main.ml` | the command |

## Without Python

The command is one executable that links only the C library.

| | Install |
|---|---|
| raven | 33 MB, plus NVRTC (134 MB) to compile kernels not yet cached |
| openai-whisper 86098128 | 5.84 GB, and ffmpeg |
| faster-whisper 1.2.1 | 2.40 GB |
| whisper.cpp v1.9.5 | 0.67 GB |
| transformers 5.23.0 | 5.74 GB |

Install is what transcription on the GPU needs, without the NVIDIA driver
and the weights: raven's `main.exe`, stripped, and CUDA's NVRTC, which raven
loads only to compile a kernel that is not in tolk's cache; whisper.cpp's
`whisper-cli` built for sm_89, with its libraries and CUDA's runtime and
cuBLAS; for the Python packages, a fresh uv environment with `bench/`'s pins,
CUDA's libraries included, and Python 3.12. Through NV, raven needs no
libcuda either. Measured on 2026-11-19 on an RTX 5000 Ada (driver 615.71.09,
CUDA 13.4, Debian 13), with raven at 72377cdbe1.

## Checkpoints

`--model` reads a checkpoint as OpenAI publishes it: a PyTorch file of the
model's dimensions and float16 weights. The SHA-256 of each file is the
directory of its URL.

| Model | URL |
|---|---|
| `tiny.en` | https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt |
| `tiny` | https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt |
| `base.en` | https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt |
| `base` | https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt |
| `small.en` | https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt |
| `small` | https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt |
| `medium.en` | https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt |
| `medium` | https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt |
| `large-v1` | https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt |
| `large-v2` | https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt |
| `large-v3`, `large` | https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt |
| `large-v3-turbo`, `turbo` | https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt |

`--tokenizer` names the directory of the rank files, from upstream's
`whisper/assets`: `gpt2.tiktoken` for an English-only model (`.en`),
`multilingual.tiktoken` for the others.

```bash
mkdir -p ranks
for f in gpt2 multilingual; do
  curl -L -o ranks/$f.tiktoken https://raw.githubusercontent.com/openai/whisper/86098128c0b4f24f0e2aa2994de830614b474227/whisper/assets/$f.tiktoken
done
curl -LO https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt
```

## Devices

`--device` says where the model runs:

- `cpu`, the default: eagerly on the host;
- `cpu-jit`: compiled by `Rune.jit` for the host;
- a GPU by name, compiled: `cuda`, `nv`, `amd` or `metal` for the first GPU
  of that kind that rig opens, `cuda:1` for the second.

## Tests

`dune build @apps/whisper/runtest` runs the tests, offline. They compare
each part with upstream's own code (openai/whisper at 86098128, on the CPU)
on trimmed rank tables, small random models and one LibriSpeech utterance,
from the goldens in `test/upstream`; `test/bin/main.t` runs the command. The
scripts in `test/gen` write every golden, each with `uv run` and its pins in
its header; `test/dune` names each test's.

`test/check/check.exe` compares the port with upstream on the full rank
files and the released checkpoints, outside runtest, since it reads the
weights and an oracle of upstream's outputs that the repository does not
hold. Each command prints a line per difference and their count, and exits 1
if there is one; `check.exe` without arguments lists each command's
arguments.

| Command | What it compares |
|---|---|
| `tokenizer` | encodings and decodings over the full rank files |
| `model` | the encoder's output, the first step's logits and the first greedy tokens of each utterance |
| `greedy` | whole greedy decodes of each utterance: tokens, sums, scores and the corpus word error rate |
| `beam` | beam searches: the finished candidates in order, the live beams, sums and word error rate |
| `e2e` | `greedy` from each utterance's FLAC file through `Audio`: its samples against ffmpeg's, its features against a float64 reference and upstream's |
| `transcribe` | long recordings through `Transcript` against upstream's `transcribe`: the loop replayed on upstream's attempts (`L`), the model at upstream's window states (`M`), and free running (`E`) |
| `excuse` | each token `greedy` or `transcribe` found differing from upstream's, against the bounds within which the timestamp rule or upstream's two greatest logits may decide otherwise |

```bash
dune build apps/whisper/test/check/check.exe
_build/default/apps/whisper/test/check/check.exe greedy WEIGHTS ASSETS \
  FEATURES GREEDY GREEDY64 SUBSET WER_TSV OUT tiny.en
```

The oracle is upstream at 86098128 on the CPU in float32, on LibriSpeech's
test-clean and test-other (https://www.openslr.org/12). `test/gen/subset.py`
picks its 219 utterances, the first three of each speaker, and
`test/gen/greedy.py` writes upstream's greedy decodes of them:

```bash
uv run apps/whisper/test/gen/subset.py LibriSpeech subset.tsv
uv run apps/whisper/test/gen/greedy.py tiny.en timestamps LibriSpeech subset.tsv weights greedy/tiny.en-timestamps.jsonl
```

The scripts of the other inputs are not in the repository. Each holds what
upstream computes, as `check.exe`'s readers take it:

| Input | Upstream's |
|---|---|
| `ORACLE_TOKENIZER`, `ORDINARY_JSON` | `Tokenizer`'s encodings and decodings of a corpus over the full rank files, and `encode_ordinary` of the texts its `encode` refuses |
| `FEATURES` | `log_mel_spectrogram` of each utterance's window, `ID.melN.short.npy` |
| `ORACLE_P1` | the encoder's output and the first step's logits of each utterance, `MODEL.ID.npz` |
| `GREEDY64` | the greedy decodes with the model in float64 |
| `WER_TSV` | the corpus word error rates of the greedy decodes |
| `BEAM` | beam searches of each utterance: candidates, beams and sums |
| `ORACLE` of `e2e` | `load_audio`'s samples from ffmpeg (`samples/`) and the float64 features (`ref64/`) |
| `ORACLE` of `transcribe` | `transcribe`'s decode calls on long recordings, with the recordings and their features |
