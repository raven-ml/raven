# gpt-oss's performance

On an NVIDIA RTX 5000 Ada, raven decodes gpt-oss-20b 2% faster than vLLM at
the checkpoint's precision and 2% behind llama.cpp packed as llama.cpp packs
it (2% ahead of it through NV), prefills 20% faster than llama.cpp and as
fast as vLLM (0.3% ahead through CUDA, 1.4% through NV), and gives a new
process its first token in 1.8 seconds, as llama.cpp does. On two of the
same machine's CPU cores it decodes 4.2 times slower than llama.cpp's CPU
build and prefills 1.5 times slower, and gives a first token twice as fast.

## Decoding and prefilling

Tokens a second, one sequence at a time, the median of three rounds and their
range. Decode is one token at position 512; prefill, a 512-token prompt; first
token, a new process from its start to its first token after a 12-token
prompt, loading the weights included. Every system keeps the experts as MXFP4,
and the second column says how it stores the attention projections and the
head.

| | Attention, head | Decode | Prefill 512 | First token |
|---|---|---|---|---|
| raven, CUDA | bfloat16 | 124.5 (124.4–124.5) | 11,970 (11,950–12,090) | 1.78 s (1.77–1.78) |
| raven, CUDA | Q8_0 | 166.1 (166.1–166.4) | 11,630 (11,620–11,740) | not measured |
| raven, NV | bfloat16 | 130.0 (130.0–130.0) | 12,100 (11,990–12,200) | 1.52 s (1.51–1.54) |
| raven, NV | Q8_0 | 171.8 (171.8–172.1) | 11,530 (11,530–11,640) | not measured |
| llama.cpp b12541, CUDA | Q8_0 | 168.9 (168.4–168.9) | 9,700 (9,700–9,710) | 1.73 s (1.72–1.73) |
| vLLM 0.34.0 | bfloat16 | 122.1 (122.1–122.1) | 11,930 (11,930–11,960) | 19.7 s (18.1–22.8) |
| transformers 5.23.0 | bfloat16 | 35.8 (35.6–36.6) | 5,480 (5,480–5,500) | 5.26 s (4.81–5.41) |
| OpenAI's PyTorch implementation | | does not fit in 32 GB: it upcasts the experts to bfloat16, about 42 GB | | |
| JAX ([jax-llm-examples](https://github.com/jax-ml/jax-llm-examples/tree/main/gpt_oss)) | | no MXFP4 path, and bfloat16 does not fit in 32 GB | | |
| tinygrad | | no inference implementation | | |
| AMD Radeon AI PRO R9700 | | not measured | | |

The Q8_0 rows do llama.cpp's work: its GGUF file of gpt-oss-20b stores the
attention projections and the head as Q8_0. The bfloat16 rows keep the
checkpoint as stored, as the chat, vLLM and transformers run it. NV is the
same card through NVIDIA's kernel driver, without CUDA: it decodes 3 to 4%
faster than CUDA and prefills as fast. Every prefill is timed back to back,
at the card's power limit, and is from 2026-11-21 at main 0df67735d8; the
first-token column is from 2026-11-20, at main 69f14bbeee: How it was
measured says why.

## Where decode's time goes

GPU time of one decode step with the attention and head packed as Q8_0, by
kernel, measured with Nsight Systems: raven's `decode-q8_0/512` over 461 steps,
llama.cpp's 65-token run at depth 512 less its 1-token run. Bandwidth is the
bytes of the weights each product reads.

| | raven, CUDA | llama.cpp, CUDA |
|---|---|---|
| Expert products, MXFP4, 1.27 GB | 2.53 ms, 502 GB/s | 2.49 ms, 510 GB/s |
| Attention projections, Q8_0, 0.68 GB | 1.45 ms, 468 GB/s | 1.41 ms, 479 GB/s |
| Head, Q8_0, 0.62 GB | 1.16 ms, 531 GB/s | 1.16 ms, 532 GB/s |
| Everything else | 0.91 ms in 439 kernels | 0.80 ms in 508 kernels |
| The step | 6.04 ms in 536 kernels | 5.86 ms in 653 kernels |

The products that read the weights run within 3% of llama.cpp's, near the
card's 576 GB/s. Most of raven's 3% lies in the small kernels around them,
the attention itself, the norms and the routing among them. Neither side
waits on launches: a step's kernel time is within 2% of its decode time in
the first table.

## Starting and memory

A new process's first token over an empty kernel cache, which compiles every
kernel, and the device memory in use while decoding, the CUDA context
included. llama.cpp, vLLM and transformers ship their kernels compiled; vLLM's
first token is with its compilation cache filled by an earlier process.

| | First token, cold | Device memory |
|---|---|---|
| raven, CUDA | 3.44 s (3.43–3.46) | 14.4 GB, 13.3 GB with Q8_0 |
| raven, NV | 3.19 s (3.19–3.21) | 14.2 GB, 13.1 GB with Q8_0 |
| llama.cpp | no kernel cache: 1.73 s | 12.4 GB |
| vLLM | about 110 s, once | 32.4 GB: it reserves 90% of the card |
| transformers | not measured | 14.6 GB |

Compiling raven's kernels adds about 1.7 s to a first token. vLLM's first
process on a machine starts its engine, compiling and warming its kernels and
CUDA graphs, in 72 s, and later ones in 13 s.

## On the host

On two P-cores of the same machine's Core Ultra 5 235, raven decodes
gpt-oss-20b 4.2 times slower than llama.cpp's CPU build and prefills 1.5
times slower, and gives a new process its first token 2.1 times faster. The
rows are the first table's, in tokens a second, on the host.

| | Attention, head | Decode | Prefill 512 | First token |
|---|---|---|---|---|
| raven, CPU | bfloat16 | 2.89 (2.88–2.90) | 24.6 (24.6–24.6) | 2.61 s (2.60–2.62) |
| raven, CPU | Q8_0 | 2.64 (2.64–2.65) | 24.6 (24.5–24.6) | not measured |
| llama.cpp b12541, CPU | Q8_0 | 12.1 (12.0–12.2) | 36.5 (36.4–36.6) | 5.37 s (5.37–5.48) |

Kernel time of one decode step at position 512 and one 512-token prefill,
the attention and head as bfloat16, from tolk's DEBUG=2 times (an untimed
run on two other P-cores, whose step took 27% longer than the table's).
Bandwidth is the bytes of the weights each product reads.

| | Decode | Prefill 512 |
|---|---|---|
| Expert products, MXFP4, 1.27 GB a step | 332 ms (76%), 3.8 GB/s | 11.2 s (53%) |
| Attention projections, bfloat16, 1.27 GB | 46 ms (11%), 27.6 GB/s | 6.0 s (29%) |
| Head, bfloat16, 1.16 GB | 40 ms (9%), 28.8 GB/s | 0.04 s |
| Everything else | 21 ms in 439 kernels | 3.8 s in 731 kernels |
| The step | 439 ms in 536 kernels | 21.1 s in 828 kernels |

The bfloat16 projections and the head stream their weights at what two
P-cores read from memory. The expert products read theirs at a seventh of
it, bound by the work in their tiles: raven decodes each 4-bit code to
float32 and multiplies floats, 8 lanes a multiply-add, its loops waiting on
the shuffles that put the decoded codes and the rows into lanes.
llama.cpp's `ggml_vec_dot_mxfp4_q8_0` (ggml-cpu/arch/x86/quants.c at
b12541) quantizes the activations to Q8_0 once and multiplies integers: for
each block of 32 values a byte shuffle looks the codes up as signed bytes
and one `maddubs` makes their 32 products, and one float multiply-add
applies both blocks' scales, so its step reads its 2.6 GB of weights at
about 31 GB/s, the memory's rate. raven's Q8_0 attention is slower than
its bfloat16 on the host for the same reason, the codes decoded in the
tile. In the prefill each weight serves many rows and the products are
bound by the cores' arithmetic on both sides: raven's expert products take
half of its time, decoded and multiplied as floats, where llama.cpp
multiplies them as integers, as above.

## How it was measured

- **Machine.** fudge: Intel Core Ultra 5 235, NVIDIA RTX 5000 Ada (sm_89,
  32 GB), driver 615.71.09, CUDA 13.4, Debian 13. Other users' jobs shared the
  host, at a load of 6 to 11 on its 14 cores, never the GPU: its process list
  was empty before and after every run. Each timed process ran on two P-cores
  (`taskset`), 2 and 3, which other work kept at most 15% busy before and
  after each run (4 and 5 on 2026-11-20).
- **Versions.** raven at main eb532024b8, but for the prefill cells, at main
  0df67735d8, and the first-token cells, warm and cold, at main 69f14bbeee.
  llama.cpp b12541's release binaries (CUDA 12.8 build) on ggml-org/gpt-oss-20b-GGUF at 41e8a94d. vLLM 0.34.0
  (torch 2.14.0+cu130), transformers 5.23.0 (torch 2.15.0+cu132, kernels
  0.18.3). raven, vLLM and transformers read openai/gpt-oss-20b at 6cee5e81.
- **Statistic.** Three rounds on 2026-11-21, each running raven over CUDA,
  raven over NV and llama.cpp in a different order. A round counted where
  llama.cpp's cells came within 5% of its rounds of 2026-11-20 and raven's
  spreads matched theirs; three more rounds, with each system's weights alone
  in the page cache, agreed within 1%. No round met that check for the first
  token, which loads 12 GB of weights through the host: llama.cpp's took 1.92
  to 2.28 s against 1.73 under the other jobs' load. The first-token cells
  are therefore the three rounds of 2026-11-20; the device memory, read again,
  has not moved. vLLM's and transformers' three rounds are from 2026-11-19,
  interleaved with raven at 3c1daa79d5, whose bfloat16 decode over CUDA was
  1% slower than this one's and its prefill 1% faster; vLLM's cold start is
  from a round before them. The prefill cells of raven and vLLM are three
  rounds of one hold on 2026-11-21, each running raven over CUDA, raven over
  NV, vLLM and llama.cpp's prefill in a rotated order, with the board's
  clock and power read every 100 ms; llama.cpp's prefill, the control, came
  within 1.2% of its cell, 9,590 to 9,680. raven's cells are thumper's
  medians (20 samples after 3 s of warmup; 3 processes for the first token),
  the others the scripts' (30 calls for llama.cpp's decode, 10 for its
  prefill, 20 calls after 3 s of warmup for vLLM's, 3 processes for a first
  token). The table gives the median of the three rounds and their range.
- **Prefill, back to back.** Every system's prefill runs a call after the one
  before, after a warmup of the same call, as a server's sustained prompts
  do, and llama-bench's do. The board then sits at its power limit: raven's
  prefills drew 239 W at 2,130 to 2,175 MHz, vLLM's 211 to 225 W at 2,310 to
  2,415 MHz, its host leaving the GPU idle about 3.5 ms of each 43. Timed
  right after a decode, as `vllm_offline.py` timed it until 2026-11-21, a
  prefill starts near the boost clock, about 2,730 MHz, and runs faster: in
  one run that day, raven's took 39.4 ms each after 128 decode steps and 42.2
  back to back, vLLM's 41.8 and 43.4. The prefill cells before these compared
  raven back to back with vLLM after a decode.
- **Same work.** Every system runs batch 1, greedy, from token ids.
  - llama.cpp's decode copies the full logits to the host, where raven reads
    back one token, and pads its KV cache.
  - vLLM's decode is a 129-token generation less a 1-token one, over 128: the
    mean over positions 512 to 639.
  - Prefill and first token end with the token read back on every system but
    llama.cpp, whose llama-bench stops at the logits.
- **Profile.** Nsight Systems 2026.3.2, CUDA trace only, CUDA graphs traced
  by node, one run a side, at main eb532024b8. Each of raven's five products
  is one program by its source's digest in a DEBUG=4 run of the row, and its
  parameters give the bytes it reads.
- **On the host.** raven at main e23edc73df with the bench's host device
  (`GPT_OSS_DEVICE=cpu`); llama.cpp b12541's CPU release build (its x86-64
  binaries, which load their Alder Lake backend), a thread on each of the two
  cores, on the same GGUF file. Both ran on P-cores 2 and 3, otherwise idle
  before and after each run, under the machine's timing lock; each round ran
  raven's rows after an untimed decode that brings its weights into the page
  cache, the order of the two alternating. Three rounds on 2026-11-21, the
  median and range given; a fourth replaced the first, in which llama.cpp's
  decode spread from 88 to 119 ms and its cells ran 5 to 20% past the others'.
  raven's cells are thumper's medians of 3 calls after the first, llama.cpp's
  of 10 decodes, 3 prefills and 3 processes. The profile is one run of
  `decode/512` and `prefill/512` under DEBUG=2, its first call left out.
- **Not measured.** The AMD card (raven, and llama.cpp's Vulkan and ROCm
  builds, which `llama_cpp.sh vulkan` and `rocm` run); batched decoding;
  llama.cpp at bfloat16, for which ggml-org publishes no file; vLLM and
  transformers with Q8_0, and either on the host; raven's first token on
  the host with Q8_0.

```bash
dune build @apps/gpt-oss/bench/bench   # raven, on the first GPU that opens
GPT_OSS_DEVICE=cpu dune build @apps/gpt-oss/bench/bench   # raven on the host
apps/gpt-oss/bench/llama_cpp.sh        # llama.cpp
apps/gpt-oss/bench/llama_cpp.sh cpu    # llama.cpp on the host
uv run apps/gpt-oss/bench/vllm_offline.py
uv run apps/gpt-oss/bench/hf_transformers.py
```

With `CUDA_VISIBLE_DEVICES=` set, the bench finds no CUDA device and measures
the card through NV.
