# gpt-oss's performance

On an NVIDIA RTX 5000 Ada, raven decodes gpt-oss-20b 2% faster than vLLM at
the checkpoint's precision and 2% behind llama.cpp packed as llama.cpp packs
it (2% ahead of it through NV), prefills 21% faster than llama.cpp and 6%
slower than vLLM, and gives a new process its first token in 1.8 seconds, as
llama.cpp does.

## Decoding and prefilling

Tokens a second, one sequence at a time, the median of three rounds and their
range. Decode is one token at position 512; prefill, a 512-token prompt; first
token, a new process from its start to its first token after a 12-token
prompt, loading the weights included. Every system keeps the experts as MXFP4,
and the second column says how it stores the attention projections and the
head.

| | Attention, head | Decode | Prefill 512 | First token |
|---|---|---|---|---|
| raven, CUDA | bfloat16 | 124.5 (124.4–124.5) | 12,020 (11,990–12,160) | 1.78 s (1.77–1.78) |
| raven, CUDA | Q8_0 | 166.1 (166.1–166.4) | 11,710 (11,690–11,880) | not measured |
| raven, NV | bfloat16 | 130.0 (130.0–130.0) | 12,080 (12,060–12,120) | 1.52 s (1.51–1.54) |
| raven, NV | Q8_0 | 171.8 (171.8–172.1) | 11,520 (11,500–11,530) | not measured |
| llama.cpp b12541, CUDA | Q8_0 | 168.9 (168.4–168.9) | 9,700 (9,700–9,710) | 1.73 s (1.72–1.73) |
| vLLM 0.34.0 | bfloat16 | 122.1 (122.1–122.1) | 12,820 (12,790–12,900) | 19.7 s (18.1–22.8) |
| transformers 5.23.0 | bfloat16 | 35.8 (35.6–36.6) | 5,480 (5,480–5,500) | 5.26 s (4.81–5.41) |
| OpenAI's PyTorch implementation | | does not fit in 32 GB: it upcasts the experts to bfloat16, about 42 GB | | |
| JAX ([jax-llm-examples](https://github.com/jax-ml/jax-llm-examples/tree/main/gpt_oss)) | | no MXFP4 path, and bfloat16 does not fit in 32 GB | | |
| tinygrad | | no inference implementation | | |
| AMD Radeon AI PRO R9700 | | not measured | | |

The Q8_0 rows do llama.cpp's work: its GGUF file of gpt-oss-20b stores the
attention projections and the head as Q8_0. The bfloat16 rows keep the
checkpoint as stored, as the chat, vLLM and transformers run it. NV is the
same card through NVIDIA's kernel driver, without CUDA: it decodes 3 to 4%
faster than CUDA and prefills as fast. The first-token column is from
2026-11-20, at main 69f14bbeee: How it was measured says why.

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

## How it was measured

- **Machine.** fudge: Intel Core Ultra 5 235, NVIDIA RTX 5000 Ada (sm_89,
  32 GB), driver 615.71.09, CUDA 13.4, Debian 13. Other users' jobs shared the
  host, at a load of 6 to 11 on its 14 cores, never the GPU: its process list
  was empty before and after every run. Each timed process ran on two P-cores
  (`taskset`), 2 and 3, which other work kept at most 15% busy before and
  after each run (4 and 5 on 2026-11-20).
- **Versions.** raven at main eb532024b8, but for the first-token cells, warm
  and cold, at main 69f14bbeee. llama.cpp b12541's release binaries
  (CUDA 12.8 build) on ggml-org/gpt-oss-20b-GGUF at 41e8a94d. vLLM 0.34.0
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
  from a round before them. raven's cells are thumper's
  medians (20 samples after 3 s of warmup; 3 processes for the first token),
  the others the scripts' (30 calls for llama.cpp's decode, 10 for its
  prefill, 3 processes for a first token). The table gives the median of the
  three rounds and their range.
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
- **Not measured.** The AMD card (raven, and llama.cpp's Vulkan and ROCm
  builds, which `llama_cpp.sh vulkan` and `rocm` run); the CPU; batched
  decoding; llama.cpp at bfloat16, for which ggml-org publishes no file; vLLM
  and transformers with Q8_0.

```bash
dune build @apps/gpt-oss/bench/bench   # raven, on the first GPU that opens
apps/gpt-oss/bench/llama_cpp.sh        # llama.cpp
uv run apps/gpt-oss/bench/vllm_offline.py
uv run apps/gpt-oss/bench/hf_transformers.py
```

With `CUDA_VISIBLE_DEVICES=` set, the bench finds no CUDA device and measures
the card through NV.
