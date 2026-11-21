# Performance

How Raven compares with the fastest stacks we know, on the workloads the road
to 1.0 is judged by. Each section names its machine, the versions compared
and the commit measured, and the benchmark code is in the repository.

## The goal post

Beta 1's goal post, JIT compilation on GPU, is met. The next goal post on the
road to 1.0 is gpt-oss-20b, jit-compiled, at least as fast as
tinygrad, JAX, llama.cpp and vLLM on NVIDIA, through CUDA and through
NVIDIA's kernel driver directly ("NV"), and on AMD. We are not there yet.

Each cell compares decoding one token at position 512 and prefilling 512
tokens, in tokens a second, with Raven storing the attention projections and
the head as the baseline does: Q8_0 against llama.cpp, bfloat16 against vLLM.
On NVIDIA the baselines run through CUDA.

| As fast as | NVIDIA, CUDA | NVIDIA, NV | AMD |
| --- | --- | --- | --- |
| llama.cpp | behind on decode, 166.1 against 168.9; ahead on prefill, 11,630 against 9,700 | ahead on decode, 171.8 against 168.9; ahead on prefill, 11,530 against 9,700 | behind on decode, 12.7 against 10.38 ms a token |
| tinygrad | blocked: tinygrad has no gpt-oss implementation to run | blocked: tinygrad has no gpt-oss implementation to run | blocked: tinygrad needs Resizable BAR on this machine |
| JAX | blocked: no MXFP4 path, and bfloat16 does not fit in 32 GB | blocked: no MXFP4 path, and bfloat16 does not fit in 32 GB | not measured |
| vLLM | ahead on decode, 124.5 against 122.1; level on prefill, 11,970 against 11,930, within their ranges | ahead on decode, 130.0 against 122.1; ahead on prefill, 12,100 against 11,930 | not measured |

The NVIDIA columns are from 2026-11-19 and 2026-11-21, below. The AMD column
is from 2026-11-08, on a Radeon AI PRO R9700 (machine "nonnormal", gfx1201)
at 502431b668, against llama.cpp b12048 through Vulkan, and has not been
measured since.

### gpt-oss-20b on an NVIDIA RTX 5000 Ada

Machine "fudge", Raven at eb532024b8 and llama.cpp b12541 on 2026-11-21,
prefills on 2026-11-21 at 0df67735d8 with vLLM's, first tokens on 2026-11-20
at 69f14bbeee, vLLM 0.34.0's decode and first token on 2026-11-19. Tokens a
second for one sequence, the median of three rounds. Every prefill is timed
back to back, a call after the one before, with the board at its power
limit. First token is a new process's, from its start to its first token,
loading the weights. Every
system keeps the experts as MXFP4.
[gpt-oss's performance](../apps/gpt-oss/PERFORMANCE.md) gives each number's
range, how each system was run, memory and the other systems, and how to
reproduce them.

| | Attention, head | Decode | Prefill, 512 tokens | First token |
| --- | --- | --- | --- | --- |
| Raven, CUDA | bfloat16 | 124.5 | 11,970 | 1.78 s |
| Raven, CUDA | Q8_0 | 166.1 | 11,630 | not measured |
| Raven, NV | bfloat16 | 130.0 | 12,100 | 1.52 s |
| Raven, NV | Q8_0 | 171.8 | 11,530 | not measured |
| llama.cpp, CUDA | Q8_0 | 168.9 | 9,700 | 1.73 s |
| vLLM | bfloat16 | 122.1 | 11,930 | 19.7 s |

Installed, Raven's chat is 32 MB, plus NVRTC (134 MB) to compile kernels it
has not cached; llama.cpp is 1.15 GB and vLLM 9.32 GB, measured on 2026-11-19
at 72377cdbe1 ([gpt-oss without Python](../apps/gpt-oss/README.md#without-python)).

## The GPU runtime: rig

Medians of release builds, 2026-10-11, from
[rig's comparison](../packages/rig/bench/compare/README.md). On NVIDIA, an RTX
5000 Ada, rig and tinygrad each run through their own driver and through
CUDA, in that order. On AMD, a Radeon AI PRO R9700, PyTorch is the only
framework run. Kernels are precompiled: compiling them at run time loads
NVRTC on NVIDIA or comgr on AMD.

| Measurement | rig: own driver / CUDA | PyTorch | JAX | tinygrad: own driver / CUDA |
| --- | --- | --- | --- | --- |
| Install size, NVIDIA | 7.6 / 115 MiB | 5,458 MiB | 5,774 MiB | 461 / 496 MiB |
| Install size, AMD | 7.6 MiB | 15,045 MiB | | |
| Time to first result, NVIDIA | 70 / 134 ms | 720 ms | 538 ms | 336 / 279 ms |
| Time to first result, AMD | 53 ms | 1,106 ms | | |
| Time per launch, NVIDIA | 2.1 / 3.2 µs | 7.7 µs | 28.9 µs | 511 / 447 µs |
| Time per launch, AMD | 4.9 µs | 8.1 µs | | |
| Loading a 256 MiB file, NVIDIA | 46.0 / 49.3 ms | 32.9 ms | 169.4 ms | 62.0 / 99.9 ms |
| Loading a 256 MiB file, AMD | 38.9 ms | 215.9 ms | | |

## Tokenization: brot

OpenWebText, one thread, 2026-11-03 on fudge (shared machine), from
[brot's benchmarks](../packages/brot/bench/README.md). The reference is
gigatoken, except for BERT-base, which gigatoken refuses: there it is HF
tokenizers.

| Pipeline | brot | Reference |
| --- | --- | --- |
| GPT-2 | 474.4 MB/s | 816.6 MB/s (gigatoken) |
| LLaMA | 190.1 MB/s | 71.4 MB/s (gigatoken) |
| Mistral | 203.5 MB/s | 145.4 MB/s (gigatoken) |
| BERT-base | 62.7 MB/s | 2.7 MB/s (HF tokenizers) |

## Dataframes: talon

Against the faster of DuckDB 1.5.6 and Polars 1.44.2, at 9b82c5b57a, under
load. Talon uses 3–10× the CPU. See
[talon's query benchmarks](../packages/talon/bench/queries/README.md).

| Workload | talon |
| --- | --- |
| Decision support, SF1 | 6.6× slower (geomean) |
| H2O group-by, 10⁷ rows | 2.1× slower |
| H2O joins | 2.0× slower |
