# Performance

How Raven compares with the fastest stacks we know, on the workloads the road
to 1.0 is judged by. Each section names its machine, the versions compared
and the commit measured, and the benchmark code is in the repository.

## The goal post

Beta 1's goal post, JIT compilation on GPU, is met. The next goal post on the
road to 1.0 is gpt-oss-20b, jit-compiled, at least as fast as
tinygrad, JAX, llama.cpp and vLLM on NVIDIA, through CUDA and through
NVIDIA's kernel driver directly ("NV"), and on AMD. We are not there yet.

| As fast as | NVIDIA, CUDA | NVIDIA, NV | AMD |
| --- | --- | --- | --- |
| llama.cpp | behind: decode 10.2 against 6.83 ms | behind: decode 12.0 against 6.83 ms (bench median; 10.7 ms of GPU time) | behind: decode 12.7 against 10.38 ms |
| tinygrad | not measured | not measured | blocked: tinygrad needs Resizable BAR on this machine |
| JAX | not measured | not measured | not measured |
| vLLM | not measured | not measured | not measured |

### gpt-oss-20b on an NVIDIA RTX 5000 Ada

Machine "fudge", under load. llama.cpp b12032, CUDA.

| Measurement | Raven, CUDA | Raven, NV | llama.cpp, CUDA |
| --- | --- | --- | --- |
| Decode, position 512 of a 1,024 cache | 10.2 ms (8ef1902f3) | 10.7 ms of GPU time; bench median 12.0 ms (31aa4521b) | 6.83 ms at depth 512 |
| Decode, with Q8_0 attention and head | 8.1–8.3 ms (f2af3251b) | not measured | |
| Prefill, 512 tokens | 122 ms, least of 4 rounds under load (4e23903949) | not measured again | 56.0 ms |
| First call, kernels cached | 0.37–0.39 s (c29acaee3) | 0.37–0.54 s, 12 tokens; 1.36 / 1.47 s at the first run (31aa4521b) | 75 ms (59 to 151 ms, no warmup) |

### gpt-oss-20b on an AMD Radeon AI PRO R9700

Machine "nonnormal" (gfx1201), at 502431b668. llama.cpp b12048, Vulkan.

| Measurement | Raven | llama.cpp, Vulkan |
| --- | --- | --- |
| Decode, position 512 | 12.7 ms, attention on WMMA | 10.38 ms at depth 512 |
| Prefill, 512 tokens | 0.418 s | 0.308 s |
| First call, kernels cached | 0.30–0.34 s, 12 tokens; 1.03 s compiling them, of which lowering and linking the first block take 69–131 ms on the host | 0.136 s; 2.19 s without Mesa's shader cache |

Where decode's time goes: the expert products take 5.4 ms at 212–254 GB/s,
against llama.cpp's 3.0. Top-k's 20-kernel sort and about 37 small kernels a
layer take about 1.5 ms, and the argmax over the logits 0.63 ms.

Where prefill's time goes: the expert products take 326 ms on WMMA at
7–8 TFLOPS, against 177. The out projection takes no WMMA: 52 ms, against
q's 10.

llama.cpp's ROCm build needs HIP libraries this machine lacks.

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
