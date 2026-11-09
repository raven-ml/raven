# todo

goalpost: jit-compiled gpt-oss at least as fast as {tinygrad,JAX,llama.cpp/vLLM} on {NVIDIA,NV,AMD}.

gpt-oss-20b, RTX 5000 Ada (fudge, under load), raven CUDA / NV /
llama.cpp b12032 CUDA; NV and the first run last measured at 31aa4521b:
- decode at position 512 of a 1024 cache 10.2 ms, GPU time and wall
  (8ef1902f3) / 10.7 ms of GPU time (bench median 12.0 ms) / 6.83 ms at
  depth 512; with Q8_0 attention and head 8.1-8.3 ms (f2af3251b) / 22.8 ms
- prefill-512 0.186 s (f2af3251b) / not measured again / 56.0 ms
- first call 0.37-0.39 s (c29acaee3) / 0.37 to 0.54 s (12 tokens, kernels
  cached; 1.36 / 1.47 s at the first run) / 75 ms (59 to 151 ms, no warmup)

gpt-oss-20b, AMD gfx1201 (nonnormal, 502431b668, the new nx and rune),
raven / llama.cpp b12048 Vulkan:
- decode at position 512 12.7 ms (attention on WMMA) / 10.38 ms at depth
  512: expert products 5.4 ms at 212-254 GB/s against 3.0; top-k's
  20-kernel sort and about 37 small kernels a layer about 1.5 ms; argmax
  over the logits 0.63 ms
- prefill-512 0.418 s / 0.308 s: expert products 326 ms on WMMA at 7-8
  TFLOPS against 177; the out projection takes no WMMA, 52 ms against q's 10
- first call 0.30-0.34 s (12 tokens, kernels cached; 1.03 s compiling
  them) / 0.136 s (2.19 s without Mesa's shader cache); lowering and
  linking the first block takes 69-131 ms of it on the host
- llama.cpp's ROCm build needs HIP libraries nonnormal lacks; tinygrad
  needs Resizable BAR here

## rig

### drivers

- host buffers on CUDA: a call that makes the process's first copy from
  the host pays one 8-11 ms `cuMemHostRegister`, and gpt-oss's prefill
  pays 17-19 ms of `cuMemHostUnregister` for weight buffers the GC freed
  just before; registration and release belong in rig's staging
- `Pinned` memory on a discrete GPU counts in the host's budget but is
  paced by the GPU's room and allocated bytes

### tests and runs

- needs hardware raven's machines lack: rig.amd over PCI on SDMA 4 (the
  page queue as `Sdma.slots`' second slot; Vega, MI100-MI300); the box and
  fill laws on SDMA 4 to 6; two GPUs as peers; mlx5 and bnxt NICs and GPU
  memory between machines over RDMA; NV over PCI with more GPU memory than
  the range from 64 GiB to `Rig_nv.windows` holds

## tolk

- placement on hybrid cores: nx's P-core cap, on fudge's CPU model
  (kimchi, 2026-10-01), timed large GEMMs at x0.96-1.04 and batched
  64x32x32 at x1.59 slower. The Mac's cap closes on fudge's numbers
  (maintainer, 2026-10-31). Host blocks' placement matters as much:
  Jit/chain-100-rooted-hit runs its 4 blocks in 12.9 µs pinned to the
  P-cores, 43.5 µs on one core, and 13.0 or 20.4 µs unpinned on a quiet
  fudge depending on which cores they reach
- AMD counters and thread traces are not collected in a profile: rig's
  AMD driver brackets each timed launch, reading the VMID filter on the
  GPU in the bracket and marking a saturated or foreign count unmeasured
  (_plans/todo-campaign/design/amd-counters.adjudication.md)
- AMD machine code without comgr (_plans/amd-isa/design-reviewed.md): an
  assembly renderer and assembler, gfx12 first, the default once its gate
  passes; HIP stays as a choice. First read the terms of AMD's ISA XML
- compile speed: Metal builds run one at a time. On the host,
  `Codegen.to_programs` starts one Clang per worker even for one small
  kernel each, and `clang -cc1` starting costs ~17 ms: a jit of a few
  small kernels spent 19.5 s CPU / 4.9 s wall cold at `PARALLEL=14`
  against 10.0 s / 10.1 s at 1 (rune's law_carries, 728 kernels, fudge
  Clang 19); batching small kernels per worker needs quiet timing
- long programs: the incomplete beta (~11,400 ops) compiles in 3 s, 10-20 s
  per derivative, past 10 min for three; a 1,000-term slice-tree sum is one
  kernel clang -O2 takes over 8 min on. Past 400 KB of C Clang compiles
  without its loop vectorizer (D167); float64 `betainccinv`'s gradient (1.0
  MB, 42k instructions) then spends, loaded, 3-4.5 s in Clang and 3.5-5.5 s
  in tolk (424 M words): about 40 passes, each linear in the program's size
  and under 0.4 s and 40 M words, rangeify's cleanups, `run_rangeify` and
  the symbolic passes before decomposition the largest. On rune's
  `special/betaincinv-grad-f64` compile row (187 M words, 35 M promoted,
  21.1 G instructions under cachegrind) the collector takes 22%, the walks
  23% (5.9 M rows, about 130 walks of the 42k-node graph, mostly first
  visits), matching 7%, hash-consing 7%, the development profile's
  `caml_apply2` 3%, the linearizer's ties broken by structure 4% and
  `caml_modify` 3%; promotions are the graphs between passes, nodes and
  their memo records a third of them
- whether memory-bound host kernels take lanes at all: no upcast ran 4-7%
  faster than 64-byte lanes in two of seven loaded fudge runs of a 4M
  float32 select-zero; decide on the scheduled quiet re-record
- device facts (762ad1c31): `core_threads = 16384` and `stream_threads =
  32768` are fudge's (sm_89, 100 SMs) and also the R9700's best, where
  D175's SMs x clock scaling put its window at 0.59x; a small part (an
  RTX 4060-class sm_89) may lose by that model: a row owed there, and
  Metal's values. `Rig.cache_bytes` has not run on hardware for NV over
  PCI
- LLVM 23 keeps the straight-line vectorizer that D167 turns off under
  LLVM 18-20 and Apple clang 17: run 73455ffe7's reduced kernel and
  rune's ALiBi attention example (test_attention.ml, the dense tq 2 tk 3
  case) through it and extend D167 if it miscompiles (LLVM 21 and 22
  compile both)
- HIP's unstaged tensor-core operands load 8 bytes again since D261's copy
  rule: unmeasured on RDNA3 (16 contiguous halves a lane) and CDNA
- a Cast from an integer to float32 on Metal relies on the conversion
  rounding to nearest, which MSL 8.6 does not promise (it allows round
  toward zero); Apple's GPUs round to nearest when tested. Make
  tolk's Cast own it, as `Decomp_dtype.narrow` does for narrow floats

## rune, kaun, vega

- sharded jit: over several devices a check or a total inside a loop or a
  switch's branch still declines, with a named error at the call
- attention under `Rune.jit`: causal saves little, since a grid under 1.3
  waves takes as long as its longest workgroup (rows already launch longest
  first): pair row tiles i and n-1-i, or split long rows' keys; the
  tensor-core backward leaves head dimensions over 64, float captures'
  cotangents, paged keys, forward mode and float32 on the composition
- attention's SIMT kernel on AMD (RDNA3, and what no core takes): K and V
  addresses are wave-uniform, so LLVM loads bfloat16 and float16 with
  s_load_u16 and a wait each, about 4100 serial round trips a tile; at
  gpt-oss's decode shape float32 (global_load_b32) takes 51 µs against
  bfloat16's 323 µs on the R9700
- a cast from bool to bfloat16 aborts comgr on gfx1201 ("Cannot select ...
  bf16 = uint_to_fp"): test_jit_gpu.ml's integer casts, seed
  s1:0072ef24ec103aaf
- a fresh process's first calls fault 72-188 pages each for 1 MiB results
  until rig's buffer cache warms

## nx

- with a large live OCaml heap, the host buffers of each talon take drove
  the major GC to ~150 M instructions a take, ~8x the take's own (probe of
  6281f607b): custom-block pressure from nx and rig host buffers

## talon, hugin, brot, munin, quill

- talon benches: record the query suite's DuckDB/Polars baselines on a
  quiet fudge (`baseline.py` refuses loaded runs; H2O 1e8 needs more than
  a run's 16 GiB), and bless `Talon_csv/read-mixed` on the Mac
- talon perf on fudge against the faster of DuckDB 1.5.6 and Polars
  1.44.2, engines re-run beside talon under load (2026-11-04 to 11-07):
  H2O 1e7 group-by geomean 2.4x at 673340f4e, q10 1.3x at eaff601a6
  (q01 then 148 -> 96 instructions a row with 5441600cd); H2O joins
  1.0-2.7x at eaff601a6 (q03 7.83 -> 2.45 G); decision support SF1 at
  cfef1ad82 q19 ~16x, q12 ~18x, q07 ~15x, q02 ~40x, since cut in
  instructions by 7e16755a1 (q02 4.71 -> 1.61 G) and c8e1f9fa2 (q19 4.66
  -> 3.68 G, q12 3.73 -> 3.14 G), and by the join placing only matching
  right rows (q05 4.13 -> 3.32 G, q03 3.62 -> 3.05 G; text gathered at 35
  instructions a row, from 48), then by Parquet decoding (e090d2d2b: q19
  3.51 -> 1.92 G, q12 2.97 -> 2.08 G, q05 3.18 -> 2.74 G), `Nx.unique`
  (fd09c766a, q05's keys 1.55 -> 1.07 G on 14 threads) and shared
  aggregate parts (b1fb8d4b4, H2O q07 -26%, q09 -10%); ratios not
  re-measured. What is left: q05's Snappy at 744 M (~32 instructions an
  element) and ~0.46 G of memcpy, part a version-1 PLAIN page copied into
  its column; `Nx.unique` on the pool still makes each key again in the
  merge and hashes keys a direct table numbers (partitioning by hash or
  splitting the table needs wall time to judge); a group-by's aggregates
  fold in one walk only if nx's reduce kernel takes segment ids
  (_plans/todo-campaign/design/talon-one-pass.md §4.5, a model of q04 and
  q05 -34%, a kernel contract change for the maintainer). The int64 cast
  before an integer sum is at most 3.8% of a query and stays.
  Data: ~/wt/tmp/talon-data on fudge (decision SF0.1/SF1, H2O 1e6/1e7)
- talon 1.0 plan (_plans/talon/product/plan.md; designs adjudicated in
  _plans/talon/design/), open units: regex, Pivot, Unicode case mapping,
  Parquet datasets, JSON and NDJSON, domain guides. After 1.0: the C Data
  Interface, the Ptree bridge, complex and validity columns, `ewm`,
  fixed-width CSV; HTML display once quill's protocol lands
- talon interchange follow-ups: `Talon_arrow.sniff` refuses decimals,
  where Parquet's `with_type` reads them as float64 or int64. Parquet
  writes a duration as bare int64 ticks and loses its unit, since no
  expression converts a duration to int64 exactly (`Expr.cast` refuses,
  and `Temporal.div` goes through float64): an exact conversion lets
  `Talon_parquet.format` refuse durations and name it
- brot vs gigatoken cd5ee648c6 on fudge (bench README, "Against
  gigatoken"): GPT-2 runs at 0.55-0.65 of its speed on one thread over
  100 MB (abbeface7). Pages: a warm pass faults 11.5k-29k times, mostly
  each batch's fresh 90 MB result; glibc's tunables remove them with no
  brot code (`mmap_max=0` and a large `trim_threshold`: 0 after the first
  pass; `glibc.malloc.hugetlb=1`: 0.8k-3k). Instructions: GC marking of
  the vocabulary's blocks is ~0.6 instr/B at 100 MB (2 only at 5 MB), so a
  flat vocabulary (~0.8%) does not pay; the walk and probe loop (~3 instr/B)
  remains.
  Branch-free merges cut callgrind's mispredicts 35% but ran no faster on
  fudge
- talon.parquet's pacing law failed once on macOS (run 42473531097: 35
  major cycles against 34); the runtime's own pace cannot explain it (the
  loop puts ~4.8k words in the major heap) and it passes at rig's 4 MiB
  floor on fudge; on failure the law now prints each collector's counts
  (0247fa6d8)
