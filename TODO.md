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

gpt-oss-20b, AMD gfx1201 (nonnormal, f24925ed77 unless stated, the new nx
and rune), raven / llama.cpp b12048 Vulkan:
- decode 20.3 ms / 9.40 ms
- prefill-512 1.13 s (baf728633, one run) / 0.308 s
- first call 1.47 s (12 tokens, kernels cached; 2.39 s compiling them) /
  0.136 s (2.19 s without Mesa's shader cache)
- llama.cpp's ROCm build needs HIP libraries nonnormal lacks; tinygrad
  needs Resizable BAR here

## rig

### drivers

- rig.amd over PCI on SDMA 4: program the page queue (regSDMA_PAGE_*,
  doorbell sdma_engine0 + 1, as sdma_v4_4_2 does) as the copy queue's
  second slot (`Sdma.slots`) and drop its 16 MiB ring (`most_ring`).
  Needs an SDMA 4 GPU (Vega, MI100-MI300); nonnormal's gfx1201 runs SDMA 7
- host buffers on CUDA: a call that makes the process's first copy from
  the host pays one 8-11 ms `cuMemHostRegister`, and gpt-oss's prefill
  pays 17-19 ms of `cuMemHostUnregister` for weight buffers the GC freed
  just before; registration and release belong in rig's staging
- `Pinned` memory on a discrete GPU counts in the host's budget but is
  paced by the GPU's room and allocated bytes

### limits and size

- NV over PCI: every GPU's memory, Device and host-mapped, comes from one
  range, 64 GiB to `Rig_nv.windows` (about 952 GiB, rig_nv_pci.ml),
  shared by all GPUs: eight 141 GB GPUs don't fit. Place Device memory in
  the GSP's 49-bit space (gsp.ml), as e81be363c did for the kernel path;
  only segments, launch descriptors and semaphores need 2^40

### API

- GPU memory between machines over RDMA (GPUDirect), as nx.rdma had: a
  copy between two machines' GPUs runs through rig.mlx5 or rig.bnxt, not
  TCP, over the memory `Rig.Buffer.dma` describes

### tests and runs

- hardware runs, none possible on fudge or nonnormal (SDMA 7, one GPU
  each, no RDMA NIC): the box and fill laws on SDMA 4 to 6; two GPUs
  (peers); mlx5 and bnxt NICs

## tolk

- placement on hybrid cores: nx's P-core cap, on fudge's CPU model
  (kimchi, 2026-10-01), timed large GEMMs at x0.96-1.04 and batched
  64x32x32 at x1.59 slower. The Mac's cap closes on fudge's numbers
  (maintainer, 2026-10-31). Host blocks' placement matters as much:
  Jit/chain-100-rooted-hit runs its 4 blocks in 12.9 µs pinned to the
  P-cores, 43.5 µs on one core, and 13.0 or 20.4 µs unpinned on a quiet
  fudge depending on which cores they reach
- AMD counters and thread traces are not collected: encode them as words
  parts
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
- AMD beam search times 16-21% of candidates below 2 GHz: hold rig's stable
  power state around a search, or ties within noise
- comgr's per-process setup is paid on every compile: each HIP compile
  starts a process (D5) that loads comgr and compiles once. NVRTC's
  processes serve compiles in turn (`Compiler_worker.call`), which took
  fudge's search/nv from 1.90 s to 1.34 s, its time in the process with
  NVRTC's cache disabled. Move comgr to
  serving workers once nonnormal times search/amd and a first call both
  ways
- device facts (762ad1c31): `core_threads = 16384` and `stream_threads =
  32768` are fudge's (sm_89, 100 SMs); a small part (an RTX 4060-class
  sm_89) may lose 1.4-2x by the model: a row owed there, and values for
  nonnormal's R9700 and Metal. `Rig.cache_bytes` has not run on hardware
  for AMD through amdgpu (L2 plus MALL) or NV over PCI
- Metal's packed kernel arguments (8869adcc8, af8961676) have not run on
  Metal hardware; check `rig_metal.mli`'s contract against them on a Mac
- comgr may compile with LLVM's verifier off, as Clang did before D275:
  check `Compiler_amd`'s commands on a ROCm build and keep the verifier on
- Apple Clang and LLVM 23 keep the straight-line vectorizer that D167
  turns off under LLVM 18-20: run 73455ffe7's reduced kernel and rune's
  ALiBi attention example (test_attention.ml, the dense tq 2 tk 3 case)
  through them on the Mac and extend D167 if they miscompile (LLVM 21
  and 22 compile both)
- programs that share a kernel load its binary once each (D146): rig could
  share a binary with no module variables and keep its promise, but on
  fudge that saves at most 1-1.5 ms of a 70 ms gpt-oss first call on CUDA
  and 2.5 ms of 60 ms on NV, against a driver function and an image table
  (about 80 lines); NV's cost is an allocation and map per image, and
  AMD's loads are unmeasured
- HIP's unstaged tensor-core operands load 8 bytes again since D261's copy
  rule: unmeasured on nonnormal
- AMD's comgr and Metal's compiler run LLVM's loop-idiom pass too; D167's
  `-mllvm -disable-loop-idiom-all` covers host Clang only, and no CI job runs
  LLVM 22+, where the pass's CRC recogniser walks a running select's paths
  (fixed upstream after 23.1.5)
- tolk's Tc case "chain › an accumulator chains…" takes 16.6 s alone and
  timed out at 60 s at load 34-50

## rune, kaun, vega

- sharded jit after its first slice (one target, 1-D sets, no loops) and
  lending per piece (898ec2b85): a place onto another set
  (_plans/rune-sharding/adjudication.md). `lanes` and `lane_index`
  compile, and per-device code, as loops that read their window's
  coordinates, makes programs that do not grow over 2 to 16 devices, each launch
  passing its window's start. nx's closing
  moves a chunk between every pair of devices: time it against tinygrad's
  ring and hierarchical allreduce on a machine with several GPUs, and if
  it is slower, write those in nx's partition
- attention under `Rune.jit`: causal saves little, since a grid under 1.3
  waves takes as long as its longest workgroup (rows already launch longest
  first): pair row tiles i and n-1-i, or split long rows' keys; the
  tensor-core backward leaves head dimensions over 64, float captures'
  cotangents, paged keys, forward mode and float32 on the composition; RDNA4
  and Metal core paths
- a fresh process's first calls fault 72-188 pages each for 1 MiB results
  until rig's buffer cache warms
- the scatter walk (I4) on a GPU: a scatter of a few updates pays one or
  two launches more than the comparison it replaced (7 rows, 1.5-3 µs);
  writing the walk's first chunk over the target (no fill) would remove one.
  Its launch-cost constants live in rune keyed on host or GPU, and
  `stream_threads` is sm_89's default on every target, the host included:
  both belong in the renderer's facts
- compiled `Nx.eigh` at k = 128 float64 replays in 24-58 ms on the host
  (9-22 sweeps of 127 rounds) against eager's 3.5 ms: each round's two
  kernels read their `% 127` gathers as scalar loads (hand C 2.7x
  faster); float32 rounds take 2.3x float64's there. Compiled `Nx.svd`
  at k = 128 float64 replays in 53-114 ms (16-36 sweeps). A GPU runs
  every sweep of both. On an idle RTX 5000 a flag read takes 5.8-6.7 us
  and stopping eigh f32 every sweep runs 1.4-2.1x faster (k = 128 11.6
  -> 6.1 ms, k = 32 1.46 -> 0.83 ms); while another process computed on
  the GPU a read took 0.86 ms and k = 32 went 3.6 -> 15.7 ms, and a stop
  makes the call synchronous: decide a GPU stop (a read after a lower
  bound, or every m sweeps, or one kernel per matrix as batched syevj)
  (_plans/todo-campaign/design/eigh-compile.adjudication.md)
- fudge's `rune.thumper` words rows moved at main since they were recorded:
  `Loops/scan-grad-100x100` 3967612 -> 4039461 (+1.8%, unexplained),
  `MlpGrad/mlp-forward-nx-eager` 2440 -> 1819, `WarmStart/*` 13-37% lower;
  explain the rise, then re-record

## nx

- with a large live OCaml heap, the host buffers of each talon take drove
  the major GC to ~150 M instructions a take, ~8x the take's own (probe of
  6281f607b): custom-block pressure from nx and rig host buffers
- nx.cpu's thread splits follow the call prices (179ea5cc5, 82e42b78d) and
  the kind tables of f6876a0ad, measured under load 1.3-4.7: time them
  quiet on fudge, with a row of a costly kind where its threads change
  (pow-f64-16K), and re-record the elementwise rows

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
  _plans/talon/design/): range joins (6f37e59c7), windows, Kit fills,
  as-of joins, interval sets and nesting landed; ewm waits for the
  maintainer's scope call; then folds, zoned time,
  hash/sample/shuffle, Pivot, dense round trips, partial
  aggregation, regex, compressed and fixed-width CSV, Parquet nested
  columns, datasets and writer, talon.arrow, talon.jsont, complex and
  validity columns, HTML display
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
- brot's dump_kernels comparison failed once on macOS arm64 CI (run
  42423619955, f8b600809): shard 2's native encode raised `519254,519240
  is not a span of the 524288 normalized bytes`, shard 5's bytecode dump
  had ':' as 0x18; not reproduced on x86 under GC stress, ASan, the debug
  runtime or checked accesses; run the 8 shards in a loop under --release
  with 3 domains on an arm64 host

## repo

- a windtrap test whose host kernel loops forever outlives its limit and
  `timeout`: the test's forked child (its own session, SIGTERM and SIGALRM
  caught, handled only at OCaml poll points) spun at 100% for 2 h while its
  parent waited on it (tolk's "hand-written kernels on the host", run
  without a fix, 2026-11-08)
- thumper: talon.thumper's Mac rows predate the Nx.Rng
  fixture (76105a215): the maintainer re-records them
