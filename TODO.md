# todo

goalpost: jit-compiled gpt-oss at least as fast as {tinygrad,JAX,llama.cpp/vLLM} on {NVIDIA,NV,AMD}.

gpt-oss-20b, CUDA (kimchi, ebfb61e268, before tolk ran on rig), raven /
llama.cpp a868c3e:
- decode 9.50 ms / 5.92 ms: attention and lm-head weights as Q8_0 as GGUF
  does (~-2.2 ms, changes numerics, so a format the caller picks); small
  kernels ~0.3 ms per step
- prefill-512 0.188 s / 58.6 ms: profile what's left after tensor cores
- first call 0.220 s with filled caches (first step 0.115 s)

gpt-oss-20b, AMD gfx1201 (nonnormal, 0dbf76af6a, before tolk ran on rig),
raven / llama.cpp (Vulkan; ROCm):
- decode 21.3 ms / 5.67; 7.41 ms
- prefill-512 1.63 s / 0.180; 0.125 s
- first call 0.39 s (12 tokens, kernels cached; 1.43 s compiling them)
- llama.cpp b11344 Vulkan, b11342 ROCm; tinygrad needs Resizable BAR here

- gpt-oss-20b on the new nx and rune: decode, prefill-512 and first call
  on fudge (CUDA, NV) and nonnormal (AMD), recorded as the GptOss
  baselines, against llama.cpp at a stated build; rune's test_jit on
  CUDA, NV and AMD

## rig

### drivers

- a CUDA module load spins while queued GPU work runs (183 ms at gpt-oss's
  first call, overlapped with compute): wait on an event instead
- rig.amd over PCI on SDMA 4: program the page queue (regSDMA_PAGE_*,
  doorbell sdma_engine0 + 1, as sdma_v4_4_2 does) as the copy queue's
  second slot (`Sdma.slots`) and drop its 16 MiB ring (`most_ring`).
  Needs an SDMA 4 GPU (Vega, MI100-MI300); nonnormal's gfx1201 runs SDMA 7

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

### perf

- AMD boxes as one COPY_LINEAR_SUB_WIND packet per 2^14 rows, where they
  run as a COPY_LINEAR per row (rig_amd.mli). Linux drives SDMA 7 with the
  v6 packet header (sdma_v7_0.c), so check SDMA 7's packet and measure on
  nonnormal against rows; SDMA 4 to 6 wait for a host. On SDMA 4 over PCI
  it also lifts a box's limit of about 170,000 rows, past which it never
  fits the ring
- kaun's GPU decode rows measure allocation again (bench_decode.ml's
  `gpu_metrics`) once a decode call's words are shown not to depend on
  the collector's phase: on fudge (CUDA) and nonnormal (AMD)
- AMD 4 KiB copies: `copy/amd/within-device-4K` on nonnormal is bimodal,
  7.9-9.2 µs in 9 samples and 14.6-15.4 µs in 11 (rig-gpu.thumper),
  against fudge's 5.5 µs CUDA and 4.0 µs NV. The copy kernel starts at
  2 MiB, so this is SDMA and the wait: find what splits the modes

## tolk

- A schedule linked per run (D173) is one rig program per part, cut
  wherever storage is made or freed, because `Rig_program` binds its inputs
  once, before its first step: each part loads its kernels' images again
  and ends a submission. Done when a step of `Rig_program` makes or frees a
  slot's memory, so a per-run schedule is one program, and a split step's
  loads and submissions, counted on a rune bench row, match a linked one's.
- `upcast_vectors` (heuristic.ml) has no term for a kernel's size: 8 nested
  float64 sines over 1024 values take 8 lanes, run 4.5x as fast, and compile
  in 2.2 s against 1.4 s without lanes. Done when an A/B on fudge of
  `trig/sin-nested-8` and the `compiled` suite's sine rows decides whether a
  kernel's size bounds its lanes. Masked axes upcast to up to 49 lanes give
  Clang straight-line blocks whose instruction scheduling takes seconds:
  rune's test_iterate case (float64 [4;2;2] over 3 lanes, a deep step)
  becomes a fully unrolled 48-lane kernel, 4.8 s in Clang and 12.8 s with
  `-march=x86-64`
- a compile on 10 domains takes about twice the CPU of one domain. At load
  25-40 on the M1 Max, a jera tsit5 solve of y'' = -sin y took 18 s of CPU on
  10 domains, 8.9 s on 1 and 10 s on 10 with `OCAMLRUNPARAM=s=4M`; of
  y'' = -tanh y, 5.0, 3.4 and 3.8 s. Counts: minor collections do not grow
  with domains (rune's test_laws compilation group, 1971 at `PARALLEL=1`,
  1874 at 10), which rules out more collections. Each compiling domain's
  exit also empties every minor heap (191 spawns among the 1874).
  Candidates: time inside each collection, waiting for the slowest domain
  (an E-core's, or one descheduled at that load); major marking and
  sweeping shared across domains; the shard lock per node; spawn and join
  per `Worker.map`. Decided by time: on fudge (6P+8E), quiet, wall and CPU
  at 1, 2, 4, 6 and 14 domains, and 6 pinned to the P-cores; `s` at the
  default and at 4M; a sample of the 14-domain run, the stop-the-world
  barrier against `pthread_mutex_lock`. nx's P-core cap, on this CPU
  (kimchi, 2026-10-01), timed large GEMMs at x0.96-1.04 and batched
  64x32x32 at x1.59 slower. The Mac's cap closes on fudge's numbers
  (maintainer, 2026-10-31). Host blocks' placement matters as much:
  Jit/chain-100-rooted-hit runs its 4 blocks in 12.9 µs pinned to the
  P-cores, 43.5 µs on one core, and 13.0 or 20.4 µs unpinned on a quiet
  fudge depending on which cores they reach
- search timing on NV clears no L2 (tinygrad's invalidate_caches); AMD
  counters and thread traces are not collected: encode both as words parts
- AMD machine code without comgr (_plans/amd-isa/design-reviewed.md): an
  assembly renderer and assembler, gfx12 first, the default once its gate
  passes; HIP stays as a choice. First read the terms of AMD's ISA XML
- divergences: tolk's entries were audited against tinygrad HEAD
  (2026-10-31); left: one store-or-recompute rule for D77, D100, D106 and
  D163; one host lane budget for D91, D108, D109, D141 and D143; the
  matrix-vector layout (D89) with its fudge rows; entries marked "Reason:
  none" decided once the port reaches HEAD. rune's DIVERGENCES.md is not
  audited yet
- compile speed: one Clang process per kernel (~30 ms each; one per program
  saves ~20%). Hypothesis, to time first: on fudge, each Clang run's wall
  time against its source's bytes, on a model-size schedule; if the process
  start dominates, one Clang invocation over several translation units, one
  object per kernel (`clang -c 0/k.c 1/k.c ...`), keeps per-kernel cache
  keys and Worker.map's balance, once each object is shown a function of its
  source alone; float64 sin/cos inline the full Payne–Hanek reduction (40-170
  KB kernels); structural comparisons in the rewriter (~6% before patterns
  tested their operations as a set): the polymorphic `List.mem` left in
  ops.ml on operation lists (1549, 2208, 2220, 2231, 2319) and dtype lists
  (1503, 1609, 1719, 2189, 2244), until a profile clears them; parallel
  compile saturates at ~4 domains (a shard lock per node, domains spawned
  per `Worker.map`); Metal builds run one at a time;
  a 5 KB r_32_32 kernel, max |x| over `where (i >= j) x xᵀ` at 32x32,
  renders as 32-lane float selects of transposed loads that clang -O2
  takes over 2 min on
- first call of a cached program (gpt-oss on CUDA: 0.44 s against 0.22 s
  on the old stack): `Worker.map` spawns up to PARALLEL−1 domains when
  every program is a disk-cache hit, and their GC stop-the-world costs
  ~600 ms (prefill) and ~200 ms (step) summed over domains (PARALLEL=1 cut
  jit_lower from 258 to 72 ms of CPU): spawn only for programs that need a
  compile; `to_program` reads and parses a cached program in 3-7 ms per
  kernel; `create_linear_with_vars` on a hit spends ~10 ms on the key
  digest and ~24 ms in other passes; the engine assembles 4 programs in
  28-57 ms
- compile time and launch counts: NUTS on eight_schools lowers to 464
  kernels, 187 of them scalar, 27 CPU-s cold to compile against 2 CPU-s to
  run; a NUTS program takes 9-14 s to build, a
  ymir aperture gradient 98 s cold; loop bodies store each carry and row in
  its own kernel (drawing-scan-256: 6 per trip against 2). Multi-output
  fusion of loop bodies
- long programs: the incomplete beta (~11,400 ops) compiles in 3 s, 10-20 s
  per derivative, past 10 min for three; a 1,000-term slice-tree sum is one
  kernel clang -O2 takes over 8 min on. Past 400 KB of C Clang compiles
  without its loop vectorizer (D167); float64 `betainccinv`'s gradient (1.3
  MB) then spends, loaded, 5.7 s in Clang, 5.1 s in codegen (no pass over
  0.6 s), 2.5 s linearizing and 1.7 s scheduling
- host upcast width per target: D141's 64-byte lanes (heuristic.ml:446)
  slowed memory-bound select-zero ~15% on x86; time 32 against 64 bytes on
  fudge
- an expensive elementwise producer (sin/cos) recomputed in each consumer's
  kernel: jera root-newton-kepler-1k compiled 2.2 ms against eager 0.57.
  The opposite also costs: `Indexing.assign_ranges` stores any elementwise
  node under an expand that feeds a reduce, so gpt-oss's
  `Nx_quant.apply_experts` (CUDA, prefill-512) decodes every expert's
  weight into float32 on each call: E_8294400_32_2 writes 2.1 GB in 4.74
  ms for gate_up, E_4147200_32_2 2.39 ms, where the decode could stay
  fused into the gated per-group loads. One store-or-recompute rule
  covers both and D77, D100, D106, D163
- apply_experts' contractions run 56 ms against the block path's 36 ms:
  one route row per thread group, so decoded weights are not shared across
  rows (D92's case); decide with D89's layout
- running sums cost ~257 ops per element (shape.ml:955): `Nx.cumsum` over
  float64 [100; 50500] takes ~1.0 s on CPU. A scan with constant work per
  element, and a compiled cumsum row
- Metal groups: after D137, lorenz-step-batch-1024's r_3_256_16_64_64 went
  10.7 -> 18.9 ms (loads stride across 16 threads). Decided without Apple
  timing (maintainer, 2026-10-31): a rule for loads that stride across a
  group's threads, checked by reading the Metal source
- AMD beam search times 16-21% of candidates below 2 GHz: hold rig's stable
  power state around a search, or ties within noise
- comgr's per-process setup is paid on every compile: each HIP compile
  starts a process (D5) that loads comgr and compiles once. NVRTC's
  processes serve compiles in turn (`Compiler_worker.call`), which took
  fudge's search/nv from 1.90 s to 1.34 s, its time in the process with
  NVRTC's cache disabled. Move comgr to
  serving workers once nonnormal times search/amd and a first call both
  ways
- CUDA contractions with a gathered operand run at 2-8% of the RTX 5000's
  SIMT float32 peak with DRAM under 46 GB/s: gpt-oss's MXFP4 expert
  products hold 4 rows x 1 column per thread, split K across 32 lanes and
  issue 20 predicated, half-coalesced x loads per 20 FMAs; a dense float32
  weight of the same shape runs at 1.39 TFLOPS holding one row per thread.
  Decide with D89's matrix-vector layout, by tinygrad's heuristic at HEAD
  and a fudge bench row
- D176 (selection and negation fold a NaN bitwise) has no rune pin: add
  one to rune's `Lower_arith` suite, a signalling NaN through a compiled
  select and negation keeping its bits
- folds of `Log2`, `Exp2` and `Pow` use the host's C library, so a folded
  constant can differ between macOS and Linux; D150 made the sine fold
  correctly rounded for that reason. Do the same for these, or state why a
  host's value is acceptable
- a call body that stores to LOCAL memory gets no barrier before the
  kernel reads it back: since tinygrad efb4cc8cb, `add_raw_barrier` does
  not look into call bodies, and tolk follows. tinygrad 6b7b018da (its
  HEAD on 2026-10-31) still walks an AFTER's dependencies without entering
  calls (`codegen/__init__.py:230-235`), so upstream has no fix to port.
  Nothing builds such a body yet; decide before one does
- a kernel's estimate counts that do not fit an `int` (a loop bounded by
  a loaded value) are exact constants, but `Search`'s compute filter,
  the engine's `split_grid` and its DEBUG=2 report evaluate them through
  `Shape.sym_compile`, which raises past an `int`: they need a magnitude,
  so evaluate exactly and saturate
- compile.thumper's fudge rows run 9-25% faster than c0682e939 does at
  equal load: find whether a regression landed unexplained before it or
  the baseline was recorded under other conditions
- a link reloads images an earlier link loaded: measure on fudge the load
  cost a search pays, and if it matters, design how rig.program takes loaded
  images (a description entry naming a Rig.Image.t)
- D175 and D110 use constants for a device's size (100 workgroups,
  `busy = 32768`): neither the renderer nor rig knows a device's compute
  units; decide where that count lives without making source depend on the
  physical part
- decode's values kernel takes 23.6 us against 12 us in the formula form: a
  core reduce split, or `decoded` covering float widening
- Metal's packed kernel arguments (8869adcc8, af8961676) have not run on
  Metal hardware; check `rig_metal.mli`'s contract against them on a Mac
- rune hands tolk one sink over several devices; D170, D173, D177 and D180
  each exist because of it. Decide whether a per-device sink, which
  tinygrad's own multi-device path also avoids, would remove them

## rune, kaun, vega

- sharded jit after its first slice (one target, 1-D sets, no loops) and
  lending per piece (898ec2b85): a place onto another set
  (_plans/rune-sharding/adjudication.md). `lanes` and `lane_index`
  compile, and per-device code, as loops that read their window's
  coordinates, makes programs that do not grow over 2 to 16 devices, each launch
  passing its window's start. nx's partials still write each tile's start
  into its program as a constant (`Partial.renumber`, `shift`: `argmax`
  along a cut axis, a scan's positions, a gather, scatter or sum over a
  cut axis), so those make a kernel per device. nx's closing
  moves a chunk between every pair of devices: time it against tinygrad's
  ring and hierarchical allreduce on a machine with several GPUs, and if
  it is slower, write those in nx's partition
- rune's `test_probe › scans › a step offers a row per row` failed once
  under load and passed on replay of its counterexample: find whether it
  is load-dependent or flaky

## nx

- sharded jit: nx's partials write each tile's start as a constant
  (`Partial.renumber`, `shift`), so argmax along a cut axis, scan
  positions, and gather, scatter or sum over a cut axis still make a
  kernel per device; pass the start as a bound variable, as maps do since
  b6dfbb6b5. An evaluated split program builds an arange the length of
  the whole axis on every device
- the attention family: `Attention`, `Attention_tangent` and
  `Attention_cotangent` with nx.cpu's forward and cotangent kernels, in
  place of `Nx.attention`'s composition, which stores three float32
  `[lead…; hq; tq; tk]` arrays a call and gives an interpreter only its
  contractions, maps and reduction. Kernels keep hidden values out of
  `o`: a partial tile with a non-finite value takes the selecting path.
  First add and record `attention/host-*` on the composition, and
  `gpu/*/attention-4096-causal` under `Rune.jit`; the family lands only if its kernels beat it by 2x on each,
  forward and cotangent
- donation's kill criterion, at least 2x over the copy on CUDA and AMD:
  `dispatch/kv-append-4096` and its `-copied` twin read 3.4 and 44.5 µs on
  fudge's host; record them on CUDA and AMD under `Rune.jit`
- nx.cpu's thread splits follow the call prices (179ea5cc5, 82e42b78d) and
  the kind tables of f6876a0ad, measured under load 1.3-4.7: time them
  quiet on fudge, with a row of a costly kind where its threads change
  (pow-f64-16K), and re-record the elementwise rows
- complex products and extremes over few terms (`few_`, `IN_ARRAY` in
  reduce_body.h) ran 1.2-1.7x slower under gcc 15 on the M1 than before
  their lanes became vectors; inlining the start step for every monoid
  costs the float extremes 15%. Add a row (prod of complex64 2x2 windows),
  time it on fudge's gcc, and act only if it loses there

## talon, hugin, brot, munin, quill

- talon benches: record the query suite's DuckDB/Polars baselines on fudge
  (H2O 1e7/1e8, decision support SF1/SF10; none committed), and bless
  `Talon_csv/read-mixed` on fudge and the Mac
- talon perf, machine and commit unrecorded: vs the faster of
  DuckDB/Polars, decision support SF1 13.2x, H2O group-by and join 2.8x;
  vs pandas 1.36x on filter, 2.05x on join. Re-measure on fudge once the
  query baselines exist
- talon 1.0 plan (_plans/talon/product/plan.md, on the Mac): windows and
  ewm, as-of and interval joins, interval sets and folds, zoned time,
  hash/sample/shuffle, Pivot and nesting, dense round trips, partial
  aggregation, regex, compressed and fixed-width CSV, Parquet nested
  columns, datasets and writer, talon.arrow, talon.jsont, complex and
  validity columns, HTML display
- brot vs gigatoken (machine unrecorded): gpt2 2.5x one thread, 3x
  multi-thread; SentencePiece pipelines 40-58 MB/s (added-token scan 28%,
  normalizer 15%); bert unfused. Re-measure on fudge at a stated gigatoken
  commit

## repo

- thumper: suites whose only x86 section is kimchi's (retired):
  gpt_oss. Record each on fudge, then delete every kimchi section.
  nx.thumper's fudge section was recorded under
  OCaml 5.5.1, so it compares nothing under 5.5.2: re-record it on a quiet
  fudge. talon.thumper's Mac rows predate the Nx.Rng
  fixture (76105a215): the maintainer re-records them
