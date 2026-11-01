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

- rig.amd over PCI, rig.pci: `peer` must take no lock (rig_edge.mli's
  Drivers section). AMD over PCI's `reaches` takes `machines_lock`, and
  rig.pci's `Memory.reaches` reads each GPU's route under `routes`
  without an IOMMU, for NV over PCI too. A callback inside those sections
  gets `Invalid_argument`. NV over PCI's atomic list is the shape for AMD;
  the route needs a first read that takes no lock
- rig's drivers keep state whole by rig.mli's rule: a synchronous raise
  leaves it whole; an asynchronous one may leak and lose the device but
  never makes a later call wrong. worklist-m.md (b8a35696e) has 89 rows
  to fix of 199. nv, nv-pci, pci, cuda, disk and remote have had their
  fix series since; rig.amd (21 rows, nonnormal), mlx5 (8), bnxt (11) and
  metal (1) have not. Recount on main, fix the rest. On Windows OCaml has
  no signal masks, so a Ctrl-C lands at any poll point below rig's core,
  rows worklist-m.md counts as unreached included (rig.cuda's page-lock
  registry under a contended lock): those rows need the same fix there
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
- compile time and launch counts: NUTS on eight_schools lowers to 464
  kernels, 187 of them scalar, 27 CPU-s cold to compile against 2 CPU-s to
  run; a NUTS program takes 9-14 s to build, a
  ymir aperture gradient 98 s cold; loop bodies store each carry and row in
  its own kernel (drawing-scan-256: 6 per trip against 2). Multi-output
  fusion of loop bodies
- long programs: the incomplete beta (~11,400 ops) compiles in 3 s, 10-20 s
  per derivative, past 10 min for three; a 1,000-term slice-tree sum is one
  kernel clang -O2 takes over 8 min on. rune's test_special builds 6.7 MB
  of C in 54 kernels (incomplete-beta gradients 430-630 KB, ndtri's f32
  derivative 951 KB), 5-14 s each in Clang; 90% of Clang's IR-pass time
  is LoopVectorizePass, and `-fno-vectorize` takes the 631 KB kernel from
  8.6 to 2.1 s, while changing ordinary kernels by ±10% and losing the
  vectors of kernels such as E_3945 (width 4). Decide beside D167 by a
  quiet bench: a size gate, a loop pragma or a flag
- host upcast width per target: D141's 64-byte lanes (heuristic.ml:446)
  slowed memory-bound select-zero ~15% on x86; time 32 against 64 bytes on
  fudge
- an expensive elementwise producer (sin/cos) recomputed in each consumer's
  kernel: jera root-newton-kepler-1k compiled 2.2 ms against eager 0.57
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
- folds of `Log2`, `Exp2` and `Pow` use the host's C library, so a folded
  constant can differ between macOS and Linux; D150 made the sine fold
  correctly rounded for that reason. Do the same for these, or state why a
  host's value is acceptable
- a call body that stores to LOCAL memory gets no barrier before the
  kernel reads it back: since tinygrad efb4cc8cb, `add_raw_barrier` does
  not look into call bodies, and tolk follows. Nothing builds such a body
  yet; check tinygrad HEAD before one does
- compile.thumper's fudge rows run 9-25% faster than c0682e939 does at
  equal load: find whether a regression landed unexplained before it or
  the baseline was recorded under other conditions
- a link reloads images an earlier link loaded: measure on fudge the load
  cost a search pays, and if it matters, design how rig.program takes loaded
  images (a description entry naming a Rig.Image.t)

## rune, kaun, vega

- sharded jit after its first slice (one target, 1-D sets, no loops):
  lending per piece, mixed-target sets, a place onto another set, loops,
  compiled `lanes` (_plans/rune-sharding/adjudication.md). A split
  program that gathers an argument so that a device's window grows still
  evaluates: a compiled backward call holds 3.1 layers per device at n=2
  and 4.2 at n=8 (d=512) against a bound of 2.2 and 1.5 (RFC 0005: two
  layers), because tolk plans copies and kernels in separate arenas,
  keeps copy buffers longer, and fuses a closing's piece slices into one
  kernel so every piece is live at once; one lane and no copy hold reach
  only 2.6 and 3.0. Per-run storage (D173) adds allocations to every split
  call, unmeasured. nx's closing
  moves a chunk between every pair of devices: time it against tinygrad's
  ring and hierarchical allreduce on a machine with several GPUs, and if
  it is slower, write those in nx's partition

## nx

- a reshape that factors or merges a cut axis gathers it whole onto every
  device where the tiles don't divide both extents: gpt-oss's 2880 inputs,
  90 MXFP4 blocks, over 4 or 8 devices. A halo exchange, as GSPMD's
  reshape realigns an uneven cut (arXiv 2105.04663 §4.1), would move only
  the rows across each window's boundary
- `reduce_ranges`'s sparse table reshapes a row-cut `x` to
  `[blocks; 2; h; ...]` and assembles it at each level (`ranges`, nx.ml):
  where a level's chunk is wider than a device's tile, the reshape reads
  the axis whole onto every device. Build the levels from slices of the
  cut axis, or after the reshape item above
- eager GPU kernel libraries, nx.cuda, nx.metal and nx.amd, onto nx's
  current kernel interface; until then a GPU set computes under
  `Rune.jit` only. Their last code is 64478fedf
  (`dev/nx2/lib/{cuda,metal,amd}`; branch `nx2-gpu` is gone, and only
  `cr/*` and `fx/*` refs reach it: tag it before those are pruned)
- the attention family: `Attention`, `Attention_tangent` and
  `Attention_cotangent` with nx.cpu's forward and cotangent kernels, in
  place of `Nx.attention`'s composition, which stores three float32
  `[lead…; hq; tq; tk]` arrays a call and gives an interpreter only its
  contractions, maps and reduction. Kernels keep hidden values out of
  `o`: a partial tile with a non-finite value takes the selecting path.
  First add and record `attention/host-*` on the composition, and
  `gpu/*/attention-4096-causal` under `Rune.jit` until eager GPU libraries
  land; the family lands only if its kernels beat it by 2x on each,
  forward and cotangent
- donation's kill criterion, at least 2x over the copy on CUDA and AMD:
  `dispatch/kv-append-4096` and its `-copied` twin read 3.4 and 44.5 µs on
  fudge's host; record them on CUDA and AMD under `Rune.jit`, or once
  eager GPU libraries land
- nx.cpu's thread splits follow the call prices (179ea5cc5, 82e42b78d) and
  the kind tables of f6876a0ad, measured under load 1.3-4.7: time them
  quiet on fudge, with a row of a costly kind where its threads change
  (pow-f64-16K), and re-record the elementwise rows
- complex products and extremes over few terms (`few_`, `IN_ARRAY` in
  reduce_body.h) ran 1.2-1.7x slower under gcc 15 on the M1 than before
  their lanes became vectors; inlining the start step for every monoid
  costs the float extremes 15%. Add a row (prod of complex64 2x2 windows),
  time it on fudge's gcc, and act only if it loses there
- a four-step FFT line past `KEPT_BYTES` (job.c) takes fresh scratch every
  call, which faults its pages in on Linux: 528 faults a call at 2^22
  complex64 points. The grid in the result (uncommitted in ~/wt/fft3b on
  fudge, on pre-squash 70b99a9fa) took 0.685 the time on six fudge cores
  and 1.03-1.07 on the M1. Weigh it against keeping large scratch in
  job.c, then port the winner; A/B c2c-4M-c64, c2c-64K-c64 and
  c2c-32x64K-c64 quiet on fudge, test_fft passing

## talon, hugin, brot, munin, quill

- talon benches: record the query suite's DuckDB/Polars baselines on fudge
  (H2O 1e7/1e8, decision support SF1/SF10; none committed), and bless
  `Talon_csv/read-mixed` on fudge and the Mac
- talon.thumper's fudge words (4e89c1604) predate talon's move to the new
  nx (cfc9ebeaf over 952791574): d38d9e44d gives them exactly, f5b5400b9
  gives main's. The rise is the new nx's words per operation, not more
  work in talon. Nx.unique is about 25 operations where the old nx's was
  one backend group: 1,437 -> 6,956 words (Group, Join, Nulls/group-sum).
  One take costs 58 -> 191 words, a slice 44 -> 179, a sum 94 -> 279; a
  read takes a claim and waits (parse-float64 +150). Sort and the other
  Nulls rows fell. Re-record the section's words with its next re-record;
  fewer words per nx operation would show in these rows. Filter's words
  vary by up to 175 a call through talon's weak Expr.Table, before and
  after, so thumper may sample them
- talon perf, machine and commit unrecorded: vs the faster of
  DuckDB/Polars, decision support SF1 13.2x, H2O group-by and join 2.8x;
  vs pandas 1.36x on filter, 2.05x on join. Re-measure on fudge once the
  query baselines exist
- talon 1.0 plan (_plans/talon/product/plan.md, on the Mac): windows and
  ewm, as-of and interval joins, interval sets and folds, zoned time,
  `cut` and temporal casts, hash/sample/shuffle, Pivot and nesting, dense
  round trips, partial aggregation, `Str.trim`/`concat`/regex, compressed
  and fixed-width CSV, Parquet nested columns, datasets and writer,
  talon.arrow, talon.jsont, complex and validity columns, HTML display
- brot vs gigatoken (machine unrecorded): gpt2 2.5x one thread, 3x
  multi-thread; SentencePiece pipelines 40-58 MB/s (added-token scan 28%,
  normalizer 15%); bert unfused. Re-measure on fudge at a stated gigatoken
  commit

## ymir, sowilo

- ymir bench: only the Mac has a section, recorded 2026-10-07 under OCaml
  5.5.1 before rig. fits-image-4096/raw-int16-rice reads 4435930 words
  there and 995245 on fudge; its +4.9% on the Mac did not show on fudge
  from fe69dde74 to main. Record a fudge section; re-record or drop the
  Mac's

## repo

- thumper: suites whose only x86 section is kimchi's (retired):
  kaun, gpt_oss, vega, sowilo; ymir has no
  x86 section. Record each on fudge, then delete every kimchi section,
  nx's and tolk's included. nx.thumper's fudge section was recorded under
  OCaml 5.5.1, so it compares nothing under 5.5.2: re-record it on a quiet
  fudge. talon.thumper's Mac rows predate the Nx.Rng
  fixture (76105a215): the maintainer re-records them
