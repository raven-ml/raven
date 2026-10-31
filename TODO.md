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

- rig.pci: a peer GPU's memory mapped through its BAR is given back by
  its owner's free while a peer that may still master can write it
  (`give_back`, pci/memory.ml); a fix needs the two GPUs' locks taken
  together
- rig.nv: memory a failed `make` could not free stays for the process,
  and the open's `Error` does not say so: `make`'s `back ()` drops every
  `Fault` (rig_nv.ml). The kernel path keeps its client
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
- rig.remote: two waits of an agent's serving thread take signals but
  never return to OCaml before they end, so a Ctrl-C lands only then: a
  join's `unblocked Thread.join` of its dials (agent.ml, at most `join_s`,
  10 s) and `Link.release_rail`'s untimed `pthread_cond_wait` for the
  rail's active transfers (rig_remote_link.c). Bound them as the waits for
  a controller and a command are (`RIG_REMOTE_STILL_NS`)

### limits and size

- NV over PCI: every GPU's memory, Device and host-mapped, comes from one
  range, 64 GiB to `Rig_nv.windows` (about 952 GiB, rig_nv_pci.ml),
  shared by all GPUs: eight 141 GB GPUs don't fit. Place Device memory in
  the GSP's 49-bit space (gsp.ml), as e81be363c did for the kernel path;
  only segments, launch descriptors and semaphores need 2^40

### API

- copies: nx's placement, allocation and assembly through
  `Buffer.submit_copy`, boxes and `Buffer.submit_fill`; done when placing on
  N GPUs costs one wait
- a read of a remote device's memory that doesn't follow its later work:
  `Nx.check` on another machine's device settles through
  `Nx_array.to_device`, whose `Rig.Buffer.copy` follows every value
  submitted there, so a check waits for its step
- GPU memory between machines over RDMA (GPUDirect), as nx.rdma had: a
  copy between two machines' GPUs runs through rig.mlx5 or rig.bnxt, not
  TCP, over the memory `Rig.Buffer.dma` describes

### tests and runs

- two-domain `stateful` tests time out on a loaded machine: test_space's
  "two domains allocate and free at once" and test_memory's "two domains'
  allocations stay within the budget" take 0.1-0.3 s, but 6 s and 50-60 s
  pinned to one core. rig's code does not spin. windtrap's start barrier
  (`Workers.run_job`) spins on `Domain.cpu_relax` without bound, a
  scheduler slice (3-5 ms) per run when the workers share a core: to fix
  in windtrap, as a bounded spin then a block. Each budget refusal calls
  `Gc.full_major` twice (`Memory.reclaim_round`), whose stop-the-world
  sections loop in every domain, backup threads included, until all
  finish: 60 ms a call on one core with two idle domains, 0.15 ms
  unpinned. Then decide whether the budget test needs 16 x 50 runs
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
- a host buffer's creation: every fresh nx result mallocs through
  `caml_rig_heap_bytes` and allocates a custom block (fudge:
  `buffer/host-create-16` 36 ns, 31 words; nx's `dispatch/add-1` 207 ns,
  51 words). Below `heap_kept_from` nothing is kept: profile malloc
  against the custom block and `host_memory`, then A/B a per-domain free
  list of small host buffers
- kaun's GPU decode rows measure allocation again (bench_decode.ml's
  `gpu_metrics`) once a decode call's words are shown not to depend on
  the collector's phase: on fudge (CUDA) and nonnormal (AMD)
- AMD 4 KiB copies: `copy/amd/within-device-4K` on nonnormal is bimodal,
  7.9-9.2 µs in 9 samples and 14.6-15.4 µs in 11 (rig-gpu.thumper),
  against fudge's 5.5 µs CUDA and 4.0 µs NV. The copy kernel starts at
  2 MiB, so this is SDMA and the wait: find what splits the modes
- every call below rig's core pays about 270 ns for its signal mask
  (5493c9821): two sigmask calls and a runtime release, the release about
  14 ns (561a0b34c). fudge: rig.disk read/4K 246 to 481 ns, write/4K 315
  to 547; copy/staged-4K +51%, device/open-close-memory +41%,
  drain/free-cache-1000 +42%. Find a mask in rig.edge that a short call
  does not pay in full

## tolk

- `upcast_vectors` (heuristic.ml) has no term for a kernel's size: 8 nested
  float64 sines over 1024 values take 8 lanes, run 4.5x as fast, and compile
  in 2.2 s against 1.4 s without lanes. Done when an A/B on fudge of
  `trig/sin-nested-8` and the `compiled` suite's sine rows decides whether a
  kernel's size bounds its lanes
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
  64x32x32 at x1.59 slower. The M1 Max's cap at its 8 P-cores
  (`hw.perflevel0.physicalcpu`, as nx's) needs the same runs on the Mac
- search timing on NV clears no L2 (tinygrad's invalidate_caches); AMD
  counters and thread traces are not collected: encode both as words parts
- AMD machine code without comgr (_plans/amd-isa/design-reviewed.md): an
  assembly renderer and assembler, gfx12 first, the default once its gate
  passes; HIP stays as a choice. First read the terms of AMD's ISA XML
- divergences: for each entry of tolk's and rune's DIVERGENCES.md, look in
  references (tinygrad HEAD, papers, other compilers) for a better
  solution; keep it with its reason or replace it
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
- compile time and launch counts: a NUTS program takes 9-14 s to build, a
  ymir aperture gradient 98 s cold; loop bodies store each carry and row in
  its own kernel (drawing-scan-256: 6 per trip against 2). Multi-output
  fusion of loop bodies
- long programs: the incomplete beta (~11,400 ops) compiles in 3 s, 10-20 s
  per derivative, past 10 min for three; a 1,000-term slice-tree sum is one
  kernel clang -O2 takes over 8 min on
- host upcast width per target: D141's 64-byte lanes (heuristic.ml:446)
  slowed memory-bound select-zero ~15% on x86; time 32 against 64 bytes on
  fudge
- an expensive elementwise producer (sin/cos) recomputed in each consumer's
  kernel: jera root-newton-kepler-1k compiled 2.2 ms against eager 0.57
- running sums cost ~257 ops per element (shape.ml:955): `Nx.cumsum` over
  float64 [100; 50500] takes ~1.0 s on CPU. A scan with constant work per
  element, and a compiled cumsum row
- Metal groups: after D137, lorenz-step-batch-1024's r_3_256_16_64_64 went
  10.7 -> 18.9 ms (loads stride across 16 threads)
- AMD beam search times 16-21% of candidates below 2 GHz: hold rig's stable
  power state around a search, or ties within noise
- comgr's per-process setup is paid on every compile: each HIP compile
  starts a process (D5) that loads comgr and compiles once. NVRTC's
  processes serve compiles in turn (`Compiler_worker.call`), which took
  fudge's search/nv from 1.90 s to 1.34 s, its time in the process with
  NVRTC's cache disabled. Move comgr to
  serving workers once nonnormal times search/amd and a first call both
  ways
- a link reloads images an earlier link loaded: measure on fudge the load
  cost a search pays, and if it matters, design how rig.program takes loaded
  images (a description entry naming a Rig.Image.t)

## rune, kaun, vega

- vega keeps optimizer state at the leaf dtype: a bfloat16 moment never
  decays by less than half its spacing (vega.mli). Design pass
- kaun's TrainStep/train-step allocated 27,332 words eagerly on the old
  rune, ~2,300 of them two traced values and a recorded op per op:
  re-measure on the new rune on fudge, and cut the per-op records if they
  remain
- an `iterate` that emits rows; jera's `Ode.path` scans every `budget`
  slot (ode.ml's `slot` mask and `put_row`). Try a scatter at a run-time
  index first, as kaun's decode cache compiles. Design pass
- fully sharded training: RFC 0005's bound (P+G+O)/n + 2 layers has no test
  since d478112cb removed the counting-device test
- kaun: 05-llama loads a GGUF checkpoint (`Nx_io.load_gguf`, Q8_0/Q4_K/Q6_K
  in `Nx_quant`) and matches the reference's logits
- a fresh _build starts each test directory's tolk cache empty
  (66f223fcd): the old rune's test_special took 348 s cold (38 s warm).
  Re-time the new rune's suites cold on fudge, and fix the slow ones at
  their cause

## nx

- words per eager operation, old nx -> new (talon's Gc.Memprof, fudge):
  take 58 -> 191, slice 44 -> 179, reshape 25 -> 91, sum 94 -> 279, cumsum
  62 -> 207. Each operation routes through `Prim`, computing its layouts
  when routing and again when executing, and each composite builds and
  lowers its own small program with two hashtables; `Nx.unique` composes
  about 25 operations (1,437 -> 6,956 words). Profile `dispatch/reshape-1`
  and `index/take-rows-1K-of-8K`, and cut the per-operation words; talon's
  Group, Join and Filter rows show it
- `Nx_ragged.take` writes an int64 position per output byte (fill,
  scatter, cumsum, gather): ~40 bytes moved per byte of text. It dominates
  talon's joins on text (join_run.ml, column.ml)
- `Nx.betaincinv` and `betainccinv`, as the gamma's inverses are
  (0947081db left them out): norn's `quantile` raises for `beta` for want
  of them. First measure rune compiling a program of their size (~20-25k
  operations)
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
- symbolic extents: `Rune.jit` keys on each array's shape, so a decode
  step retraces at each new context length or pads to a window, as kaun's
  cached attention does; rune's lowering refuses tolk's `Sym` extents.
  Design it from references before nx, rune and kaun take it
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
- `Nx.contract`'s `?a_scale`, `?b_scale` and `?segments`: `Prim.Contract`
  and nx.cpu take them and nx.quant calls them through `Prim`. Decide by
  design.md's vocabulary rule whether a user needs them from `Nx`
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

## norn, jera

- jera: Newton stops at the floats' limit by three rules (`Search.still`,
  System.newton's merit-rounding branch, `at_limit`), because `still`
  floors each component at |x|·ε, which is 0 at 0. MINPACK `hybrd`'s test
  in norm (a step within ε‖x‖ replacing all three) does not fit the
  componentwise `Tol`: it ends a lane at its largest components'
  resolution. Picard (`System.anderson ~memory:0`) on (z − x) / 2,
  z = (1, 1e-3), from (1, 0) then stalls 1000 ulps off z₂ under
  `Tol.rel 1e-13` or `Tol.ulps 256`, where the three rules converge it
  within 2 ulps, and `anderson ~memory:2` in test_system "compiled equals
  eager" stalls before the exact zero it reaches. A test in the
  tolerance's own scaling might hold both
- jera: three sums of squares still underflow for unknowns or residuals
  below about 1e-154: Newton's merit `half_square` (system.ml), where φ0,
  φ and the slope all read 0 and Armijo accepts every full step
  (f x = s atan(x / s), two unknowns from (3s, 0), s = 2^-540, ends not
  ok); Broyden's `ss = dot dx dx` (system.ml), which never updates B; and
  Levenberg–Marquardt's cost and predicted decrease (minimize.ml,
  `lm_problem`'s `cost` and `lm_trial`), where ρ = 0/0. A merit scaled by a
  per-step constant 1 / max |f x_k| leaves Armijo and the rounding test
  invariant
- jera: the contraction is measured on the whole vector's norm and
  multiplies each component's |δᵢ|, which understates a small component
  coupled to a large one. `Minimize.lbfgs ~memory:3` on
  0.25 Σ(x² − a)² from √a(1 + 1e-6) converges up to 17 times off
  `Tol.ulps 0.5` (30 of 300); `System.anderson` with memory 0 and 3 on
  M (a − x²) / (2√a), M = [[1, .5], [−.5, 1]], up to 220 and 6.5 times.
  At the floats' limit the error |δ| / (1 − q) takes the last settled q,
  whose upper end over a step of a few ulps can read 0.93 where the step
  before measured 1e-8: Newton on √a at `Tol.ulps 4` stalls 2 elements
  in 1000 held to an ulp. A limit step's bound on q cannot be told from
  f's evaluation error growing, so lowering q on it converged a lane 120
  ulps off
- norn NUTS: ~3.3 ms per eager transition, ~475 µs per leapfrog against
  HMC's ~110 µs on the same density (machine unstated). Only compiled
  rows exist (nuts/schools/64 3.44 ms, M1 Max). Add an eager row, measure
  on fudge, profile the bookkeeping
- jera eager `Sde.march`: ~1.3 ms a step at 100 lanes, depth 8
  (sde-euler-maruyama-100/eager, M1 Max); 2,000 lanes at depth 12 took
  ~6 ms. Each step descends the Brownian tree twice from the root and
  runs two eager `check_time` reads; carry W and I at a step's end into
  the next step's start
- Nested: slice moves correlate draws within a replacement, so its error
  under-covers 1.5-1.9x on gp_pois_regr (documented in
  `Evidence.expectation`, still a biased error)
- norn: `Diag.psis` and `loo` over `Norn_model.pointwise`, with each
  point's k̂; `Summary.of_weighted`
- norn: variational layer: `Gaussian.pathfinder`, `importance` (which
  `Weighted`'s doc already names as a source), `Vi`, each checked by k̂
- norn: `poisson`, `neg_binomial` and `binomial`'s log mass cancels O(n)
  terms, about 8.5e-3 nats off at float32 and n = 1e6: a saddle-point form
  (Loader 2000, stirlerr and bd0) across the discrete families
- distributions: `von_mises`, `lkj_cholesky`, `truncated`,
  `hmm` with `hmm_marginals`, and an example that sums an HMM's states out

## talon, hugin, brot, munin, quill

- talon benches: record the query suite's DuckDB/Polars baselines on fudge
  (H2O 1e7/1e8, decision support SF1/SF10; none committed), and bless
  `Talon_csv/read-mixed` on fudge and the Mac
- talon.thumper's fudge words (4e89c1604) predate talon's move to the new
  nx (cfc9ebeaf over 952791574): d38d9e44d gives them exactly, f5b5400b9
  gives main's. The rise is the new nx's words per operation, not more
  work in talon. Nx.unique is about 25 operations where the old nx's was
  one backend group: 1,437 -> 6,956 words (Group, Join, Nulls/group-sum).
  Nx_ragged.take's Nx.positions costs 5,493 at 514 rows (Filter, Join).
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
- talon under `Rune.jit`/`vmap`: state which verbs raise, and test filter,
  sort, group and join under jit
- hugin: stacked marks and time axes in hugin.mli; legends with `~ncol` and
  centred placements; CFF and CFF2 outlines; hugin_plot baselines on fudge
  (no hugin_plot.thumper exists)
- brot vs gigatoken (machine unrecorded): gpt2 2.5x one thread, 3x
  multi-thread; SentencePiece pipelines 40-58 MB/s (added-token scan 28%,
  normalizer 15%); bert unfused. Re-measure on fudge at a stated gigatoken
  commit
- brot: an option to encode added tokens in the input as text; an
  incremental decoder; walkers for o200k and Tekken; Kimi's class
  intersection and `\p{Han}` (rejected at load); T5's Precompiled
  normalizer (rejected at load)
- brot: the parity tests skip in local runtest without downloaded
  tokenizers (CI downloads them): commit trimmed ones with their
  generator, each under a licence that allows it
- munin.sys on Windows raises `Sys_error` for network and disk I/O
  counters, per-core CPU times, the process table and the load average:
  read them (GetIfTable2 needs iphlpapi, IOCTL_DISK_PERFORMANCE a handle
  per physical drive, per-core times ntdll, the table a process snapshot)
- munin GPU metrics without NVML (a vendor library, ruled out): amdgpu
  sysfs, IOKit, and NVIDIA through what rig's drivers can read; design
  first

## ymir, sowilo

- ymir.fits: read HCOMPRESS_1 (Legacy Surveys ships maskbits and nexp in
  it) and PLIO_1; both raise today (tiles.ml)
- ymir bench: only the Mac has a section, recorded 2026-10-07 under OCaml
  5.5.1 before rig. fits-image-4096/raw-int16-rice reads 4435930 words
  there and 995245 on fudge; its +4.9% on the Mac did not show on fudge
  from fe69dde74 to main. Record a fudge section; re-record or drop the
  Mac's
- ymir: HEALPix and spherical harmonics (RFC 0027 slice 3)
- ymir: time scales (UTC, TAI, TT, TDB), then time-dependent frames
  (precession, nutation, aberration) and ephemerides

## repo

- thumper: suites whose only x86 section is kimchi's (retired): brot,
  hugin_vg, kaun, kaun_decode, gpt_oss, vega, fehu, sowilo; ymir has no
  x86 section. Record each on fudge, then delete every kimchi section,
  nx's and tolk's included. talon.thumper's Mac rows predate the Nx.Rng
  fixture (76105a215): the maintainer re-records them
