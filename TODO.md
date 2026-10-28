# todo

goalpost: jit-compiled gpt-oss at least as fast as {tinygrad,JAX,llama.cpp/vLLM} on {NVIDIA,NV,AMD}.

gpt-oss-20b, CUDA (kimchi, ebfb61e268, before tolk ran on rig), raven /
llama.cpp a868c3e:
- decode 9.50 ms / 5.92 ms: attention and lm-head weights as Q8_0 as GGUF
  does (~-2.2 ms, changes numerics); small kernels ~0.3 ms per step
- prefill-512 0.188 s / 58.6 ms: profile what's left after tensor cores
- first call 0.220 s with filled caches (first step 0.115 s)

gpt-oss-20b, AMD gfx1201 (nonnormal, 0dbf76af6a, before tolk ran on rig),
raven / llama.cpp (Vulkan; ROCm):
- decode 21.3 ms / 5.67; 7.41 ms
- prefill-512 1.63 s / 0.180; 0.125 s
- first call 0.39 s (12 tokens, kernels cached; 1.43 s compiling them)
- llama.cpp b11344 Vulkan, b11342 ROCm; tinygrad needs Resizable BAR here

- re-record both on rig: GptOss rows and first call on fudge (kimchi's CPU
  and GPU; kimchi is retired) and an AMD host, llama.cpp at a stated build
- the port's gates on rig: gpt-oss decode and first call, sofo/symo, rune's
  test_jit on CUDA, NV and AMD, the tolk, rune, nx and kaun decode benches
  against the pre-port main; remat memory over its bound on NV/AMD; jit
  replay's words on x86; the engine's wrong answers on Metal+host staged
  loops

## rig

### drivers

- rig.amd over PCI on SDMA 4: program the page queue (regSDMA_PAGE_*,
  doorbell sdma_engine0 + 1, as sdma_v4_4_2 does) as the copy queue's
  second slot once an AMD host runs it, and drop its 16 MiB ring
- rig.pci: a peer GPU's memory mapped through its BAR is given back by
  its owner's free while a peer that may still master can write it; a
  fix needs the two GPUs' locks taken together
- rig.nv: from the 22nd loss in one process, each lost device keeps
  568 MiB of GPU memory (RTX 5000, kernel path); cause not isolated
- rig.nv: memory a failed `make` could not free stays for the process,
  and the open's `Error` does not say so (the kernel path keeps its
  client)
- rig.amd, rig.pci: `peer`, promised to take no lock, takes AMD over
  PCI's `machines_lock` and, without an IOMMU, rig.pci's `routes` lock; a
  callback inside those sections gets `Invalid_argument`
- every library below rig keeps its state whole under an exception: about
  200 sections take a resource and record it after an allocation, or raise
  with a change half made; each allocates its record, takes the resource,
  then publishes with one store
- rig.remote: one Ctrl-C ends an agent's `serve` two ways: inside a
  command it answers `Error "Stdlib.Sys.Break"`, in `serve`'s own waits it
  raises. An agent idle in its wait for a controller sees the Ctrl-C only
  when a controller arrives
- Windows: rig's signal mask blocks nothing there, so a Ctrl-C can land
  inside sections below rig's core that Linux and macOS keep it out of,
  such as rig.cuda's page-lock registry under a contended lock

### limits and size

- NV address spaces: Device memory in the 49-bit space on the PCI path

### API

- copies: nx2's placement, allocation and assembly through
  `Buffer.submit_copy`, boxes and `Buffer.submit_fill`; done when placing on
  N GPUs costs one wait
- GPU memory between machines over RDMA (GPUDirect), as nx.rdma had: a
  copy between two machines' GPUs runs through rig.mlx5 or rig.bnxt, not
  TCP, over the memory `Rig.Buffer.dma` describes

### tests and runs

- relock windtrap once its `workers-block` branch (a883e72, a start
  barrier that sleeps once workers stop arriving) lands: rig_pci.space
  "two domains allocate and free at once" times out at its 10 s bound in
  60 of 64 copies run 32 at once at load ~65 with windtrap's spinning
  barrier, and in none of 96 with it
- hardware runs: the box and fill laws on SDMA 4 to 6; two GPUs (peers);
  mlx5 and bnxt NICs

### perf

- AMD boxes as one COPY_LINEAR_SUB_WIND packet per 2^14 rows on SDMA 4-6,
  where they run as a COPY_LINEAR per row: measured on an AMD host against
  rows; it also lifts the row limit of a box on SDMA 4 over PCI, about
  170,000 rows, past which a box never fits the ring
- a host buffer's creation: every fresh nx result mallocs through
  `caml_rig_heap_bytes` and allocates a custom block (`buffer/host-create-16`
  54 ns, 31 words on the M1; nx's `dispatch/add-1` 214 ns): profile malloc
  against the custom block and `host_memory`, then A/B a per-domain free
  list of small host buffers
- kaun's GPU decode rows measure allocation again (bench_decode.ml's
  gpu_metrics) once a decode call's words are shown not to depend on the
  collector's phase, on Metal and CUDA
- AMD 4 KiB prepared copies are 17-26% slower since 70cca2ae5's wait change
  (the compute route itself starts at 2 MiB): name the row, explain it or
  revert the wait change for small copies
- rig.disk: each transfer pays about 220 ns for blocking signals below
  rig's core (5493c9821): two sigmask calls, and a runtime release at the
  call's end, about 14 ns of read/4K's cost. read/4K went 246 to 481 ns,
  write/4K 315 to 547 ns on fudge. Find a mask that a short transfer does
  not pay twice
- rig.disk's open/many-128 counts the frees of earlier rounds' buffers,
  which `of_io`'s drain runs when the GC has collected them, so its
  `exact` word count moves with `OCAMLRUNPARAM` and the rows run beside
  it: drain outside the measured window

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
  barrier against `pthread_mutex_lock`. nx2's P-core cap, on this CPU
  (kimchi, 2026-10-01), timed large GEMMs at x0.96-1.04 and batched
  64x32x32 at x1.59 slower. The M1 Max's cap at its 8 P-cores
  (`hw.perflevel0.physicalcpu`, as nx2's) needs the same runs on the Mac
- tolk.engine: no GPU test of a loop whose body mixes a GPU kernel and a
  host program
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
- Metal groups: after D144, lorenz-step-batch-1024's r_3_256_16_64_64 went
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

- compiled `eigh` skips a pair below u·‖A‖_F (`norm_roundoff`), svd below
  u·‖a‖‖b‖ per pair: two rules for one cause. One relative floor in
  `tangents`, |γ| ≤ u·√(|α||β|) (Demmel and Veselić, LAPACK Working Note
  15; eager's split test), removes `norm_roundoff`, its reduction and
  `lower_linalg.mli`'s graded-matrix carve-out. `norm_roundoff` overflows
  today: a float32 matrix with diagonal 2.5e38 and finite eigenvalues
  returns NaN
- baselines on fudge: the `compile` suite's four `trig/` rows, and a
  re-record of the `compiled` suite's float64 sine rows
  (`transcendental/sin*`, `special/sin*`), whose kernels now take vector lanes
- a compiled `Nx.take` gradient costs the table times the tokens:
  `indexed/take-grad-1024-of-32768x64-host` takes 94 ms on the M1 (load 20)
  against ~2 ms eager. At 128256 rows a plain loop measured 17 ms and thumper
  0.67-1.05 s, unexplained. The row has no baselines yet (Mac, fudge)
- jit compiles no complex tensor (lower.ml:36); ymir's `test_quantity`
  holds an `xfail`
- vega keeps optimizer state at the leaf dtype: a bfloat16 moment never
  decays by less than half its spacing (vega.mli:98). Design pass
- compiled Cholesky stays unrolled (lower_linalg.ml:613): compile grows with
  n (35 s at n=32); as a loop it ran 1.5x slower at n=11
- the `compile` bench suite takes ~2.4 min quiet on the Mac
  (`log_betainc-grad-b` 13 s, `betainc` 6 s, `sinkhorn-64` 3.8 s)
- kaun's TrainStep/train-step allocates 27,332 words (22,460 before the
  redesign), ~2,300 of them two traced values and a recorded op per op:
  re-record on the Mac and fudge, and cut the per-op records
- eager rune links tolk (rune's library lists tolk and tolk.engine): the
  compile path behind its own module (_plans/tolk/linkall-design.md)
- an `iterate` that emits rows; jera's `Ode.path` scans every `budget`
  slot (ode.ml:961). Design pass
- fully sharded training: RFC 0005's bound (P+G+O)/n + 2 layers has no test
  since the rune/tolk rewrite
- kaun: 05-llama loads a GGUF checkpoint (`Nx_io.load_gguf`, Q8_0/Q4_K/Q6_K
  in `Nx_quant`) and matches the reference's logits
- tests read other packages' test trees: rune reads nx's goldens and links
  nx's and tolk's test kits; ppx_ptree's suite links rune, vega and kaun
- on a cold tolk cache, rune's test_special takes 348 s (38 s warm) and
  test_laws' "a gradient through a compiled function" 28 s (3 s warm),
  past its 60 s timeout under load: the user's cache hid their compiles

## nx

- nx.wide's `chunk_levels` was tuned when each chunk compiled two kernels:
  time `compile/wide/sum-1e6` at 2-5 levels
- `betaincinv`, once tolk compiles long programs (~20-25k ops)
- GC stress for nx's C stubs: run them once under the stressed collector
- an A/B on fudge of the base-handler design (branch `effect-base`,
  bfe681ebd) against main on the eager rows decides whether it lands, and
  with it the public `Nx.Op.request`/`Nx.Op.handle`: no perform beside
  another domain's extent, no `Effect.Unhandled` raised

## nx2

- the kinds quiet the NaN they return. `nx_nan1`, `nx_nan2`, `nx_maximum`
  and `nx_minimum` keep a signalling NaN's bits (`nx_kinds.h`), while
  reductions, scans, scatters and ranges promise it quieted, so each Max or
  Min path quiets again (`reduce.c`'s `quiet_*`, `index.c`'s `QUIETED`). IEEE 754-2019 §6.2.1 and §9.6 quiet it in the
  operation. Done when the kinds quiet it and those wrappers are gone;
  elementwise bits change for signalling NaNs
- `reduce_ranges` flattens its result to quiet it, which gathers a result
  cut along its row axis onto every device; its sparse table's `assemble`
  gathers a row-cut `x` at every level
- a reshape that factors or merges a cut axis gathers it whole onto every
  device where the tiles don't divide both extents: gpt-oss's 2880 inputs,
  90 MXFP4 blocks, over 4 or 8 devices. A halo exchange, as GSPMD's
  reshape realigns an uneven cut (arXiv 2105.04663 §4.1), would move only
  the rows across each window's boundary
- a check on another machine's device waits for its step: settling reads
  its result with `Nx_array.to_device`, whose `Rig.Buffer.copy` follows
  every value submitted there. Done when rig offers a read of a remote
  device's memory that doesn't follow its later work

- frontend, `fuse` and `widen` (`float_fn`, `Ptree.fuse`): one rounding
  per element, bit for bit with `f ()` on one kernel library
- frontend, symbolic shapes: `Size.v`, `Size.dims`, `Ptree.sized`; `shape`,
  `dim` and `numel` raising on a symbolic axis under a trace; inheritance
  through `_like`, movements, broadcasting and pattern groups
- `Nx.attention`'s error bound: derive it for the identity score, state it
  in nx.mli, and tie the law's tolerance in `test_attention.ml` to it
- the attention family: `Attention`, `Attention_tangent` and
  `Attention_cotangent` with nx.cpu's forward and cotangent kernels and
  `Prog.status`, in place of `Nx.attention`'s composition, which stores each
  head's `tq × tk` scores and gives an interpreter only its contractions,
  maps and reduction. Kernels keep hidden values out of `o`: a partial
  tile with a non-finite value takes the selecting path. Record
  `attention/host-*` and `gpu/*/attention-4096-causal` on the composition
  first; the family lands only if its kernels beat it by 2x on each,
  forward and cotangent
- frontend, `contract`'s `?a_scale`, `?b_scale` (`scale`) and `?segments`,
  after `Prim.Contract` and nx.cpu take them
- record `dispatch/kv-append-4096` and its `-copied` twin; donation's kill
  criterion is at least 2x over the copy on CUDA and AMD
- order and counting operations: check their contracts, public laws and
  bench baselines
- nx.io on nx2, then Rune over nx2 with tolk lowering, before the GPT-2 and
  gpt-oss comparisons; rune.mli refuses a traced value on "another fiber",
  where `Nx.Prim` refuses one on a fiber a scheduler outside the
  interpretation forks: state rune's boundary as nx's
- the GPU kernel libraries, nx.cuda, nx.metal and nx.amd, from branch
  `nx2-gpu` (`dev/nx2/lib/{cuda,metal,amd}`) onto nx2's current kernel
  interface
- nx.ragged and nx.wide on nx2
- nx.cpu's jobs size their threads by older cost estimates, many below the
  call prices' measured kind and term costs, so costly kinds get too few
  threads (a float64 pow over 16 K elements runs on 3): derive the job
  costs from the prices' tables and measure them on thread-count rows
- complex products and extremes over few terms (reduce.c's few_, e.g. 2x2
  windows) run 1.2-1.7x slower under gcc since their lanes became vectors:
  gcc vectorises IN_ARRAY's 16-lane start step only inside few_'s own loop
  nest, not as start_'s (M1, gcc 15: prod few complex64 1.17 -> 1.78 ns an
  element); inlining it for every monoid costs the float extremes 15%:
  find a form that vectorises both. gcc's complex64 one-block product
  (prod block1) runs 1.3x slower too, untraced
- the FFT's four steps on fudge: c2c-64K-c64 and c2c-32x64K-c64 read 1.02
  and 1.03-1.04 against the commit before the column and row steps (6
  P-cores), measured with the grid-in-result code in the column step:
  re-time without it and remove what remains
- the four steps' grid in the result past the largest cache, where malloc
  maps blocks per call (Linux): the columns write it in the result's
  layout, both steps streaming, the rows transform it in place; 2^22
  points took 0.685 the time on six Arrow Lake cores (528 faults a call
  fewer) and 1.03-1.07 on the M1, which keeps scratch. Land it with
  test_fft passing on fudge
- linalg's v3 build loses 1.5-4% at order 64 on fudge (eigh-f64-64x32
  1.040, svd-f64-64 1.026, svd-f64-64x32 1.015); gcc's 128-bit vectors in
  linalg_v3.c recover it but cost lu-f64-1024-pivoting 7%: find the loops
- svd at order 64: the panel reduction's serial path (labrd) costs 2% on
  fudge (svd-f64-64 1.023, svd-f64-64x32 1.019): find what the in-order
  chunks pay over one pass
- MR-aligned GEMM row subdivision
  (`refs/audit/20261010-deep/branches/few-units`): public matmul A/Bs and
  baselines on the Mac and fudge
- engine dispatch costs: staged patches and the missing `dispatch/real-1`
  and `dispatch/sum-1` rows in `refs/audit/20261010-deep/worktrees/wt-13`;
  per-stage A/Bs and baselines, compiler internals private

## norn, jera

- jera: Newton stops at the floats' limit by three rules (`still`, the
  merit-rounding branch, `at_limit` with `bounded`), because `still` floors
  each component at |x|·ε, which is 0 at 0. One test in norm in
  `Search.judge` (MINPACK's `hybrd`: a step within xtol·‖x‖) covers Root,
  System, Broyden, Anderson and Minimize, which miss it today. Meanwhile
  `at_limit` reads a step that did not move (q = r = ∞) as at the limit,
  and the unscaled `Search.norm` underflows, so a lane of small unknowns
  converges 100 times off its tolerance; `resolved` converges without
  meeting `tol`; `jera.mli` states the limit rule for every method
- eager NUTS: ~3.3 ms per transition, ~475 µs per leapfrog against HMC's
  ~110 µs on the same density. Profile its bookkeeping; add a row
- jera eager `Sde.march`: 2,000 lanes at Brownian depth 12 take ~6 ms a step
- Law 10 flags: HMC's one pooled step size strands chains (a tenth of 128
  sblrc chains), and the stretch move barely moves gp_pois_regr. Per-chain
  adaptation
- Nested: slice moves correlate draws within a replacement, so its error
  under-covers 1.5-1.9x on gp_pois_regr
- a record parameter needs a hand-written walk (or ppx_ptree) to be an
  `Nx.Ptree`, and Summary rows lose its field names. Design pass
- variational layer: `Gaussian.pathfinder` and `importance`, `Vi`,
  `Diag.psis` and `loo`, `Summary.of_weighted`, each checked by k̂
- distributions: `binomial`, `von_mises`, `lkj_cholesky`, `truncated`,
  `hmm` with `hmm_marginals`, and an example that sums an HMM's states out

## talon, hugin, brot, munin, quill

- talon vs the faster of DuckDB/Polars: decision support SF1 13.2x, H2O
  group-by and join 2.8x; pandas 1.36x on filter and 2.05x on join.
  `Nx_ragged.take` moves ~40 bytes per byte of text and dominates joins;
  test_run 1.2 s
- talon benches: commit the query suite's DuckDB/Polars baselines (H2O
  1e7/1e8, decision support SF1/SF10) and a CSV read row; GC stress for its C stubs
- talon 1.0 plan (_plans/talon/product/plan.md): windows and ewm, as-of and
  interval joins, interval sets and folds, zoned time, `cut` and temporal
  casts, hash/sample/shuffle, Pivot and nesting, dense round trips,
  partial aggregation, `Str.trim`/`concat`/regex, compressed and
  fixed-width CSV, Parquet nested columns, datasets and writer,
  talon.arrow, talon.jsont, complex and validity columns, HTML display
- talon under `Rune.jit`/`vmap`: state which verbs raise, and test filter,
  sort, group and join under jit
- hugin: stacked marks and time axes in hugin.mli; legends with `~ncol` and
  centred placements; CFF and CFF2 outlines; hugin_plot bench baselines
- brot vs gigatoken: gpt2 2.5x one thread, 3x multi-thread; SentencePiece
  pipelines 40-58 MB/s (added-token scan 28%, normalizer 15%); bert unfused
- brot: added tokens encoded as text; an incremental decoder; walkers for
  o200k and Tekken; Kimi's class intersection and `\p{Han}`; T5's
  Precompiled normalizer; the parity tests skip without downloaded
  tokenizers: commit trimmed ones with their generator
- munin: GPU metrics (NVML by dlopen, IOKit, AMD sysfs); 7 doc fences
  mdx skips (installs, `munin` command lines naming generated run ids, the
  Kaun MNIST loop), and the guides skip Windows, which munin.sys lacks

## ymir, sowilo

- ymir.fits: HCOMPRESS_1 (Legacy Surveys ships maskbits and nexp in it);
  `fits-image-4096/raw-int16-rice` allocates +4.9% over its baseline, cause
  unknown
- ymir: HEALPix and spherical harmonics (RFC 0027 slice 3); time scales,
  time-dependent frames and ephemerides
- sowilo: `threshold` and `canny` return `Nx.bool_t` and morphology takes a
  bool kernel

## repo

- compress stays: decide its home, in nx or its own library, through a
  design campaign that audits its references (formats' specs, zlib, zstd,
  lz4, snappy) and holds it to rig's bar; `Nx_io.gunzip` moves with it
- private modules: each gets an .mli per doc/dev/documentation.md, rig as
  the reference. Left, in rig's lib/: amd/amdgpu/request.ml,
  nv/pci/gsp_defs.ml and the `defs.ml` of amd/abi, amd/pci, bnxt, mlx5,
  mlx5/abi, mlx5/uverbs, nv, nv/abi, nv/nvidia and nv/pci
- windtrap: its per-test timeout does nothing on Windows; release 0.3.0 and
  drop the git pins in dune-project and ci.yml
- tests: CI runtest under a memory cap
- talon bench: re-record Talon/{Filter,Group,Join,Sort} in talon.thumper on
  the Mac and on fudge (kimchi's section is stale): their fixture is now
  drawn from Nx.Rng
- dune stale builds: a source rewritten within seconds is digested in one
  version and copied in another; a passive server stops seeing changes yet
  answers `rpc ping`; under load it loops on "Event queue overflow".
  _plans/dune-stale-watch/
