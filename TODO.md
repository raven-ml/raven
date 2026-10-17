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

### semantics

- ledger rows (_plans/rig-semantics-refs/LEDGER.md): the exact-zero tests
  count `Gc.allocated_bytes` per call with no division; `Buffer.free` (M5); a
  staged route between two mapless devices takes both slots in one C call,
  in device-index order (G6, copy.ml:258-264); the controller keeps its
  local devices after a failed job (Q2, agent.ml:118-123); each process
  records its own cause, marked remote or local, and aborts without the
  10 s drain (R3, link.mli:47-58)
- more ledger rows: a forked child loses io devices too and runs none of
  the parent's owed stops (F8, rig_device.c:388-410); host and memory work
  runs after giving the turn up (K2); the doc sentences of F3, F5, O5, Q1,
  R4 and C4 (F3's "Other devices go on", rig.mli:502, is false on CUDA)

### drivers

- NV: each open and close of a device on fudge grows glibc's C heap by
  about 19.6 KB, against rig_nv.mli's 24 bytes: find whether rig or the RM
  leaks it (mallinfo2, heaptrack)
- rig.amd to its spec (_plans/gemini-review-2/REPORT.md,
  _plans/rig-size-audit/): unused encoders and options removed, the rest
  behind `(**/**)`; `report`, `block`, `image` hidden; holes as NV's fixed
  record; `Bus_address` into `Machine`; request.ml inlined; firmware parsing
  total over the pinned set; rig_amd.mli's 5 µs clock bound excludes KFD,
  whose GPU-to-CPU gap is one-sided and unbounded under preemption: state it
- one generator, pins file and excerpt tree per driver (AMD, NV); NV keeps
  615's headers whole (licence checked first) and only `per_release`
  structs for older releases
- `?root` behind `(**/**)` on amdgpu and nvidia
- the SQTT decoder's home (rig.amd.abi, or beside the tools that read
  traces) and whether the per-driver floor drivers stay in the benches,
  each derived from rig's identity and the rule that a row times what a
  user calls

### limits and size

- limits no hardware or spec imposes, one commit each
  (_plans/rig-device-reuse/70-, 71-): AMD's 512 parts per submission;
  AMD's segment grows as NV's segment and launch rings do, and its compute
  and copy rings resize only at idle (`AMDKFD_IOC_UPDATE_QUEUE` or the PCI
  MQD; check KFD keeps rptr/wptr), then `Rig.most_parts` goes; NV's GPFIFO
  keeps 16,384 entries and refuses Words past them: size it from the RAMFC
  field, remake the channel at idle, or state the bound on
  `Rig.Submission.words`; NV's 65,535 parts (the 16-bit index of
  `v << 16 | i + 1`): a per-channel join counter has no bound;
  PCI system memory's arena from physical memory; bnxt sized as Linux sizes
  it
- NV address spaces: Device memory in the 49-bit space on the PCI path;
  find why the RM refuses `NV_ESC_RM_MAP_MEMORY_DMA` (`NV_ERR_NO_MEMORY`)
  past ~39 GiB of 4 KiB-page host mappings on fudge (hardware-notes.md;
  rig_nv_nvidia.mli states the bound): map in larger pages if that is the
  cause
- size cuts in _plans/rig-size-audit/SUMMARY.md §A (core, PCI, NV, small):
  NV's five field accessors, the debugfs scan; classify each unused value
  (speculative, superseded, unpropagated, missed) before it goes
- test cuts in _plans/rig-simplify-audit/tests.md, each shown covered
  before it goes: rows 2, 5, 7 (once rig_pool.h stops stating its
  internals), 11-14, 18 (format models only), 19-26; decide 10, 15-17
- docs: an inventory (file:line pairs) of contracts restated across rig's
  .mli files, then each folded to one place; rig_disk_stubs.c's six
  platform branches into its Windows and POSIX sections; rig_device.c's
  Windows locks against rig_pool.c's pthreads: say why or agree

### API

- claims, one design pass: a claim held past a scope (rune's `Pins` and
  tolk's `link ~bound` go), a take over a buffer list, claim-all in rig.h
- copies: a device copy that returns once queued, copies of several runs or
  a strided box in one call, `fill` on the copy engine, run-value offsets in
  rig_program's `Move` step; done when placing on N GPUs costs one wait
- from its callers: `Image.load` reuses an equal image once GPU module
  globals are shown not shared (CUDA) or refused; `Profile.take` raises
  `Lost`; `Rig_program.load` returns `Error` only for a device's refusal
- rig.gpus: one card appears through two paths, and `openers` drops each
  path's refusal reason and names paths, not GPUs; `Rig_pci.Gpus`, which bnxt's
  NICs use too, becomes `Rig_pci.Driver`: a user-space PCI driver's match,
  hold, reset and stop, as Linux's `struct pci_driver` (API review)
- GPU memory between machines over RDMA (GPUDirect), as nx.rdma had: the
  GPU drivers give a buffer's bus ranges (PCI paths, for rig.bnxt) or a
  dma-buf (kernel paths, for rig.mlx5), and a copy between two machines'
  GPUs runs through rig.mlx5 or rig.bnxt, not TCP

### tests and runs

- failure paths with no test (_plans/rig-release/failure-paths.md, whose
  core-exception and disk rows test_interrupt and test_rig_disk now cover):
  the host relocation that does not fit its field; rig.program's
  `Out_of_memory` and `Lost`; CUDA's library that cannot load; NV's `entry`
  `No_memory` and memory above 2^40; a Metal command-buffer fault, or why
  it cannot be forced
- flakes, each failed once: test_submit "a part follows the earlier parts
  that may share memory"; test_model on one domain, `Oom_at` on device 3 or
  4 about 1 in 120 (loop it with logs kept); test_image 1 in 3 on the Mac
  before its 2026-10-12 fix (rerun 30 times)
- relock windtrap once its `workers-block` branch (a883e72, a start
  barrier that sleeps once workers stop arriving) lands: rig_pci.space
  "two domains allocate and free at once" times out at its 10 s bound in
  60 of 64 copies run 32 at once at load ~65 with windtrap's spinning
  barrier, and in none of 96 with it
- hardware runs: an AMD host (AMD over PCI since 1090e97484, 31f61d92e5 and
  dc72412596); root on an NVIDIA host (NV over PCI's open/close law and
  test_nv, the PCI clocks' span laws, VFIO behind an IOMMU); two GPUs
  (peers); mlx5 and bnxt NICs
- baselines: fudge sections for rig_cuda, rig_disk, rig_elf, rig_host,
  rig_pci_space, rig_pci_window, rig_pool, rig_program and rig_remote,
  whose only Linux rows are kimchi's; on the M1, Metal
  `launch/pipelined-100`, `buffer/read-64M`, and re-blesses of rig_program's
  `polled/*` and rig_remote's `request/alloc` and `rail/4K`, recorded under
  load; the vendor comparison (packages/rig/bench/compare/run.py) on the
  Mac, then fudge; rig's `submit/polled/pool-two-domains` (0 words warm),
  `submit/polled/stores-30` and `copy/queue-64K-two-domains` (the words of
  `copy/queue-64K`) on every machine, and every rig row a run of one
  submission changed (`submit/polled/*`, `replay/polled/*`, `device/*`,
  `drain/*`)

### perf

- a host buffer's creation: every fresh nx result mallocs through
  `caml_rig_heap_bytes` and allocates a custom block (`buffer/host-create-16`
  54 ns, 31 words on the M1; nx's `dispatch/add-1` 214 ns): profile malloc
  against the custom block and `host_memory`, then A/B a per-domain free
  list of small host buffers
- a collected buffer's release allocates in the call that drains it
  (memory.ml:628-632, ~4 words a buffer); done when `drain/dropped-1000`
  allocates only its creates and kaun's GPU decode rows measure allocation
  again (bench_decode.ml's gpu_metrics)
- results between about a quarter of the last-level cache and half of it
  rotate past it under rig's due point (rig_heap.c, heap_cycle_bytes): at
  a small heap, set-row-8M's 8 MiB results rotate through four buffers,
  32 MiB against fudge's 24 MiB L3, where three rotated before, and the
  row went 83 -> 92 us on fudge (counted on the M1 with fudge's floor;
  fudge's re-time pending). `heap/reuse-8M-x8` is unmeasured under the
  new due point
- a pool job's start and join on fudge's 6 P-cores: an empty job costs
  0.45 µs, and units run 20-40% slower than on one core (5% of it the
  frequency drop), against nx's f32 64³ gap to MKL on 6 cores (1.66 against
  1.39 µs): time MKL's or OpenMP's empty parallel region on the same cores,
  record fudge's rig_pool section, and attribute the gap by an A/B
- AMD 4 KiB prepared copies are 17-26% slower since 70cca2ae5's wait change
  (the compute route itself starts at 2 MiB): name the row, explain it or
  revert the wait change for small copies
- CUDA and NV file loads take 46-49 ms for 256 MiB against native CUDA's
  27 ms (kimchi, 593e7735a; the warm read alone is 28.7 ms, PCIe 10.7):
  rerun the compare on fudge, then profile whether rig reads the file
  before uploading instead of overlapping them
- NV: signal through the last launch's QMD release when a submission ends
  in a chained launch (a WFI release costs ~0.67 µs); first a hardware test
  that a release implies the earlier launches completed
- read a memory's atomic entry once per path (the dropped patch no longer
  applies), A/B on the M1's `submit/polled/*` and `copy/*`;
  `Submission.order`'s cheaper design against `submit/polled/make-4096` on
  fudge
- measure, then keep or cut, each with its keep threshold stated before the
  run: CUDA's second stream (a 256 MiB upload beside a compute-bound
  launch); the remote link's inline send against one sender thread
  (rig_remote's `request/*` and `rail/*` rows)

## tolk

- `upcast_vectors` (heuristic.ml) has no term for a kernel's size: 8 nested
  float64 sines over 1024 values take 8 lanes, run 4.5x as fast, and compile
  in 2.2 s against 1.4 s without lanes. Done when an A/B on fudge of
  `trig/sin-nested-8` and the `compiled` suite's sine rows decides whether a
  kernel's size bounds its lanes
- a compile on 10 domains takes about twice the CPU of one domain. At load
  25-40 on the M1 Max, a jera tsit5 solve of y'' = -sin y took 18 s of CPU on
  10 domains, 8.9 s on 1 and 10 s on 10 with `OCAMLRUNPARAM=s=4M`; of
  y'' = -tanh y, 5.0, 3.4 and 3.8 s. Candidate cause: the domain count, 10
  over the M1 Max's 8 P-cores and 2 E-cores (cap at the P-cores). Decided on
  fudge, quiet: wall and CPU time at 1, 2, 4, 8 and 10 domains; `s` at the
  default and at 4M, one domain included; minor collections and promoted
  words (`OCAMLRUNPARAM=v=0x400`); a sample of the 10-domain run, the
  stop-the-world barrier against `pthread_mutex_lock`
- tolk.engine refuses a back edge whose flag moves with the run's words
  (tolk_engine.ml:775); no GPU test of a loop whose body mixes a GPU kernel
  and a host program
- sizes in [0,1], unreachable with concrete shapes (prepare.ml:607 folds
  empty values first): a sum at n=0 reads its accumulator uninitialised; a
  loop whose index uses fold to 0 loads at invalid ids. Fix with symbolic
  shapes
- search timing on NV clears no L2 (tinygrad's invalidate_caches); AMD
  counters and thread traces are not collected: encode both as words parts
- AMD machine code without comgr (_plans/amd-isa/design-reviewed.md): an
  assembly renderer and assembler, gfx12 first, the default once its gate
  passes; HIP stays as a choice. First read the terms of AMD's ISA XML
- divergences: for each entry of tolk's and rune's DIVERGENCES.md, look in
  references (tinygrad HEAD, papers, other compilers) for a better
  solution; keep it with its reason or replace it
- compile speed: one Clang process per kernel (~30 ms each; one per program
  saves ~20%); float64 sin/cos inline the full Payne–Hanek reduction (40-170
  KB kernels); structural comparisons in the rewriter (~6%); parallel
  compile saturates at ~4 domains (a shard lock per node, domains spawned
  per `Worker.map`); Metal builds run one at a time and NVRTC in-process;
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

## rune, kaun, vega

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

- frontend, `Ptree` at a brand (RFC 0006's names at `('s, 'd) t`, `Walk`,
  `S`, `sized`, the `tensor` alias), with `check` over it, `Rng.ptree` and
  `associative_scan`
- frontend, `fuse` and `widen` (`float_fn`, `Ptree.fuse`): one rounding
  per element, bit for bit with `f ()` on one kernel library
- frontend, symbolic shapes: `Size.v`, `Size.dims`; `shape`, `dim` and
  `numel` raising on a symbolic axis under a trace; inheritance through
  `_like`, movements, broadcasting and pattern groups
- frontend, `attention` with `causal`, `kv` (`Dense | Paged`) and `pair`,
  over new primitives
- frontend, `contract`'s `?a_scale`, `?b_scale` (`scale`) and `?segments`,
  after `Prim.Contract` and nx.cpu take them
- frontend, `sync` with deferred per-set checks (a read raises the earliest
  failed one); then the samplers' parameter checks move there, and
  `Rng.truncated_normal`, `dirichlet`, `categorical`, `permutation`
- frontend, donated reuse for `set`'s and `scatter`'s target and
  `contract`'s `init`, with a `dispatch/kv-append-4096` row
- `Interp`'s `running` field is shared by fibers on one domain;
  `Nx_array.to_device` copies a strided view's whole span (probes:
  `refs/audit/20261010-deep/worktrees/wt-18`)
- order and counting operations: check their contracts, public laws and
  bench baselines
- a split axis that its device count does not divide (a [4; 3] value split
  over 2 along its last axis) raises `Invalid_argument` from `Grid.window`,
  a limit no hardware imposes: windows of unequal extent through `Grid`,
  `Devices.window` and `Route`
- a sum over split axes gathers the whole input: reduce to partials and
  combine them (`[8192; 4096]` over eight devices moves 112 KiB per device
  instead of 112 MiB)
- nx.io on nx2, then Rune over nx2 with tolk lowering, before the GPT-2 and
  gpt-oss comparisons
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
  the reference (143 lib/ modules have none; jera, norn and ymir dropped
  theirs)
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
