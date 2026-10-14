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
  and GPU; kimchi is retired) and nonnormal, llama.cpp at a stated build
- the port's gates on rig: gpt-oss decode and first call, sofo/symo, rune's
  test_jit on CUDA, NV and AMD, the tolk, rune, nx and kaun decode benches
  against the pre-port main; remat memory over its bound on NV/AMD; jit
  replay's words on x86; the engine's wrong answers on Metal+host staged
  loops

## rig

### semantics

- step 4 of _plans/rfc0036-rig-enforcement/ with the one-lock-per-thread
  check: waits do no other device's work; drains and the ladder lose no
  record on an interrupt; rounds wait for a busy device; an open runs its
  name's owed stop outside the table lock; `afters` as per-profile pending
  entries (a span recorded in `Profile.take` is never dropped)
- step 5: a submission run's take as a hold; one claim take (D8), net lines
  <= 0. Step 6: the boxed `Some` on the cache path
- ledger rows (_plans/rig-semantics-refs/LEDGER.md): decide required
  `~waits` against its callers (none in nx, rune or tolk pass waits), and
  count exact-zero tests with `Gc.allocated_bytes`; `Buffer.free`; both
  staging slots of a staged route in one C call, in device order; the
  remote controller survives a failed job; the single root cause narrowed
- more ledger rows: a forked child loses io devices too and runs none of
  the parent's owed stops (F8); host and memory work runs after giving the
  turn up (K2); `R_AMDGPU_RELATIVE64` relocations (A1); the doc sentences of
  F3, F5, O5, Q1, R4 and C4
- an opener whose `make` opens its own name raises `Invalid_argument`
  instead of waiting for good; one `STILL_MS` in C that dev.ml reads; a heap
  law for the stamp words of lost points

### drivers

- a peer view of a stopped driver-less NV GPU can crash: the stop unmaps
  the BAR windows another GPU's flush entry or view still reads
- AMD's and NV's PCI paths turn every `map_host`/`map_peer` error into
  `None` (rig_amd_pci.ml:221, rig_nv_pci.ml:205): pass a fault on as a fault
- exceptions: `Tlsf` removes a block, then allocates before reinserting
  (tlsf.ml:186, 241); `Atomic.incr`/`decr` allocate first at rig_disk.ml:222,
  262 (an exception skips `close`) and rig_mlx5.ml:729; amdgpu's and NV's
  tables must still serve stop, free and unload after an exception
- rig.amd: `Counter.places` reserves too few WGP slots on harvested RDNA3/4
  under KFD (RX 9070: 3 for 4); `fail()` in rig_amd_ring.c can overrun an
  AQL segment by 4 bytes
- rig.amd.pci: a dirty or faulted GC 9.5.0 that rig marked gets a partial
  boot (queues stopped, compute engines soft-reset), as tinygrad does: a
  mode 1 reset over its live state can stop its fabric until a power cycle
- rig.bnxt: take completions the NIC wrote before failing a queue pair; a
  terminal index outside [taken, posted] counts as none; poll and wait issue
  no firmware command; fix the leak when `Qp.make` refuses
- rig.pci (ledger X3-X6): "alone on its device" only where the reset reaches
  siblings (`reset_method`); `Function.take` gets the device's address bits;
  one VFIO container per machine; `reaches` true only under one root port
- rig remote and host pool: check for the bugs dropped with nx's runtime
  (_plans/rig-release/dropped-patches/); the remote join dials serially
  within a 10 s budget; the pool reads cgroup v2's `cpu.max` only
- rig profile: a Copy event names a side by where its bytes are (Pinned
  staging and image loads record `d→d`); a span names its submission, so a
  search under a second domain times only its own kernels
- rig.amd to its spec (_plans/gemini-review-2/REPORT.md,
  _plans/rig-size-audit/): unused encoders and options removed, the rest
  behind `(**/**)`; `report`, `block`, `image` hidden; holes as NV's fixed
  record; `Bus_address` into `Machine`; request.ml inlined; firmware parsing
  total over the pinned set; one trace window per engine; KFD's clock gap
  stated in rig_amd.mli
- one generator, pins file and excerpt tree per driver (AMD, NV); NV keeps
  615's headers whole and only `per_release` structs for older releases
- rig.mlx5 folds rig.mlx5.uverbs in (`path` internal, `?root` hidden), each
  part beyond tinygrad's RDMA justified; `?root` behind `(**/**)` on amdgpu
  and nvidia
- decide: whether the SQTT decoder stays in rig.amd.abi; whether the
  per-driver floor drivers stay in the benches; whether generators commit
  excerpts

### limits and size

- limits no hardware or spec imposes, one commit each
  (_plans/rig-device-reuse/70-, 71-): AMD's 512 parts per submission;
  rings that grow (`Rig.most_parts` goes, NV's GPFIFO refuses only `Words`
  past it); parameter bytes per compiled function; NV RM handles reused;
  PCI system memory's arena from physical memory; bnxt sized as Linux sizes
  it; mlx5's CQ from `max_cqe`
- NV address spaces: Device memory in the 49-bit space on both paths, RM's
  host-visible range to 640 GiB, VFIO's window 2^44; done when 80 borrows of
  one 1 GiB shm file succeed on fudge
- size cuts: 13 open in _plans/rig-size-audit/SUMMARY.md §A (core, PCI,
  rig.program, NV), each to its design without losing a capability
- test cuts: rows 2, 5, 7, 11-14 and 18-26 of
  _plans/rig-simplify-audit/tests.md, each shown covered before it goes
- docs: each restated contract folded to one place; rig_disk_stubs.c's six
  platform branches into its Windows and POSIX sections; rig_device.c's
  Windows locks against rig_pool.c's pthreads: say why or agree

### API

- claims and runs, one design pass: a claim held past a scope (rune's
  `Pins` and tolk's `link ~bound` go), a take over a buffer list,
  claim-all in rig.h, a submit that takes its run once
- copies: a device copy that returns once queued, copies of several runs or
  a strided box in one call, `fill` on the copy engine, run-value offsets in
  a `Rig.Buffer.copy` step; done when placing on N GPUs costs one wait
- from its callers: `Image.load` reuses an equal image; `Profile.take`
  raises `Lost`; `Run.string`; `Rig_program.load` returns `Error` only for a
  device's refusal; state `equal`, run ownership and `create`'s alignment
- rig.gpus: one card appears through two paths, and `openers` drops each
  path's refusal reason; `Rig_pci.Gpus` also holds NICs
- GPU memory between machines over RDMA (GPUDirect), as nx.rdma had: the
  GPU drivers give a buffer's bus ranges for a NIC to register, and a copy
  between two machines' GPUs runs through rig.mlx5 or rig.bnxt, not TCP

### tests and runs

- failure paths with no test (_plans/rig-release/failure-paths.md): three
  host and core rows, a Metal command-buffer fault, NV and AMD PCI faults,
  the AMD hang law, and the disk, program, CUDA and NV rows
- covers some seeds miss: test_amd "values across 2^32"
  (`s1:0ffeb6927a1ba911`), test_model "a submission's copy whose buffers
  overlap" (`s1:5dadbcc3524ab0ac`)
- flakes, each failed once: test_link fork "a child's job is failed in the
  child alone"; test_submit "a part follows the earlier parts that may share
  memory"; test_model (one domain) ~1 in 120, `Out_of_memory` on device 3 or
  4 where the reference fits; Metal "expected at most 32768, actual 40944";
  interrupt.t "Terminated: 15" under load; test_image (rerun on the Mac).
  Metal conformance must name a failing law
- hardware runs: AMD over PCI since the exception and own-word changes;
  NV over PCI's open/close law and test_nv; the PCI clocks' stamp check;
  VFIO behind an IOMMU (nonnormal needs VT-d); multi-GPU peers, mlx5, bnxt
- baselines: one quiet window per machine (Mac, fudge, nonnormal) records
  every row in _plans/device-dev/owed-rows.md, then deletes it; the vendor
  comparison (bench/compare/run.py) on each machine, the Mac first

### perf

- a host buffer's creation: every fresh nx result mallocs through
  `caml_rig_heap_bytes` and allocates a custom block, 50-75 ns an op on the
  M1 (26 ns of `dispatch/add-1`'s 220). Keep small host buffers off malloc
- a collected buffer's release allocates in the call that drains it; done
  when a drain allocates nothing and kaun's GPU decode rows measure
  allocation again (bench_decode.ml's gpu_metrics)
- a host buffer reused from rig's cache still paces the major GC as new
  memory: rig_heap.c heap_bigarray (~line 636) allocates every buffer of
  64 KiB or more with `caml_alloc_custom`'s full size even when
  `heap_take` reused it, so a loop of 8 MiB results advances the major GC
  8 MiB a call. On fudge, nx's c2c-1M on 6 P-cores in bench_nx (~300 MB
  heap) runs 1.48-1.59 ms ±20%, and 0.76-0.80 ms with OCAMLRUNPARAM=M=1000,
  with no page fault a call. Pace a cache hit as no new memory, keeping
  the full pacing for fresh allocations
- a pool job's start and join on fudge's 6 P-cores: an empty job costs
  0.45 µs, and units run 20-40% slower than on one core (5% of it the
  frequency drop). That is most of nx's f32 64³ gap to MKL on 6 cores
  (1.66 against 1.39 µs)
- AMD 4 KiB prepared copies are 17-26% slower since the compute copy route
- CUDA and NV file loads take 46-49 ms for 256 MiB against native CUDA's
  27 ms
- NV: signal through the last launch's QMD release when a submission ends
  in a chained launch (a WFI release costs ~0.67 µs); first a hardware test
  that a release implies the earlier launches completed
- read a memory's atomic entry once per path (the dropped patch no longer
  applies); `Submission.order`'s cheaper design against
  `submit/polled/make-4096` on fudge
- measure, then keep or cut: CUDA's second stream (a 256 MiB upload beside
  a compute-bound launch); the remote link's inline send against one sender
  thread

## tolk

- the disk cache never evicts (96 GB on the Mac): bound it, state the bound
  in helpers.mli, and test that its size holds past the bound
- runtest reads and writes the user's cache: set `CACHEDB` in the root
  `dune` env; done when no runtest action touches a cache outside `_build`
- tolk.engine refuses a back edge whose flag moves with the run's words
  (tolk_engine.ml:775); no GPU test of a loop whose body mixes a GPU kernel
  and a host program
- sizes in [0,1], unreachable with concrete shapes (prepare.ml:607 folds
  empty values first): a sum at n=0 reads its accumulator uninitialised; a
  loop whose index uses fold to 0 loads at invalid ids. Fix with symbolic
  shapes
- search timing on NV clears no L2 (tinygrad's invalidate_caches); AMD
  counters and thread traces are not collected: encode both as words parts
- 23 D-numbers cited from tolk and rune code (ops.ml:2607, shape.ml:2016,
  test names, rune test_jit.ml:1973): state the fact. Keep, fold or delete
  rune/DIVERGENCES.md and tolk/test/REGRESSIONS.md
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
  per `Worker.map`); Metal builds run one at a time and NVRTC in-process
- compile time and launch counts: a NUTS program takes 9-14 s to build, a
  ymir aperture gradient 98 s cold; loop bodies store each carry and row in
  its own kernel (drawing-scan-256: 6 per trip against 2). Multi-output
  fusion of loop bodies
- long programs: the incomplete beta (~11,400 ops) compiles in 3 s, 10-20 s
  per derivative, past 10 min for three; a 1,000-term slice-tree sum is one
  kernel clang -O2 takes over 8 min on
- host upcast width per target: D141's 64-byte lanes (heuristic.ml:446)
  slowed memory-bound select-zero ~15% on x86; time 32 against 64 bytes on
  fudge and nonnormal
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

- a compiled `Nx.take` gradient costs the table times the tokens:
  `indexed/take-grad-1024-of-32768x64-host` takes 94 ms on the M1 (load 20)
  against ~2 ms eager. At 128256 rows a plain loop measured 17 ms and thumper
  0.67-1.05 s, unexplained. The row has no baselines yet (Mac, fudge,
  nonnormal)
- `Total.collect` around a compiled function on a device without eager
  kernels raises `has no eager kernels` (total.ml:62,86)
- jit compiles no complex tensor (lower.ml:36); ymir's `test_quantity`
  holds an `xfail`
- covers some seeds miss: eigh "repeated eigenvalues"
  (`s1:c5a5aa5507008517`); test_bodies laws 1 and 5; test_iterate "max = 0"
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

## nx

- saves fsync unevenly: npy, npz, images and gunzip go through
  `Temp_file.write`, which never orders bytes before the rename
- compiled `Nx.svd` is inaccurate on graded float64 spectra: relative
  errors 6e-7 at n=16, 9e-9 at n=32
- `truncated_normal` loses accuracy in the tails (float64 2.5e-11, float32
  1.5e-2 at 5σ): an erfc-based tail inversion
- nx.wide's `chunk_levels` was tuned when each chunk compiled two kernels:
  time `compile/wide/sum-1e6` at 2-5 levels
- `betaincinv`, once tolk compiles long programs (~20-25k ops)
- GC stress for nx's C stubs: run them once under the stressed collector

## norn, jera

- jera: "lanes › seeded lanes reaching rounding in two steps converge" and
  "solve derivatives › compiled equals eager" each failed once
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

- talon vs the faster of DuckDB/Polars: TPC-H SF1 13.2x, H2O group-by and
  join 2.8x; pandas 1.36x on filter and 2.05x on join. `Nx_ragged.take`
  moves ~40 bytes per byte of text and dominates joins; test_run 1.2 s
- talon benches: commit the query suite's DuckDB/Polars baselines (H2O
  1e7/1e8, TPC-H SF1/SF10) and a CSV read row; GC stress for its C stubs
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
- munin: GPU metrics (NVML by dlopen, IOKit, AMD sysfs); 64 MDX-skipped doc
  fences; the metrics table omits `sys/disk_*`; `Store.latest_run` picks
  among runs started in the same second at random (a run id is a seconds
  stamp and a random suffix)
- quill's TUI suite tests a copy of quill_tui.ml, bypassing its .mli
- quill prints an `Invalid_argument` or `Failure` message raw in a cell's
  output; multi-line messages (talon's plan problems) arrive escaped

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
- licences: replace lena.png and the Gaia DR1 fixture; stop calling talon's
  bench TPC-H; rewrite talon's Schubfach (OpenJDK, GPL) and ymir's WCS pole
  (WCSLIB, LGPL) from their papers; one copyright line: the LICENSE files
  say "2025, Thibaut Mattio", the source headers "2026 The Raven authors"
- windtrap: its per-test timeout does nothing on Windows; release 0.3.0 and
  drop the git pins in dune-project and ci.yml
- tests: CI runtest under a memory cap; suites read other suites' fixtures
  (tolk, norn and ymir benches, kaun's load bench): shared data in a
  declared `test/support/`
- benches: sowilo's and talon's 10 MB of committed seeded data generated at
  bench start
- dune stale builds: a source rewritten within seconds is digested in one
  version and copied in another; a passive server stops seeing changes yet
  answers `rpc ping`; under load it loops on "Event queue overflow".
  _plans/dune-stale-watch/
