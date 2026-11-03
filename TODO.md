# todo

goalpost: jit-compiled gpt-oss at least as fast as {tinygrad,JAX,llama.cpp/vLLM} on {NVIDIA,NV,AMD}.

gpt-oss-20b, RTX 5000 Ada (fudge, 31aa4521b, the new nx and rune), raven
CUDA / NV / llama.cpp b12032 CUDA, at a load of 33 to 65:
- decode 11.0 / 10.7 ms of GPU time (bench medians 15.5 / 12.0 ms) /
  6.83 ms at depth 512; with Q8_0 attention and head 23.4 / 22.8 ms
- prefill-512 0.186 s on CUDA (f2af3251b; NV not measured again) / 56.0 ms
- first call 0.45 / 0.37 to 0.54 s (12 tokens, kernels cached; 1.36 /
  1.47 s at the first run) / 75 ms (59 to 151 ms, no warmup)

gpt-oss-20b, AMD gfx1201 (nonnormal, f24925ed77, the new nx and rune),
raven / llama.cpp b12048 Vulkan:
- decode 20.3 ms / 9.40 ms
- prefill-512 1.61 s / 0.308 s
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

- A schedule linked per run (D173) is one rig program per part, cut
  wherever storage is made or freed, because `Rig_program` binds its inputs
  once, before its first step: each part loads its kernels' images again
  and ends a submission. Done when a step of `Rig_program` makes or frees a
  slot's memory, so a per-run schedule is one program, and a split step's
  loads and submissions, counted on a rune bench row, match a linked one's.
- masked axes upcast to up to 49 lanes once gave Clang straight-line
  blocks whose scheduling took seconds (rune's test_iterate, float64
  [4;2;2] over 3 lanes, 4.8 s in Clang), but one run's 8205 host kernels
  (seed `s1:c80c8eab911c4a5e`) reach at most 16 masked lanes: find the
  property draw that makes 48, then time Clang and a replay at 48, 24 and
  8 lanes
- placement on hybrid cores: nx's P-core cap, on fudge's CPU model
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
- divergences: tolk's entries are audited against tinygrad HEAD; rune's
  DIVERGENCES.md is not audited yet
- compile speed: one Clang process per kernel (~30 ms each; one per program
  saves ~20%). Hypothesis, to time first: on fudge, each Clang run's wall
  time against its source's bytes, on a model-size schedule; if the process
  start dominates, one Clang invocation over several translation units, one
  object per kernel (`clang -c 0/k.c 1/k.c ...`), keeps per-kernel cache
  keys and Worker.map's balance, once each object is shown a function of its
  source alone; float64 sin/cos inline the full Payne–Hanek reduction (40-170
  KB kernels); parallel compile saturates at ~4 domains (a shard lock per
  node, domains spawned per `Worker.map`); Metal builds run one at a time;
  a 5 KB r_32_32 kernel, max |x| over `where (i >= j) x xᵀ` at 32x32,
  renders as 32-lane float selects of transposed loads that clang -O2
  takes over 2 min on
- gpt-oss bf16 prefill-12's first call is ~70 ms slower in wall time
  since `Heuristic.stream` (0.28-0.34 s against 0.35-0.47 s, fudge CUDA)
  though its kernels take 22 ms less GPU time; warm calls are unchanged.
  Unmeasured guess: larger cached programs take longer to load
- the host after D163's decode count (4dab9494e): inlined decoded
  products lose on the host (dense MXFP4 1024² 3-5x slower, a 256-wide MoE
  layer at 512 tokens 60-67 against 30-36 ms, fudge under load), because
  the host's codegen shares a decode over 4 rows, not 16; the scheduler
  stays target-free, so the fix is the host's tile sharing more rows, or a
  measured case for a target's exchange rate in rangeify
- dune's watch server goes stale on files a rebase or checkout rewrites
  (reported missing, or the old version built): records in
  `_plans/dune-stale-watch/` from five trees on 2026-11-02
- `Heuristic.stream` declines a product whose operand reads a reduce
  (routing ids from outside the jit); no recorded kernel carries one yet,
  so the general path's cost there is unmeasured. Its unroll law passes
  vacuously where no contracted loop is left
- first call of a cached program (gpt-oss on CUDA: 0.44 s against 0.22 s
  on the old stack): `create_linear_with_vars` on a hit spends ~10 ms on the
  key digest and ~24 ms in other passes; the engine assembles 4 programs in
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
  without its loop vectorizer (D167); float64 `betainccinv`'s gradient (1.0
  MB, 42k instructions) then spends, loaded, 3-4.5 s in Clang and 3.5-5.5 s
  in tolk (424 M words): about 40 passes, each linear in the program's size
  and under 0.4 s and 40 M words, rangeify's cleanups, `run_rangeify` and
  the symbolic passes before decomposition the largest. Rewrite tables are
  now keyed by node id (382 M words on rune's `special/betaincinv-grad-f64`
  compile row); left in tolk: table cache misses (~20% of its samples),
  interning (~10%) and matching. Clang's SROA is superlinear in vector
  `__builtin_bit_cast` (tinygrad renders them too): 4.2 of that row's 7.2
  s of Clang; rendering them as vector casts gives the same object in 2.8
  s, a divergence to admit with that row
- whether memory-bound host kernels take lanes at all: no upcast ran 4-7%
  faster than 64-byte lanes in two of seven loaded fudge runs of a 4M
  float32 select-zero; decide on the scheduled quiet re-record
- `r_32_3_2_4_3_400` (lorenz) runs 1.57x slower than beam's choice because
  the heuristic's candidates, tinygrad's, offer no output upcast; no lane
  bound fixes it
- apply_experts' contractions run 56 ms against the block path's 36 ms:
  one route row per thread group, so decoded weights are not shared across
  rows (D89's runs). No layout rule can share reuse the gather hides:
  nx.quant should present routes in blocks as `Nx.map_segments` does,
  blocks past 32 rows on a tensor core. At prefill-512 (CUDA) the
  decoded weights are stored, 64 rows an expert (D163): E_8294400_32_2
  writes 2.1 GB in 4.74 ms for gate_up, which a layout sharing them across
  rows could keep in the gated per-group loads
- D163 counts repeats pooled over a shared range, which overestimates `c`,
  and a collapsed read counts every range of its sum, which can overcount:
  tighten both if a case shows a wrong store
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
- loops bounded by a loaded value fail Spec: upcasting an axis that an
  inner loop's loaded bound reads makes a vector bound (`Lidx1 < long16`),
  and a bare loaded bound makes an AFTER of a LOAD, which tinygrad's spec
  refuses too. Decide what such loops lower to before one reaches users
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
  passing its window's start. nx's closing
  moves a chunk between every pair of devices: time it against tinygrad's
  ring and hierarchical allreduce on a machine with several GPUs, and if
  it is slower, write those in nx's partition
- sharded jit: some operations still make programs per device, so their
  compile cost grows with the device count. Program keys at 2/4/8 devices:
  sum 4/7/9 (the running fold's per-device steps), take_along_axis 7/14/22
  (programs gathering the slots' bits), cumsum 5/7/8 (around the totals'
  scan), cummax's tangent 16/15/17 and gradient 14/13/15. argmax and
  scatter are constant and sit in test_jit_devices' `uniform` list; move
  each op there as its closing or assembly becomes one program per device

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
  fudge's host, 25.9 and 43.4 µs under `Rune.jit` on AMD (1.7x); record
  them on CUDA under `Rune.jit`
- nx.cpu's thread splits follow the call prices (179ea5cc5, 82e42b78d) and
  the kind tables of f6876a0ad, measured under load 1.3-4.7: time them
  quiet on fudge, with a row of a costly kind where its threads change
  (pow-f64-16K), and re-record the elementwise rows

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

- thumper: nx.thumper's fudge section was recorded under
  OCaml 5.5.1, so it compares nothing under 5.5.2: re-record it on a quiet
  fudge. talon.thumper's Mac rows predate the Nx.Rng
  fixture (76105a215): the maintainer re-records them
