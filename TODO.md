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
- compile speed: float64 sin/cos inline the full Payne–Hanek reduction
  (40-170 KB kernels); Metal builds run one at a time
- the host after D163's decode count (4dab9494e): inlined decoded
  products lose on the host (dense MXFP4 1024² 3-5x slower, a 256-wide MoE
  layer at 512 tokens 60-67 against 30-36 ms, fudge under load), because
  the host's codegen shares a decode over 4 rows, not 16; the scheduler
  stays target-free, so the fix is the host's tile sharing more rows, or a
  measured case for a target's exchange rate in rangeify
- dune's watch server goes stale on files a rebase or checkout rewrites
  (reported missing, or the old version built): dune keys a directory's
  listing on the directory's stat and keeps it across server restarts, so
  a listing read while git rewrote the directory stays until the
  directory's mtime changes; creating and removing a file in it clears it
  (touching the files does not). Records in `_plans/dune-stale-watch/`
  (tolk-scan-aligned-20261103T150140.txt has the cause). Report upstream
  with a reproduction
- `Heuristic.stream` declines a product whose operand reads a reduce
  (routing ids from outside the jit); no recorded kernel carries one yet,
  so the general path's cost there is unmeasured. Its unroll law passes
  vacuously where no contracted loop is left
- first call of a cached program (gpt-oss's 12-token prefill on CUDA:
  0.37-0.39 s under load against 0.22 s on the old stack):
  `create_linear_with_vars` on a hit spends ~10 ms on the key digest and
  ~24 ms in other passes; the engine assembles 4 programs in 28-57 ms
- ymir's aperture gradient has no cold-compile row (22-28 -> 14-15 CPU-s
  with loop fusion, 5e509c7f1): add it to contrib ymir's bench
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
  interning (~10%) and matching
- whether memory-bound host kernels take lanes at all: no upcast ran 4-7%
  faster than 64-byte lanes in two of seven loaded fudge runs of a 4M
  float32 select-zero; decide on the scheduled quiet re-record
- host output upcasts: tinygrad's 1024-output floor gates every host
  output upcast. Lifting it gives 1.2-6x on 22 of 23 recorded host kernels
  (`r_32_3_2_4_3_400` 47.0 -> 24.9 us on a P-core, beam 23.4; gpt_oss_kv
  1888 -> 304 us; vecmat_1000 2688 -> 808 us) but raises Clang time
  sharply (gpt_oss_kv 22-34 ms -> 0.9-1.1 s, qkv 0.26 -> 1.5 s) and may lose
  1.2x on an 8-output matvec at K = 1M across all cores (estimated): a
  floor counted in D80's split-loop iterations, with both costs measured
- `Nx_quant.apply_experts` of gpt-oss's gate and up projection at 512
  tokens takes 26.4-27.6 ms on fudge's CUDA against 3.14-3.23 ms through
  `Nx.map_segments`' blocks: one route row per thread group, so no decoded
  weight is shared across rows, and no layout rule can share reuse the
  gather hides. rune's lowering of the segmented contraction owns the
  blocks, 64 rows of one expert
  (_plans/tolk-campaign/routed-experts/adjudication.md)
- D163 counts repeats pooled over a shared range, which overestimates `c`,
  and a collapsed read of a stage that reads storage counts every range of
  its sum, which can overcount: tighten both if a case shows a wrong store
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
- a float32 routed product on CUDA, which no core takes, runs at 7.4% of
  the RTX 5000's SIMT float32 peak: 17.4-17.9 ms at 512 tokens streamed in
  blocks of 16 rows (kaun `Quant/cuda/routed-product-512-tokens`), 68-69
  ms under the general rules in blocks of 64, which rune's lowering takes
  from 4 rows an expert. One token's gathered products stream at 74-93% of DRAM and
  bfloat16 blocks of 64 take the cores; what is left is tolk's no-core
  layout past 32 rows (the GPU contraction rules' 8 x 8 outputs a thread),
  on that row
- device facts (762ad1c31): `core_threads = 16384` and `stream_threads =
  32768` are fudge's (sm_89, 100 SMs); a small part (an RTX 4060-class
  sm_89) may lose 1.4-2x by the model: a row owed there, and values for
  nonnormal's R9700 and Metal. `Rig.cache_bytes` has not run on hardware
  for AMD through amdgpu (L2 plus MALL) or NV over PCI
- Metal's packed kernel arguments (8869adcc8, af8961676) have not run on
  Metal hardware; check `rig_metal.mli`'s contract against them on a Mac
- kernel parameters that name no device, the device a fact of the call's
  arguments, would rewrite D170 and delete `params_on`'s substitution,
  `rename`, `served`, `same_devices` and search's `on_device` (about -60
  lines, unverified). Audit the readers first: the device enters at
  `rangeify.ml:887-896`, public `codegen.mli:118-125` changes, the beam cache
  key includes parameter devices (`search.ml:335-346`) and is re-keyed, and
  rune's reads and hash-consing are unchecked
  (_plans/tolk-campaign/device-sink/adjudication.md §1)
- one lane for `Linked` schedules too (D41): rig orders one device's work on
  one timeline, so a copies' lane buys overlap only across devices. Decide
  on a GPU beside a memory device whose value moves into the GPU mid-chain,
  linked `Linked` with one lane and two: the time `run` takes to return and
  the GPU's timeline in `Rig.Profile`; and count the goldens one lane changes
  (`generate.py --check` with one lane in the planner and `tinygrad.patch`)
- the per-run order (D183) interleaves devices: a data-parallel step over 16
  memory devices makes 1411 `Submit` steps against 84 in tinygrad's order. A
  same-device tie-break in `Schedule.storage_order` made 1016 but raised four
  of rune's five `fully sharded` peaks over their bounds; find a tie-break
  that cuts the submissions and keeps the five peaks

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

## nx

- nx.cpu's thread splits follow the call prices (179ea5cc5, 82e42b78d) and
  the kind tables of f6876a0ad, measured under load 1.3-4.7: time them
  quiet on fudge, with a row of a costly kind where its threads change
  (pow-f64-16K), and re-record the elementwise rows

## talon, hugin, brot, munin, quill

- talon benches: record the query suite's DuckDB/Polars baselines on a
  quiet fudge (`baseline.py` refuses loaded runs; H2O 1e8 needs more than
  a run's 16 GiB), and bless `Talon_csv/read-mixed` on the Mac
- talon perf on fudge (c51b4b858, 14 threads, load 5-71, DuckDB 1.5.6,
  Polars 1.44.2), geomean vs the faster engine: decision support SF1
  11.7x, SF10 22.0x, H2O 1e7 group-by 7.9x, join 7.1x.
  SF10 q13 peaks at 14.45 GiB. Logs: ~/wt/tmp/talon-baselines on fudge
- talon 1.0 plan (_plans/talon/product/plan.md, on the Mac): windows and
  ewm, as-of and interval joins, interval sets and folds, zoned time,
  hash/sample/shuffle, Pivot and nesting, dense round trips, partial
  aggregation, regex, compressed and fixed-width CSV, Parquet nested
  columns, datasets and writer, talon.arrow, talon.jsont, complex and
  validity columns, HTML display
- brot vs gigatoken cd5ee648c6 on fudge (bench README, "Against
  gigatoken"): GPT-2 runs at 0.34 of its speed on one thread (1.4x its
  instructions, 11x its mispredicts, 6x its last-level misses, in
  `brot_byte_level_encode_ids32` and `mask_refill`) and 0.36 on eight;
  Mistral at 0.74 with equal instructions and misses, unexplained;
  SentencePiece pipelines spend 25% in `Normalizer.replace_byte`

## repo

- thumper: nx.thumper's fudge section was recorded under
  OCaml 5.5.1, so it compares nothing under 5.5.2: re-record it on a quiet
  fudge. talon.thumper's Mac rows predate the Nx.Rng
  fixture (76105a215): the maintainer re-records them
