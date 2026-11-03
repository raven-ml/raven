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
- compile speed: Metal builds run one at a time
- dune's watch server goes stale on files a rebase or checkout rewrites
  (reported missing, or the old version built): dune keys a directory's
  listing on the directory's stat and keeps it across server restarts, so
  a listing read while git rewrote the directory stays until the
  directory's mtime changes; creating and removing a file in it clears it
  (touching the files does not). Records in `_plans/dune-stale-watch/`
  (tolk-scan-aligned-20261103T150140.txt has the cause). Report upstream
  with a reproduction
- long programs: the incomplete beta (~11,400 ops) compiles in 3 s, 10-20 s
  per derivative, past 10 min for three; a 1,000-term slice-tree sum is one
  kernel clang -O2 takes over 8 min on. Past 400 KB of C Clang compiles
  without its loop vectorizer (D167); float64 `betainccinv`'s gradient (1.0
  MB, 42k instructions) then spends, loaded, 3-4.5 s in Clang and 3.5-5.5 s
  in tolk (424 M words): about 40 passes, each linear in the program's size
  and under 0.4 s and 40 M words, rangeify's cleanups, `run_rangeify` and
  the symbolic passes before decomposition the largest. On rune's
  `special/betaincinv-grad-f64` compile row (221 M words, 22.1 G
  instructions under cachegrind) what is left in tolk is the walks (a
  quarter of the instructions, their rewrites' bookkeeping included), the
  collector (a sixth), matching (a sixth) and `caml_modify` (7%)
- whether memory-bound host kernels take lanes at all: no upcast ran 4-7%
  faster than 64-byte lanes in two of seven loaded fudge runs of a 4M
  float32 select-zero; decide on the scheduled quiet re-record
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
- device facts (762ad1c31): `core_threads = 16384` and `stream_threads =
  32768` are fudge's (sm_89, 100 SMs); a small part (an RTX 4060-class
  sm_89) may lose 1.4-2x by the model: a row owed there, and values for
  nonnormal's R9700 and Metal. `Rig.cache_bytes` has not run on hardware
  for AMD through amdgpu (L2 plus MALL) or NV over PCI
- Metal's packed kernel arguments (8869adcc8, af8961676) have not run on
  Metal hardware; check `rig_metal.mli`'s contract against them on a Mac
- custom code and calls, after D265 and D271: inline custom expressions
  (`Op.Customi`) still take vector lanes; a plain store such as
  `n[0] = n[0] + 1` in a kernel with a thread axis it does not read runs
  once a workgroup; the heuristic's stride and tile checks still read only
  bare constants, so a typed range gets slower kernels (D270 fixed the
  splits)
- comgr may compile with LLVM's verifier off, as Clang did before D275:
  check `Compiler_amd`'s commands on a ROCm build and keep the verifier on
- LLVM 20+ and Apple Clang keep the straight-line vectorizer that D167
  turns off under LLVM 18 and 19: run the reduced kernel of 73455ffe7
  through them (the Mac's Homebrew LLVM) and extend D167 if they miscompile
- the host's routed decode (D92): dense at 64 rows runs 1.4x slower than
  the stored decode (12.05 against 8.5 ms on a P-core); register files of
  32 (NEON, AVX-512) take 16 rows, unmeasured
- programs that share a cubin load it once each (31 of 74 loads in a
  gpt-oss first call, about 1.8 ms): sharing needs `Rig.Image.load` to drop
  its promise of module variables per load (D146)

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
  gigatoken"): GPT-2 runs at 0.58 of its speed on one thread over 100 MB
  (1.1x its instructions, 2.1x its mispredicts, 3.0x its last-level
  misses); its pretoken table sits on 4 KB pages where gigatoken's takes
  2 MB pages (an madvise from brot's C, which brot's C rule does not allow
  yet)
- brot's Unigram (T5) splits runs of dashes or underscores unlike HF:
  "\n\n-----------" gives `-` `---` `---` against HF's `---` `---` `-`
  (3 of 1003 OpenWebText documents)

## repo

- thumper: nx.thumper's fudge section was recorded under
  OCaml 5.5.1, so it compares nothing under 5.5.2: re-record it on a quiet
  fudge. talon.thumper's Mac rows predate the Nx.Rng
  fixture (76105a215): the maintainer re-records them
