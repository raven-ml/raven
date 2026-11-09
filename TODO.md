# todo

goalpost: jit-compiled gpt-oss at least as fast as {tinygrad,JAX,llama.cpp/vLLM} on {NVIDIA,NV,AMD}.

gpt-oss-20b, RTX 5000 Ada (fudge, under load), raven CUDA / NV /
llama.cpp b12032 CUDA; NV and the first run last measured at 31aa4521b:
- decode at position 512 of a 1024 cache 10.2 ms, GPU time and wall
  (8ef1902f3) / 10.7 ms of GPU time (bench median 12.0 ms) / 6.83 ms at
  depth 512; with Q8_0 attention and head 8.1-8.3 ms (f2af3251b) / 22.8 ms
- prefill-512 122 ms (4e23903949, least of 4 rounds under load) / not
  measured again / 56.0 ms
- first call 0.37-0.39 s (c29acaee3) / 0.37 to 0.54 s (12 tokens, kernels
  cached; 1.36 / 1.47 s at the first run) / 75 ms (59 to 151 ms, no warmup)

gpt-oss-20b, AMD gfx1201 (nonnormal, 502431b668, the new nx and rune),
raven / llama.cpp b12048 Vulkan:
- decode at position 512 12.7 ms (attention on WMMA) / 10.38 ms at depth
  512: expert products 5.4 ms at 212-254 GB/s against 3.0; top-k's
  20-kernel sort and about 37 small kernels a layer about 1.5 ms; argmax
  over the logits 0.63 ms
- prefill-512 0.418 s / 0.308 s: expert products 326 ms on WMMA at 7-8
  TFLOPS against 177; the out projection takes no WMMA, 52 ms against q's 10
- first call 0.30-0.34 s (12 tokens, kernels cached; 1.03 s compiling
  them) / 0.136 s (2.19 s without Mesa's shader cache); lowering and
  linking the first block takes 69-131 ms of it on the host
- llama.cpp's ROCm build needs HIP libraries nonnormal lacks; tinygrad
  needs Resizable BAR here

## tolk

- placement on hybrid cores: where host blocks land moves a run by 1.6x.
  Jit/chain-100-rooted-hit runs its 4 blocks in 12.9 µs pinned to the
  P-cores, 43.5 µs on one core, and 13.0 or 20.4 µs unpinned on a quiet
  fudge, depending on which cores they reach
- AMD counters and thread traces are not collected in a profile: rig's
  AMD driver brackets each timed launch, reading the VMID filter on the
  GPU in the bracket and marking a saturated or foreign count unmeasured
  (_plans/todo-campaign/design/amd-counters.adjudication.md)
- long programs: tolk's passes are linear in the nodes they walk, about 0.4 us
  a row over about 130 walks, flat (no pass past 8%, no function past 4%):
  the gradient of `betainc` costs tolk and rune's tracing 0.8 s and Clang
  0.7 s, its third derivative (220k nodes) 5.3 s and 12.9 s. Past 100k nodes
  Clang decides the wait, and it grows faster than the source on one kernel:
  a 1,000-term sum of `exp(sin x)` over 64 values, one loop of 15 MB, takes it
  125 s against 10 s for 250 terms (EarlyCSE, then the machine passes; `-O1`
  changes nothing). Four kernels of 250 terms would take about 40 s; cutting
  such a kernel at a store costs one buffer per cut and has no design yet.
  Measured and declined for tolk: a per-domain cache of the last long walk
  (-2% instructions, 12 lines), and pruning rewrites by the operations a
  matcher rules on (no gain: the memo it needs costs what it saves)
- device facts (762ad1c31): `Rig.cache_bytes` has not run on hardware for
  NV over PCI
- `Cuda/cat-16-columns-of-exp-64x64`, the row that catches a lost mark on
  a concatenation, has no AMD baseline
- a fixed run of an `After` state read inside a loop still raises where the
  state is over a bitcast, or a join over other writes of its storage
- one launch pass (ecdd3203f6), views naming their storage (de0bcf38d9)
  and per-trip loop reads (293eb3f70b) have not run on AMD or Metal

## rune, kaun, vega

- attention under `Rune.jit`: causal saves little, since a grid under 1.3
  waves takes as long as its longest workgroup (rows already launch longest
  first): pair row tiles i and n-1-i, or split long rows' keys; the
  tensor-core backward leaves head dimensions over 64, float captures'
  cotangents, paged keys, forward mode and float32 on the composition
- attention's SIMT kernel on AMD (RDNA3, and what no core takes): K and V
  addresses are wave-uniform, so LLVM loads bfloat16 and float16 with
  s_load_u16 and a wait each, about 4100 serial round trips a tile; at
  gpt-oss's decode shape float32 (global_load_b32) takes 51 µs against
  bfloat16's 323 µs on the R9700

## talon, hugin, brot, munin, quill

- talon perf against the faster of DuckDB 1.5.6 and Polars 1.44.2 on fudge
  (9b82c5b57a, alternating rounds under load): decision support SF1 6.6x
  geomean (2.5-10.1x), H2O 1e7 group-by 2.1x, joins 2.0x. talon uses 3-10x
  the CPU, so the gap is work. Over the five worst SF1 queries scans are
  half the instructions (Snappy 20%, copies 14%); on q07, q08 and q12
  DuckDB's dynamic join filters and join order read far fewer rows; q01's
  aggregate is 46%. A segment-id reduce kernel in nx is unmeasured.
  Since then e4b49ffba3 cut q01 by 14%. Data: ~/wt/tmp/talon-data
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
