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
- `Cuda/cat-16-columns-of-exp-64x64`, the row that catches a lost mark on
  a concatenation, has no AMD baseline
- a fixed run of an `After` state read inside a loop still raises where the
  state is over a bitcast, or a join over other writes of its storage
- one launch pass (ecdd3203f6), views naming their storage (de0bcf38d9)
  and per-trip loop reads (293eb3f70b) have not run on AMD or Metal

## rune, kaun, vega

- attention under `Rune.jit`: the tensor-core backward leaves head
  dimensions over 64, float captures' cotangents, paged keys, forward mode
  and float32 on the composition
- attention's SIMT kernel on AMD (RDNA3, and what no core takes): K and V
  addresses are wave-uniform, so LLVM loads bfloat16 and float16 with
  s_load_u16 and a wait each, about 4100 serial round trips a tile; at
  gpt-oss's decode shape float32 (global_load_b32) takes 51 µs against
  bfloat16's 323 µs on the R9700

## talon, hugin, brot, munin, quill

- talon needs a query optimizer, a large design campaign, and the other
  optimization to match the best reference engines; 1.0 has no speed gate.
  Last measured against the faster of DuckDB 1.5.6 and Polars 1.44.2
  (9b82c5b57a, under load): decision support SF1 6.6x geomean, H2O 1e7
  group-by 2.1x, joins 2.0x, with talon using 3-10x the CPU. What the gap
  is made of: join filters pushed at run time and join reordering (DuckDB
  reads far fewer rows on q07, q08 and q12), scans (half the instructions
  of the five worst queries), aggregation (46% of q01), and possibly a
  segment-id reduce kernel in nx (unmeasured). Data: ~/wt/tmp/talon-data
- talon.parquet's pacing law failed once on macOS (run 42473531097: 35
  major cycles against 34); the runtime's own pace cannot explain it (the
  loop puts ~4.8k words in the major heap) and it passes at rig's 4 MiB
  floor on fudge; on failure the law now prints each collector's counts
  (0247fa6d8)
