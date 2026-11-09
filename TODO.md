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

- AMD thread traces are not collected in a profile. Through KFD a trace is
  unsafe until shown otherwise: an eviction may move its window and page
  directory while the trace unit writes on, so rig's traced launch test runs
  only over PCI. A probe deactivating a traced queue over a poisoned window
  decides (_plans/todo-campaign/sqtt/adjudication.md)
- spawning any process stalls a rig AMD program's GPU work about 100 ms on
  Linux before 6.16 (nonnormal, 6.12): a child's close of its copy of the
  render node, at exec or exit, runs amdgpu_flush, which waits on KFD's
  eviction fence and so evicts the process for 100 ms. rig could hold no
  descriptor on the acquired render node, mapping through dma-bufs
  (_plans/todo-campaign/spawn-stall/findings.md)
- long programs: tolk's passes are linear in the nodes they walk, about 0.4 us
  a row over about 130 walks, flat (no pass past 8%, no function past 4%):
  the gradient of `betainc` costs tolk and rune's tracing 0.8 s and Clang
  0.7 s, its third derivative (220k nodes) 5.3 s and 12.9 s. Past 100k nodes
  Clang decides the wait.
  Measured and declined for tolk: a per-domain cache of the last long walk
  (-2% instructions, 12 lines), and pruning rewrites by the operations a
  matcher rules on (no gain: the memo it needs costs what it saves). Two
  costs grow with the square of a chain: the linearizer breaks ties by
  `Ops.compare_structure` under `TUPLE_ORDER`, so 10,000 operations read
  once take 0.9 s and 25,000 take 6.6 s; and a flat chain of additions
  stays one C statement, rebuilt and scanned at every step
- not yet run on AMD: attention's backward (167d026789), stores and reads
  through views and bitcasts (1132c5b84d), a loop's non-run reads each trip
  (ecfd400a58) and cores on several reduce axes (02e163a27d); each but the
  first has run on Metal. The backward's CDNA 4 tile rule has run on no GPU:
  none here is a CDNA 4

## rune, kaun, vega

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
