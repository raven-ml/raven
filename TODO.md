# todo

goalpost: jit-compiled gpt-oss at least as fast as {tinygrad,JAX,llama.cpp/vLLM} on {NVIDIA,NV,AMD}.
Where it stands, with every measurement: doc/performance.md.

## tolk

- Dense bf16 projections on CUDA (_plans/perf-campaign/proj/): k and v at
  every token count run 8-64 workgroups with no split of K, at 88 GB/s;
  q at 512 tokens runs at 107 TFLOP/s against cuBLAS's 135; a token count
  with no factor of 16 takes no core (q 5.9 ms at 127 tokens); weights laid
  `[K; N]` take no core, about 100x their byte bound. `Stage.tile` plans
  two stages where `operands` copies in one, and sizes copies at the core's
  input dtype; one tile rule for trip depth and warp split should own both
- Since D479 groups starved reduces, a GPU `Set` election of 16384 and
  24576 rows runs 1.4x and 1.9x slower than rune's former split, and
  elections over several indices near `stream_threads` outputs up to 3.5x;
  a split decided where the renderer is known would close the gap
  (_plans/consolidate/groupgate/adjudication.md)
- tolk's "launches > a schedule does not depend on what the process built
  before it" can pass its 60 s limit under load: it runs after
  `order_independent`, whose schedule cache stays live, so its 200 full
  collections take 10-18 s; running `order_independent` uncached fixes it
- rune's paged attention with pages of 16 keys runs at 168 registers on
  CUDA, one workgroup an SM; its values spilled for a tile are refilled
  every tile, skipped ones included, which keeps K's copy from moving
  before the mask
- tolk's tensor core decodes MXFP4 in its warps before each multiply, 0.37
  SM-cycles a weight against a stream's 0.25, below the memory's rate. A
  decode into packed halves from an offline layout (Marlin, arXiv
  2408.11743 §3.4) would keep it at that rate: rune's routed blocks would
  then take the core from one row an expert, as bfloat16 slices do, where
  they stream blocks of 4 up to 2 rows, and Mixtral-like experts would
  lose nothing at 9-12 tokens
- A tensor core's kernel sums its whole reduce in each thread, so a grid
  of few blocks by few outputs reads its operand more slowly than a
  stream: on Mixtral-like experts (8 of 14336 by 4096, top 2) rune's
  routed blocks of 48 take the down product about 20% longer than blocks
  of 4 at 9-12 tokens, the MoE block 1-2% longer. A split of such a
  kernel's reduce would remove it
- Under expert-parallel placement each device's routed product holds its
  `g / D` experts and every row, those routed elsewhere in no group, and
  rune's `block_rows` reads all `n` rows over the local `g`: its rule sees
  `D` times the mean rows an expert. On 2 devices gpt-oss-20b takes the
  core from 9 tokens and blocks of 64 from 64, where one device takes
  them from 17 and 128. rune should count the rows routed to its own
  groups
- gpt-oss's router product takes 620-635 us at 16 tokens against 99 at
  12 (`r_2_8_16_2_2880`), and 85 at 127 against 39 at 128
- tolk streams a routed block of 32 rows, which `core_fits` refuses: such
  blocks took gpt-oss's MoE block 6071 us at 24 tokens
- gpt-oss's non-expert kernels take 41.7 ms over 24 layers at a 68-token
  prompt against 29.4 at 108: the attention projections at a count with a
  factor 17, `r_8_17_8_16_4_4_720_4` and `r_17_8_16_4_4_720_4`, take 13.9
  and 8.5 ms
- gpt-oss's prefill still stores gate_up's bias add as its own scatter: it
  reads 4 buffers, so the 3-buffer rule keeps it, about 5% of the MoE block
  at 512 tokens and 1-2% at 17-127. Stages are decided bottom-up, so
  inlining it pushes two removable consumers past 3 buffers; the order in
  which tolk decides stages is the question, not tinygrad's limit
- rangeify's law "a scheduled function writes what its tensors compute"
  fails under seeds `s1:63ab132b4eb9fb74` and `s1:47b3074f072c6405`: both
  graphs end in `sum 0, expand, max 0`, and the kernel graph writes
  nothing where the tensors write one element
- tolk recomputes a cheap broadcast value along every row that reads it:
  a 3-buffer polynomial of D multiply-adds broadcast along 4096 rows takes
  11.3 ms on the host against 0.64 ms stored. `recomputes`/`copies` count
  the reader's ranges; bounding it would also let fft-4096 drop 20 kernels
- On gpt-oss's cold first call, rangeify's reshape substitution
  (`Indexing`, the `apply_reshape` over placeholder ranges) allocates 8.5%
  of the call's words on the calling domain, before the parallel render
  wave starts
- AMD thread traces are not collected in a profile, and tracing through KFD
  is parked: after a GPU fault KFD runs none of the process's queues, so
  nothing can stop a running trace (one stayed on past its process's exit on
  nonnormal), and an eviction may move its window while it writes. rig's
  traced launch test runs only over PCI. The paths left are PCI and an
  amdgpu DRM submission, which holds its VMID and memory for the job

## nx, rune

- nx's `permute is a bijection` (test/test_rng.ml) covers "one element" on
  some seeds only, so the test can fail on a draw that reaches no such case
- rune's blocked complex contraction bound, `γ(β(K) + 2)`, exceeds the
  kernel contract's `√2 γ(β(2K))` at K = 1; it matters if compiled programs
  must stay within the kernel contract

## talon, hugin, brot, munin, quill

- talon needs a query optimizer, a large design of its own, and the other
  optimization to match the best reference engines; 1.0 has no speed gate.
  Last measured against the faster of DuckDB 1.5.6 and Polars 1.44.2
  (9b82c5b57a, under load): decision support SF1 6.6x geomean, H2O 1e7
  group-by 2.1x, joins 2.0x, with talon using 3-10x the CPU. What the gap
  is made of: join filters pushed at run time and join reordering (DuckDB
  reads far fewer rows on q07, q08 and q12), scans (half the instructions
  of the five worst queries), aggregation (46% of q01), and possibly a
  segment-id reduce kernel in nx (unmeasured). Data: ~/wt/tmp/talon-data
