# todo

goalpost: jit-compiled gpt-oss at least as fast as {tinygrad,JAX,llama.cpp/vLLM} on {NVIDIA,NV,AMD}.
Where it stands, with every measurement: doc/performance.md.

## tolk

- One binary per source (D485) shares nothing on a GPU between kernels over
  buffers of two sizes: cstyle names each parameter after its buffer's size
  (`data0_4096`), so the compiled texts differ. The compiled text should
  name parameters without sizes, decided where the renderer names them
  (renaming in `Codegen.without_display` would need a bijection per
  parameter; renaming in the renderer moves every source golden)
  (_plans/tessera-port/design/shapes/s3s5-notes.md)
- `Codegen.to_programs` compiles each worker's sources apart, so two kernels
  of one compiled text taken by two workers each run Clang or NVRTC: 5 of
  the 165 sources Clang compiles in tessera's cold distillation step at 2
  workers. One compile per distinct text needs the texts gathered before
  the workers split them
- Dense bf16 projections on CUDA (_plans/perf-campaign/proj/, proj2/): k
  and v run 11.3 us at 64 tokens to 24 us at 512, L2 evicted
  (`stream/key-projection-64` 15.7 us), 2-4x their 5.5 us byte bound: they
  lack a split of K across workgroups;
  q at 512 tokens runs at 107 TFLOP/s against cuBLAS's 135; a token count
  with no factor of 16 takes no core (q 5.9 ms at 127 tokens); weights laid
  `[K; N]` take no core, about 100x their byte bound.
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
- gpt-oss-20b's MXFP4 experts on fudge's RTX 5000 Ada, measured on
  2026-11-16 with the bfloat16 decode: the stream of 4 takes 0.31-0.33
  SM-cycles a weight, issue-bound at 24.4 instructions a weight, 13 of them
  decode, and a core 0.30-0.31 on gate_up and 0.31-0.32 on down,
  latency-bound at 9.0 a weight and 4 workgroups an SM, against DRAM's
  0.268; both read only the compulsory bytes. Measured on 2026-11-15 with
  rune's float32 decode, before the bfloat16 decode and the byte merge:
  of the 512-token gap to vLLM, the
  decode in the warps took 2.84 ms of gate_up's 6.27 and 1.50 of down's
  3.49, and the code loads, 20 single bytes a thread a trip, 4.14 and 2.13
  once the decode is cheap; a layout gains nothing at 64 tokens without
  that decode. At 64 tokens down's 324 workgroups on 300 slots lose 0.58
  ms to the wave tail. The decode costs 7-11% of the decode step's expert
  kernels and 22-23% of the 12-token ones; the experts are 0.11 ms of the
  decode step's 2.89 ms gap to llama.cpp
- A tensor core's kernel sums its whole reduce in each thread, so a grid
  under one wave, or just past one, reads its operand more slowly than a
  stream. With decoded slices on cores from one row a group, gpt-oss's
  down at 8-11 tokens on some routings (up to 2.8% a block) and
  Mixtral-like MXFP4 down at 4 tokens routed onto 3-4 experts (up to 7.1%
  a block; `Cuda/experts-mxfp4-4x2of8-14336x4096` +38%) lose to it. A
  split of the
  reduce, in the graph as tinygrad's `split_reduceop` does or in codegen,
  from the SM count and occupancy, would remove it
- A routed core's workgroups an SM follow the blocks a routing uses, which
  only the run knows: on random routings occupancy decides between staged
  plans, and gpt-oss's 64-token cores sit on the 128-register edge of 4
  workgroups an SM
- rune's routed products with no core: one row a block loses 12-33% to
  blocks of 4 at 3-7 tokens, and Q8_0's blocks of 64 from 16 rows a group
  take 1.5-2.9x as long as 8 or 16 (9480 against 3854 us at 128 tokens);
  the 64 came from MXFP4 under the float32 decode
- Under expert-parallel placement each device's routed product holds its
  `g / D` experts and every row, those routed elsewhere in no group, and
  rune's `block_rows` reads all `n` rows over the local `g`: its rule sees
  `D` times the mean rows an expert. On 2 devices gpt-oss-20b takes the
  core from 9 tokens and blocks of 64 from 64, where one device takes
  them from 17 and 128. rune should count the rows routed to its own
  groups
- gpt-oss's router product takes 620-635 us at 16 tokens against 99 at
  12 (`r_2_8_16_2_2880`), 567-764 us a layer at 13-15 tokens, and 85 at
  127 against 39 at 128
- With the bfloat16 decode and CUDA's byte words, how MXFP4 codes reach
  a tensor core's lanes holds about 4.4 ms of gpt-oss's prefill-512
  against vLLM, and staged Q8_0 codes ran 1.5-1.8x slower
  (`stage.ml`'s comment) for a reason nobody has measured
- MXFP4's float32 decode on Metal multiplies by a subnormal at scale byte
  0, which Metal may flush (MSL 4.1 §8.1); no test reaches it there
- tolk streams a routed block of 32 rows, which `core_fits` refuses: such
  blocks took gpt-oss's MoE block 6071 us at 24 tokens
- gpt-oss's non-expert kernels take 41.7 ms over 24 layers at a 68-token
  prompt against 29.4 at 108: the attention projections at a count with a
  factor 17, `r_8_17_8_16_4_4_720_4` and `r_17_8_16_4_4_720_4`, take 13.9
  and 8.5 ms
- At heads of 80 (32 heads, 2560 wide) the q and k projections take no
  staged tensor core: 85 and 38.9 us at 64 tokens, against 51 and 11.3 at
  heads of 64. Not profiled
- A staged core copies in at most two stages, so where twice the bytes in
  flight pass `shared_max` it can only deepen its trips; more, shallower
  stages need a counted `cp.async.wait_group` in the CUDA renderer
- A gather that reads more elements than its table holds, with no
  broadcast above it, still inlines its table's computation into every
  read
- `Rangeify.buffers` keys `homes` by a stage's value, not its node, so two
  stages of one value share a home; no failing case is known
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
- Metal's renderer emits no launch bound, and its `local_threads_max` of
  1024 is the theoretical maximum a pipeline may refuse; Mac2's AMD GPUs
  may run 64-wide SIMD groups, which `simd_threads` does not state
- `apply_opt` checks a workgroup's shared memory but not its threads
  against `local_threads_max` (`postrange.ml`), so an opt path that grows
  a group or locals, BEAM's included, can emit a launch the target refuses
  (1024 threads on gfx9's wave64, 512 on Apple2/3)
- With locals ordered innermost first outside blocked contractions,
  `matmul-float32-heads-16384x440x4x110` runs 22% slower than with the
  same order inside them, which moves other rune products by -7% to +8.7%
  (_plans/perf-campaign/norms/adjudication.md)
- `Nx.rms_norm` under a trace runs a third pass, a lane max, for its
  overflow promise: 49 kernels a gpt-oss step. The one-pass form keeps
  the bits but tolk then inlines the norm's map into q, k, v and the
  experts (+3.5 to +7 ms a prefill), until tolk keeps stored a value a
  contraction re-reads across output tiles
- `beam_search`'s disk key names no device part, so a search timed on one
  part of a target serves every other part
- `Heuristic.reused` (heuristic.ml:853) and `upcast_more`'s `reuses`
  (:926) count every access in the kernel, loads after the reduce
  included, so a bias that skips an axis both operands of a contraction
  read makes that axis look reused. Unverified with a kernel. Found by the
  I1 lend fix
- A bf16 product's tensor core at five K tiles with a float32 `exp`
  epilogue runs 1.22-1.28x slower than SIMT: at 65536x72x512, padded to
  K = 80, 0.347 against SIMT's 0.284 ms, while main takes the same core
  unpadded at K = 80 in 0.354 ms. Found by the G-D3 review
  (_plans/tessera-port/design/gemm2/gd3-review.md)

## nx, rune

- rune's attention core takes 8 warps where 8 still leave
  `local_workgroups` workgroups, else 4 (lower_attention.ml:1516). But
  renderer.mli:277-279 scopes that field to cores whose operands are not
  staged, and this core shares its tiles among its warps in workgroup
  memory. 8 against 4 was never measured. At TESSERA's shapes the core
  takes 8 (_plans/tessera-port/design/attention2/skeptic.md:66-70), so
  whether 4 would serve its rows better is unknown. Found by the
  TARGETFACTS audit (_plans/consolidate/targetfacts, A3)
- rune's attention props "every dtype" (test_attention.ml:665-669) and, on
  the GPU, "its gradient mapped by vmap" (test_jit_gpu.ml:3098-3104) take
  10-15 s on an idle fudge. At a load of 30-60 on its 14 cores they pass
  the 60 s `WINDTRAP_TIMEOUT` (dune:53) and fail the runtest; a rerun
  passes (10.7 s at load 14). Each case compiles programs of its own
  shapes. Where their time goes is unmeasured. tolk's launch law is the
  tolk entry above. Found by the TARGETFACTS audit and CODEPATH design A
- rune states a compiled complex contraction's part within
  `γ(β(K) + 2) Σ(|ar br| + |ai bi|)` (rune.mli:1696-1698,
  lower_reduce.mli:141-145), while `Nx.contract` and the kernel contract
  give `√2 γ(β(2K))` in modulus (nx.mli:2685-2688, nx_kernel.mli:300-303).
  At K = 1 alone, rune's bound is larger: `γ(5)` a part against `√2 γ(4)`
  in modulus. From K = 2 on, `β(K) + 2 ≤ β(2K)`. It matters if compiled
  programs must hold `Nx.contract`'s bound. Found by NXBOUND
- The fused attention backward refuses values of no column: `dplan`
  requires `a.dv > 0` (lower_attention.ml:2015), so an `Nx.attention` call
  over values of no column, legal by nx.mli and handled by the forward
  (:2204-2209, :2261-2263), takes the default derivative at every head dim
  and stores the weights. Lifting the condition takes more than it:
  `linear`'s operands have no guard for an empty `v` or `ō`, nor for the
  δ kernel's `o` (:2317, :2324), which all need the forward's `operand`
  (:2206-2209), one function for both lowerings rather than a second copy;
  `v̄`'s buffer of at least one element is reshaped to `[…; 0]` (:2318,
  :2359), where the forward fills its `o` at `dv = 0` (:2262); and
  `Values` launches whenever `wide` (:2340-2350). tessera's dropout
  composition needs it for its lse-only call (attention2 R7a). Found by
  the attention round-2 audit and skeptic
  (_plans/tessera-port/design/attention2/skeptic.md §1)
- `Nx.rearrange` plans for 190-252 words beyond its movements (a `b h`
  split costs 362 against 110 moved), for patterns that move nothing too.
  Found by the nx-hostcost audit
- A 0-d operand against a leaf of rank 1 or more costs 96-100 words a call,
  44 of them the broadcast, and vega pays this for its 0-d rate and weight
  decay. Found by the nx-hostcost skeptic
- An eager creation at a set without kernels holds its window whole on the
  host: `Exec.created` (exec.ml:589-600) fills each window on the host,
  then copies it, so a one-device value needs host memory of its full size
  where the device could hold it. On fudge, `capped`'s 8 GB is below an RTX
  5000 Ada's 32 GB. Slabs extend rig's own practice one layer up: rig moves
  a paged copy through staging of 32 MiB allocations whatever its size
  (rig.mli:1476-1488), and a copy through staging returns once the host's
  transfers are done, so the host may then reuse its source
  (rig.mli:1588-1590); a direct copy returns once queued (:1584-1587). The
  slabs' copies are from a plain host buffer, so they go through staging.
  So `created` would allocate the device array, then compute each slab at
  its `Prim.offset` into one host buffer and copy it into its box
  (rig.mli:1575-1582), with slab edges on whole bytes for dtypes under a
  byte (nx_array.mli:212-221). The bits are the
  same by the window law, and no second staging is needed. Where the host
  addresses the device's memory (`share`, exec.ml:576-580) `created` could
  instead compute straight into the device array. Owed by the DD
  adjudication (W1); found by its audit and skeptic
  (_plans/tessera-port/design/nx-vocab/skeptic.md §3)
- `Nx.log10` is a composite over `log2`, so its float64 bound follows only
  where `log2` is within 1 ulp: under `Rune.jit` on the host, over tolk's
  `xlog2` (transcendental.mli:16-22). Eagerly, over nx's `log2` stated at
  2 ulps, it is measured, as are the float32 bound, at every float32, and
  the exact powers of ten. On a device rune bounds `log2` at 4 ulps
  (rune.mli:1610), which leaves log10 within about 6, and the powers are
  unmeasured there. A `Log10` kind lowered over rune's `log2_parts`
  (lower_real.ml:504), which gives log2 as two parts, would prove both on
  every device, once a consumer needs log10 bounded there. Found by the DD
  audit (S2)
- `Nx.zeros_at` and `Nx.full_at` of a value whose bytes repeat one byte
  (zeros, all-ones integers) are maps of no loads, so on a set without
  kernels `Exec.created` fills them on the host and copies the window
  (exec.ml:584-600), though rig can set the bytes on the device
  (`Rig.Buffer.submit_fill`, rig.mli:1648-1660) with no host memory and
  no copy. Eager optimiser moments made at a GPU set outside a compiler
  pay a host fill of their size and a copy of it to the device.
  Unmeasured. Found by the DD design round (N3)
- rune.mli:1400-1402 says that a value a compiled function captures on
  another device and places where the program computes "is copied there
  once per program". In a split program (values whole on each of several
  devices) the capture is instead bound as it is (compile.ml:339, `Some r
  when st.split || …`), and its placement copies it at every call.
  Measured on two software devices: 6 float32 captures cost 48 bytes a
  call, 4,848 over 101 calls
  (_plans/tessera-port/design/jit-values/skeptic-probe/split.log). vega's
  schedules capture their numbers, so a data-parallel step pays this for
  each. Either upload once per device in split programs through
  `st.uploads`, or scope the sentence to one-device programs. Found by DC's
  skeptic

## vega

- A port of Barlow Twins' LARS also gives biases and normalisation
  parameters their own learning rate, 0.0048 against the weights' 0.2
  (Zbontar et al. 2021, "Optimization"; `main.py` lines 167-168). No vega
  rule sets a rate by the kind of leaf: rates differ by part
  (vega.mli:378-403), and these leaves interleave across every layer.
  Decide whether a rate by leaf kind is a rule on the leaf, as `~decays`
  and `~adapts` are; a third rule would ask whether the rules become one
  record (_plans/tessera-port/design/jit-values/skeptic.md §3). Found by
  DC's V2 audit
- Momentum schedules. PyTorch's OneCycleLR cycles beta1 by default
  (`cycle_momentum=True`), and PyTorch took tensor betas for it (PR
  #134171, issue #133898). vega's `?b1:float` is read once at the trace
  of a compiled step, so such a port is silently wrong. Decide an opt-in
  data form for b1 when a consumer needs one; until then vega.mli:48-52
  states that a sweep over any of a step's coefficients compiles once per
  value. Found by DC's skeptic

## rig

- Windows has no read depth: rig's disk opens a file without
  `FILE_FLAG_OVERLAPPED` (rig_disk_stubs.c:96-103), which serialises its
  I/O even where each `ReadFile` passes an `OVERLAPPED` offset. Found by
  the streaming-data part 2 audit

## sowilo

- A nearest resize to one column or row picks the first, not the centre:
  `compute_nearest_indices` (contrib/sowilo/lib/transform.ml:20-22)
  returns index 0 when `size_out = 1`. Pillow's NEAREST and OpenCV's
  INTER_NEAREST_EXACT, which sowilo matches at every non-tie pixel of a
  sweep of widths 1 to 96, pick `size_in / 2`: 3 from 7 columns.
  sowilo.mli:100 states no centre rule, and `round (y - 1/2)` equals PIL's
  `floor y` only because `Nx.round` takes halves away from zero. State the
  rule and compute it as `floor ((i + 1/2) scale)`, which has no tie and is
  the same at every other index. Found by the DD skeptic
  (_plans/tessera-port/design/nx-vocab/skeptic.md §2)

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
