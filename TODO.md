# todo

goalpost: jit-compiled gpt-oss at least as fast as {tinygrad,JAX,llama.cpp/vLLM} on {NVIDIA,NV,AMD}.
Where it stands, with every measurement: doc/performance.md.

## tolk

- Small products in the rows layout run too few workgroups: gpt-oss's
  router at 64 tokens or fewer and at decode, and the 12-token k and v
  projections, take 7-24 us streamed (the rows layout runs rows(W)/4
  workgroups, 8 warps for a 32-row W) where the general rules' grouped
  reduce and BEAM take 5-11. A few-workgroups case for the rows layout, as
  D110 is for columns, is undesigned (_plans/perf-campaign/routercore/)
- Bare tensor cores (`TC` with no K upcast, 180 serial K steps) take 37.9 us
  on the router at 8, 16 and 48 rows against the stream's 16-23; at R <= 16
  tolk's hand cores (projection 8 and 16: 59 and 70 us) trail BEAM's
  (48.1, 49.1) (_plans/perf-campaign/routercore/design.md)
- The Stage suite's cover "gathered rows left unstaged" (`test_stage.ml`
  ~931) is vacuous at seed `s1:9cc4952459e7d173` on main: a flaky cover
- D212's blocked contractions still lose to tinygrad's heuristic on long
  reduces: `contract/f32-2048` 1.03 of main against off's 0.92, attention's
  values of width 256 (`f0865e`) 1.04 against 0.86. BEAM's winners there
  split the reduce across 64 threads in shared memory; `Heuristic.group`
  never groups a contraction. Below D212's floor, splitting the reduce of a
  few-output contraction runs matmul_long in 0.47 of main (design b), and
  BEAM still runs the small suite products 2-4x faster
  (_plans/perf-campaign/handtiles/)
- `Heuristic.four_first` gates 4-before-3 on the fitted `core_threads`
  (16384), D212's floor on `units x 2 x 128` (25600): products between
  them, as the suite's matmul_768, take 4 by 4 unblocked with an unroll by
  4 (0.69 of main, 0.66 of tinygrad's heuristic); tinygrad's own 4 by 3 for
  them is unmeasured. Gating both on `units` would remove a fitted field
- A dense MXFP4 core never reaches `Stage.tile`: MXFP4's byte of two
  codes is still a loop at its entry (`unroll_beside` runs only in
  `unstaged ()`), so `admits` refuses. Staging it runs `stream/mxfp4-64`
  in 29.7 us against 60 (BEAM's 29.8; branch `coretiles` `07eb7efd13`).
  Parked: choosing staged against unstaged by the busiest unit's carried
  bytes needs the unroll before the first ask (reaching every bf16 staged
  row) and plans that block their sums alike, or the choice moves a sum's
  order with `units` (_plans/perf-campaign/coretiles/adjudication.md)
- D357's inner-end group split loses to tinygrad's top split on the
  bfloat16 argmax and max over 201088 values (0.67-0.77x, same axis, threads
  and registers; about 3 us a decode step) and, through `across`, on
  sum_cols_prime (0.37x: 32 threads from the top on 1571 rows padded to
  1600). The cause of the argmax case is unmeasured
  (_plans/perf-campaign/handtiles/)
- A routed tensor core over few working blocks waits on one copy's latency
  a trip: cutting its reduce into parts in rune's `Lower_reduce.rows`
  (float32 partials at the slots, summed in order by the existing gather)
  runs Mixtral-like MXFP4 down at 4-96 tokens 29-40% faster on sparse
  routings and the gate_up gradient by x 64% faster (branch `splitcount`
  `f4a4b3e16b`, tests included). Parked on two questions: the count needs
  tolk's plan of the uncut core (tile, staging, K depth), which reads the
  kernel's AST, not integers rune holds; and a cut staged core loses K depth
  in its own plan (gpt-oss +2-12%, prefill 64 +11%), whatever the divisor.
  gpt-oss gains nothing (_plans/perf-campaign/splitcount/)
- A decoded product past 16 rows takes a core (D230), but `Heuristic.core_fits`
  and `core_rows`, which rune reads, still count every pad: rune's `partial`
  keeps the flat form for Q8_0-like products of 17-32 rows and past 32 where
  the heuristic would take the core, and a decoded routed slice's `core_rows`
  stays 48 where 32 would do. Proposal: both take `~decoded`; `core_rows`'
  change waits for SPLITCOUNT's count (_plans/perf-campaign/rowdecode/)
- A decoded operand of 12-16 rows streams in one pass, but a core probably
  wins there too (kaun: 559-706 us streamed at 12 and 16 rows against cores
  of 424-492 at 17-64); it reaches gpt-oss Q8_0 prefill 12
  (_plans/perf-campaign/rowdecode/design-a.md)
- The unstaged decoded core at 17-32 rows runs 1.33x BEAM's: BEAM's own opts
  on that core take 334 us against 393-417, from K depth against
  `in_flight` (256 workgroups x 4 KB over 100 SMs); with `stream/mxfp4-64`
  (BEAM 2x) it belongs to the core tile (_plans/perf-campaign/rowdecode/design-b.md)
- Metal's simdgroup matrices share the ALUs, so a padded decoded core's
  extra multiply-adds (17 rows to 24) may cost there; D230's rule is
  unmeasured on Metal and AMD
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
  the workers split them. gpt-oss's cold call compiles no text twice, and
  its one pair of texts differing by sizes costs about 130 ms off the
  call's chain (_plans/perf-campaign/coldcall/cause.md)
- Dense bf16 projections on CUDA (_plans/perf-campaign/proj/, proj2/): k
  and v run 11.3 us at 64 tokens to 24 us at 512, L2 evicted
  (`stream/key-projection-64` 15.7 us), 2-4x their 5.5 us byte bound: they
  lack a split of K across workgroups;
  q at 512 tokens runs at 107 TFLOP/s against cuBLAS's 135; weights laid
  `[K; N]` take no core, about 100x their byte bound.
- A group beside an output axis split across keeps 16 threads across
  (D487), which on fudge's RTX 5000 Ada leaves a band open: bfloat16 column
  sums gain 10-40% at 32 across from 1024 columns while the input sits in
  the L2, and from DRAM 32 across gains 6% on bfloat16 and 21% on float32 at
  8192 columns, where float32 loses 34-41% at 1024. A rule needs the part's
  SM count, as the target's bound with a stated band, and none is derived
  (_plans/perf-campaign/groupwidth/skeptic.md §2)
- A staged tensor core pads N to its tile (D491) but not M: a token count
  whose padded M the warp tile's upcast does not divide keeps the unstaged
  core, gpt-oss-20b's q projection 115 us at 100 tokens (M padded to 112)
  against 56 at 127 and 128 (fudge,
  _plans/tessera-port/design/gemm2/gd3-notes.md). M's upcast also keeps
  routed rows within a run (`in_run`), which a pad must respect
- Where a core's rows are 8 mod 16, the heuristic takes main's swapped core
  (the rows along N, level 0, N then padded to the staged tile) before
  padding M on axis 0 (D491). Neither wins everywhere on fudge: q of
  gpt-oss-20b, swapped against axis 0 padded, 67 against 79 us at 40
  tokens, 64 against 109 at 72, 70 against 115 at 104, but 67 against 54
  at 88, and 54 against 53, 57 against 56 at 56 and 120; k 15 against 45 at
  72, 17 against 15 at 120; [88; 2880] by [2880; 4096] 0.051 against 0.033
  ms. A rule between them is a measured heuristic decision of its own
- A padded core loses to main's SIMT kernel under an exp epilogue: bfloat16
  [65536; 72] by [72; 512] into float32 with an exp after it takes 0.348
  ms against 0.283 (1.23x), by 4096 columns 2.71 against 2.11 (1.28x),
  16384 rows by 4096 0.687 against 0.532 (1.29x). Main takes the same core
  unpadded at K = 80 in 0.354 ms, and without the exp K = 72 runs 0.98 of
  main: the loss is the core's kernel at five K tiles with that epilogue,
  not the padding (fudge, _plans/tessera-port/design/gemm2/gd3-review.md)
- The 440 by 440 weight gradient over 68608 rows, a split sum of 16
  partials, takes the staged tile on 448 padded columns at about 2.27 ms
  where staging off takes 1.78 (1.28x; fudge, on PROJ-2's plans; both about
  11 times faster than main's 25.6): `Stage.tile`'s plan for a split
  partial, which the two-kernel split's design should re-derive
- Since D479 groups starved reduces, a GPU `Set` election of 16384 and
  24576 rows runs 1.4x and 1.9x slower than rune's former split, and
  elections over several indices near `stream_threads` outputs up to 3.5x;
  a split decided where the renderer is known would close the gap
  (_plans/consolidate/groupgate/adjudication.md)
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
- MXFP4 codes reach a tensor core's lanes from shared memory by 16-byte
  `cp.async.cg`, 32 bytes a row a trip, beside their held scales, which
  took gate and up and down at 512 tokens 10% faster on real weights.
  Q8_0 stays unstaged: copied through registers by one warp and waited in
  its trip, on the staged tile's smaller grid, its 34-byte blocks ran the
  output 1.34-1.43x slower at 64 tokens and query, key and value
  1.09-1.19x at 512 (a spill at 255 registers), and query, key and value
  15-18% faster at 64
- At 64 tokens the MXFP4 expert cores of 64 values a trip take 146
  registers, 3 workgroups an SM against 4 at 128: on a real routing the
  down projection runs 10% slower than with its codes unstaged (2 waves
  of its 648 workgroups become 3), gate and up 10% faster. A dense MXFP4
  product at 512 tokens takes 255 registers and spills 8 bytes. Four an SM
  needs both R <= 128 and S <= 24576 (K 64 takes 25664 B), and no count
  before ptxas separates where K 64 pays (512 tokens) from where it costs
  (_plans/perf-campaign/coretiles/)
- MXFP4's float32 decode on Metal multiplies by a subnormal at scale byte
  0, which Metal may flush (MSL 4.1 §8.1); no test reaches it there
- tolk streams a routed block of 32 rows, which `core_fits` refuses: such
  blocks took gpt-oss's MoE block 6071 us at 24 tokens
- gpt-oss's non-expert kernels take 41.7 ms over 24 layers at a 68-token
  prompt against 29.4 at 108: the attention projections at a count with a
  factor 17, `r_8_17_8_16_4_4_720_4` and `r_17_8_16_4_4_720_4`, take 13.9
  and 8.5 ms. Since D491 they pad to a core: q at 68 tokens takes 106 us
  against 511 (one layer, L2 evicted); the prompt is not re-measured
- At heads of 80 (32 heads, 2560 wide) the q and k projections take no
  staged tensor core: 85 and 38.9 us at 64 tokens, against 51 and 11.3 at
  heads of 64. Not profiled
- A staged core copies in at most two stages, so where twice the bytes in
  flight pass `shared_max` it can only deepen its trips; more, shallower
  stages need a counted `cp.async.wait_group` in the CUDA renderer
- A gather that reads more elements than its table holds, with no
  broadcast above it, still inlines its table's computation into every
  read: 2.2-2.4 ms against 0.7-1.1 ms stored on the host probe, no bench
  row moved. A count of extra evaluations against D163's 360 budget stores
  3-6x too early and prices no launch (_plans/perf-campaign/r3/)
- tolk recomputes a cheap broadcast value along every row that reads it:
  a 3-buffer polynomial of D multiply-adds broadcast along 4096 rows takes
  11.3 ms on the host against 0.64 ms stored. `recomputes`/`copies` count
  the reader's ranges; bounding it would also let fft-4096 drop 20 kernels,
  which D481's stored-readers condition holds today
- The buffer limit keeps gpt-oss's 63x64x2880 routed-rows stage stored once
  an inlined norm map makes it read four buffers: a 23 MB gather, about
  0.97 ms a prefill-512 step. This is what blocks the one-pass
  `Nx.rms_norm` (D481 names it; _plans/perf-campaign/r3/audit.md)
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
- `Nx.rms_norm` under a trace runs a third pass, a lane max, for its
  overflow promise: 49 kernels a gpt-oss step. The one-pass form keeps
  the bits but tolk then inlines the norm's map into q, k, v and the
  experts (+3.5 to +7 ms a prefill). Stored by an inline rule, it still
  trails the three passes by 0.7-1.8 ms at prefill 512, held back by the
  routed-rows gather below
- D404's launch member caps, NVRTC's 64 and Metal's 40 (an M1 Max), name
  no fudge measurement: the CUDA cap needs a fudge row, as the ledger's
  header asks of a speed measured elsewhere
- `Ops.bound_loads`, and so `Postrange.bound_by_memory` (D255), sees only
  `Op.Load`, while a scheduled kernel reads memory through bare indices:
  a loop bounded by `x[i]` in a scheduled kernel is probably still
  searched (`Codegen.gated_by_memory` reads both). Unverified on a real
  kernel. Found by SEARCHFIX
- `beam_search`'s disk key names no device part, so a search timed on one
  part of a target serves every other part. rig states arch, bus and
  cache bytes but no model or unit count, so the key needs a rig fact and
  an argument rune's `search` passes; tinygrad keys by device and suffix
- `Heuristic.reused` (heuristic.ml:853) and `upcast_more`'s `reuses`
  (:926) count every access in the kernel, loads after the reduce
  included, so a bias that skips an axis both operands of a contraction
  read makes that axis look reused. On CUDA and AMD, `empty(4,512,16) @
  empty(4,16,768) + empty(512,768)` turns `four_first` off and upcasts the
  batch whole (4x4x3, unblocked) where `matmul_768` blocks at 64; fixing
  `reused` alone gives 4x4x4, still unblocked, so `reuses` (tinygrad's
  rule, pinned by `scores_biased`) changes too, a divergence that needs a
  timed GPU row. Found by the I1 lend fix
- A bf16 product's tensor core at five K tiles with a float32 `exp`
  epilogue runs 1.22-1.28x slower than SIMT: at 65536x72x512, padded to
  K = 80, 0.347 against SIMT's 0.284 ms, while main takes the same core
  unpadded at K = 80 in 0.354 ms. Found by the G-D3 review
  (_plans/tessera-port/design/gemm2/gd3-review.md)

## nx, rune

- nx.array's `to_array` of unboxed elements and `of_array`
  (nx_array_stubs.c:1022, :1101) copy a whole array with the domain lock
  held, cold mapped pages included, since their walks touch OCaml arrays;
  the gathers (`copy`, `to_bigarray`, `to_array` of boxed elements)
  release past 256 KiB or over borrowed memory, by nx_cpu's rule. Found by
  the SI-1 review
- nx's `test_composite › together › a composite over values whole
  together is its body` (test_composite.ml:303-327) fails about 1 run in 4
  at `-j 2` on main too: it compares minor words of 100 calls, the least
  of 5 runs, and the words vary with timing under load. Found by the DD+DC
  landing
- nx's bench rows `placed/*` and `placed/matmul-f32-256x4096x256-k-split4`
  (bench_nx.ml:337-366; nx.thumper:260-267) record exact allocated words
  that vary from run to run, so `check --quick` can fail them on an
  unchanged tree. Found by the DD+DC landing
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
  shapes. Nearly all of it is cold compiles, one program a case: host
  "every dtype" 10.2 s cold (9.7 s jit for 37 programs, clang about half),
  1.9 s warm; GPU "mapped by vmap" 17.3 s cold (13.4 s jit, 36 programs),
  3.1 s warm. A case at tq/tk 7-9 compiles in 0.1 s against 0.25-0.5 s at
  300, so smaller shapes, where a case's comment allows, cut it. Found by
  the TARGETFACTS audit and CODEPATH design A
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
  `FILE_FLAG_OVERLAPPED` (rig_disk_stubs.c:98-104), which serialises its
  I/O even where each `ReadFile` passes an `OVERLAPPED` offset, so a
  copy's runs are read one after another (rig_disk_stubs.c:513-515); a
  handle opened with the flag would give it depth. Found by the
  streaming-data part 2 audit and SI-1
- rig.disk: a copy of many runs begun while another thread's blocking job
  runs reads its runs one after another, since a blocking job does not
  wait to take the pool (`take`'s trylock, rig_pool.c:867-870): 15 times
  slower at 16K rows. A warm copy runs up to 128 runnable threads over the
  host's cores beside compute. Reading each run first with `preadv2
  RWF_NOWAIT` on the caller and handing only the misses to the pool would
  keep both (on Linux 5.9 and 5.10 a 0 is to be retried there); then decide
  whether a busy pool should make a cold copy wait rather than go serial.
  Gate: the SI-1 audit's long read beside compute (§4.4) on a quiet fudge.
  Found by the SI-1 review
- rig.disk's read depth, 128 (rig_disk_stubs.c:517-532), was measured
  pinned to two CPUs, which ended the pixels' curve still rising:
  re-measure it unpinned on a quiet fudge (rows, and the port's ragged
  shape at 32, 64, 128 and 256). Then bless `bench/disk`'s read rows,
  which allocate 4 more words a copy since `Rig_edge.Io.read` takes
  arrays of runs, and `read/cold-rows-16K` (bench_disk.ml:259), which has
  no baseline, and run the long-read gate and the staged file-to-device
  copy with parking workers. Found by SI-1

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
