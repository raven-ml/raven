# todo

goalpost: jit-compiled gpt-oss at least as fast as {tinygrad,JAX,llama.cpp/vLLM} on {NVIDIA,NV,AMD}.
Where it stands, with every measurement: doc/performance.md.

## tolk

Measurements and parked work behind these items:
_plans/perf-campaign/todo-archive-2026-11-18.md. A path in parentheses is
under _plans/perf-campaign/ unless it names another directory of _plans/.

- The fused Q8_0 query, key and value product at 512 tokens is unstaged, and
  its N takes an upcast of 5, the first of `core_upcasts` dividing 640
  tiles, where 512 tiles took 4: measure 4 against 5
- The rows layout runs rows(W)/4 workgroups: the router at 64 tokens or
  fewer and the 12-token k and v take 7-24 us where a grouped reduce or BEAM
  take 5-11; bare cores at 16 rows or fewer trail BEAM's (routercore/)
- The Stage suite's cover "gathered rows left unstaged" (`test_stage.ml`
  ~931) is vacuous at seed `s1:9cc4952459e7d173`: a flaky cover
- D212's blocked contractions lose to tinygrad's heuristic on long reduces
  (`contract/f32-2048` 1.03 of main against 0.92; attention values 256 wide
  1.04 against 0.86); `Heuristic.group` never groups a contraction
- `Heuristic.four_first` gates on the fitted `core_threads` (16384) and D212
  on `units x 256`: products between them, as matmul_768, run 4 by 4
  unblocked. Gate both on `units`, which removes a fitted field
- D357's inner-end group split loses to tinygrad's top split on the bfloat16
  argmax and max over 201088 values (0.67-0.77x, about 3 us a decode step)
  and on sum_cols_prime (0.37x)
- `Heuristic.core_fits` and `core_rows`, which rune reads, count every pad
  of a decoded operand, which D230 does not: rune keeps Q8_0-like products
  of 17-32 rows flat. Both should take `~decoded` (rowdecode/)
- A decoded operand of 12-16 rows streams in one pass where a core probably
  wins (559-706 us against 424-492 at 17-64 rows); it reaches gpt-oss's
  Q8_0 prefill at 12 tokens (rowdecode/design-a.md)
- cstyle names each parameter after its buffer's size (`data0_4096`), so
  kernels over buffers of two sizes compile apart under D485: name the
  parameters without sizes in the renderer
- `Codegen.to_programs` compiles each worker's sources apart, so a text two
  workers take compiles twice (5 of 165 in tessera's cold step): gather
  the texts before the workers split them (coldcall/cause.md)
- A dense bf16 product whose weights are laid `[K; N]` takes no tensor core
  and runs about 100x its byte bound (proj/)
- A staged core pads N to its tile (D491) but not M: q at 100 tokens (M
  padded to 112) takes 115 us against 56 at 128. A pad of M must keep
  routed rows within a run (`in_run`)
- At rows 8 mod 16 neither main's swapped core nor axis 0 padded wins
  everywhere (q at 72 tokens 64 against 109 us, at 88 67 against 54); a rule
  between them is unmeasured (tessera-port/design/gemm2/gd3-notes.md)
- A bf16 core at five K tiles with a float32 `exp` epilogue runs 1.22-1.29x
  SIMT ([65536; 72] by [72; 512]: 0.348 against 0.283 ms); without the
  `exp` it runs 0.98 (tessera-port/design/gemm2/gd3-review.md)
- The 440 by 440 weight gradient over 68608 rows, 16 partials, takes the
  staged tile on 448 padded columns in 2.27 ms where staging off takes 1.78
- Since D479, GPU `Set` elections of 16384 and 24576 rows run 1.4x and 1.9x
  slower than rune's former split, and near `stream_threads` outputs up to
  3.5x: split where the renderer is known (consolidate/groupgate/)
- rune's paged attention with pages of 16 keys runs at 168 registers on
  CUDA, one workgroup an SM, and refills its spilled values every tile,
  skipped ones included
- A core sums its whole reduce in each thread, so a grid under or just past
  one wave reads slower than a stream: `Cuda/experts-mxfp4-4x2of8-14336x4096`
  +38%. A split of the reduce from the SM count would remove it (splitcount/)
- rune's routed products with no core: one row a block loses 12-33% to
  blocks of 4 at 3-7 tokens; Q8_0's blocks of 64 from 16 rows take 1.5-2.9x
  as long as 8 or 16
- Under expert-parallel placement rune's `block_rows` counts all `n` rows
  over the local `g` experts, so it sees `D` times the mean rows an expert:
  count the rows routed to its own groups
- MXFP4's float32 decode on Metal multiplies by a subnormal at scale byte
  0, which Metal may flush (MSL 4.1 §8.1); no test reaches it there
- At heads of 80 (32 heads, 2560 wide) q and k take no staged core: 85 and
  38.9 us at 64 tokens against 51 and 11.3 at heads of 64
- A gather that reads more elements than its table holds still inlines the
  table's computation into every read: 2.2-2.4 ms against 0.7-1.1 stored on
  the host (r3/)
- tolk recomputes a cheap broadcast value along every row that reads it
  (11.3 ms against 0.64 stored on the host); bounding `recomputes`/`copies`
  would also let fft-4096 drop 20 kernels
- `Nx.rms_norm` runs a third pass, a lane max, under a trace: 49 kernels a
  gpt-oss step. The one-pass form is blocked by the buffer limit, which
  stores the routed rows' 23 MB gather once the norm's map inlines (r3/)
- AMD thread traces are not collected: through KFD they are parked (after a
  fault no queue of the process runs, so a trace can't be stopped); PCI and
  an amdgpu DRM submission remain
- Metal's renderer emits no launch bound; its `local_threads_max` of 1024
  is a maximum a pipeline may refuse, and Mac2's AMD GPUs may run 64-wide
  SIMD groups, which `simd_threads` does not state
- `apply_opt` checks a workgroup's shared memory but not its threads
  against `local_threads_max` (`postrange.ml`), so BEAM can emit a launch
  the target refuses (1024 threads on gfx9's wave64, 512 on Apple2/3)
- D404's launch member caps, NVRTC's 64 and Metal's 40, name no fudge
  measurement, which the ledger's header asks of a speed measured elsewhere
- `Ops.bound_loads`, and so `Postrange.bound_by_memory` (D255), sees only
  `Op.Load`, while a scheduled kernel reads memory through bare indices:
  such a loop is probably still searched. Unverified on a real kernel
- `beam_search`'s disk key names no device part, so a search timed on one
  part serves every other part of its target; tinygrad keys by device
- `Heuristic.reused` and `upcast_more`'s `reuses` count loads after the
  reduce, so a bias read along an axis makes it look reused (`empty(4,512,16)
  @ empty(4,16,768) + empty(512,768)` upcasts the batch whole, unblocked)

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
