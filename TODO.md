# todo

goalpost: jit-compiled gpt-oss at least as fast as {tinygrad,JAX,llama.cpp/vLLM} on {NVIDIA,NV,AMD}.
Where it stands, with every measurement: doc/performance.md.

## tolk

Measurements and parked work behind these items:
_plans/perf-campaign/todo-archive-2026-11-18.md. A path in parentheses is
under _plans/perf-campaign/ unless it names another directory of _plans/.

- A bench's quiet check reads only the host's load, so a GPU group runs,
  and can be blessed, while another user's context holds the GPU: two
  foreign contexts made every CUDA row read 4.2-4.8 ms a call. A check of
  the GPU's compute apps before the GPU groups would refuse or flag it
- The rows layout runs rows(W)/4 workgroups: the router at 64 tokens or
  fewer and the 12-token k and v take 7-24 us where a grouped reduce or BEAM
  take 5-11; bare cores at 16 rows or fewer trail BEAM's (routercore/)
- D212's blocked contractions lose to tinygrad's heuristic on long reduces
  (`contract/f32-2048` 1.03 of main against 0.92; attention values 256 wide
  1.04 against 0.86); `Heuristic.group` never groups a contraction
- `Heuristic.four_first` still gates on the fitted `core_threads`. Gated
  on `units`, matmul_768 takes 4 by 3 at 1.5x the time; ungated, targets
  that state no `units` (sm_80, AMD, Metal) block small products, 1.38x by
  an old measurement. Stating `units` on every target would let it go
  (branch `fix-unitsgate` 3a8321ea33)
- D357's inner-end group split loses to tinygrad's top split where few warps
  run: the bfloat16 argmax and max over 201088 values (4096 threads) take
  8.8 us against 5.1 (ncu), 0.70-0.74x by the probe, about 3 us a decode
  step. Both read the same L2 sectors; under the top split a lane's first
  load brings the sector of its next 15 values, so 94% of L1 sectors hit
  against 31% and warps wait 2.3x less on memory. Elsewhere the top split
  runs 0.89x on sum_strided_prime (256 threads) and 1.04-1.10x on sums of
  65536-131072 threads, which it makes read 8x the L1 sectors. A gate on
  threads would also flip sum, cumsum and vecmat_1000 (32-32000 threads),
  which run 1.06-1.24x under the top split, by a timer tick, so no rule yet.
  sum_cols_prime's 0.37x is `across`, not the split end (16.4 us at
  either): its 1571 by 64 sum runs 4 workgroups of 256 threads
  (`r_4_16_16_99`), tinygrad's 64 of 32 (`r_64_32_50`). Raw:
  ~/wt/tmp/fix-argmax
- gpt-oss's decode submits each of its 24 blocks apart, so rig replays one
  chain a block and the GPU idles about 2 us between chains: +0.05 ms a step
  against llama.cpp's 3 graphs a step. kaun's per-block jit decides the
  submits (decodegap2/cause.md)
- cstyle names each parameter after its buffer's size (`data0_4096`), so
  kernels over buffers of two sizes compile apart under D485: name the
  parameters without sizes in the renderer
- A bf16 product of 6-32 rows by weights laid `[K; N]` streams by its
  columns on 4 workgroups: `contract/bf16-16x64x2880` takes 47-50 us where
  BEAM's split of the reduce over 8 threads and upcast by 4 take 8-9
- A lone per-head projection at head size 80 (kaun's `Attention.apply`
  projects q and k apart) pads N from 80 to 128, 1.6x the core's work,
  where a warp tile of 40 columns (N upcast by 5) covers it; D491's 440 to
  448 was never measured against 5 either
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
  count the rows routed to its own groups. The product holds no such fact
  at compile time; nx's `Partial.contract` knows the groups, so a field of
  `Nx_kernel.segments.Rows` would carry it: a kernel-contract change
- MXFP4's float32 decode on Metal multiplies by a subnormal at scale byte
  0, which Metal may flush (MSL 4.1 §8.1); no test reaches it there
- Whether tolk stores a computed value or recomputes it in its readers is
  decided by D77's work bar, with no device in the rule: storing a cheap
  value starts to pay at about 1100 extra operations an element on the
  host and past 2286 on CUDA (r3/audit.md). A price needs a launch cost and
  a byte rate as renderer fields and the target in the schedule's key
- Routing ids lose their value bounds through `Lower_index.sort`'s `chosen`
  (a reduce) and `Partial.local_ids` (a uint64 bitcast add), so no later
  rule can read their range
- `Nx.rms_norm` runs a third pass, a lane max, under a trace: 49 kernels a
  gpt-oss step. The one-pass form is blocked by the buffer limit, which
  stores the routed rows' 23 MB gather once the norm's map inlines (r3/)
- AMD thread traces are not collected: through KFD they are parked (after a
  fault no queue of the process runs, so a trace can't be stopped); PCI and
  an amdgpu DRM submission remain
- Metal's renderer emits no launch bound; its `local_threads_max` of 1024
  is a maximum a pipeline may refuse, and Mac2's AMD GPUs may run 64-wide
  SIMD groups, which `simd_threads` does not state
- D404's Metal launch member cap of 40 rests on an M1 Max, and no machine
  a bench may run on holds a Metal device, so the ledger's (d) cannot be met
  for it: admit a Mac row for Metal or mark that part "Reason: none"
- `Heuristic.reused` and `upcast_more`'s `reuses` count loads after the
  reduce, so a bias read along an axis makes it look reused (`empty(4,512,16)
  @ empty(4,16,768) + empty(512,768)` upcasts the batch whole, unblocked)

## nx, rune

- gpt-oss's decode copies v (`E_16_8_4`, a stride-640 slice of the fused
  q, k, v product) before rune's scatter writes the cache, since a custom
  kernel's argument that views a computed node is stored apart. Removing it
  needs a call-argument contract: tolk passing a node it stores anyway with
  the view's index, or the scatter fusing its updates' computation (one
  launch a block; branch `fix-kvwrite`)
- nx measures most errors in floats from the correctly rounded value, but
  some special functions (nx.mli's "plus half of one" least subnormal and
  an absolute `kε`) measure from the exact value: a second metric. One
  metric, stated once
- The exhaustive sweep of float32 `log` and `asinh` against nx_kinds.h's
  bits on nx_cuda and nx_amd is owed before any accuracy promise names a
  GPU
- The depthwise branch of `Nx.conv` (one input and one output channel a
  group) is slow under rune's gradient: about 8 ms for the weights and
  21 ms for the input at 8x56x56x64, the loop-sized map and fold
- Under `Rune.jit`, the weight gradient of `Nx.matmul_segments` keeps one
  `[k; n]` partial per block (lower_reduce.ml): +0.9 GB at 1,024 rows
  over AION's 40 heads, where eager keeps none
- `Nx.attention ~lengths` skips padded work in the forward only; its
  default derivative builds the whole pair loop
- rune cannot check data inside compiled code (a token range, a length
  past its row), so a model that validates its inputs, such as AION's
  codecs, checks on the host and its encode cannot run under `Rune.jit`
- No tree can check formatting: the only ocamlformat on fudge (0.29.0)
  rejects main's own files and no dev-tools lock pins the repository's
  formatter
- rune props that take 4-14 s alone pass the 60 s limit at a load of 13-30:
  `test_grad`'s dirichlet tangents and gradient (14 s), `test_ranges`' "a
  compiled call has eager's bits, every dtype" (8.6 s) and `test_total`'s
  "a map's lanes add in turn" (4.4 s). Smaller draws that keep their
  covers would hold them
- nx.cpu computes a Reduce of k reductions in k passes: `reduce_into` runs
  once per reduction (reduce.c:2468-2484), and each pass evaluates the
  fused program (:2457-2466) over every term again. var and std written as
  Sum and Max reductions of one Reduce, as R6 writes them, so run 5-7 times
  slower eagerly than main's (fudge under load: var-f32-1M 23-30 against
  159-170 us, var-f32-8192x4096-rows 2.0 against 11.1 ms). It blocks R6
  (branch `reductions-r6`: var, std, standardize and rms_norm on one
  lane-scale rescue). Found by R6
  (_plans/tessera-port/design/reductions/r6-logs/bench-abba.txt)
- nx.array's `to_array` of unboxed elements and `of_array`
  (nx_array_stubs.c:1022, :1101) copy a whole array with the domain lock
  held, cold mapped pages included, since their walks touch OCaml arrays;
  the gathers (`copy`, `to_bigarray`, `to_array` of boxed elements)
  release past 256 KiB or over borrowed memory, by nx_cpu's rule. Found by
  the SI-1 review
- nx's bench rows `placed/*` and `placed/matmul-f32-256x4096x256-k-split4`
  (bench_nx.ml:337-367; nx.thumper:261-268) record exact allocated words
  that vary from run to run, so `check --quick` can fail them on an
  unchanged tree: rig's `cache_room` (memory.ml:946-958) grows its per-size
  array when a drain returns more buffers than it held, as GC timing
  decides. Grow it at a fixed point or size it up front. Found by the DD+DC
  landing
- Allocated words in rune and tolk programs move by several tenths of a
  percent with GC timing: Ops hash-conses in Weak tables, and Shape and
  Codegen memoise in ephemerons, so an exact-allocation row can fail an
  unchanged tree. Record such rows at a fixed, larger minor heap
  (`OCAMLRUNPARAM=s=1M`), or read only differences above about 1% as
  real. Found by the DQ review
- rune's attention core takes 8 warps where 8 still leave
  `local_workgroups` workgroups, else 4 (lower_attention.ml:1516), though
  renderer.mli scopes that field to cores whose operands are not staged,
  and this core shares its tiles among its warps in workgroup memory. On
  fudge's RTX 5000 Ada, 4 warps run masked T256 rows 21-48% faster and
  unmasked T2048 rows 6-25% slower than 8, so a rule on workgroups alone
  misses the key tiles a mask skips. TESSERA's packed rows are masked.
  Found by the TARGETFACTS audit (_plans/consolidate/targetfacts, A3);
  measured by attention2 M3, in its adjudication's amendment
  (_plans/tessera-port/design/attention2/adjudication.md)
- rune's GPU attention prop "its gradient mapped by vmap"
  (test_jit_gpu.ml:3381) took 17.3 s cold on an idle fudge (36 programs)
  and passed the 60 s `WINDTRAP_TIMEOUT` (dune:53) at a load of 30-60. I-X
  cut its examples from 32 to 28, and the host's "every dtype" from 31 to
  21 (10.1 to 7.7 s cold at load 5; not measured at load 30). The vmap
  prop's time on CUDA is unmeasured until the I-X GPU gate
  (_plans/tessera-port/todo-triage/batches.md 1l): delete this entry if it
  shows headroom, else give the prop its own `~timeout`. Found by the
  TARGETFACTS audit and CODEPATH design A
- rune states a compiled complex contraction's part within
  `γ(β + 2) Σ(|ar br| + |ai bi|)`, β = β(K) (rune.mli:1706-1707,
  lower_reduce.mli:143-145), while `Nx.contract` and the kernel contract
  give `√2 γ(β(2K))` in modulus (nx.mli:2685-2688, nx_kernel.mli:303). At
  K = 1 alone, rune's bound is larger: `γ(5)` a part against `√2 γ(4)` in
  modulus. From K = 2 on, `β(K) + 2 ≤ β(2K)`. It matters if compiled
  programs must hold `Nx.contract`'s bound. Found by NXBOUND
- `Nx.rearrange` plans for 190-252 words beyond its movements (a `b h`
  split costs 362 against 110 moved), for patterns that move nothing too.
  Found by the nx-hostcost audit
- A 0-d operand against a leaf of rank 1 or more costs 96-100 words a call,
  44 of them the broadcast, and vega pays this for its 0-d rate and weight
  decay. Found by the nx-hostcost skeptic
- `Nx.log10` states 2 ulps at float64 (nx.mli:1750-1753) and derives it
  nowhere: it composes over `log2`, itself stated at 2 ulps there
  (:1746-1748), and since rune compiles nx's own sequence (b1c79a2bcc;
  rune.mli:1619-1651) a compiled log10 has eager's bits, so the 1-ulp
  route over tolk's `xlog2` under host jit is gone. The float32 bound and
  the exact powers of ten are measured at every float32. A `log2` given as
  two parts would derive the float64 bound, once a consumer needs log10
  bounded there. Found by the DD audit (S2)
- `Nx_io.Pytorch` reads torch.save files but writes none (pytorch.mli has
  `load` and `tensors` alone), so tessera's cram tests cannot make the .pt
  model file that `encode`, `targets` and `distil --init` read
  (tessera-bin/adjudication.md M9). Found by DM's skeptic

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
- rig.disk's read depth, 128 (rig_disk_stubs.c:517-536), was measured
  pinned to two CPUs, which ended the pixels' curve still rising:
  re-measure it unpinned on a quiet fudge (rows, and the port's ragged
  shape at 32, 64, 128 and 256), then run the long-read gate (the SI-1
  audit's §4.4) and the staged file-to-device copy with parking workers.
  `bench/disk`'s read rows, at 14 words a copy, and `read/cold-rows-16K`
  are recorded (d652956bb8, on cores 0-5). Found by SI-1

## talon, hugin, brot, munin, quill

- talon's stored allocation baselines for its Parquet read rows have
  drifted from main (read-nullable: 34,263 words stored, 36,671 measured
  on main); re-record them, and find which change moved them
- talon needs a query optimizer, a large design of its own, and the other
  optimization to match the best reference engines; 1.0 has no speed gate.
  Last measured against the faster of DuckDB 1.5.6 and Polars 1.44.2
  (9b82c5b57a, under load): decision support SF1 6.6x geomean, H2O 1e7
  group-by 2.1x, joins 2.0x, with talon using 3-10x the CPU. What the gap
  is made of: join filters pushed at run time and join reordering (DuckDB
  reads far fewer rows on q07, q08 and q12), scans (half the instructions
  of the five worst queries), aggregation (46% of q01), and possibly a
  segment-id reduce kernel in nx (unmeasured). Data: ~/wt/tmp/talon-data

## kaun, ymir

- `Kaun_datasets.cifar10` has no test: its loader takes only full
  10,000-entry batches (about 180 MB). A `(**/**)` seam over the
  records-to-tensor step would pin its channels-last order
- ymir_units' odoc has an unresolved reference to `Nx.Dtype.name` in
  `Unit.ratio`'s doc; odoc warnings are fatal in `@doc`
- ymir's `Grid` holds two celestial axes only, a limit no standard
  imposes; spectral axes and `-TAB` lookups need grids that are a
  celestial pair times 1-D axes (AION campaign question B4.1)
