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
- D212's blocked contractions lose to tinygrad's heuristic on long reduces
  (`contract/f32-2048` 1.03 of main against 0.92; attention values 256 wide
  1.04 against 0.86); `Heuristic.group` never groups a contraction
- `Heuristic.four_first` still gates on the fitted `core_threads`. Gated
  on `units`, matmul_768 takes 4 by 3 at 1.5x the time; ungated, targets
  that state no `units` (sm_80, AMD, Metal) block small products, 1.38x by
  an old measurement. Stating `units` on every target would let it go
  (branch `fix-unitsgate` 3a8321ea33)
- D357's inner-end group split loses to tinygrad's top split on the bfloat16
  argmax and max over 201088 values (0.67-0.77x, about 3 us a decode step)
  and on sum_cols_prime (0.37x)
- A decoded operand of 12-16 rows streams in one pass where a core probably
  wins (559-706 us against 424-492 at 17-64 rows); it reaches gpt-oss's
  Q8_0 prefill at 12 tokens. `core_fits` ignores `early`'s guards, which a
  partial form's one-block sum never meets (rowdecode/design-a.md)
- `Indexing.core_product` asks `core_fits` with a decoded factor as stored,
  so a decoded product of 17-32 rows stays grouped there; `heuristic.mli`
  (~30) says no core takes a decoded operand, but only the streamed one is
  tested
- `apply_opt`'s refusal of a tensor core past `local_threads_max`
  (`postrange.ml` ~987, D488) has no test
- cstyle names each parameter after its buffer's size (`data0_4096`), so
  kernels over buffers of two sizes compile apart under D485: name the
  parameters without sizes in the renderer
- A bf16 product of 6-32 rows by weights laid `[K; N]` streams by its
  columns; since its reduce splits into threads it is untimed (it took 476
  us at 16x64x2880, `contract/bf16-16x64x2880`), and it runs 4 workgroups:
  near its byte bound only with the reduce split across workgroups
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
  count the rows routed to its own groups
- MXFP4's float32 decode on Metal multiplies by a subnormal at scale byte
  0, which Metal may flush (MSL 4.1 §8.1); no test reaches it there
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
- D404's launch member caps, NVRTC's 64 and Metal's 40, name no fudge
  measurement, which the ledger's header asks of a speed measured elsewhere
- `beam_search`'s disk key names no device part, so a search timed on one
  part serves every other part of its target; tinygrad keys by device
- `Heuristic.reused` and `upcast_more`'s `reuses` count loads after the
  reduce, so a bias read along an axis makes it look reused (`empty(4,512,16)
  @ empty(4,16,768) + empty(512,768)` upcasts the batch whole, unblocked)

## nx, rune

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
- An eager creation at a set without kernels holds its window whole on the
  host: `Exec.created` (exec.ml:584-600) computes each result on the host,
  then places a copy, so a one-device value needs host memory of its full
  size where the device could hold it. On fudge, `capped`'s 8 GB is below
  an RTX 5000 Ada's 32 GB. Decided (nx-vocab/adjudication.md W1): slabs,
  rig's own practice one layer up, since rig moves a plain host buffer
  through staging allocations of 32 MiB whatever its size (rig.mli:
  1520-1566). `created` allocates the device array, computes each slab at
  its `Prim.offset` into one host buffer and copies it into its box
  (rig.mli:1599-1609), with slab edges on whole bytes for dtypes under a
  byte (nx_array.mli:212-221). The bits are the same by the window law.
  Where the host addresses the device's memory (`share`, exec.ml:437)
  `created` could instead compute straight into the device array. Owed by
  the DD adjudication (W1); found by its audit and skeptic
  (_plans/tessera-port/design/nx-vocab/skeptic.md §3)
- `Nx.log10` states 2 ulps at float64 (nx.mli:1750-1753) and derives it
  nowhere: it composes over `log2`, itself stated at 2 ulps there
  (:1746-1748), and since rune compiles nx's own sequence (b1c79a2bcc;
  rune.mli:1619-1651) a compiled log10 has eager's bits, so the 1-ulp
  route over tolk's `xlog2` under host jit is gone. The float32 bound and
  the exact powers of ten are measured at every float32. A `log2` given as
  two parts would derive the float64 bound, once a consumer needs log10
  bounded there. Found by the DD audit (S2)
- `Nx.zeros_at` and `Nx.full_at` of a value whose bytes repeat one byte
  (zeros, all-ones integers) are maps of no loads, so on a set without
  kernels `Exec.created` fills them on the host and copies the window
  (exec.ml:584-600), though rig can set the bytes on the device
  (`Rig.Buffer.submit_fill`, rig.mli:1700) with no host memory and no
  copy. Eager optimiser moments made at a GPU set outside a compiler pay a
  host fill of their size and a copy of it to the device. Unmeasured.
  Found by the DD design round (N3)
- rune.mli:1403-1404 says that a value a compiled function captures on
  another device and places where the program computes "is copied there
  once per program". In a split program (values whole on each of several
  devices) the capture is instead bound as it is (compile.ml:362, `Some r
  when st.split || …`), and its placement copies it at every call.
  Measured on two software devices: 6 float32 captures cost 48 bytes a
  call, 4,848 over 101 calls
  (_plans/tessera-port/design/jit-values/skeptic-probe/split.log). vega's
  schedules and rates capture their numbers as 0-d arrays (b8d7630a36), so
  a data-parallel step pays this for each. Either upload once per device
  in split programs through `st.uploads`, or scope the sentence to
  one-device programs. Found by DC's skeptic
- `Nx_io.Pytorch` reads torch.save files but writes none (pytorch.mli has
  `load` and `tensors` alone), so tessera's cram tests cannot make the .pt
  model file that `encode`, `targets` and `distil --init` read
  (tessera-bin/adjudication.md M9). Found by DM's skeptic
- `Nx.check`'s doc says a failure raises "on another set at a wait on that
  set" (nx.mli:4865-4866) and leaves out that a check no wait follows
  never raises, which the Waiting section states. Callers that link
  `Nx.check`, as jord's transforms and area rules do, reach its doc first.
  Add the clause there. Found by jord batch 11's review

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
- rig.remote's proxy answers no `Rig.capacity` (rig_remote_proxy.ml:118,
  `None` in its mli's table): the wire carries no report of the agent's
  device's memory, so a program on a remote GPU cannot check a footprint
  against it. Forward it as one request and answer, which bumps the
  protocol (DF design A). Found by DF

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
