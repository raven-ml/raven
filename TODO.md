# todo

goalpost: jit-compiled gpt-oss at least as fast as {tinygrad,JAX,llama.cpp/vLLM} on {NVIDIA,NV,AMD}.
Where it stands, with every measurement: docs/performance.md.

## tolk

Measurements and parked work behind these items:
_plans/perf-campaign/todo-archive-2026-11-18.md. A path in parentheses is
under _plans/perf-campaign/ unless it names another directory of _plans/.

- Values rangeify stores for readers that index them apart never share a
  kernel (D137 groups only the graph's stores): a dirichlet draw's jvp runs
  30 kernels against its forward's 8, 28.5 s cold in test_grad (drawfuse)
- Since D485 every CUDA kernel is `tolk_kernel` in nsys and ncu, so a
  profile names no kernel; attribution falls back to launch signatures
- f32 SIMT products run at 15-17.5 TFLOP/s where cuBLAS runs TESSERA's
  shapes at 26-30 (no TF32, `ampere_sgemm_128x64_tn` and `128x128_tn`):
  TESSERA large's model time is 73.4 s against PyTorch's 56.6, all of it in
  ffn2, q/k/v and the projections. Stage k-contiguous tiles through shared
  memory, double-buffered, for f32 SIMT products as the bf16 cores do. The
  TESSERA campaign has it (tessera-port/gpu-lane-logs/2026-11-19/gap/,
  tessera-port/design/kcontig-b2/)
- thumper decides quiet from the host's load alone (`Env.load`), so GPU
  rows run and bless under another user's compute context. Needs a public
  quiet hook in thumper and a source of the device's compute processes in
  rig (NVML by dlopen or nvidia-smi): a decision, not a fix
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
- rig builds a program's whole submit before its first launch: 120 us of
  host time ahead of a one-program decode step. Submitting in pieces would
  start the GPU sooner (perf-campaign/chains/parked.md)
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
- AMD thread traces are not collected: through KFD they are parked (after a
  fault no queue of the process runs, so a trace can't be stopped); PCI and
  an amdgpu DRM submission remain
- Metal's renderer emits no launch bound; its `local_threads_max` of 1024
  is a maximum a pipeline may refuse, and Mac2's AMD GPUs may run 64-wide
  SIMD groups, which `simd_threads` does not state
- AION-1's compiled encode does not compile on CUDA: a kernel of the
  whole program (2 objects, an image and four fluxes, 581 tokens, width
  768) needs 65536 bytes of shared memory where 49152 are allowed without
  the opt-in attribute, and `Codegen.rendered_program` raises
  `Invalid_argument "exceeds maximum shared memory size"`. `Nx.attention`
  at [2; 12; 581; 64] with lengths, the [2; 581; 768] by [768; 2048]
  product and a layer norm over 768 each compile alone, so it is a fused
  kernel. Repro: `apps/aion/test/gpu_parity.exe cuda HUB REF64` on fudge's
  RTX 5000 Ada (_plans/aion-port/logs/p5c-gpu2.log); the narrow fixtures
  of `test_gpu` run
- Compiled AION-1 on the host runs 4-6x slower than nx.cpu's eager kernels
  on two of fudge's CPUs: an encode of 2 objects 27.7 s against 7.1 s, a
  273-token decode 20.5 s against 3.3 s (medians of three, load 9-24,
  _plans/aion-port/logs/p5c-cpu.log), unsearched; the narrow fixtures'
  compiled calls are faster than eager. A `BEAM` run and one kernel's
  profile would tell whether search or the kernels' shape is short

## nx, rune

- Under `DEBUG=4`, rune's `memory › a split program holds at its return
  at most its result and its plan` fails: find what the debug
  output keeps alive.
- gpt-oss's decode copies v (`E_16_8_4`, a stride-640 slice of the fused
  q, k, v product) before rune's scatter writes the cache, since a custom
  kernel's argument that views a computed node is stored apart. Removing it
  needs a call-argument contract: tolk passing a node it stores anyway with
  the view's index, or the scatter fusing its updates' computation (one
  launch a block; branch `fix-kvwrite`)
- The exhaustive sweep of float32 `log` and `asinh` against nx_kinds.h's
  bits on nx_cuda and nx_amd is owed before any accuracy promise names a
  GPU
- The depthwise branch of `Nx.conv` (one input and one output channel a
  group) is slow under rune's gradient: about 8 ms for the weights and
  21 ms for the input at 8x56x56x64, the loop-sized map and fold
  (depthwise/parked.md)
- nx.cpu's Reduce still runs a program once per reduction for Args across
  rows, Moments and Logsumexp across rows or in groups, and a call of two
  Logsumexps alone
- Under `Rune.jit`, `Nx.matmul_segments`' weight gradient stores one
  `[k; n]` partial per block (+0.9 GB at 1,024 rows over 40 heads): it
  needs a tolk reduce with a per-workgroup trip count read from memory
  (segpartials/parked.md)
- `Nx.attention ~lengths`'s compiled derivative builds the whole pair loop
  on the host, in forward mode, with pages or float captures, and past 160
  dimensions; eager `grad` under vmap with per-lane lengths does too
- No tree can check formatting: the only ocamlformat on fudge (0.29.0)
  rejects main's own files and no dev-tools lock pins the repository's
  formatter
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
- nx has no `nan_to_num`: aion's `Spectrum.nan_to_num` composes `isnan`,
  `clamp` and `where` to replace NaN and the infinities as torch's does
- `Nx_io.Hdf5.window` reads one box, every chunk it covers whole, so
  aion's `Survey.read` reads a source's rows run by run: a scattered row of
  a Legacy Survey file, chunked (27, 1, 20, 40), decodes 128 chunks of
  86,400 bytes, 11.1 MB for a 0.41 MB image. A row-subset read needs its
  own design once a reader measures the cost
- `Nx_io.Hdf5.values` decodes a whole chunked dataset into memory: 22.5 GB
  for one Legacy Survey file's `image_array`. Only `window` fits such files
- `Nx_io.Hdf5` reads no strings or booleans: fixed-length strings read as
  their bytes at `uint8` (aion's `Survey` trims and decodes band names), and
  h5py's booleans as `int8`
- nx states no batch-extent invariance: whether on one device an element's
  result depends only on the extents of the axes it reduces over, not the
  others' (nx.mli:2271-2274 ties nx.cpu's order of a reduction to "the
  shape alone", 2706-2709 a contraction's to "the shapes alone"; tolk may
  split by shape). If it held, a law states it and aion's draws per object
  stop depending on `--batch`. Found by aion batch 7's skeptic

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

- rig's host cache (rig_heap.c) misses 46-50 buffers a call of attention's
  eager lengths gradient (peak-minus-live bound at its floor, exact sizes),
  and glibc unmaps them: 1.5 us a fault, up to 55 ms a call (hostfaults/)
- gpt-oss decode runs 26 CUDA chains, one a submit; the 32 cap binds only
  at prefill. 24 boundaries of 0.8 us and a 23 us host gap before the
  first layer cost 42-76 us of 6.12 ms (perf-campaign/chaincap/cause.md)
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

- A right y axis sets its title on the line of a right legend's title, so
  the two read as one header, as row facet titles did before eaf19326e1
- hugin's 16-styles image differs on fudge in 13 pixels of one symbol by
  up to 3 levels: the rasterizer's arcs read libm, so bytes vary by system
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
- raven has no library that fetches byte ranges over HTTPS: kaun.hf,
  kaun.datasets and tessera's `Fetch` each wrap curl. `Nx_io.Partial`
  decodes from fetched ranges, so with one aion could read a Legacy Survey
  object's 0.41 MB rows where it now needs the whole HEALPix file, 62.7 GB
  on average
