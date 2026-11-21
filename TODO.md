# todo

goalpost: jit-compiled gpt-oss at least as fast as {tinygrad,JAX,llama.cpp/vLLM} on {NVIDIA,NV,AMD}.
Where it stands, with every measurement: docs/performance.md.

## tolk

Measurements and parked work behind these items:
_plans/perf-campaign/todo-archive-2026-11-18.md. A path in parentheses is
under _plans/perf-campaign/ unless it names another directory of _plans/.

- f32 SIMT products run at 15-17.5 TFLOP/s where cuBLAS runs TESSERA's
  shapes at 26-30 (no TF32, `ampere_sgemm_128x64_tn` and `128x128_tn`):
  TESSERA large's model time is 73.4 s against PyTorch's 56.6, all of it in
  ffn2, q/k/v and the projections. Stage k-contiguous tiles through shared
  memory, double-buffered, for f32 SIMT products as the bf16 cores do. The
  TESSERA campaign has it (tessera-port/gpu-lane-logs/2026-11-19/gap/,
  tessera-port/design/kcontig-b2/)
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

## nx, rune

- gpt-oss's decode copies v (`E_16_8_4`, a stride-640 slice of the fused
  q, k, v product) before rune's scatter writes the cache, since a custom
  kernel's argument that views a computed node is stored apart. Removing it
  needs a call-argument contract: tolk passing a node it stores anyway with
  the view's index, or the scatter fusing its updates' computation (one
  launch a block; branch `fix-kvwrite`)
- main differs from its pinned ocamlformat (0.29.0) in 1,837 files, hand
  wrapping since September's style commits; one `dune fmt` makes the tree
  clean (+117,607/-102,814). Before it: ignore entries or generators for the
  21 generated files it rewrites, every runtest (74 files use `__LOC__`), a
  moment with no branch in flight, then `.git-blame-ignore-revs`, a CI step
  and AGENTS.md's interim sentence dropped (_plans/todo-sweep/n8-fmt/)
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

## rig

- gpt-oss decode runs 26 CUDA chains, one a submit; the 32 cap binds only
  at prefill. 24 boundaries of 0.8 us and a 23 us host gap before the
  first layer cost 42-76 us of 6.12 ms (perf-campaign/chaincap/cause.md)

## talon, hugin, brot, munin, quill

- A right y axis sets its title on the line of a right legend's title, so
  the two read as one header, as row facet titles did before eaf19326e1
- hugin's 16-styles image differs on fudge in 13 pixels of one symbol by
  up to 3 levels: the rasterizer's arcs read libm, so bytes vary by system
- talon's stored allocation baselines for its Parquet read rows have
  drifted from main (read-nullable: 34,263 words stored, 36,671 measured
  on main); re-record them, and find which change moved them
