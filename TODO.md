# todo

goalpost: jit-compiled gpt-oss at least as fast as {tinygrad,JAX,llama.cpp/vLLM} on {NVIDIA,NV,AMD}.
Where it stands, with every measurement: docs/performance.md.

## tolk

Measurements and parked work behind these items:
_plans/perf-campaign/todo-archive-2026-11-18.md. A path in parentheses is
under _plans/perf-campaign/ unless it names another directory of _plans/.

- D212's blocked contractions lose to tinygrad's heuristic on long reduces
  (`contract/f32-2048` 1.03 of main against 0.92; attention values 256 wide
  1.04 against 0.86); `Heuristic.group` never groups a contraction
- Since D479, GPU `Set` elections of 16384 and 24576 rows run 1.4x and 1.9x
  slower than rune's former split, and near `stream_threads` outputs up to
  3.5x: split where the renderer is known (consolidate/groupgate/)

## nx, rune

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

## talon, hugin, brot, munin, quill

- A right y axis sets its title on the line of a right legend's title, so
  the two read as one header, as row facet titles did before eaf19326e1
- hugin's 16-styles image differs on fudge in 13 pixels of one symbol by
  up to 3 levels: the rasterizer's arcs read libm, so bytes vary by system
- talon's stored allocation baselines for its Parquet read rows have
  drifted from main (read-nullable: 34,263 words stored, 36,671 measured
  on main); re-record them, and find which change moved them
