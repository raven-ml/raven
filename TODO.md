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
