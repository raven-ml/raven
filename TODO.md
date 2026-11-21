# todo

goalpost: jit-compiled gpt-oss at least as fast as {tinygrad,JAX,llama.cpp/vLLM} on {NVIDIA,NV,AMD}.
Where it stands, with every measurement: docs/performance.md.

## nx, rune

- main differs from its pinned ocamlformat (0.29.0) in 1,837 files, hand
  wrapping since September's style commits; one `dune fmt` makes the tree
  clean (+117,607/-102,814). Before it: ignore entries or generators for the
  21 generated files it rewrites, every runtest (74 files use `__LOC__`), a
  moment with no branch in flight, then `.git-blame-ignore-revs`, a CI step
  and AGENTS.md's interim sentence dropped (_plans/todo-sweep/n8-fmt/)
