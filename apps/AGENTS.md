# apps/AGENTS.md

- An app's library lives in `lib/`, its command in `bin/main.ml`, and
  `test/` and `bench/` beside them.
- An app uses raven's public interfaces only. Code another program would
  reuse goes into a library.
- Checkpoints are downloaded, never committed. Tests read small copies that
  the generators in `test/gen` write.
- `dune build @apps/runtest` runs offline.
