# apps/AGENTS.md

- An app uses raven's public interfaces only. Code another program would
  reuse goes into a library.
- Checkpoints are downloaded, never committed. Tests read small copies that
  the generators in `test/gen` write.
- `dune build @apps/runtest` runs offline.
