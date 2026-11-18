# Apps

Programs that do real work with raven, written to be read.

- An app uses raven's public interfaces only. Code another program would
  reuse goes into a library.
- Weights are downloaded, never committed.
- `dune build @apps/runtest` runs every app's fast tests. Checks against a
  reference implementation need downloaded weights and run outside it; each
  app's README says how.

| App | What it does |
|---|---|
| [gpt-oss](gpt-oss/) | OpenAI's gpt-oss, a mixture-of-experts model: chat in the harmony format, on the CPU or compiled for GPUs |
