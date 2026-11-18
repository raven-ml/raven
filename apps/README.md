# Apps

Programs that do real work with raven, written to be read.

- An app uses raven's public interfaces only. Code another program would
  reuse goes into a library.
- Checkpoints are downloaded, never committed. Tests read small copies
  their generators in `test/gen` write.
- `dune build @apps/runtest` runs every app's tests, offline. A check that
  needs a real model's weights runs outside it, and its app's README says
  how.

| App | What it does |
|---|---|
| [gpt-oss](gpt-oss/) | OpenAI's gpt-oss, a mixture-of-experts model: chat in the harmony format, on the CPU or compiled for GPUs |
