# Apps

Programs that use raven for real work, to read as much as to run.

- An app uses raven's public interfaces only. Code another program would
  reuse goes into a library.
- Weights and datasets are fetched, never committed.
- `dune build @apps/runtest` runs each app's fast tests on small fixtures.
  Parity with an upstream implementation runs outside it, and the app's
  README records the last result.

| App | What it does |
|---|---|
| [gpt-oss](gpt-oss/) | OpenAI's gpt-oss, a mixture-of-experts model: chat in the harmony format, eagerly on the CPU or compiled on GPUs |
