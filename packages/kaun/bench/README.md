# kaun benchmarks

| Path | Role | Gated? |
|---|---|---|
| `bench_kaun.ml` → `bench_kaun.exe` | **Gate.** Train-step and layer microbenchmarks with a committed baseline; run with `dune build @bench`. | yes (`kaun.thumper`) |
| `decode/bench_decode.ml` → `bench_decode.exe` | **Gate.** The jitted decode step of a GPT-2 124M shaped decoder built from kaun layers (one token through donated key-value caches, token read back on the host) at two cache lengths, so a change to cached attention is measured against a committed step time. Zero weights, no download. | yes (`decode/kaun_decode.thumper`) |
| `compare/` | **Comparison.** The same model trained end to end in PyTorch, Raven and tinygrad from one shared fixture: cold start, warm start, steady step time and peak memory, with timings withheld from any run whose losses disagree. Run on demand; see [compare/README.md](compare/README.md). | no |

The comparison lives here because kaun is the top of the training stack: its
Raven models are written against kaun, vega and rune exactly as a user would
write them, so its numbers cover every package underneath.
