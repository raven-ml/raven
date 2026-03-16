# Munin Examples

| Example | What you'll learn |
|---------|-------------------|
| [01-basic](01-basic/) | Start a session, log metrics, store an artifact, finish |
| [02-metrics](02-metrics/) | Define metrics with summaries, goals, and custom x-axes |
| [03-artifacts](03-artifacts/) | Version artifacts, attach aliases, track cross-run lineage |
| [04-media](04-media/) | Log images, files, and structured tables |
| [05-parameter-sweep](05-parameter-sweep/) | Group runs under a sweep, compare results |
| [06-inspect](06-inspect/) | Query the store, browse experiments, examine provenance |
| [07-system-monitor](07-system-monitor/) | Record CPU and memory usage automatically |
| [x-kaun-mnist](x-kaun-mnist/) | End-to-end MNIST training with kaun integration |

Run any example with:

```sh
dune exec packages/munin/examples/01-basic/main.exe
```

Examples write to a local `_munin/` directory.
