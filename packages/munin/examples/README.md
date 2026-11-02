# Munin Examples

| Example | What it shows |
|---------|---------------|
| [01-basic](01-basic/) | Start a run, log a metric, write a summary |
| [02-metrics](02-metrics/) | Goals, an x axis of another metric, series and statistics read back |
| [03-artifacts](03-artifacts/) | Versions, aliases, and which run produced and used each one |
| [04-media](04-media/) | Files a run logs at a step, found through its history |
| [05-parameter-sweep](05-parameter-sweep/) | Runs whose parent is a sweep, compared by their best error |
| [06-inspect](06-inspect/) | Runs, status, provenance, metrics and artifacts read from a store |
| [07-system-monitor](07-system-monitor/) | Readings of the machine and the process, sampled into a run |
| [x-kaun-mnist](x-kaun-mnist/) | MNIST training with kaun, tracked by munin |
| [x-kaun-mnist-jit](x-kaun-mnist-jit/) | Jitted MNIST training on a GPU, with readings of the device |

Run an example with:

```sh
dune exec packages/munin/examples/01-basic/main.exe
```

Examples write to the default store. To keep them out of it, point
`RAVEN_TRACKING_DIR` at another directory:

```sh
RAVEN_TRACKING_DIR=/tmp/munin dune exec packages/munin/examples/01-basic/main.exe
```

`06-inspect` reads what the others wrote. The `x-kaun-mnist` examples
download MNIST on their first run.
