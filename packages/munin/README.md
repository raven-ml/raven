# Munin

Local experiment tracking for [Raven](https://github.com/raven-ml/raven).

Track metrics, save artifacts, and compare runs on the local filesystem.
Comes with a terminal dashboard for live monitoring. Data is plain JSON,
readable with `jq`. No server, no accounts.

## Quick Start

```ocaml
let session =
  Munin.Session.start ~experiment:"mnist"
    ~params:[ ("lr", `Float 0.001); ("epochs", `Int 10) ]
    ()
in
for step = 1 to 1000 do
  let loss = train_step () in
  Munin.Session.log_metric session ~step "train/loss" loss
done;
Munin.Session.finish session ()
```

```sh
munin watch          # live terminal dashboard
munin compare a b c  # side-by-side params + summary
```

## Features

- **Scalar metrics** with `define_metric` for auto-computed summaries (min, max, mean, last) and custom x-axes
- **Media logging** -- images, tables, and files at specific steps
- **Versioned artifacts** with content-addressed storage, aliases, and lineage tracking
- **Terminal dashboard** with live metric charts, system resource panels, and EMA smoothing
- **CLI** -- `runs`, `show`, `compare`, `metrics`, `watch`, `artifacts`, `delete`, `gc`
- **System monitoring** -- opt-in CPU and memory tracking via background thread
- **Run grouping** for hyperparameter sweeps, parent/child for nested runs
- **Provenance** -- git commit, command line, hostname, environment captured automatically
- **Plain JSON storage** -- append-only JSONL event logs, `jq`-friendly

## Libraries

| Library | Description |
|---------|-------------|
| `munin` | Core tracking: Session, Run, Store, Artifact |
| `munin.tui` | Terminal dashboard (`munin watch`) |
| `munin.sys` | Background system monitoring (CPU, memory) |

## Examples

- **01-basic** -- Minimal run with scalar metrics
- **02-metrics** -- Metric definitions, auto-summaries, epoch tracking
- **03-artifacts** -- Versioned checkpoints with aliases and lineage
- **04-media** -- Logging images and tables
- **05-parameter-sweep** -- Run grouping for hyperparameter search
- **06-inspect** -- Reading runs programmatically
- **07-system-monitor** -- Background CPU and memory tracking

## Contributing

See the [Raven monorepo README](../../README.md) for guidelines.

## License

ISC License. See [LICENSE](../../LICENSE) for details.
