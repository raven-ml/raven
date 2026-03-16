# Munin

Munin is a local-first experiment tracker for OCaml. It records
hyperparameters, metrics, media, and versioned artifacts to disk with
no external services. A CLI and live TUI let you inspect and compare
runs from the terminal.

## Features

- **Scalar tracking**: `log_metric`, `log_metrics`, auto-computed summaries
- **Metric definitions**: summary modes (min/max/mean/last), goals (minimize/maximize), custom x-axes
- **Media logging**: images, files, audio, and structured tables
- **Versioned artifacts**: content-addressed deduplication, aliases, cross-run lineage
- **Provenance**: git commit, command line, environment variables, captured automatically
- **System monitoring**: background CPU and memory sampling via `Munin_sys`
- **CLI**: `munin runs`, `munin show`, `munin compare`, `munin metrics`, `munin artifacts`
- **Live TUI**: `munin watch` with real-time metric charts and system stats

## Quick Start

<!-- $MDX skip -->
```ocaml
let () =
  Munin.Session.with_run ~experiment:"demo"
    ~params:[ ("lr", `Float 0.001); ("epochs", `Int 10) ]
  @@ fun session ->
  for step = 1 to 100 do
    let loss = 1.0 /. Float.of_int step in
    Munin.Session.log_metric session ~step "loss" loss
  done;
  Munin.Session.set_summary session [ ("final_loss", `Float 0.01) ]
```

Inspect the run from the terminal:

<!-- $MDX skip -->
```sh
munin runs
munin show <RUN_ID>
munin metrics <RUN_ID> --key loss
```

## Next Steps

- [Getting Started](01-getting-started/) -- installation, key concepts, first example
- [Tracking Metrics](02-tracking/) -- scalars, metric definitions, media, Kaun integration
- [Artifacts](03-artifacts/) -- versioned files, aliases, lineage, deduplication
