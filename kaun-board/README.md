# Kaun Board

Training run logger and terminal dashboard for [Kaun](../kaun/)

Kaun Board logs scalar metrics during training and provides a live TUI
dashboard with charts and system resource monitoring. Runs are stored as
append-only JSONL files that can be inspected incrementally while training
is in progress.

## Quick Start

Log metrics from your training loop:

```ocaml
let logger =
  Kaun_board.Log.create ~experiment:"mnist"
    ~config:[ ("lr", Jsont.Json.number 0.001);
              ("batch_size", Jsont.Json.int 64) ]
    ()
in
Printf.printf "To monitor: kaun-board %s\n" (Kaun_board.Log.run_id logger);

(* In your training loop *)
Kaun_board.Log.log_scalar logger ~step ~epoch ~tag:"train/loss" loss;
Kaun_board.Log.log_scalars logger ~step ~epoch
  [ ("train/loss_avg", avg_loss); ("test/accuracy", test_acc) ];

Kaun_board.Log.close logger
```

Then launch the dashboard in another terminal:

```sh
kaun-board              # auto-selects the most recent run
kaun-board <RUN_ID>     # monitor a specific run
```

## Features

- **Logging**: thread-safe scalar event logging to append-only JSONL files
- **Run management**: timestamped run directories with JSON manifests storing experiment name, tags, and config
- **Incremental reads**: stream events from disk without rescanning history
- **Live dashboard**: terminal UI with metric charts, latest values, and best-value tracking
- **System monitoring**: CPU, memory, and GPU usage in the dashboard sidebar
- **Keyboard navigation**: number keys for metric detail view, arrow keys for pagination

## Libraries

| Library | opam package | Description |
|---------|-------------|-------------|
| `kaun_board` | `kaun-board` | Core: logging, run management, event store |
| `kaun_board_tui` | `kaun-board.tui` | Terminal dashboard UI |

## Run Directory Layout

Each training run creates a directory under `$RAVEN_RUNS_DIR` (defaults to
`$XDG_CACHE_HOME/raven/runs`):

```
2026-02-26_14-30-00_a1b2_mnist/
  run.json       -- manifest (experiment, tags, config, timestamps)
  events.jsonl   -- append-only scalar events
```

## Examples

- **01-mnist** -- MNIST training with kaun-board logging and live monitoring

## Contributing

See the [Raven monorepo README](../README.md) for guidelines.

## License

ISC License. See [LICENSE](../LICENSE) for details.
