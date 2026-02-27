# Kaun Board

Kaun Board logs scalar metrics during training and provides a live
terminal dashboard with charts and system resource monitoring. Runs
are stored as append-only JSONL files that can be inspected while
training is in progress.

## Installation

<!-- $MDX skip -->
```bash
opam install kaun-board
```

Add to your `dune` file:

<!-- $MDX skip -->
```dune
(executable
 (name main)
 (libraries kaun kaun-board))
```

## Logging Metrics

Create a logger at the start of training and log scalar values at
each step:

<!-- $MDX skip -->
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

(* Log multiple metrics at once *)
Kaun_board.Log.log_scalars logger ~step ~epoch
  [ ("train/loss_avg", avg_loss); ("test/accuracy", test_acc) ];

Kaun_board.Log.close logger
```

Key points:

- `Log.create` creates a run directory and returns a logger. The
  `~experiment` name is appended to the run ID. `~config` stores
  hyperparameters in the run manifest.
- `Log.log_scalar` appends a single metric observation. `~step` is
  required, `~epoch` is optional.
- `Log.log_scalars` appends multiple metrics with the same step and
  epoch.
- `Log.close` marks the session as closed. Always call it when
  training ends.

### Integrating with Train.fit

Use the `~report` callback to log metrics during `Train.fit`:

<!-- $MDX skip -->
```ocaml
let logger = Kaun_board.Log.create ~experiment:"mnist" () in

let st = Train.fit trainer st
  ~report:(fun ~step ~loss _st ->
    Kaun_board.Log.log_scalar logger ~step ~tag:"train/loss" loss)
  data
in

Kaun_board.Log.close logger
```

### Integrating with Train.step

For custom training loops using `Train.step`:

<!-- $MDX skip -->
```ocaml
let logger = Kaun_board.Log.create ~experiment:"mnist" () in

let step = ref 0 in
Data.iter (fun (x, y) ->
  incr step;
  let loss_val, st' =
    Train.step trainer !st ~training:true
      ~loss:(fun logits -> Loss.cross_entropy_sparse logits y) x
  in
  st := st';
  let loss = Nx.item [] loss_val in
  Kaun_board.Log.log_scalar logger ~step:!step ~tag:"train/loss" loss
) data;

Kaun_board.Log.close logger
```

## The Dashboard

Launch the dashboard from another terminal while training runs:

<!-- $MDX skip -->
```bash
kaun-board              # auto-selects the most recent run
kaun-board <RUN_ID>     # monitor a specific run
```

The dashboard shows:

- **Metric charts**: line charts for each logged metric, updated live
- **Latest values**: current value of each metric
- **Best values**: best observed value per metric (minimum for loss/error
  tags, maximum otherwise)
- **System resources**: CPU, memory, and GPU usage in the sidebar

### Keyboard shortcuts

- `1`–`9` — focus a specific metric chart
- `←` `→` — paginate metrics
- `q` — quit

## Run Directory Layout

Each training run creates a directory under `$RAVEN_RUNS_DIR` (defaults
to `$XDG_CACHE_HOME/raven/runs`):

```
2026-02-26_14-30-00_a1b2_mnist/
  run.json       -- manifest (experiment, tags, config, timestamps)
  events.jsonl   -- append-only scalar events
```

The manifest (`run.json`) stores the experiment name, tags,
hyperparameter config, and creation timestamp. Events are one JSON
object per line with `step`, `tag`, `value`, `wall_time`, and an
optional `epoch`.

## Reading Runs Programmatically

Use `Run` and `Store` to read and aggregate logged metrics:

<!-- $MDX skip -->
```ocaml
(* Load a run from its directory *)
let run = Kaun_board.Run.load run_dir |> Option.get in
Printf.printf "Run: %s\n" (Kaun_board.Run.run_id run);

(* Stream events incrementally *)
let stream = Kaun_board.Run.open_events run in
let events = Kaun_board.Run.read_events stream in

(* Aggregate into a store *)
let store = Kaun_board.Store.create () in
Kaun_board.Store.update store events;

(* Query metrics *)
let latest = Kaun_board.Store.latest_metrics store in
List.iter (fun (tag, m) ->
  Printf.printf "%s: %.4f (step %d)\n" tag m.value m.step
) latest;

Kaun_board.Run.close_events stream
```

`Store` maintains per-tag latest values, full history, and best values
as new events arrive, without rescanning history on each update.
