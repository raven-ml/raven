# Getting Started

This guide covers installation, key concepts, and a complete first
example that tracks a run, inspects it via the CLI, and compares two
runs.

## Installation

<!-- $MDX skip -->
```bash
opam install munin
```

Or build from source:

<!-- $MDX skip -->
```bash
git clone https://github.com/raven-ml/raven
cd raven && dune build munin
```

## Key Concepts

**Session.** A session is the write handle for a single run. All
mutations go through append-only events -- no direct state editing.
`Session.start` opens a session, `Session.finish` closes it.
`Session.with_run` wraps both and handles exceptions.

**Run.** A run is the persisted, read-only view of a tracked
experiment. It materializes its state by replaying the event log.
`Run.params`, `Run.summary`, `Run.metric_history`, and other
accessors expose the data.

**Store.** A store is the root directory containing all experiments
and artifacts. `Store.open_` creates or opens it.
`Store.list_runs`, `Store.find_run`, and `Store.latest_run`
discover runs across experiments.

**Artifact.** An artifact is a versioned, content-addressed file or
directory. Versions are auto-incremented (v1, v2, ...). Aliases like
`"latest"` or `"best"` resolve to a specific version.
`Session.log_artifact` produces one, `Session.use_artifact` records
consumption.

**Value.** Parameters, summaries, and metadata use a simple scalar
type: `` [`Bool of bool | `Int of int | `Float of float | `String of string] ``.

## Example: First Tracked Run

This example starts a session, logs hyperparameters and metrics,
saves an artifact, then reads everything back.

<!-- $MDX skip -->
```ocaml
open Munin

let () =
  let session =
    Session.start ~experiment:"demo" ~name:"baseline"
      ~params:[ ("lr", `Float 0.001); ("hidden", `Int 64) ]
      ()
  in
  (* Log metrics at each step. *)
  Session.define_metric session "loss" ~summary:`Min ~goal:`Minimize ();

  for step = 1 to 50 do
    let loss = 1.0 /. Float.of_int step in
    let acc = 1.0 -. loss in
    Session.log_metrics session ~step [ ("loss", loss); ("accuracy", acc) ]
  done;

  (* Write a summary value explicitly. *)
  Session.set_summary session [ ("note", `String "first run") ];

  Session.finish session ();
  Printf.printf "run: %s\n" (Run.id (Session.run session))
```

After running, inspect from the terminal:

<!-- $MDX skip -->
```sh
# List all runs.
munin runs

# Show full details for a run.
munin show <RUN_ID>

# Dump metric history as TSV.
munin metrics <RUN_ID> --key loss

# Export as JSON.
munin metrics <RUN_ID> --key loss --format json
```

## Example: Comparing Two Runs

Run the same experiment with different hyperparameters, then compare.

<!-- $MDX skip -->
```ocaml
open Munin

let train ~name ~lr =
  Session.with_run ~experiment:"demo" ~name
    ~params:[ ("lr", `Float lr) ]
  @@ fun session ->
  Session.define_metric session "loss" ~summary:`Min ~goal:`Minimize ();
  for step = 1 to 50 do
    let loss = (1.0 /. Float.of_int step) *. (1.0 /. lr) in
    Session.log_metric session ~step "loss" loss
  done

let () =
  train ~name:"slow" ~lr:0.01;
  train ~name:"fast" ~lr:0.1
```

Compare them side by side:

<!-- $MDX skip -->
```sh
munin compare <RUN_ID_1> <RUN_ID_2>
```

The compare command prints a table with parameters and summary values.
When a metric has a `goal` declared, the best value is marked with `*`.

## Provenance

Every run automatically captures:

- The command line (`Sys.argv`)
- Working directory
- Hostname and PID
- Git commit hash and dirty status

Pass `~capture_env:["CUDA_VISIBLE_DEVICES"; "OMP_NUM_THREADS"]` to
`Session.start` to also record specific environment variables.

## Store Location

By default, runs are stored in `$XDG_DATA_HOME/raven/munin`. Override
with the `RAVEN_TRACKING_DIR` environment variable, or pass `~root`
to `Session.start` and `Store.open_`.

## Next Steps

- [Tracking Metrics](../02-tracking/) -- scalars, metric definitions, media, Kaun integration
- [Artifacts](../03-artifacts/) -- versioned files, aliases, lineage, deduplication
