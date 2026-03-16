# Tracking Metrics

This page covers scalar metric logging, metric definitions with
summaries and goals, media logging, and integration with Kaun's
training loop.

## Logging Scalars

`Session.log_metric` records a single scalar at a given step.
`Session.log_metrics` records several atomically with the same
timestamp.

<!-- $MDX skip -->
```ocaml
open Munin

let () =
  Session.with_run ~experiment:"tracking-demo" @@ fun session ->
  for step = 1 to 100 do
    let loss = 1.0 /. Float.of_int step in
    let acc = 1.0 -. loss in
    Session.log_metrics session ~step [ ("loss", loss); ("accuracy", acc) ]
  done
```

Each call appends to an event log. The `step` is your x-axis counter
(typically the global training step). A wall-clock timestamp is added
automatically; pass `~timestamp` to override it.

Read metrics back through the `Run` module:

<!-- $MDX skip -->
```ocaml
let run = Session.run session in
Run.metric_keys run        (* ["accuracy"; "loss"] *)
Run.latest_metrics run     (* latest value per key *)
Run.metric_history run "loss"  (* full chronological history *)
```

## Defining Metrics

`Session.define_metric` declares how a metric should be summarized,
compared, and plotted. Call it once per key, before or after logging
values.

<!-- $MDX skip -->
```ocaml
Session.define_metric session "loss"
  ~summary:`Min
  ~goal:`Minimize
  ();

Session.define_metric session "accuracy"
  ~summary:`Max
  ~goal:`Maximize
  ();
```

### Summary Modes

The `~summary` parameter controls the auto-computed run summary value:

| Mode   | Summary value |
|--------|---------------|
| `` `Min ``  | Minimum over all samples |
| `` `Max ``  | Maximum over all samples |
| `` `Mean `` | Arithmetic mean of all samples |
| `` `Last `` | Most recent sample (default) |
| `` `None `` | No auto-summary |

When the run is loaded, the summary is computed from the full metric
history. You do not need to compute it yourself.

### Explicit Summaries

`Session.set_summary` writes explicit summary values that always take
precedence over auto-computed ones:

<!-- $MDX skip -->
```ocaml
Session.set_summary session
  [ ("best_loss", `Float 0.023); ("note", `String "converged early") ]
```

Use this for values that are not simple aggregations of a metric
history, or for non-float summaries.

### Goal

The `~goal` parameter declares whether lower (`` `Minimize ``) or
higher (`` `Maximize ``) values are better. It is used by:

- `munin compare` to mark the best value with `*`
- `munin watch` TUI for "best" badges
- `Run_monitor.best` to find the best observation

### Step Metric

The `~step_metric` parameter specifies another metric as the x-axis:

<!-- $MDX skip -->
```ocaml
Session.define_metric session "val/accuracy"
  ~summary:`Max ~goal:`Maximize ~step_metric:"epoch" ();
```

This tells renderers to plot `val/accuracy` against the `epoch`
metric instead of the raw step counter.

## Epoch Tracking

Epochs are not a special concept -- log them as a regular metric and
reference them with `~step_metric`:

<!-- $MDX skip -->
```ocaml
Session.define_metric session "train/loss"
  ~summary:`Min ~goal:`Minimize ~step_metric:"epoch" ();
Session.define_metric session "val/accuracy"
  ~summary:`Max ~goal:`Maximize ~step_metric:"epoch" ();

for epoch = 1 to 10 do
  let steps_per_epoch = 100 in
  for batch = 1 to steps_per_epoch do
    let step = ((epoch - 1) * steps_per_epoch) + batch in
    let loss = 1.0 /. Float.of_int step in
    Session.log_metrics session ~step
      [ ("train/loss", loss); ("epoch", Float.of_int epoch) ]
  done;
  let step = epoch * steps_per_epoch in
  Session.log_metric session ~step "val/accuracy" (Float.of_int epoch *. 0.1)
done
```

## Media Logging

### Images and Files

`Session.log_media` copies a file into the run's `media/` directory
and records it in the event log. The `~kind` is metadata for
renderers.

<!-- $MDX skip -->
```ocaml
(* Log an image at a specific step. *)
Session.log_media session ~step:100 ~key:"viz/confusion"
  ~kind:`Image ~path:"/tmp/confusion_matrix.png";

(* Log a text file. *)
Session.log_media session ~step:1 ~key:"config"
  ~kind:`File ~path:"config.yaml"
```

Keys may contain `/` separators to organize media into a hierarchy.
The file is stored at `<run_dir>/media/<key_path>_<step>.<ext>`.

Read media back:

<!-- $MDX skip -->
```ocaml
let run = Session.run session in
Run.media_keys run                    (* ["config"; "viz/confusion"] *)
Run.media_history run "viz/confusion" (* list of media_entry records *)
```

### Structured Tables

`Session.log_table` stores a table as a JSON file. Useful for
confusion matrices, per-class metrics, or data samples.

<!-- $MDX skip -->
```ocaml
Session.log_table session ~step:1 ~key:"results/per_class"
  ~columns:[ "class"; "precision"; "recall"; "f1" ]
  ~rows:[
    [ `String "cat";  `Float 0.92; `Float 0.88; `Float 0.90 ];
    [ `String "dog";  `Float 0.89; `Float 0.93; `Float 0.91 ];
    [ `String "bird"; `Float 0.95; `Float 0.91; `Float 0.93 ];
  ]
```

## Integration with Kaun

Munin has no compile-time dependency on Kaun. Integration happens
through `Train.fit`'s `~report` callback:

<!-- $MDX skip -->
```ocaml
open Kaun

let () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let session =
    Munin.Session.start ~experiment:"mnist" ~name:"cnn-adam"
      ~params:[
        ("lr", `Float 0.001);
        ("batch_size", `Int 64);
        ("optimizer", `String "adam");
      ]
      ()
  in
  Munin.Session.define_metric session "train/loss"
    ~summary:`Min ~goal:`Minimize ();
  Munin.Session.define_metric session "val/accuracy"
    ~summary:`Max ~goal:`Maximize ();

  let (x_train, y_train), (x_test, y_test) = Kaun_datasets.mnist () in
  let trainer =
    Train.make ~model ~optimizer:(Vega.adam (Vega.Schedule.constant 0.001))
  in
  let st = ref (Train.init trainer ~dtype:Nx.float32) in

  for epoch = 1 to 3 do
    let train_data =
      Data.prepare ~shuffle:true ~batch_size:64 (x_train, y_train)
      |> Data.map (fun (x, y) ->
          (x, fun logits -> Loss.cross_entropy_sparse logits y))
    in
    st :=
      Train.fit trainer !st
        ~report:(fun ~step ~loss _st ->
          Munin.Session.log_metrics session ~step
            [ ("train/loss", loss); ("epoch", Float.of_int epoch) ])
        train_data;

    (* Evaluate and log validation accuracy. *)
    let test_batches = Data.prepare ~batch_size:64 (x_test, y_test) in
    let acc =
      Metric.eval
        (fun (x, y) ->
          let logits = Train.predict trainer !st x in
          Metric.accuracy logits y)
        test_batches
    in
    Munin.Session.log_metric session ~step:(epoch * 937)
      "val/accuracy" acc
  done;

  Munin.Session.finish session ()
```

## System Monitoring

`Munin_sys.start` spawns a background thread that samples CPU and
memory usage every 15 seconds (configurable via `~interval`):

<!-- $MDX skip -->
```ocaml
let sysmon = Munin_sys.start session () in
(* ... training ... *)
Munin_sys.stop sysmon
```

Logged metrics: `sys/cpu_user`, `sys/cpu_system`, `sys/mem_used_pct`,
`sys/mem_used_gb`, `sys/proc_cpu_pct`, `sys/proc_mem_mb`.

## Next Steps

- [Artifacts](../03-artifacts/) -- versioned files, aliases, lineage, deduplication
