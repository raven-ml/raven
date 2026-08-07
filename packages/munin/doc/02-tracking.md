# Tracking Metrics

This page covers scalar metric logging, metric definitions with
summaries and goals, media logging, and integration with Kaun's
training loop.

## Logging Scalars

`Session.metric` declares a metric and returns the handle that logs
it. The key is spelled once, where the metric is declared, so a typo
cannot silently create a second channel. `Metric.log` records a single
scalar at a given step; `Session.log_metrics` records several under
one timestamp.

<!-- $MDX skip -->
```ocaml
open Munin

let () =
  Session.with_run ~experiment:"tracking-demo" @@ fun session ->
  let loss = Session.metric session ~goal:`Minimize "loss" in
  let accuracy = Session.metric session ~goal:`Maximize "accuracy" in
  for step = 1 to 100 do
    let l = 1.0 /. Float.of_int step in
    Session.log_metrics session ~step [ (loss, l); (accuracy, 1.0 -. l) ]
  done
```

Each call appends to an event log. The `step` is your x-axis counter
(typically the global training step). A wall-clock timestamp is added
automatically; pass `~timestamp` to override it.

Read metrics back through the `Run` module. Reading is keyed by
string: keys come off disk, so there is nothing for a handle to
check.

<!-- $MDX skip -->
```ocaml
let run = Session.run session in
Run.metric_keys run        (* ["accuracy"; "loss"] *)
Run.latest_metrics run     (* latest value per key *)
Run.metric_history run "loss"  (* full chronological history *)
```

## Declaring Metrics

The optional arguments of `Session.metric` say how a metric should be
summarized, compared, and plotted. Declaring the same key again
replaces its definition; readers see the last one.

<!-- $MDX skip -->
```ocaml
let loss = Session.metric session ~goal:`Minimize "loss" in
let accuracy = Session.metric session ~goal:`Maximize "accuracy" in
```

### Summary Modes

The `~summary` parameter controls the auto-computed run summary value:

| Mode   | Summary value |
|--------|---------------|
| `` `Min ``  | Minimum over all samples |
| `` `Max ``  | Maximum over all samples |
| `` `Mean `` | Arithmetic mean of all samples |
| `` `Last `` | Most recent sample |
| `` `None `` | No auto-summary |

`~summary` defaults to the mode `~goal` implies -- `` `Min `` for
`` `Minimize ``, `` `Max `` for `` `Maximize `` -- and to `` `Last ``
when no goal is given. Pass it explicitly only to override that.

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

The `~step_metric` parameter takes another metric to use as the
x-axis, so the reference is checked by the compiler:

<!-- $MDX skip -->
```ocaml
let epoch = Session.metric session "epoch" in
let accuracy =
  Session.metric session ~goal:`Maximize ~step_metric:epoch "val/accuracy"
in
```

This tells renderers to plot `val/accuracy` against the `epoch`
metric instead of the raw step counter. The x-axis metric must be
declared before the metrics that reference it.

## Epoch Tracking

Epochs are not a special concept -- declare one as a regular metric
and reference it with `~step_metric`:

<!-- $MDX skip -->
```ocaml
let epoch_metric = Session.metric session "epoch" in
let loss =
  Session.metric session ~goal:`Minimize ~step_metric:epoch_metric "train/loss"
in
let accuracy =
  Session.metric session ~goal:`Maximize ~step_metric:epoch_metric
    "val/accuracy"
in

for epoch = 1 to 10 do
  let steps_per_epoch = 100 in
  for batch = 1 to steps_per_epoch do
    let step = ((epoch - 1) * steps_per_epoch) + batch in
    Session.log_metrics session ~step
      [ (loss, 1.0 /. Float.of_int step);
        (epoch_metric, Float.of_int epoch) ]
  done;
  let step = epoch * steps_per_epoch in
  Metric.log accuracy ~step (Float.of_int epoch *. 0.1)
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

Munin has no compile-time dependency on Kaun. Since the training loop is
ordinary code you own, logging is a line inside it:

<!-- $MDX skip -->
```ocaml
let () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let session =
    Munin.Session.start ~experiment:"mnist" ~name:"mlp-adamw"
      ~params:[
        ("lr", `Float 0.001);
        ("batch_size", `Int 64);
        ("optimizer", `String "adamw");
      ]
      ()
  in
  let train_loss =
    Munin.Session.metric session ~goal:`Minimize "train/loss"
  in
  let val_accuracy =
    Munin.Session.metric session ~goal:`Maximize "val/accuracy"
  in
  let epoch_metric = Munin.Session.metric session "epoch" in

  let train_x, train_y, test_x, test_y = Kaun_datasets.mnist () in
  let params = Model.init () in
  let state = ref (params, Vega.adamw_init (module Model) params) in
  let step = ref 0 in

  for epoch = 1 to 3 do
    Kaun.Data.batches2 ~shuffle:true ~batch_size:64 (train_x, train_y)
    |> Seq.iter (fun (x, y) ->
        let params, ostate = !state in
        let loss, grads =
          Rune.value_and_grad (module Model)
            (fun p -> Kaun.Loss.softmax_cross_entropy_sparse (Model.apply p x) y)
            params
        in
        let params, ostate =
          Vega.adamw_step (module Model) ~lr:0.001 ostate ~params ~grads
        in
        state := (params, ostate);
        incr step;
        Munin.Session.log_metrics session ~step:!step
          [ (train_loss, Nx.item [] loss);
            (epoch_metric, Float.of_int epoch) ]);

    (* Evaluate and log validation accuracy. *)
    let acc = Kaun.Metric.accuracy (Model.apply (fst !state) test_x) test_y in
    Munin.Metric.log val_accuracy ~step:!step acc
  done;

  Munin.Session.finish session
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
