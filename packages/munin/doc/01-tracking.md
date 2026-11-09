# Tracking runs

A run is one execution of a program. Its session writes it, and readers
list it and fold its log. This page follows one store through the life of
a few runs. Its examples open `Munin` and share one store:

```ocaml
open Munin

let store = Store.open_ ()
```

## Sessions

A session is the one writer of one run. `Session.start` creates the run
and writes its start: the experiment, an optional name, params, the
largest step it will log, and the provenance of the process. The session
ends the run with `Session.finish`, which writes how it ended: finished,
failed with an error, or killed. `Session.within` brackets a function with
`finish`, so the run ends however the function returns:

```ocaml
let diverged =
  let s = Session.start ~store ~experiment:"tracking" ~name:"diverged" () in
  (try Session.within s (fun _ -> failwith "loss is NaN") with Failure _ -> ());
  Session.id s
```

```ocaml
# Run.status (Option.get (Store.run store diverged));;
- : Run.status = `Failed "Failure(\"loss is NaN\")"
```

`within` caught the exception, ended the run as failed with its text, and
raised it again, which the program caught.

A run whose program dies before its end, such as one killed by the system
for lack of memory, has no end in its log. While a session is open its
process holds the run's lock, and the system releases the lock when the
process exits, however it exits. A reader on the same machine probes the
lock, and reads the run as running while it is held and as crashed once it
is free. A reader that cannot probe it, such as one on another machine,
reads the run as silent. The status section of
[`store_format.mli`](../lib/store_format.mli) states the whole rule, and
`Run.status` in [`run.mli`](../lib/run.mli) names each case.

## Metrics and steps

`Session.metric` declares a metric and returns the handle that logs it.
`Session.log` logs one sample through a handle, and `Session.log_metrics`
logs a sample of several metrics in one record, all at one step and one
time. A step is the program's own counter, an integer from 0 to 2^53, and
steps need not increase from one sample to the next. A sample's time
defaults to now.

`~goal` says whether lower or higher values are better. `~step_metric`
names another metric that readers use as this one's x axis: a sample at
step `s` pairs with that metric's latest sample at `s` (the reading
section of [`store_format.mli`](../lib/store_format.mli)). This run logs a
loss every step and an accuracy once an epoch, against the epoch:

```ocaml
let epochs =
  Session.within (Session.start ~store ~experiment:"tracking" ~steps:300 ())
  @@ fun s ->
  let epoch = Session.metric s "epoch" in
  let loss = Session.metric s ~goal:`Minimize "train/loss" in
  let acc = Session.metric s ~goal:`Maximize ~step_metric:epoch "val/acc" in
  for step = 1 to 300 do
    let e = Float.of_int ((step + 99) / 100) in
    Session.log_metrics s ~step
      [ (loss, Float.of_int (301 - step) /. 100.); (epoch, e) ];
    if step mod 100 = 0 then Session.log acc ~step (0.5 +. (e /. 10.))
  done;
  Session.id s
```

A reader finds the pairs from the two series. `Metric.step` and
`Metric.value` index a series from 0 in log order:

```ocaml
# let h = History.read (Option.get (Store.run store epochs));;
val h : History.t = <abstr>
# History.step_metric h "val/acc";;
- : string option = Some "epoch"
# let at series step =
    let v = ref None in
    for i = 0 to Metric.length series - 1 do
      if Metric.step series i = step then v := Some (Metric.value series i)
    done;
    !v
  in
  let acc = History.series h "val/acc" and epoch = History.series h "epoch" in
  List.init (Metric.length acc) (fun i ->
      (at epoch (Metric.step acc i), Metric.value acc i));;
- : (float option * float) list =
[(Some 1., 0.6); (Some 2., 0.7); (Some 3., 0.8)]
```

In one run a key is a metric's or a reading's ([System
monitoring](03-system-monitoring.md)), never both, and the session
refuses a key of the other kind (the log section of
[`store_format.mli`](../lib/store_format.mli)).

## Reading a run

`Store.runs` lists a store's runs, newest first, and `Store.run` finds one
by id. Each is a `Run.t`: what its start states, its tags and notes, and
its status, read from the first and last records of its log.
`History.read` folds the whole log into the run's series, statistics and
summary. With `~series`, the history keeps the samples of the keys named
and the statistics of every key.

A history is a value. `History.follow` folds what the log gained since,
and costs only the new records, so a reader follows a live run by calling
it at intervals. A reader sees each record once the call that wrote it
returns:

```ocaml
# let s = Session.start ~store ~experiment:"tracking" ~name:"live" ();;
val s : Session.t = <abstr>
# let loss = Session.metric s "train/loss" in
  for step = 0 to 9 do Session.log loss ~step 1. done;;
- : unit = ()
# let h =
    History.read ~series:[ "train/loss" ]
      (Option.get (Store.run store (Session.id s)));;
val h : History.t = <abstr>
# (Run.status (History.run h), Metric.length (History.series h "train/loss"));;
- : Run.status * int = (`Running, 10)
# let loss = Session.metric s "train/loss" in
  for step = 10 to 19 do Session.log loss ~step 0.5 done;
  Session.finish s;;
- : unit = ()
# let h = History.follow h;;
val h : History.t = <abstr>
# (Run.status (History.run h), Metric.length (History.series h "train/loss"));;
- : Run.status * int = (`Finished, 20)
```

The second `Session.metric` declares the key again and returns another
handle to it.

`History.stats` gives a key's count, latest sample, and the least,
greatest and mean of its finite samples. A NaN or an infinity is counted
apart and never becomes a least, greatest, mean or best value (the
preamble of [`metric.mli`](../lib/metric.mli)):

```ocaml
let nan_run =
  Session.within (Session.start ~store ~experiment:"tracking" ()) @@ fun s ->
  let loss = Session.metric s ~goal:`Minimize "train/loss" in
  List.iteri (fun step v -> Session.log loss ~step v) [ 2.; 1.; nan; 3. ];
  Session.id s
```

```ocaml
# let h = History.read (Option.get (Store.run store nan_run)) in
  let st = Option.get (History.stats h "train/loss") in
  ( st.count,
    Option.map (fun (n, (first : Metric.sample)) -> (n, first.step))
      st.non_finite,
    Option.map (fun (b : Metric.sample) -> b.value)
      (Option.bind (History.goal h "train/loss") (fun g ->
           Metric.best g st)) );;
- : int * (int * int) option * float option = (4, Some (1, 2), Some 1.)
```

## Resuming a run

A run that crashed can be continued. The program restores a state it
saved, such as a checkpoint at step 200, and `Session.resume run ~step:200`
opens a new session of the run that continues from that step. Samples the
run logged with a step above 200 belong to the history the restart
abandoned. They become orphans: the log keeps them, and readers leave
them out of series and statistics (the reading and resume sections of
[`store_format.mli`](../lib/store_format.mli)).

Here a child process stands for a program that logged steps 0 to 299 and
was killed, with its state saved at step 200:

```ocaml
let crashed =
  match Unix.fork () with
  | 0 ->
      let s = Session.start ~store ~experiment:"resume" () in
      let loss = Session.metric s ~goal:`Minimize "train/loss" in
      for step = 0 to 299 do
        Session.log loss ~step (Float.of_int (300 - step))
      done;
      Unix._exit 1
  | pid ->
      ignore (Unix.waitpid [] pid);
      List.find (fun r -> Run.experiment r = "resume") (Store.runs store)
```

```ocaml
# Run.status crashed;;
- : Run.status = `Crashed
```

The restarted program resumes from step 200 and logs on from step 201:

```ocaml
let () =
  Session.within (Session.resume crashed ~step:200) @@ fun s ->
  let loss = Session.metric s ~goal:`Minimize "train/loss" in
  for step = 201 to 399 do
    Session.log loss ~step (Float.of_int (400 - step) /. 2.)
  done
```

```ocaml
# let h = History.read (Option.get (Store.run store (Run.id crashed)));;
val h : History.t = <abstr>
# ( Metric.length (History.series h "train/loss"),
    Metric.length (History.orphans h "train/loss"),
    List.map (fun (r : History.resume) -> r.step) (History.resumes h) );;
- : int * int * int list = (400, 99, [200])
```

The series holds steps 0 to 200 of the first process and 201 to 399 of the
second. Steps 201 to 299 of the first are orphans. A resume needs no
checkpoint format: the program saves and restores its own state, and the
step it gives `resume` says what that state includes. An artifact version
is one place to keep it ([Artifacts](02-artifacts.md)).

Samples of readings are never orphans, since the machine's state at a time
happened whatever the run's steps did.

## Summaries

A summary holds what the program states about the run, such as a test
score. `Session.set_summary` appends values, each replacing the run's
earlier value of its key, and `History.summary` reads them:

```ocaml
let scored =
  Session.within (Session.start ~store ~experiment:"tracking" ()) @@ fun s ->
  Session.set_summary s [ ("test/acc", `Float 0.91); ("note", `String "a") ];
  Session.set_summary s [ ("note", `String "early stop") ];
  Session.id s
```

```ocaml
# History.summary (History.read (Option.get (Store.run store scored)));;
- : (string * Value.t) list =
[("note", `String "early stop"); ("test/acc", `Float 0.91)]
```

Readers compute the statistics of metrics, so a summary need not hold
them. When a run ends, its session writes the statistics of each key
beside the log, and `History.stats_of` reads them without folding the log.
A listing of many long runs shows each one's best values this way (the
stats section of [`store_format.mli`](../lib/store_format.mli)):

```ocaml
# List.filter_map
    (fun (key, (ks : History.key_stats)) ->
      match ks.goal with
      | None -> None
      | Some goal ->
          Option.map
            (fun (b : Metric.sample) -> (key, b.step, b.value))
            (Metric.best goal ks.stats))
    (History.stats_of (Option.get (Store.run store epochs)));;
- : (string * int * float) list =
[("train/loss", 300, 0.01); ("val/acc", 300, 0.8)]
```

## Tags and notes

A run's tags and notes are the parts of a run that change after it
starts. Any process may change them, while the run is written and after it
ended (the annotations section of
[`store_format.mli`](../lib/store_format.mli)). `Session.start ~tags`
gives the first tags, `Store.add_tags` and `Store.remove_tags` change
them, and `Store.set_notes` replaces the notes whole:

```ocaml
# Store.add_tags store epochs [ "baseline"; "cpu" ];;
- : unit = ()
# Store.set_notes store epochs (Some "Accuracy still rising at epoch 3.");;
- : unit = ()
# let r = Option.get (Store.run store epochs) in
  (Run.tags r, Run.notes r);;
- : string list * string option =
(["baseline"; "cpu"], Some "Accuracy still rising at epoch 3.")
```

## Files

`Session.log_file` copies a file into the run under a key and a step, such
as an image of samples at the end of an epoch, and `History.files` lists
them:

```ocaml
let with_files =
  Session.within (Session.start ~store ~experiment:"tracking" ()) @@ fun s ->
  let path = Filename.temp_file "grid" ".ppm" in
  for step = 1 to 2 do
    Out_channel.with_open_bin path (fun oc ->
        Printf.fprintf oc "P6\n1 1\n255\n%c%c%c" '\255' '\000' '\000');
    Session.log_file s ~step ~key:"samples/grid" path
  done;
  Session.id s
```

```ocaml
# List.map
    (fun (f : History.file) -> (f.key, f.step, Filename.basename f.path))
    (History.files (History.read (Option.get (Store.run store with_files))));;
- : (string * int * string) list =
[("samples/grid", 1, "1.ppm"); ("samples/grid", 2, "2.ppm")]
```

The copy is complete before its record is written, so a reader that finds
the record finds the whole file (the files section of
[`store_format.mli`](../lib/store_format.mli)).

## Provenance and parents

A run's start records its provenance: the command, the working directory,
the git commit and whether the tree was dirty, the toolchain, the machine,
and the seeds the program names. When the tree is dirty, the run also
keeps the output of `git diff --binary HEAD`, so the code it ran can be
recovered. `Provenance.detect` reads it from the process, and the program
adds its seeds and the environment variables it wants kept:

```ocaml
let seeded =
  let provenance =
    Provenance.detect ~seeds:[ ("init", 42); ("data", 7) ]
      ~capture_env:[ "OMP_NUM_THREADS" ] ()
  in
  Session.within (Session.start ~store ~experiment:"tracking" ~provenance ())
  @@ Session.id
```

```ocaml
# (Run.provenance (Option.get (Store.run store seeded))).seeds;;
- : (string * int) list = [("init", 42); ("data", 7)]
```

A run may belong to another run of the store, its parent, such as the run
that drives a sweep. `~parent` names it, and a reader finds a parent's
runs by filtering the listing:

```ocaml
let sweep =
  Session.within (Session.start ~store ~experiment:"sweep" ()) @@ fun sweep ->
  let parent = Session.id sweep in
  List.iter
    (fun lr ->
      Session.within
        (Session.start ~store ~experiment:"sweep" ~parent
           ~params:[ ("lr", `Float lr) ] ())
      @@ fun s ->
      let loss = Session.metric s ~goal:`Minimize "train/loss" in
      Session.log loss ~step:0 (Float.abs (lr -. 0.01)))
    [ 0.1; 0.01; 0.001 ];
  parent
```

```ocaml
# List.filter_map
    (fun r ->
      if Run.parent r <> Some sweep then None
      else List.assoc_opt "lr" (Run.params r))
    (Store.runs store);;
- : Value.t list = [`Float 0.001; `Float 0.01; `Float 0.1]
```

The children are listed newest first, so the last one started comes first.
