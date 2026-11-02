# Artifacts

An artifact is a name, such as `model`, with versions numbered from 1. A
run publishes a version from a file or a directory: a checkpoint, a
dataset, an exported model. Another run finds a version by its number, as
the latest, or by an alias, and records that it used it. The preamble of
[`artifact.mli`](../lib/artifact.mli) defines the terms, and the artifacts
and blobs sections of [`store_format.mli`](../lib/store_format.mli)
specify the files.

The examples on this page share one store and a helper that writes a file:

```ocaml
open Munin

let store = Store.open_ ()

let write path text =
  Out_channel.with_open_bin path (fun oc -> output_string oc text)
```

## Publishing versions

`Session.log_artifact s ~name path` copies `path` into the store as a new
version of `name`, produced by `s`'s run, and returns the version.
`~step` records the step it was produced at, and `~metadata` attaches
values. A version never changes. This run publishes a checkpoint
directory twice, and names the first one `best`:

```ocaml
let train =
  Session.within (Session.start ~store ~experiment:"artifacts" ()) @@ fun s ->
  let dir = Filename.temp_dir "ckpt" "" in
  let checkpoint step =
    write (Filename.concat dir "weights.bin") (Printf.sprintf "at %d" step);
    Session.log_artifact s ~step ~name:"model"
      ~metadata:[ ("format", `String "raw") ]
      dir
  in
  let first = checkpoint 100 in
  ignore (checkpoint 200);
  Store.set_alias store first "best";
  Session.id s
```

```ocaml
# List.map
    (fun a -> (Artifact.name a, Artifact.version a, Artifact.step a))
    (Store.artifacts store);;
- : (string * int * int option) list =
[("model", 1, Some 100); ("model", 2, Some 200)]
# Store.aliases store "model";;
- : (string * int) list = [("best", 1)]
# Option.map Artifact.version (Store.artifact store "model" `Latest);;
- : int option = Some 2
```

`log_artifact` copies the directory once and computes the digest of the
bytes it copies, so version 1 holds the checkpoint as it was at step 100,
although the program overwrote the directory afterwards.

An alias names one version and moves when set again. Setting `best` on
version 2 later makes `` `Alias "best" `` select version 2, and versions
1 and 2 stay as they are.

## Digests

A version's content is stored once, as a blob named by its digest. Two
versions with the same bytes share one blob, and each is still a version
of its own:

```ocaml
# let config = Filename.temp_file "config" ".yaml" in
  write config "lr: 0.001\n";
  Session.within (Session.start ~store ~experiment:"artifacts" ()) @@ fun s ->
  let a = Session.log_artifact s ~name:"config" config in
  let b = Session.log_artifact s ~name:"config" config in
  ( Artifact.version a,
    Artifact.version b,
    Artifact.digest a = Artifact.digest b,
    Artifact.path a = Artifact.path b );;
- : int * int * bool * bool = (1, 2, true, true)
```

`Artifact.path` is the blob itself, shared by every version with the same
content, so a program reads it and never writes to it.

## Using a version

A run that reads a version records it with `Session.use_artifact`. The
version knows the run that produced it, and the using run's history lists
what it used:

```ocaml
let evaluation =
  Session.within
    (Session.start ~store ~experiment:"artifacts" ~name:"eval" ())
  @@ fun s ->
  let best = Option.get (Store.artifact store "model" (`Alias "best")) in
  Session.use_artifact s best;
  let weights =
    In_channel.with_open_bin
      (Filename.concat (Artifact.path best) "weights.bin")
      In_channel.input_all
  in
  Session.set_summary s [ ("weights", `String weights) ];
  Session.id s
```

```ocaml
# let best = Option.get (Store.artifact store "model" (`Alias "best"));;
val best : Artifact.t = <abstr>
# Artifact.run best = train;;
- : bool = true
# let h = History.read (Option.get (Store.run store evaluation));;
val h : History.t = <abstr>
# (History.uses h, History.summary h);;
- : (string * int) list * (string * Value.t) list =
([("model", 1)], [("weights", `String "at 100")])
```

A use is a record in the using run's log, so finding every run that used a
version reads every log:

```ocaml
# List.filter_map
    (fun r ->
      if List.mem ("model", 1) (History.uses (History.read r)) then Run.name r
      else None)
    (Store.runs store);;
- : string list = ["eval"]
```

## Resuming from a checkpoint

`log_artifact` syncs the version and the producing run's log before it
returns (the durability section of
[`store_format.mli`](../lib/store_format.mli)). A program that resumes
from a version therefore finds every sample up to the version's step in
the log, even after a power loss. The run's latest checkpoint gives both
the state and the step to resume from. Here a child process stands for a
program killed after its checkpoint at step 20:

```ocaml
let crashed =
  match Unix.fork () with
  | 0 ->
      let s = Session.start ~store ~experiment:"checkpointed" () in
      let loss = Session.metric s "train/loss" in
      let state = Filename.temp_file "state" ".txt" in
      for step = 0 to 29 do
        Session.log loss ~step (Float.of_int (30 - step));
        if step mod 10 = 0 then begin
          write state (string_of_int step);
          ignore (Session.log_artifact s ~step ~name:"state" state)
        end
      done;
      Unix._exit 1
  | pid ->
      ignore (Unix.waitpid [] pid);
      List.find (fun r -> Run.experiment r = "checkpointed") (Store.runs store)
```

```ocaml
let restart run =
  let mine =
    List.filter
      (fun a -> Artifact.run a = Run.id run)
      (Store.artifacts ~name:"state" store)
  in
  let last = List.nth mine (List.length mine - 1) in
  let step = Option.get (Artifact.step last) in
  (* The program restores its state from [Artifact.path last] here. *)
  Session.within (Session.resume run ~step) @@ fun s ->
  Session.use_artifact s last;
  let loss = Session.metric s "train/loss" in
  for step = step + 1 to 29 do
    Session.log loss ~step (Float.of_int (30 - step))
  done

let () = restart crashed
```

```ocaml
# let h = History.read (Option.get (Store.run store (Run.id crashed)));;
val h : History.t = <abstr>
# ( Run.status (History.run h),
    Metric.length (History.series h "train/loss"),
    Metric.length (History.orphans h "train/loss") );;
- : Run.status * int * int = (`Finished, 30, 9)
```

`Store.artifacts ~name` lists a name's versions by number, so the last of
the run's own is its latest checkpoint. The first process logged steps 21
to 29 after it, and the resume made them orphans.

## Deleting versions

`Store.delete_artifact` deletes one version and the aliases that name it.
Its number never names another version. Its blob stays until `Store.gc`
finds that no version names it:

```ocaml
# let v2 = Option.get (Store.artifact store "model" (`Version 2)) in
  Store.delete_artifact store v2;;
- : unit = ()
# Option.map Artifact.version (Store.artifact store "model" `Latest);;
- : int option = Some 1
# Store.gc store;;
- : int = 1
```

`gc` waits for the publications in progress, and publications that start
wait for it, so it never removes a blob a version is about to name (the
blobs section of [`store_format.mli`](../lib/store_format.mli)).
