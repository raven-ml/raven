# Artifacts

Artifacts are versioned, content-addressed files or directories with
cross-run lineage tracking. Use them for datasets, model checkpoints,
and any other outputs you want to version alongside your runs.

## Logging an Artifact

`Session.log_artifact` copies a file or directory into the blob store,
assigns a version, and records it as an output of the current run.

<!-- $MDX skip -->
```ocaml
open Munin

let () =
  Session.with_run ~experiment:"pipeline" ~name:"prepare-data"
    ~tags:[ "data" ]
  @@ fun session ->
  (* ... produce a dataset file ... *)
  let _artifact =
    Session.log_artifact session
      ~name:"measurements"
      ~kind:`dataset
      ~path:"data/measurements.csv"
      ~metadata:[ ("rows", `Int 10000); ("format", `String "csv") ]
      ~aliases:[ "latest" ]
      ()
  in
  ()
```

Parameters:

- **`~name`** -- logical name for the artifact (e.g. `"measurements"`, `"mnist-cnn"`).
- **`~kind`** -- one of `` `checkpoint ``, `` `model ``, `` `dataset ``, `` `file ``, `` `dir ``, `` `other ``.
- **`~path`** -- path to the file or directory to store.
- **`~metadata`** -- optional key-value pairs attached to the version.
- **`~aliases`** -- optional alias list (e.g. `["latest"; "best"]`).

## Artifact Kinds

The `kind` field is a semantic label. It does not affect storage; all
artifacts are stored the same way.

| Kind | Use for |
|------|---------|
| `` `checkpoint `` | Training checkpoints (model + optimizer state) |
| `` `model `` | Final model weights |
| `` `dataset `` | Datasets and data splits |
| `` `file `` | Single files (configs, logs, reports) |
| `` `dir `` | Directory trees |
| `` `other `` | Anything else |

## Versioning

Each call to `log_artifact` with the same `~name` creates a new
version: `v1`, `v2`, `v3`, and so on. Versions are immutable once
created.

<!-- $MDX skip -->
```ocaml
(* First call creates v1. *)
let v1 =
  Session.log_artifact session ~name:"model" ~kind:`model
    ~path:"model_epoch1.safetensors" ()
in
(* Second call creates v2. *)
let v2 =
  Session.log_artifact session ~name:"model" ~kind:`model
    ~path:"model_epoch2.safetensors" ()
in
Printf.printf "%s %s\n" (Artifact.version v1) (Artifact.version v2)
(* prints: v1 v2 *)
```

## Aliases

Aliases are mutable pointers to a specific version. Common aliases:

- `"latest"` -- the most recent version
- `"best"` -- the best-performing version

Pass `~aliases` when logging to attach them. When a new version gets
the same alias, it moves from the old version to the new one.

Resolve an alias through the store:

<!-- $MDX skip -->
```ocaml
let store = Store.open_ () in
match Store.find_artifact store ~name:"model" ~version:"latest" with
| Some artifact ->
    Printf.printf "resolved to %s\n" (Artifact.version artifact)
| None ->
    Printf.printf "not found\n"
```

`Store.find_artifact` accepts both explicit versions (`"v2"`) and
aliases (`"latest"`).

## Content-Addressed Deduplication

Artifact payloads are stored in a blob directory keyed by their
SHA-256 digest. If two versions have identical content, only one copy
is stored on disk.

<!-- $MDX skip -->
```ocaml
(* These share the same blob if the file content is identical. *)
let a = Session.log_artifact session ~name:"config" ~kind:`file
  ~path:"config.yaml" () in
let b = Session.log_artifact session ~name:"config" ~kind:`file
  ~path:"config.yaml" () in
assert (Artifact.digest a = Artifact.digest b)
```

`Store.gc` removes blobs that are no longer referenced by any artifact
version.

## Lineage

### Producer

When you call `Session.log_artifact`, the current run is automatically
recorded as the producer:

<!-- $MDX skip -->
```ocaml
let artifact =
  Session.log_artifact session ~name:"features" ~kind:`dataset
    ~path:"features.csv" ()
in
Artifact.producer_run_id artifact  (* Some "<run_id>" *)
```

### Consumer

`Session.use_artifact` records the current run as a consumer of an
existing artifact:

<!-- $MDX skip -->
```ocaml
(* Run 2 consumes the artifact produced by Run 1. *)
let store = Store.open_ () in
match Store.find_artifact store ~name:"features" ~version:"latest" with
| Some artifact ->
    Session.use_artifact session artifact;
    let path = Artifact.path artifact in
    Printf.printf "loading from: %s\n" path
| None ->
    failwith "artifact not found"
```

After this, the lineage is recorded in both directions:

<!-- $MDX skip -->
```ocaml
Artifact.producer_run_id artifact   (* run that created it *)
Artifact.consumer_run_ids artifact  (* runs that consumed it *)

Run.output_artifacts run1  (* artifacts produced by run1 *)
Run.input_artifacts run2   (* artifacts consumed by run2 *)
```

## Loading Artifacts

### From a Store

`Store.find_artifact` resolves by name and version (or alias):

<!-- $MDX skip -->
```ocaml
let store = Store.open_ () in
match Store.find_artifact store ~name:"mnist-cnn" ~version:"best" with
| Some artifact ->
    let path = Artifact.path artifact in
    Printf.printf "path: %s (%d bytes)\n" path (Artifact.size_bytes artifact)
| None ->
    Printf.printf "not found\n"
```

### Listing Artifacts

`Store.list_artifacts` supports filtering by name, kind, alias, and
lineage:

<!-- $MDX skip -->
```ocaml
let store = Store.open_ () in

(* All artifacts. *)
let all = Store.list_artifacts store () in

(* Only checkpoints. *)
let checkpoints = Store.list_artifacts store ~kind:`checkpoint () in

(* Only artifacts produced by a specific run. *)
let from_run = Store.list_artifacts store ~producer_run:"<RUN_ID>" () in

Printf.printf "total: %d, checkpoints: %d, from run: %d\n"
  (List.length all) (List.length checkpoints) (List.length from_run)
```

### From the CLI

<!-- $MDX skip -->
```sh
# List all artifacts.
munin artifacts

# Filter by name.
munin artifacts --name mnist-cnn
```

## Complete Example: Cross-Run Lineage

A data-preparation run produces a dataset. A training run consumes it
and produces a model checkpoint.

<!-- $MDX skip -->
```ocaml
open Munin

let write_file path text =
  let oc = open_out path in
  Fun.protect ~finally:(fun () -> close_out oc)
    (fun () -> output_string oc text)

let () =
  (* Run 1: produce a dataset. *)
  Session.with_run ~experiment:"pipeline" ~name:"prepare-data"
    ~tags:[ "data" ]
  @@ fun session ->
  write_file "/tmp/data.csv" "x,y\n1.0,2.0\n3.0,4.0\n";
  ignore
    (Session.log_artifact session ~name:"training-data" ~kind:`dataset
       ~path:"/tmp/data.csv"
       ~metadata:[ ("rows", `Int 2) ]
       ~aliases:[ "latest" ] ())

let () =
  (* Run 2: consume the dataset, produce a model. *)
  let store = Store.open_ () in
  Session.with_run ~experiment:"pipeline" ~name:"train"
    ~tags:[ "training" ]
  @@ fun session ->
  let dataset =
    match Store.find_artifact store ~name:"training-data" ~version:"latest" with
    | Some a -> a
    | None -> failwith "dataset not found"
  in
  Session.use_artifact session dataset;
  Printf.printf "training on: %s\n" (Artifact.path dataset);

  (* ... train model ... *)

  write_file "/tmp/model.bin" "model weights";
  ignore
    (Session.log_artifact session ~name:"my-model" ~kind:`model
       ~path:"/tmp/model.bin"
       ~aliases:[ "latest" ] ())
```

## Garbage Collection

`Store.gc` removes blobs not referenced by any artifact version:

<!-- $MDX skip -->
```ocaml
let store = Store.open_ () in
let removed = Store.gc store in
Printf.printf "removed %d unreferenced blobs\n" removed
```
