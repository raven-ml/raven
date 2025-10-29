module Snapshot = Checkpoint.Snapshot

type t = {
  step : int;
  params : Ptree.t;
  opt_state : Optimizer.state;
  rng : Rune.Rng.key;
  metrics : Metrics.Collection.t option;
}

let init ~model ~optimizer ?metrics ~rngs ~dtype () =
  let params = model.Layer.init ~rngs ~dtype in
  let opt_state = Optimizer.init optimizer params in
  { step = 0; params; opt_state; rng = rngs; metrics }

let create ?(step = 0) ~params ~opt_state ~rng ?metrics () =
  { step; params; opt_state; rng; metrics }

let apply_gradients ~optimizer ~grads state =
  let updates, opt_state =
    Optimizer.step optimizer state.opt_state state.params grads
  in
  Optimizer.apply_updates_inplace state.params updates;
  { state with opt_state; step = state.step + 1 }

let next_rng state =
  let split = Rune.Rng.split state.rng in
  if Array.length split < 2 then
    invalid_arg "Train_state.next_rng: expected Rune.Rng.split to return 2 keys";
  (split.(0), { state with rng = split.(1) })

let reset_metrics state =
  (match state.metrics with
  | Some metrics -> Metrics.Collection.reset metrics
  | None -> ());
  state

let update_metrics state ~predictions ~targets ?loss ?weights () =
  match state.metrics with
  | None -> ()
  | Some metrics ->
      Metrics.Collection.update metrics ~predictions ~targets ?loss ?weights ()

let compute_metrics state =
  match state.metrics with
  | None -> []
  | Some metrics -> Metrics.Collection.compute metrics

let schema_key = "schema"
let schema_value = "kaun.train_state/1"
let checkpoint_slug = "state"

let to_snapshot ?encode_metrics ({ step; params; opt_state; rng; metrics } : t)
    =
  let open Snapshot in
  let base_entries =
    [
      (schema_key, scalar_string schema_value);
      ("step", scalar_int step);
      ("params", of_ptree params);
      ("optimizer_state", Optimizer.serialize opt_state);
      ("rng", scalar_int (Rune.Rng.to_int rng));
    ]
  in
  match (metrics, encode_metrics) with
  | None, _ -> record_of base_entries
  | Some metrics_value, Some encode ->
      record_of (("metrics", encode metrics_value) :: base_entries)
  | Some _, None ->
      invalid_arg
        "Train_state.to_snapshot: metrics present but encode_metrics missing"

let of_snapshot ~optimizer ?decode_metrics snapshot =
  let open Result in
  let open Snapshot in
  let ( let* ) = bind in
  let error msg = Error ("Train_state.of_snapshot: " ^ msg) in

  let* record =
    match snapshot with Record r -> Ok r | _ -> error "expected record"
  in

  let validate_schema record =
    match Snapshot.Record.find_opt schema_key record with
    | None -> Ok ()
    | Some (Scalar (String value)) ->
        if String.equal value schema_value then Ok ()
        else error ("unsupported schema " ^ value)
    | Some _ -> error "invalid schema field"
  in

  let* () = validate_schema record in

  let find field =
    match Snapshot.Record.find_opt field record with
    | Some value -> Ok value
    | None -> error ("missing field " ^ field)
  in

  let decode_step = function
    | Scalar (Int i) -> Ok i
    | Scalar (Float f) -> Ok (int_of_float f)
    | _ -> error "expected integer step"
  in

  let decode_rng = function
    | Scalar (Int seed) -> Ok (Rune.Rng.key seed)
    | Scalar (Float f) -> Ok (Rune.Rng.key (int_of_float f))
    | _ -> error "expected RNG scalar"
  in

  let decode_metrics_field () =
    match Snapshot.Record.find_opt "metrics" record with
    | None -> Ok None
    | Some value -> (
        match decode_metrics with
        | None ->
            error
              "metrics present but decode_metrics missing; provide a decoder"
        | Some decode -> (
            match decode value with
            | Ok metrics -> Ok (Some metrics)
            | Error msg -> error msg))
  in

  let* params_node = find "params" in
  let* params =
    match Snapshot.to_ptree params_node with
    | Ok params -> Ok params
    | Error msg -> error msg
  in

  let* opt_state_node = find "optimizer_state" in
  let* opt_state = Optimizer.restore optimizer opt_state_node in

  let* rng_node = find "rng" in
  let* rng = decode_rng rng_node in

  let* step_node = find "step" in
  let* step = decode_step step_node in

  let* metrics = decode_metrics_field () in
  Ok { step; params; opt_state; rng; metrics }

let make_artifact ?encode_metrics state =
  let snapshot = to_snapshot ?encode_metrics state in
  Checkpoint.artifact ~label:"state" ~kind:(Checkpoint.Custom checkpoint_slug)
    ~snapshot ()

let find_artifact artifacts =
  List.find_map
    (fun artifact ->
      if String.equal (Checkpoint.artifact_slug artifact) checkpoint_slug then
        Some artifact
      else None)
    artifacts

let save ~repository ?step ?tags ?metadata ?encode_metrics state =
  let step = Option.value ~default:state.step step in
  let metadata = Option.value ~default:[] metadata in
  let artifact = make_artifact ?encode_metrics state in
  Checkpoint.write repository ~step ?tags ~metadata ~artifacts:[ artifact ]

let load ~repository ~step ~optimizer ?decode_metrics () =
  match Checkpoint.read repository ~step with
  | Error err -> Error ("Train_state.load: " ^ Checkpoint.error_to_string err)
  | Ok (_manifest, artifacts) -> (
      match find_artifact artifacts with
      | None -> Error "Train_state.load: missing state artifact"
      | Some artifact ->
          let snapshot = Checkpoint.artifact_snapshot artifact in
          of_snapshot ?decode_metrics ~optimizer snapshot)
