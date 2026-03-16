type status = [ `running | `finished | `failed | `killed ]
type metric = { step : int; timestamp : float; value : float }

type provenance = {
  notes : string option;
  command : string list;
  cwd : string;
  hostname : string option;
  pid : int;
  git_commit : string option;
  git_dirty : bool option;
  env : (string * string) list;
}

type metric_def = {
  summary : [ `Min | `Max | `Mean | `Last | `None ];
  step_metric : string option;
  goal : [ `Minimize | `Maximize ] option;
}

type media_entry = {
  step : int;
  timestamp : float;
  kind : [ `Image | `Audio | `Table | `File ];
  path : string;
}

type t = {
  root : string;
  id : string;
  dir : string;
  experiment_name : string;
  name : string option;
  group : string option;
  parent_id : string option;
  started_at : float;
  ended_at : float option;
  status : status;
  provenance : provenance;
  tags : string list;
  params : (string * Value.t) list;
  summary : (string * Value.t) list;
  latest_metrics : (string * metric) list;
  histories : (string * metric list) list;
  metric_defs : (string * metric_def) list;
  media : (string * media_entry list) list;
  input_artifacts : Artifact.t list;
  output_artifacts : Artifact.t list;
}

let schema_version = 2
let id t = t.id
let dir t = t.dir
let experiment_name t = t.experiment_name
let name t = t.name
let group t = t.group
let parent_id t = t.parent_id
let started_at t = t.started_at
let ended_at t = t.ended_at
let status t = t.status
let provenance t = t.provenance
let notes t = t.provenance.notes
let tags t = t.tags
let params t = t.params
let summary t = t.summary
let latest_metrics t = t.latest_metrics
let input_artifacts t = t.input_artifacts
let output_artifacts t = t.output_artifacts
let metric_defs t = t.metric_defs
let media_keys t = List.map fst t.media

let media_history t key =
  match List.assoc_opt key t.media with Some entries -> entries | None -> []

let resumable t = t.status = `running
let find_param t key = List.assoc_opt key t.params
let find_summary t key = List.assoc_opt key t.summary
let metric_keys t = List.map fst t.latest_metrics

let metric_history t key =
  match List.assoc_opt key t.histories with
  | Some history -> history
  | None -> []

let manifest_path root experiment id =
  Filename.concat
    (Filename.concat
       (Filename.concat
          (Filename.concat (Filename.concat root "experiments") experiment)
          "runs")
       id)
    "run.json"

let events_path dir = Filename.concat dir "events.jsonl"

let status_of_string = function
  | "finished" -> `finished
  | "failed" -> `failed
  | "killed" -> `killed
  | _ -> `running

let push_tag seen acc tag =
  if Hashtbl.mem seen tag then acc
  else (
    Hashtbl.replace seen tag ();
    tag :: acc)

let provenance_of_json json =
  let env_json = Json_utils.json_mem "env" json in
  {
    notes = Json_utils.json_mem "notes" json |> Json_utils.json_string;
    command = Json_utils.json_mem "command" json |> Json_utils.json_string_list;
    cwd =
      Option.value
        (Json_utils.json_mem "cwd" json |> Json_utils.json_string)
        ~default:"";
    hostname = Json_utils.json_mem "hostname" json |> Json_utils.json_string;
    pid =
      Option.value
        (Json_utils.json_mem "pid" json |> Json_utils.json_number)
        ~default:0.0
      |> int_of_float;
    git_commit = Json_utils.json_mem "git_commit" json |> Json_utils.json_string;
    git_dirty = Json_utils.json_mem "git_dirty" json |> Json_utils.json_bool;
    env =
      Json_utils.json_assoc env_json
      |> List.filter_map (fun (key, value) ->
          Json_utils.json_string value |> Option.map (fun text -> (key, text)));
  }

let sorted_of_hashtbl tbl =
  Hashtbl.to_seq tbl |> List.of_seq
  |> List.sort (fun (a, _) (b, _) -> String.compare a b)

let materialize root experiment id dir manifest_json =
  let tag_seen = Hashtbl.create 8 in
  let tags =
    Json_utils.json_mem "tags" manifest_json
    |> Json_utils.json_string_list
    |> List.fold_left (push_tag tag_seen) []
    |> List.rev
  in
  let params =
    Json_utils.json_mem "params" manifest_json
    |> Json_utils.json_assoc
    |> List.map (fun (k, v) -> (k, Value.of_json v))
  in
  let summary_table = Hashtbl.create 8 in
  let history_table = Hashtbl.create 16 in
  let latest_table = Hashtbl.create 16 in
  let metric_def_table = Hashtbl.create 8 in
  let media_table = Hashtbl.create 8 in
  let input_seen = Hashtbl.create 8 in
  let output_seen = Hashtbl.create 8 in
  let input_artifacts = ref [] in
  let output_artifacts = ref [] in
  let tags = ref tags in
  let status = ref `running in
  let ended_at = ref None in
  let notes =
    ref
      (Json_utils.json_mem "provenance" manifest_json |> provenance_of_json)
        .notes
  in
  List.iter
    (function
      | Event_log.Metric { step; timestamp; key; value } ->
          let metric = { step; timestamp; value } in
          let history =
            match Hashtbl.find_opt history_table key with
            | Some history -> history
            | None -> []
          in
          Hashtbl.replace history_table key (metric :: history);
          Hashtbl.replace latest_table key metric
      | Define_metric { key; summary; step_metric; goal } ->
          Hashtbl.replace metric_def_table key { summary; step_metric; goal }
      | Media { step; timestamp; key; kind; path } ->
          let abs_path = Filename.concat dir path in
          let entry = { step; timestamp; kind; path = abs_path } in
          let prev =
            match Hashtbl.find_opt media_table key with
            | Some l -> l
            | None -> []
          in
          Hashtbl.replace media_table key (entry :: prev)
      | Summary values ->
          List.iter
            (fun (key, value) -> Hashtbl.replace summary_table key value)
            values
      | Notes value -> notes := value
      | Tags values -> tags := List.fold_left (push_tag tag_seen) !tags values
      | Artifact_output { name; version } ->
          let key = name ^ ":" ^ version in
          if not (Hashtbl.mem output_seen key) then (
            Hashtbl.replace output_seen key ();
            match Artifact.load ~root ~name ~version with
            | Some artifact -> output_artifacts := artifact :: !output_artifacts
            | None -> ())
      | Artifact_input { name; version } ->
          let key = name ^ ":" ^ version in
          if not (Hashtbl.mem input_seen key) then (
            Hashtbl.replace input_seen key ();
            match Artifact.load ~root ~name ~version with
            | Some artifact -> input_artifacts := artifact :: !input_artifacts
            | None -> ())
      | Resumed _ ->
          ended_at := None;
          status := `running
      | Finished { status = status_string; ended_at = finished_at } ->
          status := status_of_string status_string;
          ended_at := Some finished_at)
    (Event_log.read (events_path dir));
  let latest_metrics = sorted_of_hashtbl latest_table in
  let histories =
    sorted_of_hashtbl history_table
    |> List.map (fun (key, history) -> (key, List.rev history))
  in
  let metric_defs = sorted_of_hashtbl metric_def_table in
  (* Auto-compute summaries from define_metric declarations. Explicit
     set_summary always wins; auto-summary only fills gaps. *)
  Hashtbl.iter
    (fun key (def : metric_def) ->
      if not (Hashtbl.mem summary_table key) then
        match Hashtbl.find_opt history_table key with
        | None | Some [] -> ()
        | Some history ->
            (* history is in reverse chronological order (newest first) *)
            let auto =
              match def.summary with
              | `Min ->
                  Some
                    (List.fold_left
                       (fun acc (m : metric) -> Float.min acc m.value)
                       Float.infinity history)
              | `Max ->
                  Some
                    (List.fold_left
                       (fun acc (m : metric) -> Float.max acc m.value)
                       Float.neg_infinity history)
              | `Mean ->
                  let n = List.length history in
                  let sum =
                    List.fold_left
                      (fun acc (m : metric) -> acc +. m.value)
                      0. history
                  in
                  Some (sum /. Float.of_int n)
              | `Last ->
                  (* newest is first in reversed list *)
                  Some (List.hd history).value
              | `None -> None
            in
            Option.iter
              (fun v -> Hashtbl.replace summary_table key (`Float v))
              auto)
    metric_def_table;
  let summary = sorted_of_hashtbl summary_table in
  let media =
    sorted_of_hashtbl media_table
    |> List.map (fun (key, entries) -> (key, List.rev entries))
  in
  let base_provenance =
    Json_utils.json_mem "provenance" manifest_json |> provenance_of_json
  in
  {
    root;
    id;
    dir;
    experiment_name = experiment;
    name = Json_utils.json_mem "name" manifest_json |> Json_utils.json_string;
    group = Json_utils.json_mem "group" manifest_json |> Json_utils.json_string;
    parent_id =
      Json_utils.json_mem "parent_id" manifest_json |> Json_utils.json_string;
    started_at =
      Option.value
        (Json_utils.json_mem "started_at" manifest_json
        |> Json_utils.json_number)
        ~default:0.0;
    ended_at = !ended_at;
    status = !status;
    provenance = { base_provenance with notes = !notes };
    tags = List.rev !tags;
    params;
    summary;
    latest_metrics;
    histories;
    metric_defs;
    media;
    input_artifacts = List.rev !input_artifacts;
    output_artifacts = List.rev !output_artifacts;
  }

let load ~root ~experiment ~id =
  let path = manifest_path root experiment id in
  if not (Sys.file_exists path) then None
  else
    try
      let json = Fs.read_file path |> Json_utils.json_of_string in
      let schema_ok =
        match
          Json_utils.json_mem "schema_version" json |> Json_utils.json_number
        with
        | Some value -> int_of_float value = schema_version
        | None -> false
      in
      if not schema_ok then None
      else
        let dir = Filename.dirname path in
        Some (materialize root experiment id dir json)
    with _ -> None

let list ~root ~experiment ?status:status_filter ?tag ?parent
    ?group:group_filter () =
  let runs_dir =
    Filename.concat
      (Filename.concat (Filename.concat root "experiments") experiment)
      "runs"
  in
  Fs.list_dirs runs_dir
  |> List.filter_map (fun id -> load ~root ~experiment ~id)
  |> List.filter (fun run ->
      Option.fold ~none:true ~some:(fun s -> status run = s) status_filter
      && Option.fold ~none:true
           ~some:(fun tag -> List.exists (String.equal tag) (tags run))
           tag
      && Option.fold ~none:true
           ~some:(fun parent -> parent_id run = Some parent)
           parent
      && Option.fold ~none:true ~some:(fun g -> group run = Some g) group_filter)
  |> List.sort (fun a b -> String.compare (id b) (id a))

let children t = list ~root:t.root ~experiment:t.experiment_name ~parent:t.id ()
