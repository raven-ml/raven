(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type status = [ `Running | `Finished | `Failed | `Killed ]

type media_entry = {
  step : int;
  timestamp : float;
  kind : [ `Image | `Audio | `Table | `File ];
  path : string;
}

(* What the run declared when it started. It never changes afterwards, and it is
   what every fold of the event log starts from. *)
type manifest = {
  params : (string * Value.t) list;
  provenance : Provenance.t;
  notes : string option;
  tags : string list;
}

(* State folded from the event log. [consumed] is how many bytes of the log it
   accounts for, which is where [reload] resumes. Histories and media are kept
   newest first so that folding more events is a prepend; accessors reverse
   them. [summary_writes] is what the run wrote explicitly, [summary] adds the
   values computed from metric definitions on top. *)
type full = {
  manifest : manifest;
  consumed : int;
  status : status;
  tags : string list;
  notes : string option;
  ended_at : float option;
  last_event_at : float option;
  summary_writes : (string * Value.t) list;
  summary : (string * Value.t) list;
  histories : (string * Metric.sample list) list;
  metric_defs : (string * Metric.def) list;
  media : (string * media_entry list) list;
  input_artifacts : Artifact.t list;
  output_artifacts : Artifact.t list;
}

(* Header fields are always available without I/O *)
type t = {
  root : string;
  id : string;
  dir : string;
  experiment : string;
  name : string option;
  group : string option;
  parent_id : string option;
  started_at : float;
  status : status;
  tags : string list;
  full : full Lazy.t;
}

let schema_version = 2

(* Header accessors — no I/O *)

let id t = t.id
let dir t = t.dir
let experiment t = t.experiment
let name t = t.name
let group t = t.group
let parent_id t = t.parent_id
let started_at t = t.started_at
let status t = t.status
let tags t = t.tags
let resumable t = t.status = `Running

(* Full accessors — forces lazy on first access *)

let full t = Lazy.force t.full
let params t = (full t).manifest.params
let provenance t = (full t).manifest.provenance
let notes t = (full t).notes
let ended_at t = (full t).ended_at
let last_event_at t = (full t).last_event_at
let summary t = (full t).summary
let find_param t key = List.assoc_opt key (params t)
let find_summary t key = List.assoc_opt key (summary t)
let metric_keys t = List.map fst (full t).histories
let input_artifacts t = (full t).input_artifacts
let output_artifacts t = (full t).output_artifacts
let metric_defs t = (full t).metric_defs
let media_keys t = List.map fst (full t).media

let latest_metrics t =
  List.filter_map
    (fun (key, history) ->
      match history with [] -> None | sample :: _ -> Some (key, sample))
    (full t).histories

let media_history t key =
  match List.assoc_opt key (full t).media with
  | Some entries -> List.rev entries
  | None -> []

let metric_history t key =
  match List.assoc_opt key (full t).histories with
  | Some history -> List.rev history
  | None -> []

(* A goal is what makes one observation better than another: without one there
   is nothing to optimize, and so no best sample to report. *)
let best t key =
  match List.assoc_opt key (full t).metric_defs with
  | None | Some { Metric.goal = None; _ } -> None
  | Some { goal = Some goal; _ } -> (
      match List.assoc_opt key (full t).histories with
      | None | Some [] -> None
      | Some (first :: rest) ->
          let better (a : Metric.sample) (b : Metric.sample) =
            match goal with
            | `Minimize -> a.value < b.value
            | `Maximize -> a.value > b.value
          in
          Some
            (List.fold_left
               (fun best sample -> if better sample best then sample else best)
               first rest))

(* Paths *)

let run_dir ~root ~experiment id =
  Filename.concat
    (Filename.concat
       (Filename.concat (Filename.concat root "experiments") experiment)
       "runs")
    id

let manifest_path root experiment id =
  Filename.concat (run_dir ~root ~experiment id) "run.json"

let events_path dir = Filename.concat dir "events.jsonl"

(* Parsing helpers *)

let status_of_string = function
  | "finished" -> `Finished
  | "failed" -> `Failed
  | "killed" -> `Killed
  | _ -> `Running

(* [notes] is stored inside the manifest's "provenance" object but is run
   metadata, not provenance; it is read out separately. *)
let notes_of_json json =
  Json_utils.json_mem "notes" json |> Json_utils.json_string

let provenance_of_json json : Provenance.t =
  let env_json = Json_utils.json_mem "env" json in
  {
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

let manifest_of_json json =
  {
    params =
      Json_utils.json_mem "params" json
      |> Json_utils.json_assoc
      |> List.map (fun (key, value) -> (key, Value.of_json value));
    provenance = Json_utils.json_mem "provenance" json |> provenance_of_json;
    notes = Json_utils.json_mem "provenance" json |> notes_of_json;
    tags = Json_utils.json_mem "tags" json |> Json_utils.json_string_list;
  }

(* Fold state *)

(* Tables while folding, sorted association lists once frozen. Lists grow at the
   head, so they are reversed on the way out. *)
type acc = {
  root : string;
  dir : string;
  manifest : manifest;
  tag_seen : (string, unit) Hashtbl.t;
  histories : (string, Metric.sample list) Hashtbl.t;
  metric_defs : (string, Metric.def) Hashtbl.t;
  media : (string, media_entry list) Hashtbl.t;
  summary_writes : (string, Value.t) Hashtbl.t;
  input_seen : (string, unit) Hashtbl.t;
  output_seen : (string, unit) Hashtbl.t;
  mutable tags : string list;
  mutable input_artifacts : Artifact.t list;
  mutable output_artifacts : Artifact.t list;
  mutable notes : string option;
  mutable status : status;
  mutable ended_at : float option;
  mutable last_event_at : float option;
}

let sorted_of_hashtbl tbl =
  Hashtbl.to_seq tbl |> List.of_seq
  |> List.sort (fun (a, _) (b, _) -> String.compare a b)

let table_of_assoc entries =
  let tbl = Hashtbl.create (List.length entries) in
  List.iter (fun (key, value) -> Hashtbl.replace tbl key value) entries;
  tbl

let artifact_key artifact =
  Artifact.name artifact ^ ":" ^ Artifact.version artifact

let add_tag acc tag =
  if not (Hashtbl.mem acc.tag_seen tag) then begin
    Hashtbl.replace acc.tag_seen tag ();
    acc.tags <- tag :: acc.tags
  end

let empty_acc ~root ~dir manifest =
  {
    root;
    dir;
    manifest;
    tag_seen = Hashtbl.create 8;
    histories = Hashtbl.create 16;
    metric_defs = Hashtbl.create 8;
    media = Hashtbl.create 8;
    summary_writes = Hashtbl.create 8;
    input_seen = Hashtbl.create 8;
    output_seen = Hashtbl.create 8;
    tags = [];
    input_artifacts = [];
    output_artifacts = [];
    notes = manifest.notes;
    status = `Running;
    ended_at = None;
    last_event_at = None;
  }

(* Fold state for a log that has not been read yet. *)
let acc_of_manifest ~root ~dir manifest =
  let acc = empty_acc ~root ~dir manifest in
  List.iter (add_tag acc) manifest.tags;
  acc

(* Fold state that resumes where [f] stopped. *)
let acc_of_full ~root ~dir (f : full) =
  let acc =
    {
      (empty_acc ~root ~dir f.manifest) with
      histories = table_of_assoc f.histories;
      metric_defs = table_of_assoc f.metric_defs;
      media = table_of_assoc f.media;
      summary_writes = table_of_assoc f.summary_writes;
      tags = List.rev f.tags;
      input_artifacts = List.rev f.input_artifacts;
      output_artifacts = List.rev f.output_artifacts;
      notes = f.notes;
      status = f.status;
      ended_at = f.ended_at;
      last_event_at = f.last_event_at;
    }
  in
  List.iter (fun tag -> Hashtbl.replace acc.tag_seen tag ()) f.tags;
  List.iter
    (fun a -> Hashtbl.replace acc.input_seen (artifact_key a) ())
    f.input_artifacts;
  List.iter
    (fun a -> Hashtbl.replace acc.output_seen (artifact_key a) ())
    f.output_artifacts;
  acc

let fold_event acc = function
  | Event_log.Metric { step; timestamp; key; value } ->
      let history =
        match Hashtbl.find_opt acc.histories key with
        | Some history -> history
        | None -> []
      in
      Hashtbl.replace acc.histories key
        ({ Metric.step; timestamp; value } :: history);
      acc.last_event_at <- Some timestamp
  | Define_metric { key; summary; step_metric; goal } ->
      Hashtbl.replace acc.metric_defs key { Metric.summary; step_metric; goal }
  | Media { step; timestamp; key; kind; path } ->
      let entry =
        { step; timestamp; kind; path = Filename.concat acc.dir path }
      in
      let entries =
        match Hashtbl.find_opt acc.media key with
        | Some entries -> entries
        | None -> []
      in
      Hashtbl.replace acc.media key (entry :: entries);
      acc.last_event_at <- Some timestamp
  | Summary values ->
      List.iter
        (fun (key, value) -> Hashtbl.replace acc.summary_writes key value)
        values
  | Notes value -> acc.notes <- value
  | Tags values -> List.iter (add_tag acc) values
  | Artifact_output { name; version } ->
      let key = name ^ ":" ^ version in
      if not (Hashtbl.mem acc.output_seen key) then begin
        Hashtbl.replace acc.output_seen key ();
        match Artifact.load ~root:acc.root ~name ~version with
        | Some artifact ->
            acc.output_artifacts <- artifact :: acc.output_artifacts
        | None -> ()
      end
  | Artifact_input { name; version } ->
      let key = name ^ ":" ^ version in
      if not (Hashtbl.mem acc.input_seen key) then begin
        Hashtbl.replace acc.input_seen key ();
        match Artifact.load ~root:acc.root ~name ~version with
        | Some artifact ->
            acc.input_artifacts <- artifact :: acc.input_artifacts
        | None -> ()
      end
  | Resumed _ ->
      acc.ended_at <- None;
      acc.status <- `Running
  | Finished { status; ended_at } ->
      acc.status <- status_of_string status;
      acc.ended_at <- Some ended_at

(* [history] is newest first and non-empty. *)
let auto_summary (def : Metric.def) (history : Metric.sample list) =
  match def.summary with
  | `Min ->
      Some
        (List.fold_left
           (fun acc (m : Metric.sample) -> Float.min acc m.value)
           Float.infinity history)
  | `Max ->
      Some
        (List.fold_left
           (fun acc (m : Metric.sample) -> Float.max acc m.value)
           Float.neg_infinity history)
  | `Mean ->
      let sum =
        List.fold_left
          (fun acc (m : Metric.sample) -> acc +. m.value)
          0. history
      in
      Some (sum /. Float.of_int (List.length history))
  | `Last -> Some (List.hd history).value
  | `None -> None

let freeze acc ~consumed =
  let histories = sorted_of_hashtbl acc.histories in
  let metric_defs = sorted_of_hashtbl acc.metric_defs in
  let summary_writes = sorted_of_hashtbl acc.summary_writes in
  (* Metric definitions fill in the summary entries the run did not write
     itself. Both lists are key-sorted, so merging keeps them so. *)
  let auto =
    List.filter_map
      (fun (key, def) ->
        if List.mem_assoc key summary_writes then None
        else
          match Hashtbl.find_opt acc.histories key with
          | None | Some [] -> None
          | Some history ->
              auto_summary def history
              |> Option.map (fun value -> (key, `Float value)))
      metric_defs
  in
  {
    manifest = acc.manifest;
    consumed;
    status = acc.status;
    tags = List.rev acc.tags;
    notes = acc.notes;
    ended_at = acc.ended_at;
    last_event_at = acc.last_event_at;
    summary_writes;
    summary =
      List.merge (fun (a, _) (b, _) -> String.compare a b) summary_writes auto;
    histories;
    metric_defs;
    media = sorted_of_hashtbl acc.media;
    input_artifacts = List.rev acc.input_artifacts;
    output_artifacts = List.rev acc.output_artifacts;
  }

(* Materialize full data from manifest JSON + event log *)
let materialize ~root ~dir manifest_json =
  let acc = acc_of_manifest ~root ~dir (manifest_of_json manifest_json) in
  let events, consumed = Event_log.read_from (events_path dir) ~pos:0 in
  List.iter (fold_event acc) events;
  freeze acc ~consumed

let reload t =
  let previous = full t in
  let events, consumed =
    Event_log.read_from (events_path t.dir) ~pos:previous.consumed
  in
  let folded =
    if consumed = previous.consumed then previous
    else
      (* A log shorter than what we consumed was truncated or replaced, so the
         read restarted at its beginning and the fold restarts too. *)
      let acc =
        if consumed < previous.consumed then
          acc_of_manifest ~root:t.root ~dir:t.dir previous.manifest
        else acc_of_full ~root:t.root ~dir:t.dir previous
      in
      List.iter (fold_event acc) events;
      freeze acc ~consumed
  in
  {
    t with
    status = folded.status;
    tags = folded.tags;
    full = Lazy.from_val folded;
  }

(* Full eager load — reads manifest + events immediately *)
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
        let dir = run_dir ~root ~experiment id in
        let name = Json_utils.json_mem "name" json |> Json_utils.json_string in
        let group =
          Json_utils.json_mem "group" json |> Json_utils.json_string
        in
        let parent_id =
          Json_utils.json_mem "parent_id" json |> Json_utils.json_string
        in
        let started_at =
          Option.value
            (Json_utils.json_mem "started_at" json |> Json_utils.json_number)
            ~default:0.0
        in
        let folded = materialize ~root ~dir json in
        Some
          {
            root;
            id;
            dir;
            experiment;
            name;
            group;
            parent_id;
            started_at;
            status = folded.status;
            tags = folded.tags;
            full = Lazy.from_val folded;
          }
    with _ -> None

(* Build from already-known header fields — reads manifest + events only when
   full data is accessed *)
let of_header ~root ~id ~experiment ~name ~group ~parent_id ~status ~tags
    ~started_at =
  let dir = run_dir ~root ~experiment id in
  let full =
    lazy
      (let path = manifest_path root experiment id in
       materialize ~root ~dir (Fs.read_file path |> Json_utils.json_of_string))
  in
  {
    root;
    id;
    dir;
    experiment;
    name;
    group;
    parent_id;
    started_at;
    status;
    tags;
    full;
  }

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
