(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let err_not_resumable = "Munin.Session.resume: run is not resumable"
let err_missing_manifest = "Munin.Session.run: missing run manifest"
let err_closed_session = "Munin.Session.log_artifact: closed session"
let err_path_missing = "Munin.Session.log_artifact: path does not exist: "
let err_media_missing = "Munin.Session.log_media: path does not exist: "

type t = {
  root : string;
  experiment : string;
  id : string;
  dir : string;
  mutex : Mutex.t;
  mutable closed : bool;
}

let schema_version = 2
let manifest_path dir = Filename.concat dir "run.json"
let events_path dir = Filename.concat dir "events.jsonl"
let random_state = lazy (Random.State.make_self_init ())

let generate_id () =
  let state = Lazy.force random_state in
  let now = Unix.gettimeofday () in
  let tm = Unix.localtime now in
  let stamp =
    Printf.sprintf "%04d-%02d-%02d_%02d-%02d-%02d" (tm.Unix.tm_year + 1900)
      (tm.Unix.tm_mon + 1) tm.Unix.tm_mday tm.Unix.tm_hour tm.Unix.tm_min
      tm.Unix.tm_sec
  in
  let suffix = Printf.sprintf "%04x" (Random.State.int state 0x10000) in
  stamp ^ "_" ^ suffix

let status_to_string = function
  | `Finished -> "finished"
  | `Failed -> "failed"
  | `Killed -> "killed"

let root_of_run_dir dir =
  Filename.dirname (Filename.dirname (Filename.dirname (Filename.dirname dir)))

let with_lock t f =
  Mutex.lock t.mutex;
  Fun.protect ~finally:(fun () -> Mutex.unlock t.mutex) f

let append_event t event =
  Fs.append_line (events_path t.dir) (Event_log.encode event)

let write_manifest path json =
  Fs.write_file path (Json_utils.json_to_string ~pretty:true json ^ "\n")

let optional_field key f = function None -> [] | Some v -> [ (key, f v) ]

(* [notes] is run metadata but shares the manifest's "provenance" object; see
   Run.notes_of_json. *)
let provenance_json ?notes (p : Provenance.t) =
  Json_utils.json_obj
    ([
       ("command", Jsont.Json.list (List.map Jsont.Json.string p.command));
       ("cwd", Jsont.Json.string p.cwd);
       ("pid", Jsont.Json.int p.pid);
       ( "env",
         Json_utils.json_obj
           (List.map (fun (k, v) -> (k, Jsont.Json.string v)) p.env) );
     ]
    @ optional_field "notes" Jsont.Json.string notes
    @ optional_field "hostname" Jsont.Json.string p.hostname
    @ optional_field "git_commit" Jsont.Json.string p.git_commit
    @ optional_field "git_dirty" Jsont.Json.bool p.git_dirty)

let make_manifest ~id ~experiment ~started_at ?name ?group ?parent ~tags ~params
    ~provenance () =
  Json_utils.json_obj
    ([
       ("schema_version", Jsont.Json.int schema_version);
       ("id", Jsont.Json.string id);
       ("experiment", Jsont.Json.string experiment);
       ("started_at", Jsont.Json.number started_at);
       ("tags", Jsont.Json.list (List.map Jsont.Json.string tags));
       ( "params",
         Json_utils.json_obj
           (List.map (fun (k, v) -> (k, Value.to_json v)) params) );
       ("provenance", provenance);
     ]
    @ optional_field "name" Jsont.Json.string name
    @ optional_field "group" Jsont.Json.string group
    @ optional_field "parent_id"
        (fun run -> Jsont.Json.string (Run.id run))
        parent)

let start ?store ?name ?group ?parent ?(tags = []) ?(params = []) ?notes
    ?provenance ~experiment () =
  let store = match store with Some store -> store | None -> Store.open_ () in
  let root = Store.root store in
  let id = generate_id () in
  let dir =
    Filename.concat
      (Filename.concat
         (Filename.concat (Filename.concat root "experiments") experiment)
         "runs")
      id
  in
  Fs.ensure_dir dir;
  let provenance =
    match provenance with Some p -> p | None -> Provenance.detect ()
  in
  let started_at = Unix.gettimeofday () in
  let parent_id = Option.map Run.id parent in
  let manifest =
    make_manifest ~id ~experiment ~started_at ?name ?group ?parent ~tags ~params
      ~provenance:(provenance_json ?notes provenance)
      ()
  in
  write_manifest (manifest_path dir) manifest;
  Index.add root ~id
    { experiment; name; group; parent_id; status = `Running; tags; started_at };
  { root; experiment; id; dir; mutex = Mutex.create (); closed = false }

let finish ?(status = `Finished) t =
  with_lock t (fun () ->
      if not t.closed then (
        append_event t
          (Event_log.Finished
             {
               status = status_to_string status;
               ended_at = Unix.gettimeofday ();
             });
        Index.update_status t.root ~id:t.id (status :> Index.status);
        t.closed <- true))

let with_run ?store ?name ?group ?parent ?tags ?params ?notes ?provenance
    ~experiment f =
  let session =
    start ?store ?name ?group ?parent ?tags ?params ?notes ?provenance
      ~experiment ()
  in
  match f session with
  | value ->
      finish session;
      value
  | exception exn ->
      finish ~status:`Failed session;
      raise exn

let resume run =
  if not (Run.resumable run) then invalid_arg err_not_resumable;
  let root = root_of_run_dir (Run.dir run) in
  let t =
    {
      root;
      experiment = Run.experiment run;
      id = Run.id run;
      dir = Run.dir run;
      mutex = Mutex.create ();
      closed = false;
    }
  in
  append_event t (Event_log.Resumed { at = Unix.gettimeofday () });
  Index.update_status root ~id:(Run.id run) `Running;
  t

let id t = t.id
let dir t = t.dir

let run t =
  match Run.load ~root:t.root ~experiment:t.experiment ~id:t.id with
  | Some run -> run
  | None -> failwith err_missing_manifest

let set_notes t notes =
  with_lock t (fun () ->
      if not t.closed then append_event t (Event_log.Notes notes))

let metric t ?summary ?goal ?step_metric key =
  let summary =
    match (summary, goal) with
    | Some summary, _ -> summary
    | None, Some `Minimize -> `Min
    | None, Some `Maximize -> `Max
    | None, None -> `Last
  in
  let step_metric = Option.map Metric.key step_metric in
  with_lock t (fun () ->
      if not t.closed then
        append_event t
          (Event_log.Define_metric { key; summary; step_metric; goal }));
  Metric.make ~key ~append:(fun ~step ~timestamp value ->
      with_lock t (fun () ->
          if not t.closed then
            append_event t (Event_log.Metric { step; timestamp; key; value })))

let log_metrics t ~step ?timestamp pairs =
  with_lock t (fun () ->
      if not t.closed then
        let timestamp =
          Option.value timestamp ~default:(Unix.gettimeofday ())
        in
        List.iter
          (fun (m, value) ->
            append_event t
              (Event_log.Metric { step; timestamp; key = Metric.key m; value }))
          pairs)

let rel_path_of ~run_dir abs_path =
  String.sub abs_path
    (String.length run_dir + 1)
    (String.length abs_path - String.length run_dir - 1)

let media_dest_path run_dir key step ext =
  let parts = String.split_on_char '/' key in
  let dir_parts, leaf =
    match List.rev parts with
    | [] -> ([], "media")
    | [ single ] -> ([], single)
    | last :: rest -> (List.rev rest, last)
  in
  let media_dir =
    List.fold_left Filename.concat (Filename.concat run_dir "media") dir_parts
  in
  let filename = Printf.sprintf "%s_%d%s" leaf step ext in
  (media_dir, Filename.concat media_dir filename)

let log_media t ~step ~key ~kind ~path =
  with_lock t (fun () ->
      if not t.closed then begin
        if not (Sys.file_exists path) then invalid_arg (err_media_missing ^ path);
        let ext = Filename.extension path in
        let media_dir, dest = media_dest_path t.dir key step ext in
        Fs.ensure_dir media_dir;
        Fs.copy_file path dest;
        let timestamp = Unix.gettimeofday () in
        append_event t
          (Event_log.Media
             {
               step;
               timestamp;
               key;
               kind;
               path = rel_path_of ~run_dir:t.dir dest;
             })
      end)

let log_table t ~step ~key ~columns ~rows =
  with_lock t (fun () ->
      if not t.closed then begin
        let json =
          Json_utils.json_obj
            [
              ("columns", Jsont.Json.list (List.map Jsont.Json.string columns));
              ( "rows",
                Jsont.Json.list
                  (List.map
                     (fun row -> Jsont.Json.list (List.map Value.to_json row))
                     rows) );
            ]
        in
        let media_dir, dest = media_dest_path t.dir key step ".json" in
        Fs.ensure_dir media_dir;
        Fs.write_file dest (Json_utils.json_to_string ~pretty:true json ^ "\n");
        let timestamp = Unix.gettimeofday () in
        append_event t
          (Event_log.Media
             {
               step;
               timestamp;
               key;
               kind = `Table;
               path = rel_path_of ~run_dir:t.dir dest;
             })
      end)

let set_summary t values =
  with_lock t (fun () ->
      if not t.closed then append_event t (Event_log.Summary values))

let add_tags t tags =
  with_lock t (fun () ->
      if (not t.closed) && tags <> [] then append_event t (Event_log.Tags tags))

let log_artifact t ~name ~kind ~path ?(metadata = []) ?(aliases = []) () =
  with_lock t (fun () ->
      if t.closed then failwith err_closed_session;
      if not (Sys.file_exists path) then invalid_arg (err_path_missing ^ path);
      let digest = Fs.sha256_path path in
      let blob_rel_path =
        Filename.concat (Filename.concat "blobs" "sha256") digest
      in
      let blob_abs_path = Filename.concat t.root blob_rel_path in
      if not (Sys.file_exists blob_abs_path) then
        Fs.copy_tree path blob_abs_path;
      let payload : Artifact.payload =
        if Fs.is_directory path then `Dir else `File
      in
      let json_metadata =
        List.map (fun (k, v) -> (k, Value.to_json v)) metadata
      in
      let artifact =
        Artifact.create ~root:t.root ~name ~kind ~payload ~digest
          ~path:blob_rel_path ~metadata:json_metadata ~aliases
          ~producer_run_id:(Some t.id)
      in
      append_event t
        (Event_log.Artifact_output { name; version = Artifact.version artifact });
      artifact)

let use_artifact t artifact =
  with_lock t (fun () ->
      if not t.closed then (
        Artifact.add_consumer ~root:t.root ~name:(Artifact.name artifact)
          ~version:(Artifact.version artifact)
          t.id;
        append_event t
          (Event_log.Artifact_input
             {
               name = Artifact.name artifact;
               version = Artifact.version artifact;
             })))
