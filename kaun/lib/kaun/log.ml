(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* ───── Backend Abstraction ───── *)

(* A backend handles the actual writing of events to files. This allows
   supporting multiple output formats (JSONL, TensorBoard, etc.) *)

type writer = {
  write_scalar : step:int -> epoch:int option -> tag:string -> float -> unit;
  write_hparams : (string * Yojson.Basic.t) list -> unit;
  close : unit -> unit;
}

type backend = run_dir:string -> writer

(* ───── JSONL Backend ───── *)

let jsonl_writer ~run_dir =
  let events_path = Filename.concat run_dir "events.jsonl" in
  let channel = open_out_gen [ Open_append; Open_creat ] 0o644 events_path in
  let mutex = Mutex.create () in
  let write_line line =
    Mutex.lock mutex;
    Fun.protect
      ~finally:(fun () -> Mutex.unlock mutex)
      (fun () ->
        output_string channel line;
        output_char channel '\n';
        flush channel)
  in
  {
    write_scalar =
      (fun ~step ~epoch ~tag value ->
        let epoch_field =
          Option.map (fun e -> [ ("epoch", `Int e) ]) epoch
          |> Option.value ~default:[]
        in
        let event =
          `Assoc
            ([
               ("type", `String "scalar");
               ("step", `Int step);
               ("wall_time", `Float (Unix.gettimeofday ()));
               ("tag", `String tag);
               ("value", `Float value);
             ]
            @ epoch_field)
        in
        write_line (Yojson.Basic.to_string event));
    write_hparams =
      (fun params ->
        let event =
          `Assoc
            [
              ("type", `String "hparams");
              ("wall_time", `Float (Unix.gettimeofday ()));
              ("params", `Assoc params);
            ]
        in
        write_line (Yojson.Basic.to_string event));
    close =
      (fun () ->
        Mutex.lock mutex;
        Fun.protect
          ~finally:(fun () -> Mutex.unlock mutex)
          (fun () -> close_out channel));
  }

let jsonl : backend = jsonl_writer

(* ───── TensorBoard Backend ───── *)

(* Writes a simple CSV format that can be visualized or imported into TensorBoard.
   For full TensorBoard protobuf format, a dedicated library would be needed. *)

let tensorboard_writer ~run_dir =
  let csv_path = Filename.concat run_dir "scalars.csv" in
  let channel = open_out_gen [ Open_append; Open_creat ] 0o644 csv_path in
  let mutex = Mutex.create () in
  let header_written = ref false in
  let write_line line =
    Mutex.lock mutex;
    Fun.protect
      ~finally:(fun () -> Mutex.unlock mutex)
      (fun () ->
        if not !header_written then (
          output_string channel "wall_time,step,epoch,tag,value\n";
          header_written := true);
        output_string channel line;
        output_char channel '\n';
        flush channel)
  in
  {
    write_scalar =
      (fun ~step ~epoch ~tag value ->
        let epoch_str = Option.fold ~none:"" ~some:string_of_int epoch in
        let line =
          Printf.sprintf "%.6f,%d,%s,%s,%.6f" (Unix.gettimeofday ()) step epoch_str tag
            value
        in
        write_line line);
    write_hparams =
      (fun params ->
        (* TensorBoard CSV doesn't have a standard hparams format, skip *)
        let _ = params in
        ());
    close =
      (fun () ->
        Mutex.lock mutex;
        Fun.protect
          ~finally:(fun () -> Mutex.unlock mutex)
          (fun () -> close_out channel));
  }

let tensorboard : backend = tensorboard_writer

(* ───── Multi Backend ───── *)

let multi_writer backends ~run_dir =
  let writers = List.map (fun backend -> backend ~run_dir) backends in
  {
    write_scalar =
      (fun ~step ~epoch ~tag value ->
        List.iter (fun w -> w.write_scalar ~step ~epoch ~tag value) writers);
    write_hparams =
      (fun params -> List.iter (fun w -> w.write_hparams params) writers);
    close = (fun () -> List.iter (fun w -> w.close ()) writers);
  }

let multi backends : backend = multi_writer backends

(* ───── Run ID and Directory Management ───── *)

let generate_run_id ?experiment () =
  let now = Unix.gettimeofday () in
  let tm = Unix.localtime now in
  let date =
    Printf.sprintf "%04d-%02d-%02d_%02d-%02d-%02d" (tm.Unix.tm_year + 1900)
      (tm.Unix.tm_mon + 1) tm.Unix.tm_mday tm.Unix.tm_hour tm.Unix.tm_min
      tm.Unix.tm_sec
  in
  Option.fold ~none:date ~some:(Printf.sprintf "%s_%s" date) experiment

let ensure_dir_recursive path =
  let rec ensure acc parts =
    match parts with
    | [] -> ()
    | part :: rest ->
        let next = if acc = "" then part else acc ^ "/" ^ part in
        if next <> "" && not (Sys.file_exists next) then Unix.mkdir next 0o755;
        ensure next rest
  in
  ensure "" (String.split_on_char '/' path)

(* ───── Run Manifest ───── *)

(* The manifest (run.json) stores immutable run metadata written once at creation.
   Separating this from the event stream avoids re-parsing events for run info. *)

let write_manifest ~run_dir ~run_id ?experiment ~tags ~config () =
  let experiment_field =
    Option.map (fun e -> ("experiment", `String e)) experiment |> Option.to_list
  in
  let manifest =
    `Assoc
      ([
         ("schema_version", `Int 1);
         ("run_id", `String run_id);
         ("created_at", `Float (Unix.gettimeofday ()));
         ("tags", `List (List.map (fun t -> `String t) tags));
         ("config", `Assoc config);
       ]
      @ experiment_field)
  in
  let path = Filename.concat run_dir "run.json" in
  let oc = open_out path in
  output_string oc (Yojson.Basic.pretty_to_string manifest);
  output_char oc '\n';
  close_out oc

(* ───── Logger Type ───── *)

type t = { run_id : string; run_dir : string; writer : writer }

(* ───── Public API ───── *)

let create ?(backend = jsonl) ?(base_dir = "./runs") ?experiment ?(tags = [])
    ?(config = []) () =
  let run_id = generate_run_id ?experiment () in
  let run_dir = Filename.concat base_dir run_id in
  ensure_dir_recursive run_dir;
  write_manifest ~run_dir ~run_id ?experiment ~tags ~config ();
  let writer = backend ~run_dir in
  let t = { run_id; run_dir; writer } in
  (* Auto-log config as hparams *)
  if config <> [] then writer.write_hparams config;
  t

let run_id t = t.run_id
let run_dir t = t.run_dir
let close t = t.writer.close ()

let log_scalar t ~step ~epoch ~tag value =
  t.writer.write_scalar ~step ~epoch:(Some epoch) ~tag value

let log_scalars t ~step ~epoch metrics =
  List.iter (fun (tag, value) -> log_scalar t ~step ~epoch ~tag value) metrics

let log_metrics t ~step ~epoch ~prefix collection =
  let metrics = Metrics.Collection.compute collection in
  List.iter
    (fun (name, value) ->
      let tag = prefix ^ "/" ^ name in
      log_scalar t ~step ~epoch ~tag value)
    metrics

let log_hparams t params = t.writer.write_hparams params
