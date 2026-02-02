(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* ───── Backend Abstraction ───── *)

(* A backend handles the actual writing of events. This allows supporting
   multiple output formats (JSONL via kaun_runlog, TensorBoard CSV, etc.) *)

type writer = {
  write_scalar : step:int -> epoch:int option -> tag:string -> float -> unit;
  close : unit -> unit;
}

type backend = run:Kaun_runlog.Run.t -> writer

(* ───── JSONL Backend ───── *)

let jsonl_writer ~run =
  let mutex = Mutex.create () in
  {
    write_scalar =
      (fun ~step ~epoch ~tag value ->
        let event =
          Kaun_runlog.Event.Scalar
            { step; epoch; tag; value; wall_time = Unix.gettimeofday () }
        in
        Mutex.lock mutex;
        Fun.protect
          ~finally:(fun () -> Mutex.unlock mutex)
          (fun () -> Kaun_runlog.Run.append_event run event));
    close = (fun () -> ());
  }

let jsonl : backend = jsonl_writer

(* ───── TensorBoard Backend ───── *)

(* Writes a simple CSV format that can be visualized or imported into
   TensorBoard. For full TensorBoard protobuf format, a dedicated library would
   be needed. *)

let tensorboard_writer ~run =
  let run_dir = Kaun_runlog.Run.dir run in
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
          Printf.sprintf "%.6f,%d,%s,%s,%.6f" (Unix.gettimeofday ()) step
            epoch_str tag value
        in
        write_line line);
    close =
      (fun () ->
        Mutex.lock mutex;
        Fun.protect
          ~finally:(fun () -> Mutex.unlock mutex)
          (fun () -> close_out channel));
  }

let tensorboard : backend = tensorboard_writer

(* ───── Multi Backend ───── *)

let multi_writer backends ~run =
  let writers = List.map (fun backend -> backend ~run) backends in
  {
    write_scalar =
      (fun ~step ~epoch ~tag value ->
        List.iter (fun w -> w.write_scalar ~step ~epoch ~tag value) writers);
    close = (fun () -> List.iter (fun w -> w.close ()) writers);
  }

let multi backends : backend = multi_writer backends

(* ───── Logger Type ───── *)

type t = { run : Kaun_runlog.Run.t; writer : writer }

(* ───── Public API ───── *)

let create ?(backend = jsonl) ?base_dir ?experiment ?(tags = []) ?(config = [])
    () =
  let run = Kaun_runlog.create_run ?base_dir ?experiment ~tags ~config () in
  let writer = backend ~run in
  { run; writer }

let run_id t = Kaun_runlog.Run.run_id t.run
let run_dir t = Kaun_runlog.Run.dir t.run
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
