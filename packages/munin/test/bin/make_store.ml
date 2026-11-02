(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The store of the CLI's tests, [make_store MODE ROOT ARGS...]:

   [build ROOT] records three runs and two artifact versions, and prints
   [base=ID], [wide=ID] and [sweep=ID] lines for the shell to eval.

   [crash ROOT] starts a run, logs steps 0 to 4, prints its id and exits without
   finishing it.

   [resume ROOT ID STEP] resumes the run ID at STEP, logs steps 3 and 4 again,
   and finishes.

   [hold ROOT] starts a run, prints its id, waits for the end of its standard
   input and exits without finishing the run.

   Samples carry fixed times, so that their output does not change from one test
   run to the next. Runs' start times and ids do. *)

open Munin

let provenance =
  {
    Provenance.command = [ "make_store"; "a\tb" ];
    cwd = "/";
    hostname = None;
    pid = 0;
    ocaml = "5.0.0";
    raven = "test";
    os = "test";
    arch = "test";
    cpu = None;
    git = Some { commit = String.make 40 'a'; dirty = false };
    seeds = [ ("init", 7) ];
    env = [];
  }

(* The first time is an epoch whose microseconds round down:
   1709385958.67742347... prints as .677423. *)
let time step =
  if step = 0 then 1709385958.6774235 else 1793541788.25 +. Float.of_int step

let start store ?name ?parent ?tags ?params experiment =
  Session.start ~store ?name ?parent ?tags ?params ~provenance ~experiment ()

let log s step pairs = Session.log_metrics s ~step ~time:(time step) pairs

(* [publish s ?step contents] publishes [contents] as a version of mnist-ckpt,
   from a file it removes after. *)
let publish s ?step contents =
  let path = "ckpt" in
  Out_channel.with_open_bin path (fun oc -> output_string oc contents);
  let v = Session.log_artifact s ?step ~name:"mnist-ckpt" path in
  Sys.remove path;
  v

let params lr =
  [
    ("lr", `Float lr);
    ("batch", `Int 64);
    ("opt", `String "adam");
    ("init", `Float Float.nan);
  ]

(* Losses with a NaN and an infinity, an accuracy whose best ties, and a metric
   without a goal; a reading; a summary; version 1 of mnist-ckpt. *)
let base store =
  let s =
    start store ~name:"base" ~tags:[ "baseline" ] ~params:(params 0.001) "mnist"
  in
  let loss = Session.metric s ~goal:`Minimize "train/loss" in
  let acc = Session.metric s ~goal:`Maximize "val/acc" in
  let lr = Session.metric s "lr" in
  log s 0 [ (loss, 2.); (acc, 0.5); (lr, 1e6) ];
  log s 1 [ (loss, Float.nan); (acc, 0.75); (lr, 4.58e-72) ];
  log s 2 [ (loss, 0.5); (acc, 0.75) ];
  log s 3 [ (loss, Float.infinity) ];
  log s 4 [ (loss, 0.25) ];
  Session.sample s ~every:3600. [ (fun () -> [ ("sys/load", 0.5) ]) ];
  Session.set_summary s
    [ ("test/acc", `Float 0.98); ("comment", `String "tab\there") ];
  let v1 = publish s ~step:4 "v1\n" in
  Session.finish s;
  (Session.id s, v1)

(* A loss that ties base's best, an accuracy that is never finite, a failed end,
   and version 2 of mnist-ckpt. *)
let wide store =
  let s = start store ~name:"wide" ~params:(params 0.01) "mnist" in
  let loss = Session.metric s ~goal:`Minimize "train/loss" in
  let acc = Session.metric s ~goal:`Maximize "val/acc" in
  log s 0 [ (loss, 1.); (acc, Float.nan) ];
  log s 1 [ (loss, 0.25); (acc, Float.nan) ];
  ignore (publish s "v2 longer\n");
  Session.finish ~status:(`Failed "diverged\nat step 1") s;
  Session.id s

let sweep store ~parent v1 =
  let s = start store ~parent ~tags:[ "sweep"; "night" ] "sweep" in
  Session.use_artifact s v1;
  Session.finish ~status:`Killed s;
  Session.id s

let build store =
  let base, v1 = base store in
  let wide = wide store in
  let sweep = sweep store ~parent:base v1 in
  Printf.printf "base=%s\nwide=%s\nsweep=%s\n" base wide sweep

let crash store =
  let s = start store ~name:"resumed" "mnist" in
  let loss = Session.metric s ~goal:`Minimize "train/loss" in
  List.iteri (fun step v -> log s step [ (loss, v) ]) [ 5.; 4.; 3.; 2.; 1. ];
  print_endline (Session.id s)

let resume store id step =
  let run = Option.get (Store.run store id) in
  let s = Session.resume ~provenance run ~step in
  let loss = Session.metric s ~goal:`Minimize "train/loss" in
  log s 3 [ (loss, 2.5) ];
  log s 4 [ (loss, 2.25) ];
  Session.finish s

let hold store =
  let s = start store "held" in
  print_endline (Session.id s);
  ignore (In_channel.input_all stdin)

let () =
  (* Windows ends a line with CR LF on a text channel, which the shell would
     keep in the ids it reads. *)
  set_binary_mode_out stdout true;
  let store = Store.open_ ~root:Sys.argv.(2) () in
  match Sys.argv.(1) with
  | "build" -> build store
  | "crash" -> crash store
  | "resume" -> resume store Sys.argv.(3) (int_of_string Sys.argv.(4))
  | "hold" -> hold store
  | m -> failwith ("make_store: unknown mode " ^ m)
