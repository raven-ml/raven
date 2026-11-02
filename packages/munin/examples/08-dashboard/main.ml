(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The dashboard over a store made for it: a sweep of learning rates, one run
   whose loss turns NaN, one that failed, and one that trains while the
   dashboard shows it. Quitting ends that run and removes the store. *)

open Munin

let steps = 3_000

(* A loss that falls faster with a larger learning rate, with some noise. *)
let loss ~lr step =
  (2.0 *. exp (-.lr *. Float.of_int step)) +. 0.1 +. Random.float 0.05

let start store ~name ~lr =
  let s =
    Session.start ~store ~experiment:"demo" ~name ~steps
      ~params:[ ("lr", `Float lr) ]
      ()
  in
  (s, Session.metric s ~goal:`Minimize "loss")

(* Logs the loss of steps [first] to [last], every 10 steps. *)
let train m ~lr ?(nan_from = max_int) first last =
  for step = first to last do
    if step mod 10 = 0 then
      let v = if step >= nan_from then Float.nan else loss ~lr step in
      Session.log m ~step v
  done

let sweep store =
  List.iter
    (fun lr ->
      let s, m = start store ~name:(Printf.sprintf "lr-%g" lr) ~lr in
      train m ~lr 0 steps;
      Session.finish s)
    [ 0.0005; 0.001; 0.002 ];
  let s, m = start store ~name:"diverged" ~lr:0.004 in
  train m ~lr:0.004 ~nan_from:400 0 steps;
  Session.finish s;
  let s, m = start store ~name:"out-of-memory" ~lr:0.003 in
  train m ~lr:0.003 0 800;
  Session.finish ~status:(`Failed "out of memory") s

(* Logs a step every 40 ms until [stop] is set or the run reaches [steps]. *)
let live m ~lr stop =
  let step = ref 0 in
  while (not (Atomic.get stop)) && !step <= steps do
    train m ~lr !step !step;
    incr step;
    Thread.delay 0.04
  done

let rec remove path =
  if Sys.is_directory path then begin
    Array.iter (fun f -> remove (Filename.concat path f)) (Sys.readdir path);
    Sys.rmdir path
  end
  else Sys.remove path

let () =
  let root = Filename.temp_dir "munin-dashboard" "" in
  let store = Store.open_ ~root () in
  sweep store;
  let s, m = start store ~name:"training" ~lr:0.0015 in
  Session.sample s ~every:1.0 [ Munin_sys.host (); Munin_sys.process () ];
  let stop = Atomic.make false in
  let trainer = Thread.create (live m ~lr:0.0015) stop in
  let quit () =
    Atomic.set stop true;
    Thread.join trainer;
    Session.finish ~status:`Killed s;
    remove root
  in
  match Fun.protect ~finally:quit (fun () -> Munin_tui.run store Runs) with
  | Ok () -> ()
  | Error why ->
      prerr_endline ("08-dashboard: " ^ why);
      exit 1
