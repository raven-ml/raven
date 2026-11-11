(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The dashboard over a store made for it: a sweep of learning rates, one run
   whose loss turns NaN, one that failed, and one that trains while the
   dashboard shows it. Each run logs what a classifier's training logs: its
   training loss and gradient norm, the learning rate, throughput and epoch
   every 10 steps, and a validation loss and accuracy every 100. The live run
   also samples the machine, the process and a made-up GPU. Quitting ends the
   live run and removes the store. *)

open Munin

let steps = 3_000
let warmup = 200
let steps_per_epoch = 469

type metrics = {
  train_loss : Session.metric;
  grad_norm : Session.metric;
  lr : Session.metric;
  throughput : Session.metric;
  epoch : Session.metric;
  val_loss : Session.metric;
  val_acc : Session.metric;
}

let start store ~name ~lr =
  let s =
    Session.start ~store ~experiment:"demo" ~name ~steps
      ~params:[ ("lr", `Float lr); ("batch", `Int 128) ]
      ()
  in
  let m = Session.metric s in
  ( s,
    {
      train_loss = m ~goal:`Minimize "train/loss";
      grad_norm = m "train/grad_norm";
      lr = m "train/lr";
      throughput = m "train/samples_per_s";
      epoch = m "epoch";
      val_loss = m ~goal:`Minimize "val/loss";
      val_acc = m ~goal:`Maximize "val/acc";
    } )

(* Noise in [-a, a], heavier near zero. *)
let noise a = a *. (Random.float 1. +. Random.float 1. -. 1.)

(* A linear warmup, then a cosine decay to zero. *)
let schedule ~lr step =
  if step < warmup then lr *. Float.of_int step /. Float.of_int warmup
  else
    let t = Float.of_int (step - warmup) /. Float.of_int (steps - warmup) in
    lr *. 0.5 *. (1. +. cos (Float.pi *. t))

(* Logs step [step]: the training metrics every 10 steps, validation every
   100. From [nan_from] the losses are NaN and the gradient norm infinite. *)
let log_step s m ~lr ?(nan_from = max_int) step =
  let progress = lr *. Float.of_int step in
  let diverged = step >= nan_from in
  if step mod 10 = 0 then begin
    let train_loss =
      if diverged then Float.nan
      else (2.0 *. exp (-.progress)) +. 0.08 +. noise 0.06
    in
    let grad_norm =
      if diverged then Float.infinity
      else
        let g = 0.4 +. (3.0 *. exp (-.progress)) +. noise 0.15 in
        if Random.int 150 = 0 then 4. *. g else g
    in
    Session.log_metrics s ~step
      [
        (m.train_loss, train_loss);
        (m.grad_norm, grad_norm);
        (m.lr, schedule ~lr step);
        (m.throughput, 1450. +. noise 60.);
        (m.epoch, Float.of_int step /. Float.of_int steps_per_epoch);
      ]
  end;
  if step mod 100 = 0 then
    let val_loss =
      if diverged then Float.nan
      else (2.0 *. exp (-0.9 *. progress)) +. 0.14 +. noise 0.01
    in
    let val_acc = 0.97 -. (0.87 *. exp (-1.1 *. progress)) +. noise 0.004 in
    Session.log_metrics s ~step [ (m.val_loss, val_loss); (m.val_acc, val_acc) ]

let train s m ~lr ?nan_from first last =
  for step = first to last do
    log_step s m ~lr ?nan_from step
  done

let sweep store =
  List.iter
    (fun lr ->
      let s, m = start store ~name:(Printf.sprintf "lr-%g" lr) ~lr in
      train s m ~lr 0 steps;
      Session.finish s)
    [ 0.0005; 0.001; 0.002 ];
  let s, m = start store ~name:"diverged" ~lr:0.004 in
  train s m ~lr:0.004 ~nan_from:400 0 steps;
  Session.finish s;
  let s, m = start store ~name:"out-of-memory" ~lr:0.003 in
  train s m ~lr:0.003 0 800;
  Session.finish ~status:(`Failed "out of memory") s

(* A GPU that this demo makes up, busy while the live run trains. A real
   program reads its devices through rig; munin reads none itself. *)
let gib = 1024 * 1024 * 1024

let gpu () =
  {
    Munin_sys.unknown with
    memory = Some (24 * gib);
    used = Some ((14 * gib) + Random.full_int gib);
    busy = Some (90. +. noise 8.);
    temperature = Some (71. +. noise 3.);
    power = Some (230. +. noise 25.);
  }

(* Logs a step every 40 ms until [stop] is set or the run reaches [steps]. *)
let live s m ~lr stop =
  let step = ref 0 in
  while (not (Atomic.get stop)) && !step <= steps do
    log_step s m ~lr !step;
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
  Session.sample s ~every:1.0
    [ Munin_sys.host (); Munin_sys.process (); Munin_sys.device "gpu0" gpu ];
  let stop = Atomic.make false in
  let trainer = Thread.create (live s m ~lr:0.0015) stop in
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
