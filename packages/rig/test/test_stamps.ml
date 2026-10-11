(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Stamps against a model: what a submit and a Buffer.wait wait for, on one
   domain and on two, and a replay over two run copies, as a linked program
   runs. Polled devices run their queues only when a wait sleeps or the test
   runs them, so a wait that returns early leaves its work unrun. *)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module P = Rig_support.Polled
module Support = Rig_support

let submit ?(buffers = [||]) ?waits s =
  Rig.submit ?waits s ~run:(Sub.Run.make ()) ~buffers

let timeout = 120.

let pp_access ppf = function
  | Rig.Read -> Format.pp_print_string ppf "Read"
  | Rig.Read_write -> Format.pp_print_string ppf "Read_write"

let access = Gen.of_list ~pp:pp_access [ Rig.Read; Rig.Read_write ]

(* Two devices and one memory *)

(* Each value of the API is two devices and one memory of the first. Devices go
   back to a pool when their program ends, and the next program counts their
   values from where they stood, once their earlier work ran. *)
type system = { devices : (Rig.t * P.t) array; base : int array; m : B.t }

let pool = Mutex.create ()
let free = ref []
let opened = Atomic.make 0

let take () =
  match
    Mutex.protect pool (fun () ->
        match !free with
        | ds :: rest ->
            free := rest;
            Some ds
        | [] -> None)
  with
  | Some ds -> ds
  | None ->
      let n = Atomic.fetch_and_add opened 1 in
      Array.init 2 (fun i -> P.open_ (Printf.sprintf "stamps:%d-%d" n i))

let give t = Mutex.protect pool (fun () -> free := t.devices :: !free)
let device = Gen.of_list ~pp:Format.pp_print_int [ 0; 1 ]
let other i = 1 - i

(* The model: the values each device assigned, the last write of the memory and
   each device's last use. *)
type model = {
  values : int array;
  mutable write : (int * int) option;
  uses : int array;
}

let pp_model ppf r =
  Format.fprintf ppf "values %d %d, write %s, uses %d %d" r.values.(0)
    r.values.(1)
    (match r.write with
    | None -> "none"
    | Some (i, v) -> Printf.sprintf "%d:%d" i v)
    r.uses.(0) r.uses.(1)

let memory = abstract ~pp:pp_model ~release:give "m"
let signaled t i = Rig.signaled (fst t.devices.(i)) - t.base.(i)
let open_model () = { values = [| 0; 0 |]; write = None; uses = [| 0; 0 |] }

let open_system () =
  let devices = take () in
  let base =
    Array.map
      (fun (d, _) ->
        let p = Rig.submit (Sub.make d [||]) ~run:(Sub.Run.make ()) ~buffers:[||] in
        Rig.Point.wait p;
        Rig.Point.value p)
      devices
  in
  { devices; base; m = B.create (fst devices.(0)) 64 }

(* A submit on device [i] that reads or writes the memory: its value, and the
   other device's word once the submit returned. Writing waits for the other
   device's every use, reading for its write. *)
let submit_system i access t =
  let d = fst t.devices.(i) in
  let use : Rig.access -> _ = function
    | Read -> Sub.read
    | Read_write -> Sub.write
  in
  let s = Sub.make d [| Support.touching [| use access |] |] in
  let m = require_some (B.borrow d t.m) in
  let v = Rig.Point.value (submit s ~buffers:[| m |]) - t.base.(i) in
  (v, signaled t (other i))

let submit_model i access r = function
  | Error e -> raise e
  | Ok (v, seen) ->
      let j = other i in
      equal ~msg:"value" int (r.values.(i) + 1) v;
      (match (access, r.write) with
      | Rig.Read, Some (w, at) when w = j ->
          at_least ~msg:"the other device's write" int ~than:at seen
      | Rig.Read, _ -> ()
      | Rig.Read_write, _ ->
          at_least ~msg:"the other device's use" int ~than:r.uses.(j) seen);
      r.values.(i) <- v;
      r.uses.(i) <- v;
      if access = Rig.Read_write then r.write <- Some (i, v)

(* A host access waits for the memory's last write, or for every use. *)
let wait_system access t =
  B.wait t.m access;
  (signaled t 0, signaled t 1)

let wait_model access r = function
  | Error e -> raise e
  | Ok (s0, s1) -> (
      let seen = [| s0; s1 |] in
      match access with
      | Rig.Read -> (
          match r.write with
          | None -> ()
          | Some (i, at) -> at_least ~msg:"the last write" int ~than:at seen.(i)
          )
      | Rig.Read_write ->
          at_least ~msg:"device 0's use" int ~than:r.uses.(0) s0;
          at_least ~msg:"device 1's use" int ~than:r.uses.(1) s1)

(* A copy of the memory into a fresh buffer of device [i] reads it: it waits for
   the memory's last write. *)
let copy_system i t =
  let dst = B.create (fst t.devices.(i)) 64 in
  B.copy ~src:t.m ~dst;
  (signaled t 0, signaled t 1)

let commands =
  [
    command "open" (Gen.unit @-> makes memory) open_model open_system;
    command "read"
      (device @-> memory ^-> judges (pair int int))
      (fun i -> submit_model i Rig.Read)
      (fun i -> submit_system i Rig.Read);
    (* A write excludes every other submit naming the memory, as its caller's
       exclusive claim does: its [pre] keeps it out of the parallel calls. *)
    command "write"
      ~pre:(fun _ _ -> true)
      (device @-> memory ^-> judges (pair int int))
      (fun i -> submit_model i Rig.Read_write)
      (fun i -> submit_system i Rig.Read_write);
    command "wait"
      (access @-> memory ^-> judges (pair int int))
      wait_model wait_system;
    command "copy"
      (device @-> memory ^-> judges (pair int int))
      (fun _ -> wait_model Rig.Read)
      copy_system;
    command "run"
      (device @-> memory ^-> returns unit)
      (fun _ _ -> ())
      (fun i t -> ignore (P.run (snd t.devices.(i))));
  ]

(* A replay over two run copies *)

(* A linked program's run [n] uses copy [n mod 2]: it waits for the copy's
   memory, which run [n - 2] used, writes run [n]'s arguments into it, and
   submits the copy's prepared submission. Its fill stores [n] where the
   arguments say, into run [n]'s output. The host goes on to the next run
   without waiting for the output. *)
type replay = {
  d : Rig.t;
  p : P.t;
  runs_storage : Sub.Run.t array;
  subs : Sub.t array;
  outs : (int, B.t) Hashtbl.t;
  points : (int, int) Hashtbl.t;
  mutable runs : int;
}

let poke =
  Sub.fill ~refs:[| Sub.write 0 ~at:0 |] Support.poke ~params:16

let replays = Atomic.make 0

let open_replay () =
  let name =
    Printf.sprintf "stamps:replay-%d" (Atomic.fetch_and_add replays 1)
  in
  let d, p = P.open_ name in
  let runs_storage = Array.init 2 (fun _ -> Sub.Run.make ()) in
  let subs =
    Array.init 2 (fun _ -> Sub.make d [| poke |])
  in
  {
    d;
    p;
    runs_storage;
    subs;
    outs = Hashtbl.create 8;
    points = Hashtbl.create 8;
    runs = 0;
  }

let run_replay t =
  let n = t.runs in
  let run = t.runs_storage.(n mod 2) and s = t.subs.(n mod 2) in
  cover "a copy reused before its last run ran"
    (n >= 2 && Rig.signaled t.d < Hashtbl.find t.points (n - 2));
  let out = B.create t.d 8 in
  B.wait out Rig.Read_write;
  Support.store (Support.Reader.host out) (-1);
  let block = Sub.block s 0 in
  Sub.Run.int64 run block 0 0;
  Sub.Run.int64 run block 8 n;
  let v = Rig.Point.value (Rig.submit s ~run ~buffers:[| out |]) in
  Hashtbl.replace t.outs n out;
  Hashtbl.replace t.points n v;
  t.runs <- n + 1

let output t n =
  let out = Hashtbl.find t.outs n in
  B.wait out Rig.Read;
  Support.load (Support.Reader.host out)

let replay = abstract "r"
let run_index = among int replay (fun runs -> List.init !runs Fun.id)
let run_command = command "run" (replay ^-> returns unit) incr run_replay

(* Runs are drawn twice as often as the rest, so the host runs ahead. *)
let replay_commands =
  [
    command "open" (Gen.unit @-> makes replay) (fun () -> ref 0) open_replay;
    run_command;
    run_command;
    command "device runs"
      (replay ^-> returns unit)
      (fun _ -> ())
      (fun t -> ignore (P.run t.p));
    command "output"
      (replay ^-> run_index ^-> returns int)
      (fun _ n -> n)
      output;
  ]

let tests =
  [
    group ~timeout "stamps"
      [
        stateful "submits and waits follow the memory's stamps" commands;
        stateful ~count:30 ~domains:2
          "stamps raised from two domains are their maxima" commands;
      ];
    group ~timeout "replay"
      [
        stateful ~steps:12 "a run copy is rewritten only after its last run ran"
          replay_commands;
      ];
  ]

let () = exit (run "rig.stamps" tests)
