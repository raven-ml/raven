(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Times of one program on an agent from two domains, in a suite of its own: its
   job, which its cases share, is the process's one until the suite ends. *)

open Windtrap
open Remote_job
open Remote_program
module G = Rig_program

(* Times of one program on an agent from two domains each answer their own
   run's durations, within the host's window around their call, and leave
   their run's bytes. A large time's fill is timed, at least [large / per_ns]
   ns: one that answered another run's times, or left its fill untimed (0),
   falls short of it, as a fill of 8 groups does unless the machine stalls
   it. A program makes at most one call before its branches, a load, and
   none after: each call is round trips to the agent, and calls on one
   domain are [times]'s. Each run loads its own program, so no program
   carries state from one run to the next. About half the cases load
   nothing and make no call, so 60 cases time about 30 programs.

   Under load each round trip waits for the agent's threads and this
   process's to be scheduled, so a program makes few of them one after the
   other. The programs share the job's Polled devices, which the agent opens
   once, and a pool of inputs: a time zeroes its pair with fills that rig
   orders before the run by their memory, and reads the pair back with two
   copies that it waits for together. *)

(* The job's Polled devices and host, and the pairs of inputs that no time
   holds. A time makes a pair only when none is idle, so there are at most
   two, one per domain. *)
type pool = {
  ds : Rig.t array;
  lock : Mutex.t;
  mutable idle : (B.t * B.t) list;
}

let remote =
  fixture
    ~teardown:(fun (j, a, _) ->
      Rig_remote.close j;
      kill a)
    (fun () ->
      let a = with_key_file start in
      let j = connect [ a ] in
      let h = List.hd (Rig_remote.hosts j) in
      let ds = with_host (Array.of_list (polled h)) in
      (j, a, { ds; lock = Mutex.create (); idle = [] }))

let timed = abstract "p"

let load_timed () =
  let _, _, pool = remote () in
  let p =
    require_ok ~pp:Format.pp_print_string
      (G.load (unlooped (archs pool.ds)) (Iarray.of_array pool.ds))
  in
  (p, pool)

let take pool =
  Mutex.protect pool.lock @@ fun () ->
  match pool.idle with
  | io :: rest ->
      pool.idle <- rest;
      Some io
  | [] -> None

let give pool io =
  Mutex.protect pool.lock (fun () -> pool.idle <- io :: pool.idle)

(* The bytes of [x] and [c], through two copies to this process. *)
let read_pair x c =
  let hx = B.create Rig.host (B.length x) in
  let hc = B.create Rig.host (B.length c) in
  let px = B.submit_copy ~src:x ~dst:hx [| B.row (B.length x) |] in
  let pc = B.submit_copy ~src:c ~dst:hc [| B.row (B.length c) |] in
  Rig.Point.wait px;
  Rig.Point.wait pc;
  (read_host hx, read_host hc)

let time_law ~least = function
  | Error e -> failf "raised %s" (Printexc.to_string e)
  | Ok (times, window, inputs) ->
      window_law ~least times window;
      equal ~msg:"the inputs" (pair string string) expected inputs

(* A time of [p] whose fill runs [groups] groups, on a pair of inputs from
   [pool] that it zeroes first. *)
let time_with groups (p, pool) =
  let x, c =
    match take pool with
    | Some io -> io
    | None -> (B.create pool.ds.(1) size, B.create pool.ds.(0) 8)
  in
  Fun.protect ~finally:(fun () -> give pool (x, c)) @@ fun () ->
  ignore (B.submit_fill x '\000' : Rig.Point.t);
  ignore (B.submit_fill c '\000' : Rig.Point.t);
  let f = { G.inputs = [| x; c |]; ints = [| groups; 0 |] } in
  let t0 = Rig.Profile.now () in
  let times = G.time p f in
  let window = Rig.Profile.now () - t0 in
  (times, window, read_pair x c)

let timed_commands =
  [
    command "load" (Gen.unit @-> makes timed) (fun () -> ()) load_timed;
    command "time"
      (timed ^-> judges (triple (array int) int (pair string string)))
      (fun () -> time_law ~least:0)
      (time_with 8);
    command "time large"
      (timed ^-> judges (triple (array int) int (pair string string)))
      (fun () -> time_law ~least:(large / per_ns))
      (time_with large);
  ]

let () =
  Watchdog.start ();
  exit
    (run "rig_remote.times"
       [
         (* A hang fails by name, with room left in the watchdog's 100 s. *)
         stateful ~timeout:60. ~count:60 ~steps:1 ~domains:2
           "times of one program on an agent from two domains" timed_commands;
       ])
