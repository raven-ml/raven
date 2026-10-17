(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A job fails on the controller only by a loss of its own hosts and
   proxies: a loss of another device of the controller, or of an earlier
   job's, reaches nothing of it. A process of its own, since rig keeps a
   process's first failure for good (Rig.failure), which no test here may
   take for the job's. A start that finds a job open first fails it if a
   loss fails it, and raises otherwise, so [still_open] tells at once
   whether a loss failed the job, without waiting for its watcher. *)

open Windtrap
open Remote_job
module Polled = Rig_support.Polled

(* Raises unless [j] is open: a second job's start fails [j] first if a loss
   on the controller fails it, then raises while [j] stays open. *)
let still_open j agents =
  raises_match ~msg:"a second job while the first is open"
    (function Invalid_argument _ -> true | _ -> false)
    (fun () ->
      ignore (Rig_remote.connect ~key:(as_key key) (List.map address agents)));
  equal ~msg:"the job's failure" (option string) None (Rig_remote.failed j)

(* A loss of one of the controller's own devices loses that device alone, and
   the job goes on. *)
let own_device_lost () =
  let d, p = Polled.open_ "OWN" in
  with_job @@ fun j agents ->
  let far = List.hd (mem (List.hd (Rig_remote.hosts j))) in
  Polled.fault p "the controller's device faulted";
  (try ignore (Rig.Buffer.create d 16) with Rig.Lost _ -> ());
  equal ~msg:"OWN lost" bool true (Option.is_some (Rig.lost d));
  still_open j agents;
  equal ~msg:"a round trip in the job" string "abc"
    (read (far_of_string far "abc"))

(* A job failed by an agent's death loses its hosts and proxies with the root
   cause, and the controller's own devices go on: it starts another job. *)
let job_failed () =
  let d, _ = Polled.open_ "OTHER" in
  with_agents ~n:2 @@ fun agents ->
  let j = connect agents in
  let hs = Rig_remote.hosts j in
  let far = List.hd (mem (List.hd hs)) in
  kill (List.hd agents);
  until ~what:"the job's failure" (fun () -> Rig_remote.failed j <> None);
  let root = Option.get (Rig_remote.failed j) in
  List.iter
    (fun h ->
      raises_match ~msg:(Rig.name h)
        (fun e -> lost_why e = Some root)
        (fun () -> Rig.Buffer.create h 16))
    (far :: hs);
  equal ~msg:"OTHER's loss" (option string) None (Rig.lost d);
  equal ~msg:"a round trip on OTHER" string "abc"
    (read (far_of_string d "abc"));
  with_job @@ fun j' _ ->
  let far' = List.hd (mem (List.hd (Rig_remote.hosts j'))) in
  equal ~msg:"a round trip in the next job" string "abc"
    (read (far_of_string far' "abc"))

(* A proxy of a failed job, first used once the next job runs, is lost then,
   and the next job goes on. *)
let old_proxy_lost () =
  with_agents ~n:2 @@ fun agents ->
  let j = connect agents in
  let old = List.hd (mem (List.hd (Rig_remote.hosts j))) in
  kill (List.hd agents);
  until ~what:"the job's failure" (fun () -> Rig_remote.failed j <> None);
  with_job @@ fun j' agents' ->
  let far' = List.hd (mem (List.hd (Rig_remote.hosts j'))) in
  raises_match ~msg:"the old proxy"
    (fun e -> Option.is_some (lost_why e))
    (fun () -> Rig.Buffer.create old 16);
  still_open j' agents';
  equal ~msg:"a round trip in the next job" string "abc"
    (read (far_of_string far' "abc"))

let () =
  Watchdog.start ();
  exit
    (run "rig_remote.own"
       [
         group ~timeout:60. "rig_remote"
           [
             test "a loss of a device of the controller leaves the job going"
               own_device_lost;
             test "a failed job leaves the controller's devices going"
               job_failed;
             test "an earlier job's proxy lost leaves the next job going"
               old_proxy_lost;
           ];
       ])
