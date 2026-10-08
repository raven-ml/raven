(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A device of the controller lost fails its job: a process of its own, since a
   failed job fails the process for good. *)

open Windtrap
open Remote_job
module Polled = Rig_support.Polled

let root = "OWN lost: the controller's device faulted"

(* The process notices its own loss within a second. *)
let notice_within = 1.5

let own_device_lost () =
  let d, p = Polled.open_ "OWN" in
  with_agents @@ fun agents ->
  let j = connect agents in
  let far = List.hd (mem (List.hd (Rig_remote.hosts j))) in
  Polled.fault p "the controller's device faulted";
  (try ignore (Rig.Buffer.create d 16) with Rig.Lost _ -> ());
  let t0 = Unix.gettimeofday () in
  until ~what:"the job's failure" (fun () -> Rig_remote.failure j <> None);
  let took = Unix.gettimeofday () -. t0 in
  equal (option string) (Some root) (Rig_remote.failure j);
  less ~msg:"seconds to notice" float_exact ~than:notice_within took;
  raises_match
    (fun e -> lost_why e = Some root)
    (fun () -> Rig.Buffer.create far 16);
  (* The agent ends with the job. That it learns the root cause is
     test_rig_remote_abort's, which holds the race this test leaves open. *)
  equal ~msg:"the agent's exit" int 2 (fst (finish (List.hd agents)));
  let failed = require_some (Rig.failure ()) in
  with_agents @@ fun fresh ->
  equal (result pass string) (Error failed)
    (Rig_remote.connect ~key (List.map address fresh))

let () =
  Watchdog.start ();
  exit
    (run "rig_remote.own"
       [
         group ~timeout:60. "rig_remote"
           [
             test
               "a device of the controller lost fails the job within a second"
               own_device_lost;
           ];
       ])
