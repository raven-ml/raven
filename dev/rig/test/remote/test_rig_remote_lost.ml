(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A device lost in one agent fails the job: a process of its own, since a
   failed job fails the process for good. *)

open Windtrap
open Remote_job

let root = "FAULTY:0 lost: the agent's device faulted"

(* The FAULTY device's first allocation faults, in the first agent. The job
   fails with that loss as its root cause; every device here is lost with it,
   both agents end with it, and the process starts no other job. *)
let agent_device_lost () =
  with_agents ~n:2 @@ fun agents ->
  let j = connect agents in
  let hs = Rig_remote.hosts j in
  let others = List.concat_map mem hs in
  let faulty =
    match Rig_remote.devices (List.hd hs) "FAULTY" with
    | Ok [ d ] -> d
    | Ok _ -> fail "one FAULTY device"
    | Error why -> fail why
  in
  (try ignore (Rig.Buffer.create faulty 16) with _ -> ());
  until ~what:"the job's failure" (fun () -> Rig_remote.failure j <> None);
  equal (option string) (Some root) (Rig_remote.failure j);
  List.iter
    (fun d ->
      raises_match ~msg:(Rig.name d)
        (fun e -> lost_why e = Some root)
        (fun () -> Rig.Buffer.create d 16))
    (hs @ others);
  (* The agent whose device faulted reports that loss, and the other ends with
     the job (test_rig_remote_abort holds whether the cause reaches it). *)
  (match List.map finish agents with
  | [ first; (code, _) ] ->
      equal ~msg:"the faulted agent"
        (pair int (list string))
        (2, [ "failed: " ^ root ])
        first;
      equal ~msg:"the other agent's exit" int 2 code
  | _ -> fail "two agents");
  let failed = require_some (Rig.failure ()) in
  with_agents @@ fun fresh ->
  equal (result pass string) (Error failed)
    (Rig_remote.connect ~key (List.map address fresh))

let () =
  Watchdog.start ();
  exit
    (run "rig_remote.lost"
       [
         group ~timeout:60. "rig_remote"
           [
             test "a device lost in an agent fails the job with its loss"
               agent_device_lost;
           ];
       ])
