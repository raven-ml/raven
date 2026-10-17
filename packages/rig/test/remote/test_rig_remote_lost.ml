(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A device lost in one agent fails the job: a process of its own, since rig
   keeps the first loss of the job's devices here for good (Rig.failure). *)

open Windtrap
open Remote_job

let root = "FAULTY:0 lost: the agent's device faulted"

(* The FAULTY device's first allocation faults in the agent. The job fails
   with that loss as the agent's root cause, which this process records from
   the agent's abort, marked as the agent's. devices answers that cause for a
   kind opened before or not, every device of the job is lost with it, a rail
   is its Error, close leaves it failed, and the agent ends with the job. *)
let agent_device_lost () =
  with_agents @@ fun agents ->
  let agent = List.hd agents in
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
  until ~what:"the job's failure" (fun () -> Rig_remote.failed j <> None);
  let cause = Option.get (Rig_remote.failed j) in
  equal ~msg:"the cause" string (machine agent ^ " aborted: " ^ root) cause;
  List.iter
    (fun kind ->
      match Rig_remote.devices (List.hd hs) kind with
      | Ok _ -> failf "devices of %s after the failure" kind
      | Error why -> equal ~msg:("devices of " ^ kind) string cause why)
    [ "MEM"; "POLLED" ];
  List.iter
    (fun d ->
      raises_match ~msg:(Rig.name d)
        (fun e -> lost_why e = Some cause)
        (fun () -> Rig.Buffer.create d 16))
    (hs @ others);
  (match Rig.capability (List.hd hs) Rig_remote_abi.key with
  | Some (Rig_remote_abi.Host r) ->
      let t = { Rig_remote_abi.src = 0; dst = 0; length = 1 } in
      equal ~msg:"a rail after the failure" (result pass string) (Error cause)
        (r.rail None ~send:[| t |] ~receive:[||])
  | _ -> fail "a host's record is Host");
  (* close on a failed job returns at once, and leaves it failed. *)
  Rig_remote.close j;
  equal ~msg:"after close" (option string) (Some cause) (Rig_remote.failed j);
  equal ~msg:"the agent" exit_w (2, [ "failed: " ^ root ]) (finish agent)

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
