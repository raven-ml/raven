(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The root cause reaches every process of a failed job, also while the
   connection is busy: a process of its own, since a failed job fails the
   process for good. *)

open Windtrap
open Remote_job
module Polled = Rig_support.Polled

let root = "OWN lost: the controller's device faulted"

(* A rail from this process whose ready count runs far ahead keeps the
   connection's sending thread sending when the job fails. *)
let busy_connection () =
  let d, p = Polled.open_ "OWN" in
  with_agents @@ fun agents ->
  let j = connect agents in
  let r =
    match Rig.capability (List.hd (Rig_remote.hosts j)) Rig_remote_abi.key with
    | Some (Rig_remote_abi.Host r) -> r
    | _ -> fail "a host's record is Host"
  in
  let t = { Rig_remote_abi.src = 0; dst = 0; length = 1 lsl 16 } in
  let rail = require_ok (r.rail None ~send:[||] ~receive:[| t |]) in
  let e = require_some rail.local in
  e.ready 1_000_000_000;
  Polled.fault p "the controller's device faulted";
  (try ignore (Rig.Buffer.create d 16) with Rig.Lost _ -> ());
  until ~what:"the job's failure" (fun () -> Rig_remote.failure j <> None);
  equal (option string) (Some root) (Rig_remote.failure j);
  equal
    (pair int (list string))
    (2, [ "failed: " ^ root ])
    (finish (List.hd agents))

let () =
  Watchdog.start ();
  exit
    (run "rig_remote.abort"
       [
         group ~timeout:60. "rig_remote"
           [
             test "the root cause reaches the agent while the link is busy"
               busy_connection;
           ];
       ])
