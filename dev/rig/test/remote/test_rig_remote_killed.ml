(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* An agent's process killed fails the job: a process of its own, since a failed
   job fails the process for good. *)

open Windtrap
open Remote_job

(* Every device here is lost with the root cause, which names the killed agent's
   machine, and the other agent ends with the job. *)
let agent_killed () =
  with_agents ~n:2 @@ fun agents ->
  let j = connect agents in
  let victim = List.hd agents and other = List.nth agents 1 in
  kill victim;
  until ~what:"the job's failure" (fun () -> Rig_remote.failure j <> None);
  let root = Option.get (Rig_remote.failure j) in
  starts_with ~msg:"names the killed agent's machine" ~affix:(machine victim)
    root;
  List.iter
    (fun h ->
      raises_match ~msg:(Rig.name h)
        (fun e -> lost_why e = Some root)
        (fun () -> Rig.Buffer.create h 16))
    (Rig_remote.hosts j);
  (* The other agent may notice the loss itself before the abort comes, and
     names its peer by its place in the job. *)
  match finish other with
  | 2, [ why ] -> ends_with ~affix:"closed its connection" why
  | code, lines ->
      failf "the other agent exited %d, printing [%s]" code
        (String.concat "; " lines)

let () =
  Watchdog.start ();
  exit
    (run "rig_remote.killed"
       [
         group ~timeout:60. "rig_remote"
           [ test "an agent's process killed fails the job" agent_killed ];
       ])
