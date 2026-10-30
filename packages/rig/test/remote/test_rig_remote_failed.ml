(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The controller's own failure fails its job: a process of its own, since it
   fails the process for good. *)

open Windtrap
open Remote_job

(* Rig.fail loses the job's hosts and proxies, which fails the job: a connect
   at once finds it failed, and returns the process's failure from the first
   host's open. *)
let process_failed () =
  with_agents @@ fun agents ->
  let agent = List.hd agents in
  let j = connect agents in
  let host = List.hd (Rig_remote.hosts j) in
  let far = List.hd (mem host) in
  let why = "the test fails the process" in
  Rig.fail why;
  with_agents @@ fun fresh ->
  let first = List.hd fresh in
  (match Rig_remote.connect ~key:(as_key key) (List.map address fresh) with
  | Ok _ -> fail "a job of a failed process"
  | Error e ->
      starts_with ~msg:"names the first machine" ~prefix:(machine first) e;
      ends_with ~msg:"the process's failure" ~suffix:(": " ^ why) e);
  let cause = Option.get (Rig_remote.failed j) in
  let lost d = Printf.sprintf "%s lost: %s" (Rig.name d) why in
  (match List.find_opt (fun d -> cause = lost d) [ host; far ] with
  | Some _ -> ()
  | None -> failf "the job's cause %S names no host or proxy of it" cause);
  equal ~msg:"the job's agent" exit_w
    (2, [ "failed: controller aborted: " ^ cause ])
    (finish agent);
  equal ~msg:"the fresh agent's exit" int 2 (fst (finish first))

let () =
  Watchdog.start ();
  exit
    (run "rig_remote.failed"
       [
         group ~timeout:60. "rig_remote"
           [
             test "the controller's own failure fails its job, and connect's"
               process_failed;
           ];
       ])
