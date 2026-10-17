(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* An agent's process killed fails the job: a process of its own, since rig
   keeps the first loss of the job's devices here for good (Rig.failure).

   The controller and the other agent both see the death. Each test stops one of
   them while the agent dies, so the other reports first. Each process records
   the death as its own cause, which names the killed agent's machine, or as
   the cause another process relayed in its abort. *)


open Windtrap
open Remote_job

(* The process's report of [cause], its own or relayed. *)
let reports ~msg cause = function
  | 2, [ line ] ->
      starts_with ~msg ~affix:"failed: " line;
      ends_with ~msg ~affix:cause line
  | code, lines ->
      failf "%s exited %d, printing [%s]" msg code (String.concat "; " lines)

let stopped p f =
  if Sys.win32 then
    skip ~reason:"Windows has no signal that stops a process" ();
  Unix.kill p.pid Sys.sigstop;
  Fun.protect ~finally:(fun () -> Unix.kill p.pid Sys.sigcont) f

(* The other agent is stopped: this process, the controller, notices first.
   Every host of the job is lost with the root cause. *)
let controller_first () =
  with_agents ~n:2 @@ fun agents ->
  let j = connect agents in
  let victim = List.hd agents and other = List.nth agents 1 in
  stopped other (fun () ->
      kill victim;
      until ~what:"the job's failure" (fun () -> Rig_remote.failed j <> None));
  let root = Option.get (Rig_remote.failed j) in
  starts_with ~msg:"names the killed agent's machine" ~affix:(machine victim)
    root;
  List.iter
    (fun h ->
      raises_match ~msg:(Rig.name h)
        (fun e -> lost_why e = Some root)
        (fun () -> Rig.Buffer.create h 16))
    (Rig_remote.hosts j);
  reports ~msg:"the other agent" root (finish other)

(* The controller, support/controller.exe, is stopped: the other agent notices
   first. *)
let agent_first () =
  with_key_file @@ fun file ->
  let agents = List.init 2 (fun _ -> start file) in
  let c = start_controller file "watch" agents in
  Fun.protect
    ~finally:(fun () -> List.iter kill (c :: agents))
    (fun () ->
      let victim = List.hd agents and other = List.nth agents 1 in
      equal ~msg:"the controller" string "connected" (input_line c.out);
      let why =
        stopped c (fun () ->
            kill victim;
            match finish other with
            | 2, [ line ] -> line
            | code, lines ->
                failf "the other agent exited %d, printing [%s]" code
                  (String.concat "; " lines))
      in
      starts_with ~msg:"names the killed agent's machine"
        ~affix:("failed: " ^ machine victim)
        why;
      let cause = String.sub why 8 (String.length why - 8) in
      reports ~msg:"the controller" cause (finish c))

let () =
  Watchdog.start ();
  exit
    (run "rig_remote.killed"
       [
         group ~timeout:60. "rig_remote"
           [
             test "an agent's death, seen first by another agent" agent_first;
             test "an agent's death, seen first by the controller"
               controller_first;
           ];
       ])
