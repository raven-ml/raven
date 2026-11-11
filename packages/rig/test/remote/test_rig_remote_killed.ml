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
      starts_with ~msg ~prefix:"failed: " line;
      ends_with ~msg ~suffix:cause line
  | code, lines ->
      failf "%s exited %d, printing [%s]" msg code (String.concat "; " lines)

(* Runs [f] with [p], a child, stopped, and resumes it after. A process of
   several threads stops only once one of them takes the signal, and until then
   the others run and may notice the death first. The wait returns once all of
   them stopped. *)
let stopped p f =
  if Sys.win32 then
    skip ~reason:"Windows has no signal that stops a process" ();
  Unix.kill p.pid Sys.sigstop;
  (match Unix.waitpid [ Unix.WUNTRACED ] p.pid with
  | _, Unix.WSTOPPED _ -> ()
  | _, Unix.WEXITED n ->
      p.status <- Some (n, []);
      failf "process %d exited %d before it stopped" p.pid n
  | _, Unix.WSIGNALED n ->
      p.status <- Some (-n, []);
      failf "process %d died of %s before it stopped" p.pid
        (Sys.signal_to_string n));
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
  starts_with ~msg:"names the killed agent's machine" ~prefix:(machine victim)
    root;
  List.iter
    (fun h ->
      raises_match ~msg:(Rig.name h)
        (fun e -> lost_why e = Some root)
        (fun () -> Rig.Buffer.create h 16))
    (Rig_remote.hosts j);
  reports ~msg:"the other agent" root (finish other)

(* The controller, support/controller.exe, is stopped: the other agent notices
   first. The other agent reports its failure on its standard output as soon
   as it has its root cause, and its serve returns only once the controller
   has ended its side of their connection, so the controller resumes once
   the report has come. *)
let agent_first () =
  with_key_file @@ fun file ->
  let victim = start file in
  let other = start ~vars:[ ("RIG_REMOTE_REPORT", "1") ] file in
  let c = start_controller file "watch" [ victim; other ] in
  Fun.protect
    ~finally:(fun () -> List.iter kill [ c; victim; other ])
    (fun () ->
      equal ~msg:"the controller" string "connected" (input_line c.out);
      let report =
        stopped c (fun () ->
            kill victim;
            input_line other.out)
      in
      starts_with ~msg:"names the killed agent's machine"
        ~prefix:("failed " ^ machine victim)
        report;
      let cause = String.sub report 7 (String.length report - 7) in
      reports ~msg:"the other agent" cause (finish other);
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
