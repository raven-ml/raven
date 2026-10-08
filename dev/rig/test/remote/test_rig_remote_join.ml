(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Two agents that cannot reach each other: a process of its own, since what a
   failed join leaves of the process is no other test's state. *)

open Windtrap
open Remote_job

(* connect waits at most 10 s for each answer. *)
let answer_bound = 10.

(* The second agent stops listening once the controller reached it, and answers
   nothing, so the first cannot join it. If connect has not returned within the
   bound and a second, the test ends the second agent to end the wait. *)
let unjoined () =
  with_key_file @@ fun file ->
  let a1 = start file and a2 = start ~mode:"deaf" file in
  Fun.protect
    ~finally:(fun () -> List.iter kill [ a1; a2 ])
    (fun () ->
      let r = ref None in
      let t0 = Unix.gettimeofday () in
      let t =
        Thread.create
          (fun () ->
            r := Some (Rig_remote.connect ~key [ address a1; address a2 ]))
          ()
      in
      while !r = None && Unix.gettimeofday () -. t0 < answer_bound +. 1. do
        Thread.delay 0.05
      done;
      let took = Unix.gettimeofday () -. t0 in
      if !r = None then kill a2;
      Thread.join t;
      less ~msg:"seconds connect took" float_exact ~than:(answer_bound +. 1.)
        took;
      match Option.get !r with
      | Ok j ->
          Rig_remote.close j;
          fail "a job whose agents cannot join"
      | Error why ->
          starts_with ~affix:"127.0.0.1:" why;
          contains ~msg:"names the first agent" ~sub:(machine a1) why;
          contains ~msg:"names the second agent" ~sub:(machine a2) why;
          equal ~msg:"the first agent, told the job failed" int 2
            (fst (finish a1)))

let () =
  Watchdog.start ();
  exit
    (run "rig_remote.join"
       [
         group ~timeout:60. "rig_remote"
           [
             xfail
               ~reason:
                 "connect waits for an unanswered join past the first agent's \
                  failure"
               (test "two agents that cannot reach each other fail the join"
                  unjoined);
           ];
       ])
