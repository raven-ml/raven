(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Joins: agents that hold the dials they take, and two agents that cannot
   reach each other, in a process of its own, since what a failed join leaves
   of the process is no other test's state. *)

open Windtrap
open Remote_job

(* connect waits at most 10 s for each answer. *)
let answer_bound = 10.

(* Three agents after the first hold every agent's dial until told to let
   them through, so each dial that starts stays in flight: the [k]-th agent
   takes one from each agent before it. All six come before any passes, so
   each agent dials every later one at once; dialled one after another, an
   agent's second dial would wait for its first. *)
let held () =
  with_key_file @@ fun file ->
  let gates = List.init 3 (fun _ -> Unix.pipe ~cloexec:true ()) in
  let holding =
    List.map (fun (r, _) -> start ~mode:"held" ~input:r file) gates
  in
  List.iter (fun (r, _) -> Unix.close r) gates;
  let agents = start file :: holding in
  Fun.protect ~finally:(fun () -> List.iter kill agents) @@ fun () ->
  let r = ref None in
  let t =
    Thread.create
      (fun () ->
        r :=
          Some (Rig_remote.connect ~key:(as_key key) (List.map address agents)))
      ()
  in
  List.iteri
    (fun i a ->
      for k = 1 to i + 1 do
        let msg = Printf.sprintf "%s's dial %d" (machine a) k in
        equal ~msg string "dialled" (input_line a.out)
      done)
    holding;
  List.iter
    (fun (_, w) ->
      ignore (Unix.write_substring w "go\n" 0 3);
      Unix.close w)
    gates;
  Thread.join t;
  match Option.get !r with
  | Error why -> failf "connect: %s" why
  | Ok j ->
      Rig_remote.close j;
      List.iter
        (fun a -> equal ~msg:(machine a) exit_w (0, [ "closed" ]) (finish a))
        agents

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
            r :=
              Some
                (Rig_remote.connect ~key:(as_key key)
                   [ address a1; address a2 ]))
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
             test "a join dials its later agents at once" held;
             test "two agents that cannot reach each other fail the join"
               unjoined;
           ];
       ])
