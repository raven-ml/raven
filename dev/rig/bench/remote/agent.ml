(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The agent of the copy rows and of lan.exe: [agent.exe HOST PORT] reads the
   job's key on standard input, listens at HOST and PORT (0: the system
   chooses), prints the port, and serves one job, its machine's host alone. It
   exits 0 once the job closed and 2 once it failed, printing why, and at once
   if the process that started it ends, so that a killed bench leaves no agent
   behind. *)

let fail why =
  prerr_endline ("agent.exe: " ^ why);
  exit 1

(* Exits once this process's parent is no longer the one that started it. *)
let orphaned () =
  let parent = Unix.getppid () in
  let rec watch () =
    Unix.sleepf 0.1;
    if Unix.getppid () <> parent then Unix._exit 3;
    watch ()
  in
  ignore (Thread.create watch ())

let () =
  match Array.to_list Sys.argv |> List.tl with
  | [ host; port ] -> (
      let key = In_channel.input_all stdin |> String.trim in
      let a =
        match Rig_remote.listen ~key host (int_of_string port) with
        | Ok a -> a
        | Error why -> fail why
      in
      orphaned ();
      Printf.printf "%d\n%!" (Rig_remote.port a);
      match Rig_remote.serve a [] with
      | Ok () -> exit 0
      | Error why ->
          prerr_endline ("agent.exe: " ^ why);
          exit 2)
  | _ -> fail "usage: agent.exe HOST PORT"
