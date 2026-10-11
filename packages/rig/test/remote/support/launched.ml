(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A launched program for the suites: [launched.exe MODE] calls
   Rig_remote.launched, with the variables its launcher set, and prints what it
   sees on its standard output, which carries no report. It prints "none" and
   exits 0 without the variables, and "error: WHY" and exits 1 if launched
   answers an error. Otherwise it prints "connected" and its hosts' names, then:

   - "close" closes the job and exits 0; - "exit" exits 0 without closing it; -
   "watch" waits until the process failed, then kills itself with SIGKILL, so
   that nothing written later can reach its launcher; - "fork" forks a child
   that exits 0, then prints "going on" if its job did not fail, and closes it;
   - "exec" runs [sh -c SCRIPT], SCRIPT its second argument, then closes the
   job.

   In mode "open", its second argument a PORT, it first connects to the agent
   at the loopback's PORT with the launcher's key, then calls launched with
   that job open and prints "raised" if it raised Invalid_argument, closes the
   job and exits 0. *)

let patience = 60.

let rec failed t0 =
  match Rig.failure () with
  | Some _ -> ()
  | None when Unix.gettimeofday () -. t0 > patience -> exit 3
  | None ->
      Unix.sleepf 0.01;
      failed t0

let run mode j =
  match mode with
  | "close" -> Rig_remote.close j
  | "exit" -> ()
  | "watch" ->
      failed (Unix.gettimeofday ());
      Unix.kill (Unix.getpid ()) Sys.sigkill
  | "fork" -> (
      match Unix.fork () with
      | 0 -> exit 0
      | child ->
          ignore (Unix.waitpid [] child);
          if Rig_remote.failed j = None then print_endline "going on";
          Rig_remote.close j)
  | "exec" ->
      let sh = "/bin/sh" in
      let pid =
        Unix.create_process sh
          [| sh; "-c"; Sys.argv.(2) |]
          Unix.stdin Unix.stdout Unix.stderr
      in
      ignore (Unix.waitpid [] pid);
      Rig_remote.close j
  | m -> failwith ("no mode " ^ m)

let open_first port =
  let key = Result.get_ok (Rig_remote.key (Sys.getenv "RIG_REMOTE_KEY")) in
  match Rig_remote.connect ~key [ ("127.0.0.1", int_of_string port) ] with
  | Error why ->
      Printf.printf "error: %s\n%!" why;
      exit 1
  | Ok j ->
      (match Rig_remote.launched () with
      | _ -> print_endline "returned"
      | exception Invalid_argument _ -> print_endline "raised");
      Rig_remote.close j;
      exit 0

let () =
  if Sys.argv.(1) = "open" then open_first Sys.argv.(2);
  match Rig_remote.launched () with
  | None ->
      print_endline "none";
      exit 0
  | Some (Error why) ->
      Printf.printf "error: %s\n%!" why;
      exit 1
  | Some (Ok j) ->
      print_endline "connected";
      List.iter (fun h -> print_endline (Rig.name h)) (Rig_remote.hosts j);
      run Sys.argv.(1) j;
      exit 0
