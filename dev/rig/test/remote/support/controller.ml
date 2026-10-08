(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A controller for the suites: [controller.exe KEYFILE MODE PORT...] connects
   to the agents at the loopback ports, prints "connected", then in mode "exit"
   exits without closing the job, in mode "wait" waits to be killed, and in mode
   "watch" waits for the job to fail, prints "failed: WHY" and exits 2. It
   prints why and exits 1 if it cannot connect. *)

(* The longest it waits, in modes "wait" and "watch". *)
let patience = 60.

let rec watch j t0 =
  match Rig_remote.failure j with
  | Some why ->
      Printf.printf "failed: %s\n%!" why;
      exit 2
  | None when Unix.gettimeofday () -. t0 > patience -> exit 1
  | None ->
      Unix.sleepf 0.01;
      watch j t0

let () =
  let key =
    match Rig_remote.read_key Sys.argv.(1) with
    | Ok k -> k
    | Error why ->
        print_endline why;
        exit 1
  in
  let ports =
    Array.to_list (Array.sub Sys.argv 3 (Array.length Sys.argv - 3))
  in
  let agents = List.map (fun p -> ("127.0.0.1", int_of_string p)) ports in
  match Rig_remote.connect ~key agents with
  | Error why ->
      print_endline why;
      exit 1
  | Ok j ->
      print_endline "connected";
      if Sys.argv.(2) = "wait" then Unix.sleepf patience;
      if Sys.argv.(2) = "watch" then watch j (Unix.gettimeofday ());
      exit 0
