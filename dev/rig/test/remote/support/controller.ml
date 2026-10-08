(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A controller for the suites: [controller.exe KEYFILE MODE PORT...] connects
   to the agents at the loopback ports, prints "connected", then in mode "exit"
   exits without closing the job, and in mode "wait" waits to be killed. It
   prints why and exits 1 if it cannot connect. *)

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
  | Ok _ ->
      print_endline "connected";
      if Sys.argv.(2) = "wait" then Unix.sleepf 60.;
      exit 0
