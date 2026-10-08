(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* An agent for the suites: [agent.exe KEYFILE [MODE]] listens on the
   loopback at a port the system chooses, prints it, and serves one job, then
   prints "closed" and exits 0 once the job closed, or prints "failed: WHY" and
   exits 2 once it failed. In mode "again" it then listens at the same port and
   serves a second job likewise; in mode "linger" it waits 2 s before it exits.

   It serves three kinds. "MEM" is two memory devices. "FAULTY" is a device
   whose next fallible call faults with "the agent's device faulted", which
   loses it. "NONE" is an opener that fails with "no such hardware". *)

let fail why =
  print_endline why;
  exit 1

let mem () =
  let open_ i = Rig.memory_device (Printf.sprintf "MEM:%d" i) in
  match (open_ 0, open_ 1) with
  | Ok a, Ok b -> Ok [ a; b ]
  | Error e, _ | _, Error e -> Error e

let faulty () =
  let d, p = Rig_support.Polled.open_ "FAULTY:0" in
  Rig_support.Polled.fail_at p 1 (`Fault "the agent's device faulted");
  Ok [ d ]

let kinds =
  [
    ("MEM", mem);
    ("FAULTY", faulty);
    ("NONE", fun () -> Error "no such hardware");
  ]

let serve key port =
  match Rig_remote.listen ~key "127.0.0.1" port with
  | Error why -> fail why
  | Ok a -> (
      Printf.printf "%d\n%!" (Rig_remote.port a);
      match Rig_remote.serve a kinds with
      | Ok () ->
          print_endline "closed";
          Rig_remote.port a
      | Error why ->
          Printf.printf "failed: %s\n%!" why;
          exit 2)

let () =
  let key =
    match Rig_remote.read_key Sys.argv.(1) with
    | Ok k -> k
    | Error why -> fail why
  in
  let mode = if Array.length Sys.argv > 2 then Sys.argv.(2) else "" in
  let port = serve key 0 in
  if mode = "again" then ignore (serve key port);
  if mode = "linger" then Unix.sleepf 2.;
  exit 0
