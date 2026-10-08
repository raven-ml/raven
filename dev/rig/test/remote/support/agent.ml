(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* An agent for the suites: [agent.exe KEYFILE] listens on the loopback at a
   port the system chooses, prints it, and serves one job whose "MEM" kind is
   two memory devices. It exits 0 once the job closed and 2 once it failed,
   printing why. *)

let () =
  let key =
    match Rig_remote.read_key Sys.argv.(1) with
    | Ok k -> k
    | Error why ->
        prerr_endline why;
        exit 1
  in
  let a =
    match Rig_remote.listen ~key "127.0.0.1" 0 with
    | Ok a -> a
    | Error why ->
        prerr_endline why;
        exit 1
  in
  Printf.printf "%d\n%!" (Rig_remote.port a);
  let mem () =
    let open_ i = Rig.memory_device (Printf.sprintf "MEM:%d" i) in
    match (open_ 0, open_ 1) with
    | Ok a, Ok b -> Ok [ a; b ]
    | Error e, _ | _, Error e -> Error e
  in
  match Rig_remote.serve a [ ("MEM", mem) ] with
  | Ok () -> exit 0
  | Error why ->
      prerr_endline why;
      exit 2
