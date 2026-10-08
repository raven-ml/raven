(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* An agent for the suites: [agent.exe KEYFILE [MODE]] listens on the loopback
   at a port the system chooses, prints it, and serves one job, then prints
   "closed" and exits 0 once the job closed, or prints "failed: WHY" and exits 2
   once it failed.

   In mode "again" it then listens at the same port and serves a second job
   likewise. In mode "linger" it waits 2 s before it exits. In mode "twice" it
   then calls serve again, and prints "serve raised" if that raised
   Invalid_argument. In mode "deaf" it is no agent of rig.remote: it proves the
   key to the controller, then stops listening, so no other agent of the job
   reaches it, and waits for the job to end.

   It serves three kinds. "MEM" is two memory devices. "FAULTY" is a device
   whose next fallible call faults with "the agent's device faulted", which
   loses it. "NONE" is an opener that fails with "no such hardware". *)

module Wire = Rig_remote_proxy.Wire
module Link = Rig_remote_proxy.Link

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

let serve a =
  match Rig_remote.serve a kinds with
  | Ok () -> print_endline "closed"
  | Error why ->
      Printf.printf "failed: %s\n%!" why;
      exit 2

let listen key port =
  match Rig_remote.listen ~key "127.0.0.1" port with
  | Error why -> fail why
  | Ok a ->
      Printf.printf "%d\n%!" (Rig_remote.port a);
      a

let deaf key =
  let l = Unix.socket ~cloexec:true Unix.PF_INET Unix.SOCK_STREAM 0 in
  Unix.bind l (Unix.ADDR_INET (Unix.inet_addr_loopback, 0));
  Unix.listen l 1;
  (match Unix.getsockname l with
  | Unix.ADDR_INET (_, p) -> Printf.printf "%d\n%!" p
  | Unix.ADDR_UNIX _ -> fail "no port");
  let fd, _ = Unix.accept ~cloexec:true l in
  Unix.close l;
  match Wire.accept fd ~key ~admit:(fun _ -> Ok ()) with
  | Error why -> fail why
  | Ok _ ->
      let j = Link.job () in
      let l = Link.make j fd ~name:"controller" ~peer:Wire.Controller in
      let rec drain () =
        match Link.next l with
        | Ok Wire.Close -> print_endline "closed"
        | Ok _ -> drain ()
        | Error why ->
            Printf.printf "failed: %s\n%!" why;
            exit 2
      in
      drain ()

let () =
  let key =
    match Rig_remote.read_key Sys.argv.(1) with
    | Ok k -> k
    | Error why -> fail why
  in
  let mode = if Array.length Sys.argv > 2 then Sys.argv.(2) else "" in
  if mode = "deaf" then deaf key
  else begin
    let a = listen key 0 in
    serve a;
    if mode = "again" then serve (listen key (Rig_remote.port a));
    if mode = "linger" then Unix.sleepf 2.;
    if mode = "twice" then
      match Rig_remote.serve a kinds with
      | _ -> print_endline "serve returned"
      | exception Invalid_argument _ -> print_endline "serve raised"
  end;
  exit 0
