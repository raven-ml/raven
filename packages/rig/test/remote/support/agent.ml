(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* An agent for the suites: [agent.exe KEYFILE [MODE]] listens on the loopback
   at a port the system chooses, prints it, and serves one job, then prints
   "closed" and exits 0 once the job closed, or prints "failed: WHY" and exits 2
   once it failed. After "closed" it prints "open: NAME" for each device it
   opened that is not closed.

   In mode "again" it then listens at the same port and serves a second job
   likewise. In mode "linger" it waits 2 s before it exits. In mode "twice" it
   then calls serve again, and prints "serve raised" if that raised
   Invalid_argument. In mode "helper" it starts a process, [agent.exe helper],
   that prints "helper ready" and waits for a line or the end of stdin. It
   reaps that process before exiting. In mode "deaf" it is no agent of
   rig.remote: it proves the key to the controller, then stops listening, so no
   other agent of the job reaches it, and waits for the job to end. In mode
   "crowded" it limits its descriptors to [crowd] and, while it serves, takes
   lines on stdin: "fill" opens descriptors until the system refuses one, then
   prints "full"; "free" closes them, then prints "freed". In mode "high" it
   holds [past] descriptors before it listens, its limit raised to [high], so
   that every socket it makes has a number past [past].

   It serves eleven kinds. "MEM" is two memory devices. "POLLED" is two Polled
   devices, which launch Polled's host functions. "TIGHT" is a Polled device
   whose queue holds 3 parts. "DISK" is the io device of files. "HIDDEN" is a
   Polled device whose [Device] memory the host does not address. "MAPLESS"
   is a Polled device that maps no host memory. "FAULTY" is a device whose
   next fallible call faults with "the agent's device faulted", which loses it.
   "NONE" is an opener that fails with "no such hardware". "COLLECT" runs a full
   major collection and opens no device. "RAISE" is an opener that raises
   [Failure "the opener raised"]. "LOUD" prints "opening LOUD" and opens no
   device. *)

module Wire = Rig_remote_proxy.Wire
module Link = Rig_remote_proxy.Link

let fail why =
  print_endline why;
  exit 1

(* Crowding *)

(* The descriptors a crowded agent may hold: more than it opens to serve. *)
let crowd = 256

(* Descriptors past select's FD_SETSIZE, 1024 on Linux and macOS, and the
   limit of descriptors that holds them and the job's. *)
let past = 1100
let high = 4096

(* Holds [past] descriptors, its limit raised to [high]. *)
let hold_high () =
  let e = Rig_support.set_descriptor_limit high in
  if e <> 0 then fail (Printf.sprintf "raising descriptors: %d" e);
  List.init past (fun _ -> Unix.dup ~cloexec:true Unix.stdout)

(* Runs the commands of mode "crowded" from stdin until its end. *)
let crowding () =
  let rec fill held =
    match Unix.dup ~cloexec:true Unix.stdout with
    | fd -> fill (fd :: held)
    | exception Unix.Unix_error (Unix.EMFILE, _, _) -> held
    | exception Unix.Unix_error (e, _, _) ->
        fail ("filling descriptors: " ^ Unix.error_message e)
  in
  let rec take held =
    match In_channel.input_line stdin with
    | Some "fill" ->
        let held = fill held in
        print_endline "full";
        take held
    | Some "free" ->
        List.iter Unix.close held;
        print_endline "freed";
        take []
    | Some _ | None -> ()
  in
  take []

(* The devices the openers made. *)
let opened = ref []

let mem () =
  let open_ i = Rig.memory_device (Printf.sprintf "MEM:%d" i) in
  match (open_ 0, open_ 1) with
  | a, b ->
      opened := a :: b :: !opened;
      Ok [ a; b ]
  | exception Failure e -> Error e

let polled () =
  let open_ i = fst (Rig_support.Polled.open_ (Printf.sprintf "POLLED:%d" i)) in
  let a = open_ 0 in
  let b = open_ 1 in
  opened := a :: b :: !opened;
  Ok [ a; b ]

let tight () =
  let d, _ = Rig_support.Polled.open_ ~capacity:3 "TIGHT:0" in
  opened := d :: !opened;
  Ok [ d ]

let hidden () =
  let d, _ = Rig_support.Polled.open_ ~host_visible:false "HIDDEN:0" in
  opened := d :: !opened;
  Ok [ d ]

let mapless () =
  let d, _ = Rig_support.Polled.open_ ~maps_host:false "MAPLESS:0" in
  opened := d :: !opened;
  Ok [ d ]

let faulty () =
  let d, p = Rig_support.Polled.open_ "FAULTY:0" in
  Rig_support.Polled.fail_at p 1 (`Fault "the agent's device faulted");
  opened := d :: !opened;
  Ok [ d ]

let kinds =
  [
    ("MEM", mem);
    ("POLLED", polled);
    ("TIGHT", tight);
    ("DISK", fun () -> Ok [ Rig_disk.device ]);
    ("FAULTY", faulty);
    ("HIDDEN", hidden);
    ("MAPLESS", mapless);
    ("NONE", fun () -> Error "no such hardware");
    ("RAISE", fun () -> failwith "the opener raised");
    ( "COLLECT",
      fun () ->
        Gc.full_major ();
        Ok [] );
    ( "LOUD",
      fun () ->
        print_endline "opening LOUD";
        Ok [] );
  ]

let serve a =
  match Rig_remote.serve a kinds with
  | Ok () ->
      print_endline "closed";
      List.iter
        (fun d ->
          if Rig.lost d <> Some "closed" then
            Printf.printf "open: %s\n%!" (Rig.name d))
        (List.rev !opened);
      opened := []
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
  if Sys.argv.(1) = "helper" then begin
    print_endline "helper ready";
    ignore (In_channel.input_line stdin);
    exit 0
  end;
  let key =
    match Rig_remote.read_key Sys.argv.(1) with
    | Ok k -> k
    | Error why -> fail why
  in
  let mode = if Array.length Sys.argv > 2 then Sys.argv.(2) else "" in
  if mode = "deaf" then
    deaf (In_channel.with_open_bin Sys.argv.(1) In_channel.input_all)
  else begin
    if mode = "crowded" then begin
      let e = Rig_support.set_descriptor_limit crowd in
      if e <> 0 then fail (Printf.sprintf "limiting descriptors: %d" e)
    end;
    let held = if mode = "high" then hold_high () else [] in
    let a = listen key 0 in
    if mode = "crowded" then ignore (Thread.create crowding ());
    let helper =
      if mode <> "helper" then None
      else
        let exe = Sys.executable_name in
        Some
          (Unix.create_process exe [| exe; "helper" |] Unix.stdin Unix.stdout
             Unix.stderr)
    in
    if mode = "together" then begin
      (* Two domains serve [a] at once: one raises, the other serves. *)
      let other =
        Domain.spawn (fun () ->
            match serve a with
            | () -> ()
            | exception Invalid_argument _ -> print_endline "serve raised")
      in
      (match serve a with
      | () -> ()
      | exception Invalid_argument _ -> print_endline "serve raised");
      Domain.join other
    end
    else serve a;
    if mode = "again" then serve (listen key (Rig_remote.port a));
    if mode = "linger" then Unix.sleepf 2.;
    Option.iter
      (fun pid ->
        match snd (Unix.waitpid [] pid) with
        | Unix.WEXITED 0 -> print_endline "helper exited"
        | _ -> fail "helper failed")
      helper;
    List.iter Unix.close held;
    if mode = "twice" then
      match Rig_remote.serve a kinds with
      | _ -> print_endline "serve returned"
      | exception Invalid_argument _ -> print_endline "serve raised"
  end;
  exit 0
