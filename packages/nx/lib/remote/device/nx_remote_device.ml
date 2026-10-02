(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Remote = Nx_device_support.Remote
module Remote_server = Nx_device_support.Remote_server
module Mmio = Nx_device_support.Mmio
module Driver = Nx_device.Driver

let default_port = 6667

(* The hosts connected to, by host and port. *)
let lock = Mutex.create ()
let hosts : (string * int, Nx_device.t * Remote.t) Hashtbl.t = Hashtbl.create 4

let make r =
  let memory =
    {
      Driver.alloc =
        (fun n ->
          Option.map
            (fun a -> Driver.Region.v ~host:a ~handle:a a n)
            (Remote.alloc r n));
      free = (fun m -> Remote.free r (Driver.Region.address m));
    }
  in
  let io =
    {
      Driver.read = Remote.read r;
      write = Remote.write r;
      copy = Remote.copy r;
    }
  in
  (* A program is gone with a failed connection: the server dropped it. A
     program the server refuses leaves the connection usable. *)
  let load ~binary ~entry =
    match Remote.load r ~binary ~name:entry with
    | id ->
        Ok
          ( Nativeint.of_int id,
            fun () -> if Remote.failed r = None then Remote.unload r id )
    | exception Failure why when Remote.failed r = None -> Error why
  in
  let call h buffers values =
    Remote.call r (Nativeint.to_int h) buffers values
  in
  Driver.host ~address:(Remote.name r) ~arch:(Remote.arch r)
    ~programs:{ load; call }
    ~synchronized:(fun () -> Remote.ping r)
    ~finalize:(fun ~failed:_ -> Remote.close r)
    ~memory io

let connect ?(port = default_port) ?(timeout_ms = Driver.default_timeout) ~key
    host =
  if timeout_ms <= 0 then
    invalid_arg
      (Printf.sprintf "Nx_remote_device.connect: timeout %d ms" timeout_ms);
  Mutex.protect lock (fun () ->
      match Hashtbl.find_opt hosts (host, port) with
      | Some (d, _) -> Ok d
      | None -> (
          match Remote.connect ~timeout_ms ~key host port with
          | exception Failure why -> Error why
          | r ->
              let d = make r in
              Nx_device.set_timeout d timeout_ms;
              Hashtbl.replace hosts (host, port) (d, r);
              Ok d))

(* Serving: host programs, loaded here and kept until the client drops them. *)

let programs () =
  let table = Hashtbl.create 16 and next = Atomic.make 0 in
  let lock = Mutex.create () in
  let host = Driver.host_programs in
  let load ~binary ~name =
    match host.load ~binary ~entry:name with
    | Error why -> failwith why
    | Ok loaded ->
        let id = Atomic.fetch_and_add next 1 in
        Mutex.protect lock (fun () -> Hashtbl.replace table id loaded);
        id
  in
  let call id buffers values =
    match Mutex.protect lock (fun () -> Hashtbl.find_opt table id) with
    | None -> failwith (Printf.sprintf "no program %d" id)
    | Some (handle, _) ->
        host.call handle
          (Array.map (fun m -> (Mmio.address m, Mmio.length m)) buffers)
          values
  in
  let unload id =
    Option.iter
      (fun (_, free) -> free ())
      (Mutex.protect lock (fun () ->
           let loaded = Hashtbl.find_opt table id in
           Hashtbl.remove table id;
           loaded))
  in
  { Remote_server.load; call; unload }

let listen ~key addr = Remote_server.listen ~key ~programs:(programs ()) addr

let remote d =
  if d == Nx_device.host then None
  else
    match
      Mutex.protect lock (fun () ->
          Hashtbl.fold
            (fun _ (d', r) acc -> if d' == d then Some r else acc)
            hosts None)
    with
    | Some _ as r -> r
    | None ->
        invalid_arg
          (Printf.sprintf "Nx_remote_device.remote: %s is no host"
             (Nx_device.name d))
