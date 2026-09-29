(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The RDMA runtime on machines without an adapter: this one, and another this
   process serves on the loopback. *)

open Windtrap

let no_adapter host =
  if Nx_rdma_device.count ~host () > 0 then
    skip ~reason:"the machine has an adapter" ()

let refused host =
  match Nx_rdma_device.get ~host 0 with
  | Ok _ -> fail "an adapter opened"
  | Error msg -> contains ~msg:"the runtime names it" ~sub:"RDMA: " msg

(* Served for the life of the process, which disconnects at exit. *)
let other_machine () =
  let key = "a key of the test, long enough" in
  let s =
    Nx_remote_device.listen ~key (Unix.ADDR_INET (Unix.inet_addr_loopback, 0))
  in
  let port =
    match Nx_device_support.Remote_server.address s with
    | Unix.ADDR_INET (_, p) -> p
    | Unix.ADDR_UNIX _ -> assert false
  in
  match Nx_remote_device.connect ~port ~key "127.0.0.1" with
  | Ok host -> host
  | Error why -> fail why

let () =
  exit
    (run "nx.rdma.device"
       [
         test "without an adapter, an open fails with a message" (fun () ->
             no_adapter Nx_device.host;
             refused Nx_device.host;
             raises_match (Exn.invalid_arg ~substring:"RDMA: ") (fun () ->
                 Nx_rdma_device.v 0));
         test "another machine's adapters" (fun () ->
             let host = other_machine () in
             no_adapter host;
             refused host);
         test "a negative index and a device that is no host are refused"
           (fun () ->
             raises_match (Exn.invalid_arg ~substring:"-1 < 0") (fun () ->
                 Nx_rdma_device.get (-1));
             let gpu =
               Nx_device.make ~name:"GPU" ~arch:"x" ~budget:0
                 ~memory:{ alloc = (fun _ -> None); free = ignore }
                 ()
             in
             raises_match (Exn.invalid_arg ~substring:"no host") (fun () ->
                 Nx_rdma_device.get ~host:gpu 0));
       ])
