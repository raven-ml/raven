(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The AMD runtime on a machine without an AMD GPU: opening refuses with a
   message, fixes no interface, and the queries refuse devices of other
   vendors. *)

open Windtrap

let no_gpu () =
  if
    Nx_amd_device.count ~interface:Kernel () > 0
    || Nx_amd_device.count ~interface:Pci () > 0
  then skip ~reason:"this machine has an AMD GPU" ()

let not_amd = Exn.invalid_arg ~substring:"CPU is not an AMD device"

(* Another machine without GPUs: this process serves it on the loopback, then
   stops serving it. *)
let test_other_machine () =
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
  | Error why -> fail why
  | Ok host ->
      if Nx_amd_device.count ~host () > 0 then
        skip ~reason:"the machine has a GPU" ();
      let named = Printf.sprintf "AMD@127.0.0.1:%d: " port in
      (match Nx_amd_device.get ~host 0 with
      | Ok _ -> fail "a GPU opened"
      | Error msg ->
          is_true ~msg:"the GPU's name starts it"
            (String.starts_with ~prefix:named msg));
      (match Nx_amd_device.get ~host ~interface:Kernel 0 with
      | Ok _ -> fail "a GPU opened"
      | Error msg -> contains ~msg:"over PCI only" ~sub:"over PCI" msg);
      Nx_device_support.Remote_server.stop s;
      let lost = function Nx_device.Lost (d, _) -> d == host | _ -> false in
      raises_match ~msg:"count, the machine gone" lost (fun () ->
          Nx_amd_device.count ~host ());
      raises_match ~msg:"get, the machine gone" lost (fun () ->
          Nx_amd_device.get ~host 1)

let () =
  exit
    (run "nx.amd.device"
       [
         test "without a GPU, an open fails with a message and fixes nothing"
           (fun () ->
             no_gpu ();
             List.iter
               (fun interface ->
                 match Nx_amd_device.get ~interface 0 with
                 | Ok _ -> fail "a GPU opened"
                 | Error msg ->
                     is_true ~msg:"the vendor names the failure"
                       (String.starts_with ~prefix:"AMD: " msg))
               [ Nx_amd_device.Kernel; Pci; Kernel ];
             is_error (Nx_amd_device.get 0);
             raises_match (Exn.failure ~substring:"AMD: ") (fun () ->
                 Nx_amd_device.v 0));
         test "another machine's GPUs are opened over PCI" test_other_machine;
         test "a negative index is refused" (fun () ->
             raises_match (Exn.invalid_arg ~substring:"-1 < 0") (fun () ->
                 Nx_amd_device.get (-1)));
         test "the queries refuse devices of other vendors" (fun () ->
             let host = Nx_device.host in
             raises_match not_amd (fun () -> Nx_amd_device.interface host);
             raises_match not_amd (fun () -> Nx_amd_device.handles host);
             raises_match not_amd (fun () -> Nx_amd_device.scratch host 16));
       ])
