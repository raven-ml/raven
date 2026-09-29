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
             raises_match (Exn.invalid_arg ~substring:"AMD: ") (fun () ->
                 Nx_amd_device.v 0));
         test "a negative index is refused" (fun () ->
             raises_match (Exn.invalid_arg ~substring:"-1 < 0") (fun () ->
                 Nx_amd_device.get (-1)));
         test "the queries refuse devices of other vendors" (fun () ->
             let host = Nx_device.host in
             raises_match not_amd (fun () -> Nx_amd_device.interface host);
             raises_match not_amd (fun () -> Nx_amd_device.handles host);
             raises_match not_amd (fun () -> Nx_amd_device.scratch host 16));
       ])
