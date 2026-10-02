(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* NVIDIA GPUs as nx devices through the kernel driver, on real hardware: what
   Nx_nv opens is the runtime's GPU with no eager backend, and only a call of
   [get_pci] or [device_pci] reaches PCI. Tests needing a GPU skip without
   one. *)

open Windtrap

let device = Testable.make ~pp:Nx.Device.pp ~equal:Nx.Device.equal
let memory = Testable.make ~pp:Nx_device.pp ~equal:Nx_device.equal
let gpu () = match Nx_nv.get 0 with Ok d -> d | Error e -> skip ~reason:e ()

let opening =
  group "opening"
    [
      test "device i is the runtime's GPU i of the kernel driver" (fun () ->
          let d = gpu () in
          let m = Result.get_ok (Nx_nv_device.get ~interface:Kernel 0) in
          equal device d (Nx_nv.device 0);
          equal memory m (Nx.Device.memory d);
          equal string (Nx_device.name m) (Nx.Device.name d);
          not_equal device Nx.Device.host d);
      test "get fails without touching PCI where the kernel driver is absent"
        (fun () ->
          if Sys.file_exists "/dev/nvidiactl" then
            skip ~reason:"/dev/nvidiactl exists" ();
          let why = require_error (Nx_nv.get 0) in
          raises (Failure why) (fun () -> ignore (Nx_nv.device 0)));
      test "a negative index raises Invalid_argument" (fun () ->
          List.iter
            (fun f -> raises_match Exn.invalid_arg (fun () -> ignore (f (-1))))
            [ Nx_nv.get; Nx_nv.get_pci ];
          List.iter
            (fun f -> raises_match Exn.invalid_arg (fun () -> ignore (f (-1))))
            [ Nx_nv.device; Nx_nv.device_pci ]);
    ]

let computing =
  group "computing"
    [
      test "values placed on it read back" (fun () ->
          let p = Nx.Placement.on (gpu ()) in
          let x = Nx.create Nx.float32 [| 3 |] [| 1.; -0.; 3.5 |] in
          let y = Nx.place p x in
          equal bool true (Nx.Placement.equal p (Nx.placement y));
          equal (array float_exact) (Nx.to_array x) (Nx.to_array y));
      test "an eager operation raises, naming the remedies" (fun () ->
          let x =
            Nx.place (Nx.Placement.on (gpu ())) (Nx.ones Nx.float32 [| 2 |])
          in
          raises_match (Exn.invalid_arg ~substring:"has no eager kernels")
            (fun () -> ignore (Nx.exp x)));
    ]

let () = exit (run "nx.nv" [ opening; computing ])
