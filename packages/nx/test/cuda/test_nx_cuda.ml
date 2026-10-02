(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* CUDA GPUs as nx devices, on real hardware: what Nx_cuda opens is the
   runtime's GPU with no eager backend. Tests needing a GPU skip without one. *)

open Windtrap

let device = Testable.make ~pp:Nx.Device.pp ~equal:Nx.Device.equal
let memory = Testable.make ~pp:Nx_device.pp ~equal:Nx_device.equal
let count = Nx_cuda_device.count ()
let gpu () = match Nx_cuda.get 0 with Ok d -> d | Error e -> skip ~reason:e ()

let opening =
  group "opening"
    [
      test "device i is the runtime's GPU i, one device per index" (fun () ->
          let d = gpu () in
          equal device d (Nx_cuda.device 0);
          equal memory
            (Result.get_ok (Nx_cuda_device.get 0))
            (Nx.Device.memory d);
          equal string "CUDA" (Nx.Device.name d);
          not_equal device Nx.Device.host d);
      test "a GPU past the last is an Error, and device raises it as Failure"
        (fun () ->
          let why = require_error (Nx_cuda.get count) in
          raises (Failure why) (fun () -> ignore (Nx_cuda.device count)));
      test "a negative index raises Invalid_argument" (fun () ->
          raises_match Exn.invalid_arg (fun () -> ignore (Nx_cuda.get (-1)));
          raises_match Exn.invalid_arg (fun () -> ignore (Nx_cuda.device (-1))));
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

let () = exit (run "nx.cuda" [ opening; computing ])
