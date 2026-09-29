(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Values placed on runtime devices, over test runtimes whose memory is host
   memory: the laws every runtime keeps, and what makes a runtime a device. *)

open Windtrap
open Nx_test

type bytes_ba =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

(* A runtime over host memory. *)
let runtime name =
  let memory : (nativeint, bytes_ba) Hashtbl.t = Hashtbl.create 16 in
  let alloc n =
    let ba =
      Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout n
    in
    let a = Nx_device.Buffer.host_address (Nx_device.Buffer.of_bigarray ba) in
    Hashtbl.add memory a ba;
    Some { Nx_device.host = Some a; device = a; handle = a }
  in
  let free (m : Nx_device.memory) = Hashtbl.remove memory m.device in
  Nx_device.make ~name ~arch:"test" ~budget:max_int ~memory:{ alloc; free } ()

let r1 = runtime "R1"
let r2 = runtime "R2"
let d1 = Nx.Device.of_runtime r1
let d2 = Nx.Device.of_runtime r2

let devices =
  group "devices"
    [
      test "each runtime is one device, named as it, and the host's is the host"
        (fun () ->
          is_true (Nx.Device.of_runtime Nx_device.host == Nx.Device.host);
          is_true (Nx.Device.of_runtime r1 == d1);
          equal string "R1" (Nx.Device.name d1);
          is_false (Nx.Device.equal d1 d2));
      test "a placement mixing runtime devices with the host is refused"
        (fun () ->
          raises_invalid_arg (fun () ->
              Nx.Placement.replicated [ d1; Nx.Device.host ]));
    ]

let () = exit (run "nx runtime devices" (devices :: Runtimes.laws [ r1; r2 ]))
