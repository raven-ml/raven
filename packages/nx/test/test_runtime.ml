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
      test "the host shares the runtimes' memory: a copy on it and on a runtime \
            computes and reads back"
        (fun () ->
          let p = Nx.Placement.replicated [ d1; Nx.Device.host ] in
          let x = Nx.place p (Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |]) in
          let y = Nx.add x x in
          is_true (Nx.Placement.equal (Nx.placement y) p);
          equal (array float_exact) [| 2.; 4.; 6. |] (Nx.to_array y));
    ]

(* A buffer that reaches the host engine is on the host and of the value's
   format, whether a caller hands it over or a device's memory reads it back. *)
let host_buffers =
  let not_host = Exn.invalid_arg ~substring:"not CPU"
  and other_format = Exn.invalid_arg ~substring:"float64 buffer read as float32"
  and on_device = Nx_device.Buffer.create r1 Nx_dtype.Scalar.Float32 4
  and float64 = Nx_core.Elements.create Nx.float64 4 in
  group "host buffers"
    [
      test "from_host refuses a buffer on a device or of another format"
        (fun () ->
          let from_host b =
            Nx_effect.from_host Nx_effect.host_tensor_context Nx.float32 b
          in
          raises_match not_host (fun () -> from_host on_device);
          raises_match other_format (fun () -> from_host float64));
      test "a read of a buffer on a device or of another format raises"
        (fun () ->
          let reading b =
            let d =
              Nx_effect.Device.make "READS"
                { Devices.memory with read = (fun _ -> b) }
            in
            let x =
              Nx.place (Nx.Placement.device d) (Nx.zeros Nx.float32 [| 4 |])
            in
            fun () -> Nx.place Nx.Placement.host x
          in
          raises_match not_host (reading on_device);
          raises_match other_format (reading float64));
    ]

let () =
  exit
    (run "nx runtime devices"
       (devices :: host_buffers :: Runtimes.laws [ r1; r2 ]))
