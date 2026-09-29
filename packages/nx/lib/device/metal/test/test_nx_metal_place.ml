(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Values placed on the Metal GPU through nx: they live in its buffers, read
   back unchanged, and fail with nx's exception when it has no memory. *)

open Windtrap

let metal = Nx_metal_device.v 0
let gpu = Nx.Device.of_runtime metal
let on_gpu = Nx.Placement.device gpu

let test_round_trip () =
  let s0 = Nx_device.stats metal in
  let x = Nx.rand Nx.float32 [| 64; 32 |] in
  let y = Nx.place on_gpu x in
  equal ~msg:"device" string "METAL" (Nx.Device.name gpu);
  is_true ~msg:"placed" (Nx.Placement.equal (Nx.placement y) on_gpu);
  equal ~msg:"elements" (array float_exact) (Nx.to_array x) (Nx.to_array y);
  let d = Nx_device.Stats.diff s0 (Nx_device.stats metal) in
  equal ~msg:"uploaded" int (64 * 32 * 4) (Nx_device.Stats.bytes_in d);
  let i4 = Nx.create Nx.int4 [| 3 |] [| -8; 7; 1 |] in
  equal ~msg:"int4" (array int) (Nx.to_array i4)
    (Nx.to_array (Nx.place on_gpu i4))

let test_views_and_compute () =
  let x = Nx.arange Nx.int32 0 12 1 |> Nx.reshape [| 3; 4 |] in
  let y = Nx.place on_gpu x in
  equal ~msg:"transpose" (array int32)
    (Nx.to_array (Nx.transpose x))
    (Nx.to_array (Nx.transpose y));
  let z = Nx.mul y y in
  is_true ~msg:"result on the GPU" (Nx.Placement.equal (Nx.placement z) on_gpu);
  equal ~msg:"product" (array int32) (Nx.to_array (Nx.mul x x)) (Nx.to_array z)

let test_out_of_memory () =
  let budget = Nx_device.budget metal in
  Fun.protect ~finally:(fun () -> Nx_device.set_budget metal budget)
  @@ fun () ->
  Gc.full_major ();
  Nx_device.set_budget metal (Nx_device.Stats.allocated (Nx_device.stats metal));
  raises_match
    (function Nx.Device.Out_of_memory (d, 4096) -> d == gpu | _ -> false)
    (fun () -> Nx.place on_gpu (Nx.zeros Nx.float32 [| 1024 |]))

let () =
  exit
    (run "nx on Metal"
       [
         test "round trip" test_round_trip;
         test "views and compute" test_views_and_compute;
         test "out of memory" test_out_of_memory;
       ])
