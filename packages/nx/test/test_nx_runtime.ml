(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Values placed on runtime devices, over test devices whose memory is host
   memory: they place, read back, view, compute and fail as nx's contracts say,
   whatever the device. *)

open Windtrap

type bytes_ba =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

(* A runtime device over host memory. *)
let runtime ?(budget = max_int) name =
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
  Nx_device.make ~name ~arch:"test" ~budget ~memory:{ alloc; free } ()

let r1 = runtime "R1"
let r2 = runtime "R2"
let d1 = Nx.Device.of_runtime r1
let d2 = Nx.Device.of_runtime r2
let on d = Nx.Placement.device d
let floats = array float_exact
let ints = array int

let test_devices () =
  is_true ~msg:"the host" (Nx.Device.of_runtime Nx_device.host == Nx.Device.host);
  is_true ~msg:"memoised" (Nx.Device.of_runtime r1 == d1);
  equal ~msg:"name" string "R1" (Nx.Device.name d1);
  is_false ~msg:"distinct" (Nx.Device.equal d1 d2)

let test_round_trip () =
  let x = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let s0 = Nx_device.stats r1 in
  let y = Nx.place (on d1) x in
  is_true ~msg:"placed" (Nx.Placement.equal (Nx.placement y) (on d1));
  equal ~msg:"shape" ints [| 2; 3 |] (Nx.shape y);
  equal ~msg:"elements" floats (Nx.to_array x) (Nx.to_array y);
  let back = Nx.place Nx.Placement.host y in
  equal ~msg:"back on the host" floats (Nx.to_array x) (Nx.to_array back);
  let d = Nx_device.Stats.diff s0 (Nx_device.stats r1) in
  equal ~msg:"uploaded once" int 24 (Nx_device.Stats.bytes_in d)

let test_dtypes () =
  let i4 = Nx.create Nx.int4 [| 5 |] [| -8; 7; 0; -1; 3 |] in
  equal ~msg:"int4" ints (Nx.to_array i4) (Nx.to_array (Nx.place (on d1) i4));
  let b = Nx.create Nx.bool [| 3 |] [| true; false; true |] in
  equal ~msg:"bool" (array bool) (Nx.to_array b)
    (Nx.to_array (Nx.place (on d1) b));
  let h = Nx.create Nx.bfloat16 [| 2 |] [| 1.5; -2. |] in
  equal ~msg:"bfloat16" floats (Nx.to_array h)
    (Nx.to_array (Nx.place (on d1) h));
  let e = Nx.place (on d1) (Nx.zeros Nx.float64 [| 0; 4 |]) in
  equal ~msg:"empty" ints [| 0; 4 |] (Nx.shape e)

let test_views () =
  let x = Nx.arange Nx.int32 0 24 1 |> Nx.reshape [| 2; 3; 4 |] in
  let y = Nx.place (on d1) x in
  let same msg f =
    equal ~msg (array int32) (Nx.to_array (f x)) (Nx.to_array (f y))
  in
  same "transpose" Nx.transpose;
  same "slice" (Nx.slice [ Nx.I 1; Nx.R (1, 3) ]);
  same "flip" (Nx.flip ~axes:[ 2 ]);
  same "broadcast" (fun t -> Nx.broadcast_to [| 2; 2; 3; 4 |] t);
  equal ~msg:"item" int32 13l (Nx.item [ 1; 0; 1 ] y)

let test_compute () =
  let x = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let y = Nx.place (on d1) x in
  let z = Nx.add y y in
  is_true ~msg:"the result stays on the device"
    (Nx.Placement.equal (Nx.placement z) (on d1));
  equal ~msg:"sum" floats [| 2.; 4.; 6. |] (Nx.to_array z)

let test_sharded () =
  let x = Nx.arange Nx.float32 0 8 1 |> Nx.reshape [| 4; 2 |] in
  let p = Nx.Placement.sharded ~axis:0 [ d1; d2 ] in
  let y = Nx.place p x in
  equal ~msg:"whole" floats (Nx.to_array x) (Nx.to_array y);
  equal ~msg:"shard" floats [| 4.; 5.; 6.; 7. |]
    (Nx.to_array
       (Nx.place (on d2)
          (Nx.place Nx.Placement.host y |> Nx.slice [ Nx.R (2, 4) ])))

let test_out_of_memory () =
  let small = Nx.Device.of_runtime (runtime ~budget:16 "SMALL") in
  raises_match
    (function Nx.Device.Out_of_memory (d, 400) -> d == small | _ -> false)
    (fun () -> Nx.place (on small) (Nx.zeros Nx.float32 [| 100 |]))

let test_mixed_engines () =
  raises_match (Exn.invalid_arg ~substring:"") (fun () ->
      Nx.Placement.replicated [ d1; Nx.Device.host ])

let () =
  exit
    (run "nx runtime devices"
       [
         test "devices" test_devices;
         test "round trip" test_round_trip;
         test "dtypes" test_dtypes;
         test "views" test_views;
         test "compute" test_compute;
         test "sharded" test_sharded;
         test "out of memory" test_out_of_memory;
         test "one engine per placement" test_mixed_engines;
       ])
