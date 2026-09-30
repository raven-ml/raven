(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The core of the lowering: dtypes, captures, movements, placements and reads.
   A traced value is checked by evaluating its graph with its captures bound
   (Traces.value) against the value nx gives eagerly. *)

open Windtrap
open Nx_test
open Traces
open Rune_next
module Ops = Tolk_next.Ops
module Dtype = Tolk_next.Dtype

let pp_float ppf x = Format.fprintf ppf "%.17g" x
let floats = viewed ~pp:pp_float Nx.float32 Gen.any_float
let arange n = Nx.create Nx.float32 [| n |] (Array.init n float_of_int)
let grid r c = Nx.reshape [| r; c |] (arange (r * c))
let jit_error = function Lower.Jit_error _ -> true | _ -> false

(* A traced copy of [x], and its value. *)
let copied x =
  let s, y = trace (fun () -> Nx.copy x) in
  (s, value s y)

(* Dtypes *)

let tolk_dtype = option (Testable.make ~pp:Dtype.pp ~equal:Dtype.equal)

let dtypes =
  group "dtypes"
    [
      test "floats, integers and booleans keep their width and signedness"
        (fun () ->
          let is dt expected =
            equal tolk_dtype (Some expected) (Lower.dtype dt)
          in
          is Nx.float16 Float16;
          is Nx.bfloat16 Bfloat16;
          is Nx.float32 Float32;
          is Nx.float64 Float64;
          is Nx.int8 Int8;
          is Nx.uint8 Uint8;
          is Nx.int16 Int16;
          is Nx.uint16 Uint16;
          is Nx.int32 Int32;
          is Nx.uint32 Uint32;
          is Nx.int64 Int64;
          is Nx.uint64 Uint64;
          is Nx.bool Bool);
      test "the 8-bit floats are the OCP formats" (fun () ->
          equal tolk_dtype (Some Dtype.Fp8e4m3) (Lower.dtype Nx.float8_e4m3);
          equal tolk_dtype (Some Dtype.Fp8e5m2) (Lower.dtype Nx.float8_e5m2));
      test "4-bit integers and complex numbers have no counterpart" (fun () ->
          is_none (Lower.dtype Nx.int4);
          is_none (Lower.dtype Nx.uint4);
          is_none (Lower.dtype Nx.complex64);
          is_none (Lower.dtype Nx.complex128));
    ]

(* Captures *)

let bound s = List.length (Lower.captures s)

let captures =
  group "captures"
    [
      prop "a captured view computes its elements, whatever its strides" floats
        (fun x -> exact x (snd (copied x)));
      test "overlapping windows are read from their storage" (fun () ->
          let x = Nx.sliding_window ~window:3 ~step:1 (arange 6) in
          exact x (snd (copied x)));
      test "windows of windows are read from their storage" (fun () ->
          let x =
            Nx.sliding_window ~axis:1 ~window:2
              (Nx.sliding_window ~window:3 (grid 4 5))
          in
          exact x (snd (copied x)));
      test "a value that reaches one element is a constant, bound nowhere"
        (fun () ->
          let s, y =
            copied (Nx.broadcast_to [| 3; 2 |] (Nx.scalar Nx.float32 2.))
          in
          exact (Nx.full Nx.float32 [| 3; 2 |] 2.) y;
          equal int 0 (bound s));
      test "a single element of a larger storage is a constant" (fun () ->
          let s, y = copied (Nx.slice [ I 3 ] (arange 8)) in
          exact (Nx.scalar Nx.float32 3.) y;
          equal int 0 (bound s));
      test "a borrowed single element is storage, which its owner may change"
        (fun () ->
          let ba = Bigarray.Array1.of_array Float32 C_layout [| 7. |] in
          let x = Nx.of_bigarray (Bigarray.genarray_of_array1 ba) in
          let s, y = copied x in
          exact x y;
          equal int 1 (bound s));
      test "a capture met twice is bound once" (fun () ->
          let x = grid 2 3 in
          let s, _ = trace (fun () -> (Nx.copy x, Nx.copy x)) in
          equal int 1 (bound s));
      test "two views of one storage are bound apart" (fun () ->
          let x = grid 2 3 in
          let t = Nx.transpose x in
          let s, _ = trace (fun () -> (Nx.copy x, Nx.copy t)) in
          equal int 2 (bound s));
      test "storage is bound from the 16-byte boundary at or below its view"
        (fun () ->
          let s, _ = copied (Nx.slice [ R (5, 9) ] (arange 16)) in
          match Lower.captures s with
          | [ (u, [ b ]) ] ->
              equal int 5 (Nx_device.Buffer.length b);
              equal (list int) [ 5 ]
                (List.map
                   (function Ops.Int n -> n | Ops.Sym _ -> -1)
                   (Ops.shape u))
          | _ -> fail "one capture on one device");
      test "an empty view is bound nowhere" (fun () ->
          let s, y = copied (Nx.slice [ R (2, 2) ] (arange 4)) in
          equal (array int) [| 0 |] (Nx.shape y);
          equal int 0 (bound s));
    ]

(* Devices *)

let runtime name =
  Nx_device.Driver.device ~name ~arch:"test" ~budget:max_int
    (Host_visible { memory = Nx_device.Driver.host_memory; mapping = None })

let d1 = Nx.Device.of_runtime (runtime "CPU:1")
let d2 = Nx.Device.of_runtime (runtime "CPU:2")
let twin = Nx.Device.of_runtime (runtime "CPU:1")
let names s = List.map fst (Lower.devices s)

(* Parameters *)

let parameter x =
  let s = scope () in
  let y = within s (fun () -> Nx.copy (argument s x)) in
  value s y

let parameters =
  group "parameters"
    [
      prop "a parameter views its argument's storage, whatever its strides"
        floats (fun x -> exact x (parameter x));
      test "a parameter binds nothing when traced" (fun () ->
          let s = scope () in
          ignore (Lower.param s ~slot:0 (grid 2 3));
          equal int 0 (bound s));
      test "a parameter is at its argument's placement" (fun () ->
          let x = Nx.place (Nx.Placement.device d1) (grid 2 3) in
          let s = scope () in
          let y = Lower.param s ~slot:0 x in
          is_true (Nx.Placement.equal (Nx.placement x) (Nx.placement y));
          equal (list string) [ "CPU:1" ] (names s));
    ]

(* Movements *)

let movements =
  group "movements"
    [
      prop "a traced movement computes what the eager one gives"
        (Gen.pair layout floats) (fun (steps, x) ->
          let s, y = trace (fun () -> lay_out steps (Nx.copy x)) in
          exact (lay_out steps x) (value s y));
      test "windows along an inner axis take its place, their elements last"
        (fun () ->
          let x = grid 3 5 in
          let s, y =
            trace (fun () -> Nx.sliding_window ~axis:1 ~window:2 ~step:2 x)
          in
          exact (Nx.sliding_window ~axis:1 ~window:2 ~step:2 x) (value s y));
      test "a movement of a capture moves its storage, bound once" (fun () ->
          let x = grid 3 4 in
          let s, y = trace (fun () -> Nx.flip (Nx.transpose x)) in
          exact (Nx.flip (Nx.transpose x)) (value s y);
          equal int 1 (bound s));
    ]

(* Placements *)

let placements =
  group "placements"
    [
      test "a traced value placed on a device is copied there" (fun () ->
          let x = grid 2 3 in
          let s, y =
            trace (fun () -> Nx.place (Nx.Placement.device d1) (Nx.transpose x))
          in
          exact (Nx.transpose x) (value s y);
          is_true (Nx.Placement.equal (Nx.Placement.device d1) (Nx.placement y));
          equal (list string) [ "CPU"; "CPU:1" ] (names s));
      test "a traced value placed on devices is a copy on each" (fun () ->
          let p = Nx.Placement.replicated [ d1; d2 ] in
          let s, y = trace (fun () -> Nx.place p (Nx.flip (arange 4))) in
          exact (Nx.flip (arange 4)) (value s y);
          equal (list string) [ "CPU"; "CPU:1"; "CPU:2" ] (names s));
      test "a capture on another placement is placed once, then bound there"
        (fun () ->
          let p = Nx.Placement.device d1 and x = grid 2 2 in
          let s, y = trace (fun () -> Nx.place p x) in
          exact (grid 2 2) (value s y);
          match Lower.captures s with
          | [ (_, [ b ]) ] ->
              equal string "CPU:1" (Nx_device.name (Nx_device.Buffer.device b))
          | _ -> fail "one capture on one device");
      test "a placed capture is bound where it lies, without a copy" (fun () ->
          let x = Nx.place (Nx.Placement.device d1) (grid 2 3) in
          let s, y = copied x in
          exact (grid 2 3) y;
          equal (list string) [ "CPU:1" ] (names s));
      test "a traced value split over devices holds its slices" (fun () ->
          let p = Nx.Placement.sharded ~axis:0 [ d1; d2 ] and x = grid 4 3 in
          let s, y = trace (fun () -> Nx.place p (Nx.copy x)) in
          exact x (value s y);
          is_true (Nx.Placement.equal p (Nx.placement y)));
      test "a split capture is its shards, reassembled" (fun () ->
          let p = Nx.Placement.sharded ~axis:1 [ d1; d2 ] in
          let x = Nx.place p (grid 3 4) in
          let s, y = copied x in
          exact (grid 3 4) y;
          equal int 1 (bound s));
      test "two devices of one name cannot meet in one trace" (fun () ->
          let a = Nx.place (Nx.Placement.device d1) (grid 2 2) in
          let b = Nx.place (Nx.Placement.device twin) (grid 2 2) in
          raises_match (Exn.invalid_arg ~substring:"CPU:1") (fun () ->
              trace (fun () -> (Nx.copy a, Nx.copy b))));
    ]

(* Reads *)

let reads =
  group "reads"
    [
      test "a capture's elements are read eagerly" (fun () ->
          let x = grid 2 2 in
          let _, v = trace (fun () -> Nx.to_array x) in
          equal (array float_exact) [| 0.; 1.; 2.; 3. |] v);
      test "a traced value's elements cannot be read" (fun () ->
          raises_match jit_error (fun () ->
              trace (fun () -> Nx.to_array (Nx.copy (grid 2 2)))));
    ]

(* Refusals *)

let metal _ =
  match Tolk_next.Device.renderer "METAL" with
  | Ok r -> r
  | Error e -> failwith e

let refusals =
  group "refusals"
    [
      test "a dtype the device's renderer lacks is refused, naming the device"
        (fun () ->
          raises
            (Lower.Jit_error
               "cannot compile contiguous: float64 is not supported on CPU")
            (fun () ->
              trace ~renderer:metal (fun () ->
                  Nx.copy (Nx.create Nx.float64 [| 2 |] [| 1.; 2. |]))));
      test "a dtype with no counterpart is refused" (fun () ->
          raises
            (Lower.Jit_error "cannot compile contiguous: int4 is not supported")
            (fun () ->
              trace (fun () -> Nx.copy (Nx.create Nx.int4 [| 2 |] [| 1; 2 |]))));
      test "a Fourier transform is refused" (fun () ->
          let z = Nx.cast Nx.complex64 (arange 4) in
          raises (Lower.Jit_error "cannot compile fft") (fun () ->
              trace (fun () -> Nx.fft z)));
    ]

let () =
  exit
  @@ run "lower"
       [ dtypes; captures; parameters; movements; placements; reads; refusals ]
