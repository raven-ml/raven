(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The dispatch and the loads of the lowering, through operations run on the
   host against nx's eager result. *)

open Windtrap
module D = Nx_array.Dtype
module P = Nx.Prim

let copy x = Lowered.run (fun () -> P.eval ~by:"t" (P.Copy x))

let smoke =
  group "copies"
    [
      test "a transposed view is its elements" (fun () ->
          let x =
            Nx.transpose
              (Nx.create D.Int32 [| 2; 3 |] [| 0l; 1l; 2l; 3l; 4l; 5l |])
          in
          equal (array int32) (Nx.to_array x) (copy x));
      test "an int4 value is its elements" (fun () ->
          let x = Nx.create D.Int4 [| 4 |] [| -8; -1; 0; 7 |] in
          equal (array int) (Nx.to_array x) (copy x));
      test "a map of coordinates is their values" (fun () ->
          let prog = Nx_kernel.Prog.v ~ins:[||] [| Coord 1 |] ~outs:[| 0 |] in
          let map () =
            let y, () =
              P.eval ~by:"t"
                (P.Map
                   {
                     layout = Nx_array.Layout.contiguous [| 2; 3 |];
                     prog;
                     outs = [ D.Int64 ];
                     loads = Iarray.of_array [||];
                   })
            in
            y
          in
          equal (array int64) [| 0L; 1L; 2L; 0L; 1L; 2L |] (Lowered.run map));
    ]

(* Movements and padded loads move elements: each gives eager's elements. *)

module M = Nx_array.Move

let ints shape =
  Nx.create D.Int32 shape
    (Array.init (Array.fold_left ( * ) 1 shape) Int32.of_int)

let complexes shape =
  let n = Array.fold_left ( * ) 1 shape in
  Nx.create D.Complex64 shape
    (Array.init n (fun i ->
         { Complex.re = Float.of_int i; im = -.Float.of_int i }))

let moved m x = P.eval ~by:"t" (P.Move (m, x))

let agrees_move t m x =
  equal (array t) (Nx.to_array (moved m x)) (Lowered.run (fun () -> moved m x))

let range start count step = { M.start; count; step }
let window axis size step dilation = { M.axis; size; step; dilation }

let complex =
  Testable.make
    ~pp:(fun ppf (c : Complex.t) -> Format.fprintf ppf "%h%+hi" c.re c.im)
    ~equal:(fun (a : Complex.t) (b : Complex.t) ->
      Int64.equal (Int64.bits_of_float a.re) (Int64.bits_of_float b.re)
      && Int64.equal (Int64.bits_of_float a.im) (Int64.bits_of_float b.im))

let moves =
  let a = Iarray.of_array in
  group "movements"
    [
      test "a reshape" (fun () ->
          agrees_move int32 (M.Reshape (a [| 3; 4 |])) (ints [| 2; 6 |]));
      test "a broadcast adds and repeats axes" (fun () ->
          agrees_move int32 (M.Broadcast (a [| 2; 3; 4 |])) (ints [| 3; 1 |]));
      test "a permutation" (fun () ->
          agrees_move int32 (M.Permute (a [| 2; 0; 1 |])) (ints [| 2; 3; 4 |]));
      test "a slice by steps of both signs" (fun () ->
          agrees_move int32
            (M.Slice (a [| range 1 3 2; range 5 3 (-2) |]))
            (ints [| 7; 6 |]));
      test "a slice that keeps nothing" (fun () ->
          agrees_move int32
            (M.Slice (a [| range 0 0 3; range 0 2 1 |]))
            (ints [| 4; 2 |]));
      test "a step past the run's end" (fun () ->
          agrees_move int32 (M.Slice (a [| range 2 1 5 |])) (ints [| 4 |]));
      test "windows with a step and a dilation" (fun () ->
          agrees_move int32
            (M.Window (a [| window 0 2 2 1; window 1 2 1 2 |]))
            (ints [| 5; 6 |]));
      test "a complex value moves part by part" (fun () ->
          agrees_move complex
            (M.Slice (a [| range 3 2 (-1); range 0 2 2 |]))
            (complexes [| 4; 3 |]));
    ]

let identity dt =
  Nx_kernel.Prog.v
    ~ins:(Iarray.of_array [| D.Any dt |])
    (Iarray.of_array [| Nx_kernel.Prog.In 0 |])
    ~outs:(Iarray.of_array [| 0 |])

let padded (type v s) (x : (v, s, _) Nx.t) (fill : v) pad =
  let dt = Nx.dtype x in
  let shape = Result.get_ok (Nx_kernel.Shape.padded pad (Nx.shape x)) in
  let y, () =
    P.eval ~by:"t"
      (P.Map
         {
           layout = Nx_array.Layout.contiguous shape;
           prog = identity dt;
           outs = [ dt ];
           loads = Iarray.of_array [| P.Padded { x; fill; pad } |];
         })
  in
  y

let pad ?(windows = [||]) lo hi interior =
  let a = Iarray.of_array in
  { Nx_kernel.lo = a lo; hi = a hi; interior = a interior; windows = a windows }

let agrees_pad t x fill pad =
  equal (array t)
    (Nx.to_array (padded x fill pad))
    (Lowered.run (fun () -> padded x fill pad))

let loads =
  group "padded loads"
    [
      test "ends padded with the fill" (fun () ->
          agrees_pad int32
            (ints [| 2; 3 |])
            (-7l)
            (pad [| 1; 0 |] [| 2; 3 |] [| 0; 0 |]));
      test "negative ends crop" (fun () ->
          agrees_pad int32
            (ints [| 4; 5 |])
            9l
            (pad [| -1; 2 |] [| 1; -3 |] [| 0; 0 |]));
      test "interior padding spaces elements" (fun () ->
          agrees_pad int32
            (ints [| 3; 2 |])
            (-1l)
            (pad [| 0; 1 |] [| 1; 0 |] [| 2; 1 |]));
      test "windows over the padded value" (fun () ->
          agrees_pad int32
            (ints [| 5; 4 |])
            0l
            (pad
               ~windows:[| window 0 3 2 1; window 1 2 1 2 |]
               [| 1; 1 |] [| 1; 1 |] [| 0; 1 |]));
      test "a complex fill pads each part" (fun () ->
          agrees_pad complex
            (complexes [| 2; 2 |])
            { Complex.re = -0.; im = Float.nan }
            (pad [| 1; 0 |] [| 0; 1 |] [| 1; 0 |]));
    ]

let () = exit (run "Lower" [ smoke; moves; loads ])
