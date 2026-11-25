(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Prim.Sort: the kernel library's order and stability, reversed where
   descending, the first k of each slice, every dtype sub-byte ones included,
   and its rule before any kernel. Where a Sort over cut values lies is
   test_windows' window law. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module C = Nx_support.Counting
module Count = (val Nx.devices ~kernels:(module C) [ Nx_support.memory 0 ])

let read x =
  A.to_array (Option.get (Nx.Repr.array ~by:"test" (Nx.place Nx.Host.on x)))

let value a = Nx.Repr.of_array Nx.Host.v a

let sort ?(descending = false) ?k axis x =
  Nx.Prim.eval ~by:"t" (Sort { axis; descending; k; x })

let bits = Int64.bits_of_float
let nan_with k = Int64.float_of_bits (Int64.logor 0x7ff8_0000_0000_0000L k)

(* Floats in the sort order, NaNs last; descending, its exact reverse. Equal
   elements, NaNs of two payloads and the two [-0.] among them, keep their order
   of position either way, each with its bits. *)
let test_float_order () =
  let a = nan_with 3L and b = nan_with 5L in
  let xs =
    [| 1.; a; -0.; Float.neg_infinity; 0.; b; Float.infinity; -0.; -2. |]
  in
  let v, p = sort 0 (value (A.of_array D.Float64 [| 9 |] xs)) in
  equal ~msg:"ascending" (array int64)
    (Array.map bits
       [| Float.neg_infinity; -2.; -0.; -0.; 0.; 1.; Float.infinity; a; b |])
    (Array.map bits (read v));
  equal ~msg:"positions" (array int64)
    [| 3L; 8L; 2L; 7L; 4L; 0L; 6L; 1L; 5L |]
    (read p);
  let v, p =
    sort ~descending:true 0 (value (A.of_array D.Float64 [| 9 |] xs))
  in
  equal ~msg:"descending" (array int64)
    (Array.map bits
       [| a; b; Float.infinity; 1.; 0.; -0.; -0.; -2.; Float.neg_infinity |])
    (Array.map bits (read v));
  equal ~msg:"descending positions" (array int64)
    [| 1L; 5L; 6L; 0L; 4L; 2L; 7L; 8L; 3L |]
    (read p)

let test_k () =
  let x =
    value (A.of_array D.Int32 [| 2; 4 |] [| 5l; 1l; 4l; 1l; 0l; 9l; 9l; 3l |])
  in
  let v, p = sort ~descending:true ~k:2 1 x in
  equal ~msg:"shape" (array int) [| 2; 2 |] (Nx.shape v);
  equal ~msg:"the first two of each row" (array int32) [| 5l; 4l; 9l; 9l |]
    (read v);
  equal ~msg:"positions" (array int64) [| 0L; 2L; 1L; 2L |] (read p);
  let v, p = sort ~k:0 1 x in
  equal ~msg:"none kept"
    (pair (array int) (array int))
    ([| 2; 0 |], [| 2; 0 |])
    (Nx.shape v, Nx.shape p)

(* A dtype packed below a byte, its elements, their sorted values and the
   positions they come from. *)
type packed =
  | Packed :
      string * ('v, 's) D.t * 'v testable * 'v array * 'v array * int64 array
      -> packed

let sub_byte (Packed (_, dt, w, xs, want, positions)) =
  let v, p = sort 0 (value (A.of_array dt [| Array.length xs |] xs)) in
  equal ~msg:"values" (array w) want (read v);
  equal ~msg:"positions" (array int64) positions (read p)

(* float4_e2m1fn's [-0.] sorts below its [+0.]: compared by bits. *)
let float4 =
  Testable.make ~pp:Format.pp_print_float ~equal:(fun a b -> bits a = bits b)

let packed =
  [
    Packed
      ( "int4",
        D.Int4,
        int,
        [| 3; -8; 7; 0; -8 |],
        [| -8; -8; 0; 3; 7 |],
        [| 1L; 4L; 3L; 0L; 2L |] );
    Packed
      ("uint4", D.Uint4, int, [| 15; 0; 9 |], [| 0; 9; 15 |], [| 1L; 2L; 0L |]);
    Packed
      ( "bit",
        D.Bit,
        bool,
        [| true; false; true |],
        [| false; true; true |],
        [| 1L; 0L; 2L |] );
    Packed
      ( "float4_e2m1fn",
        D.Float4_e2m1fn,
        float4,
        [| 0.; -0.; 6.; -1.5 |],
        [| -1.5; -0.; 0.; 6. |],
        [| 3L; 1L; 0L; 2L |] );
  ]

(* A k past the axis, an axis past the rank and a negative k raise before any
   kernel is called. *)
let refused (_, (k, axis)) =
  let x =
    Nx.place Count.on (value (A.of_array D.Int32 [| 2; 3 |] (Array.make 6 0l)))
  in
  C.reset ();
  raises_match (Exn.invalid_arg ~sub:"t: ") (fun () -> sort ?k axis x);
  equal ~msg:"kernel calls" int 0 (C.calls ())

let () =
  exit
    (run "nx sort"
       [
         group "Prim.Sort"
           [
             test "floats sort as the sort order, NaNs last, stably both ways"
               test_float_order;
             test "k keeps each slice's first k" test_k;
             cases "a dtype packed below a byte sorts with its positions"
               ~id:(fun (Packed (name, _, _, _, _, _)) -> name)
               packed sub_byte;
             cases "the rule raises before any kernel" ~id:fst
               [
                 ("k past the axis", (Some 4, 1));
                 ("an axis past the rank", (None, 2));
                 ("a negative k", (Some (-1), 0));
               ]
               refused;
           ];
       ])
