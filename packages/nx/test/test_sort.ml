(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Sort through Nx.Prim: the order and stability Nx_kernel.S.sort states,
   sub-byte dtypes in one kernel call, its rule before any kernel, and a split
   placement. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module C = Nx_support.Counting

let m = Nx_support.memory

module Count = (val Nx.devices ~kernels:(module C) [ m 0 ])
module Count2 = (val Nx.devices ~kernels:(module C) [ m 0; m 1 ])

let host_array x = Option.get (Nx.Repr.array (Nx.place Nx.Host.on x))
let read x = A.to_array (host_array x)
let value a = Nx.Repr.of_array Nx.Host.v a

let sort ?(descending = false) ?k axis x =
  Nx.Prim.eval ~by:"t" (Sort { axis; descending; k; x })

let bits = Int64.bits_of_float
let nan_with k = Int64.float_of_bits (Int64.logor 0x7ff8_0000_0000_0000L k)

(* The order *)

let test_float_order () =
  let a = nan_with 3L and b = nan_with 5L in
  let xs =
    [| 1.; a; -0.; Float.neg_infinity; 0.; b; Float.infinity; -0.; -2. |]
  in
  let v, p = sort 0 (value (A.of_array D.Float64 [| 9 |] xs)) in
  equal ~msg:"ascending, NaNs last, each its bits, in position order"
    (array int64)
    (Array.map bits
       [| Float.neg_infinity; -2.; -0.; -0.; 0.; 1.; Float.infinity; a; b |])
    (Array.map bits (read v));
  equal ~msg:"positions" (array int64)
    [| 3L; 8L; 2L; 7L; 4L; 0L; 6L; 1L; 5L |]
    (read p);
  let v, p =
    sort ~descending:true 0 (value (A.of_array D.Float64 [| 9 |] xs))
  in
  equal ~msg:"descending keeps equal elements in increasing position"
    (array int64)
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

let test_sub_byte () =
  let check (type v s) name (dt : (v, s) D.t) (pp : v testable) (xs : v array)
      (want : v array) (positions : int64 array) =
    let x = Nx.place Count.on (value (A.of_array dt [| Array.length xs |] xs)) in
    C.reset ();
    let v, p = sort 0 x in
    equal ~msg:(name ^ " in one kernel call") int 1 (C.calls ());
    equal ~msg:(name ^ " values") (array pp) want (read v);
    equal ~msg:(name ^ " positions") (array int64) positions (read p)
  in
  check "int4" D.Int4 int [| 3; -8; 7; 0; -8 |] [| -8; -8; 0; 3; 7 |]
    [| 1L; 4L; 3L; 0L; 2L |];
  check "uint4" D.Uint4 int [| 15; 0; 9 |] [| 0; 9; 15 |] [| 1L; 2L; 0L |];
  check "bit" D.Bit bool [| true; false; true |] [| false; true; true |]
    [| 1L; 0L; 2L |];
  let x =
    Nx.place Count.on
      (value (A.of_array D.Float4_e2m1fn [| 4 |] [| 0.; -0.; 6.; -1.5 |]))
  in
  C.reset ();
  let v, p = sort 0 x in
  equal ~msg:"float4_e2m1fn in one kernel call" int 1 (C.calls ());
  equal ~msg:"float4_e2m1fn, -0 below +0" (array int64)
    (Array.map bits [| -1.5; -0.; 0.; 6. |])
    (Array.map bits (read v));
  equal ~msg:"float4_e2m1fn positions" (array int64) [| 3L; 1L; 0L; 2L |]
    (read p)

(* Rules and placements *)

let test_rules () =
  let x =
    Nx.place Count.on (value (A.of_array D.Int32 [| 2; 3 |] (Array.make 6 0l)))
  in
  C.reset ();
  raises_match (Exn.invalid_arg ~substring:"t: ") (fun () -> sort ~k:4 1 x);
  raises_match (Exn.invalid_arg ~substring:"t: ") (fun () -> sort 2 x);
  raises_match (Exn.invalid_arg ~substring:"t: ") (fun () -> sort ~k:(-1) 0 x);
  equal ~msg:"kernel calls" int 0 (C.calls ())

let test_split () =
  let xs = [| 4l; 1l; 3l; 3l; 0l; 2l; 7l; -1l |] in
  let x = value (A.of_array D.Int32 [| 4; 2 |] xs) in
  let one = sort 1 (Nx.place Count.on x) in
  let two = sort 1 (Nx.place (Count2.split ~axis:0) x) in
  equal ~msg:"values" (array int32) (read (fst one)) (read (fst two));
  equal ~msg:"positions" (array int64) (read (snd one)) (read (snd two));
  equal ~msg:"split as its operand"
    (Testable.make ~pp:Nx.Placement.pp ~equal:Nx.Placement.equal)
    (Count2.split ~axis:0)
    (Nx.placement (fst two))

let () =
  exit
    (run "nx sort"
       [
         group "order"
           [
             test "floats: -inf < -0 < +0 < +inf < NaN, stable" test_float_order;
             test "k keeps the first elements" test_k;
             test "sub-byte dtypes sort in one kernel call" test_sub_byte;
           ];
         group "rules and placements"
           [
             test "the rule raises before any kernel" test_rules;
             test "a split value sorts as on one device" test_split;
           ];
       ])
