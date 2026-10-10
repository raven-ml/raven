(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Sort through Nx.Prim: the order and stability Nx_kernel.Spec.sort states,
   against a stable sort of the slices, its rule before any kernel, a split
   placement, a sub-byte dtype the host's kernels decline, and a library that
   declines every sort. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module C = Nx_support.Counting

let m = Nx_support.memory

module Count = (val Nx.devices ~kernels:(module C) [ m 0 ])
module Count2 = (val Nx.devices ~kernels:(module C) [ m 0; m 1 ])

(* nx.cpu, declining every sort. *)
module Unsorted = struct
  include Nx_cpu

  let name = "nx.unsorted"
  let sort _ ~values:_ ~positions:_ _ = A.Declined
end

module Dec = (val Nx.devices ~kernels:(module Unsorted) [ m 2 ])

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

(* A stable sort of each slice along [axis], as positions. *)
let reference descending s axis xs =
  let r = Array.length s in
  let n = s.(axis) in
  let inner = Array.fold_left ( * ) 1 (Array.sub s (axis + 1) (r - axis - 1)) in
  let outer = Array.fold_left ( * ) 1 (Array.sub s 0 axis) in
  let values = Array.copy xs and positions = Array.make (Array.length xs) 0L in
  for o = 0 to outer - 1 do
    for i = 0 to inner - 1 do
      let at j = (((o * n) + j) * inner) + i in
      let order =
        List.stable_sort
          (fun j l ->
            let c = Int32.compare xs.(at j) xs.(at l) in
            if descending then -c else c)
          (List.init n Fun.id)
      in
      List.iteri
        (fun t j ->
          values.(at t) <- xs.(at j);
          positions.(at t) <- Int64.of_int j)
        order
    done
  done;
  (values, positions)

let case =
  let open Gen in
  let* s = array ~size:(int_range 1 3) (int_range 0 5) in
  let* axis = int_range 0 (Array.length s - 1) in
  let* descending = bool in
  let+ xs =
    array
      ~size:(constant (Array.fold_left ( * ) 1 s))
      (map Int32.of_int (int_range (-3) 3))
  in
  (s, axis, descending, xs)

let law_stable (s, axis, descending, xs) =
  cover "an empty slice" (Array.exists (( = ) 0) s);
  cover "descending" descending;
  let v, p = sort ~descending axis (value (A.of_array D.Int32 s xs)) in
  let want_v, want_p = reference descending s axis xs in
  equal ~msg:"values" (array int32) want_v (read v);
  equal ~msg:"positions" (array int64) want_p (read p)

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
  equal ~msg:"split as its operand" bool true
    (Option.equal Nx.Placement.equal
       (Nx.placement (fst two))
       (Some (Count2.split ~axis:0)))

(* Declines *)

let test_sub_byte () =
  let check (type v s) name (dt : (v, s) D.t) (pp : v testable) (xs : v array)
      (want : v array) (positions : int64 array) =
    let v, p = sort 0 (value (A.of_array dt [| Array.length xs |] xs)) in
    equal ~msg:(name ^ " values") (array pp) want (read v);
    equal ~msg:(name ^ " positions") (array int64) positions (read p)
  in
  check "int4" D.Int4 int [| 3; -8; 7; 0; -8 |] [| -8; -8; 0; 3; 7 |]
    [| 1L; 4L; 3L; 0L; 2L |];
  check "uint4" D.Uint4 int [| 15; 0; 9 |] [| 0; 9; 15 |] [| 1L; 2L; 0L |];
  check "bit" D.Bit bool [| true; false; true |] [| false; true; true |]
    [| 1L; 0L; 2L |];
  let v, p =
    sort 0 (value (A.of_array D.Float4_e2m1fn [| 4 |] [| 0.; -0.; 6.; -1.5 |]))
  in
  equal ~msg:"float4_e2m1fn, -0 below +0" (array int64)
    (Array.map bits [| -1.5; -0.; 0.; 6. |])
    (Array.map bits (read v));
  equal ~msg:"float4_e2m1fn positions" (array int64) [| 3L; 1L; 0L; 2L |]
    (read p)

let test_declined () =
  let x =
    Nx.place Dec.on (value (A.of_array D.Float32 [| 3 |] [| 2.; 1.; 0. |]))
  in
  match sort 0 x with
  | _ -> fail "a declined sort computed"
  | exception Invalid_argument e ->
      List.iter
        (fun sub -> contains ~msg:sub ~sub e)
        [
          "t: ";
          "nx.unsorted";
          "Sort";
          "m2";
          "float32";
          "not available yet";
          "Nx.place";
        ]

let () =
  exit
    (run "nx sort"
       [
         group "order"
           [
             test "floats: -inf < -0 < +0 < +inf < NaN, stable" test_float_order;
             test "k keeps the first elements" test_k;
             prop "each slice is a stable sort" case law_stable;
           ];
         group "rules and placements"
           [
             test "the rule raises before any kernel" test_rules;
             test "a split value sorts as on one device" test_split;
           ];
         group "declines"
           [
             test "a sub-byte sort computes at its accumulator" test_sub_byte;
             test "a declined sort raises naming the move" test_declined;
           ];
       ])
