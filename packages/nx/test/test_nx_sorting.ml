(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Sorting and searching tests for Nx *)

open Windtrap
open Test_nx_support

(* ───── Where Tests ───── *)

let test_where_1d () =
  let mask = Nx.create Nx.bool [| 3 |] [| true; false; true |] in
  let a = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let b = Nx.create Nx.float32 [| 3 |] [| 4.; 5.; 6. |] in
  let result = Nx.where mask a b in
  check_t "where 1D" [| 3 |] [| 1.; 5.; 3. |] result

let test_where_broadcast () =
  let mask = Nx.create Nx.bool [| 2; 1 |] [| true; false |] in
  let a = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let b = Nx.create Nx.float32 [| 2; 3 |] [| 7.; 8.; 9.; 10.; 11.; 12. |] in
  let result = Nx.where mask a b in
  check_t "where with broadcasting" [| 2; 3 |]
    [| 1.; 2.; 3.; 10.; 11.; 12. |]
    result

let test_where_scalar_inputs () =
  let mask =
    Nx.create Nx.bool [| 2; 3 |] [| true; false; true; false; true; false |]
  in
  let a = Nx.scalar Nx.float32 5.0 in
  let b = Nx.scalar Nx.float32 10.0 in
  let result = Nx.where mask a b in
  check_t "where with scalar inputs" [| 2; 3 |]
    [| 5.0; 10.0; 5.0; 10.0; 5.0; 10.0 |]
    result

let test_where_invalid_shapes () =
  let mask = Nx.create Nx.bool [| 2 |] [| true; false |] in
  let a = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let b = Nx.create Nx.float32 [| 2 |] [| 4.; 5. |] in
  raises ~msg:"where invalid shapes"
    (Invalid_argument
       "broadcast: cannot broadcast [3] with [2] (dim 0: 3\226\137\1602)")
    (fun () -> ignore (Nx.where mask a b))

(* ───── Sort Tests ───── *)

let test_sort_2d_axis0 () =
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 4.; 1.; 3.; 2.; 5.; 6. |] in
  let result, indices = Nx.sort ~axis:0 t in
  check_t "sort 2D axis 0 values" [| 2; 3 |] [| 2.; 1.; 3.; 4.; 5.; 6. |] result;
  check_t "sort 2D axis 0 indices" [| 2; 3 |]
    [| 1l; 0l; 0l; 0l; 1l; 1l |]
    indices

let test_sort_2d_axis1 () =
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 4.; 1.; 3.; 2.; 5.; 6. |] in
  let result, indices = Nx.sort ~axis:1 t in
  check_t "sort 2D axis 1 values" [| 2; 3 |] [| 1.; 3.; 4.; 2.; 5.; 6. |] result;
  check_t "sort 2D axis 1 indices" [| 2; 3 |]
    [| 1l; 2l; 0l; 0l; 1l; 2l |]
    indices

let test_sort_invalid_axis () =
  let t = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  check_invalid_arg "sort invalid axis"
    "sort: axis 2 out of bounds for 2D tensor" (fun () -> Nx.sort ~axis:2 t)

let test_sort_nan_handling () =
  let t = Nx.create Nx.float32 [| 5 |] [| 3.; nan; 1.; 2.; nan |] in
  let result, _ = Nx.sort t in
  (* NaN values should be sorted to the end *)
  let first_three = Nx.slice [ Nx.R (0, 3) ] result in
  check_t "sort NaN handling - non-NaN values" [| 3 |] [| 1.; 2.; 3. |]
    first_three;
  (* Check that last two values are NaN *)
  equal ~msg:"sort NaN handling - NaN at end" bool true
    (Float.is_nan (Nx.item [ 3 ] result) && Float.is_nan (Nx.item [ 4 ] result))

let test_sort_stable () =
  (* Test sort stability with repeated values *)
  let t = Nx.create Nx.float32 [| 6 |] [| 3.; 1.; 2.; 1.; 3.; 2. |] in
  let _, indices = Nx.sort t in
  (* For stable sort, original order should be preserved for equal elements *)
  check_t "sort stable indices" [| 6 |] [| 1l; 3l; 2l; 5l; 0l; 4l |] indices

(* ───── Argsort Tests ───── *)

let test_argsort_1d () =
  let t = Nx.create Nx.float32 [| 5 |] [| 3.; 1.; 4.; 1.; 5. |] in
  let result = Nx.argsort t in
  check_t "argsort 1D" [| 5 |] [| 1l; 3l; 0l; 2l; 4l |] result

let test_argsort_2d_axis0 () =
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 4.; 1.; 3.; 2.; 5.; 6. |] in
  let result = Nx.argsort ~axis:0 t in
  check_t "argsort 2D axis 0" [| 2; 3 |] [| 1l; 0l; 0l; 0l; 1l; 1l |] result

let test_argsort_2d_axis1 () =
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 4.; 1.; 3.; 2.; 5.; 6. |] in
  let result = Nx.argsort ~axis:1 t in
  check_t "argsort 2D axis 1" [| 2; 3 |] [| 1l; 2l; 0l; 0l; 1l; 2l |] result

let test_argsort_empty () =
  let t = Nx.create Nx.float32 [| 0 |] [||] in
  let result = Nx.argsort t in
  check_t "argsort empty" [| 0 |] [||] result

(* ───── Argmax Tests ───── *)

let test_argmax_1d () =
  let t = Nx.create Nx.float32 [| 5 |] [| 3.; 1.; 4.; 1.; 5. |] in
  let result = Nx.argmax t in
  check_t "argmax 1D" [||] [| 4l |] result

let test_argmax_2d_axis0 () =
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let result = Nx.argmax ~axis:0 t in
  check_t "argmax 2D axis 0" [| 3 |] [| 1l; 1l; 1l |] result

let test_argmax_2d_axis1 () =
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let result = Nx.argmax ~axis:1 t in
  check_t "argmax 2D axis 1" [| 2 |] [| 2l; 2l |] result

let test_argmax_keepdims () =
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let result = Nx.argmax ~axis:1 ~keepdims:true t in
  check_shape "argmax keepdims shape" [| 2; 1 |] result;
  check_t "argmax keepdims values" [| 2; 1 |] [| 2l; 2l |] result

let test_argmax_nan () =
  let t = Nx.create Nx.float32 [| 4 |] [| 1.; nan; 3.; 2. |] in
  let result = Nx.argmax t in
  (* NaN handling may vary - just check it doesn't crash *)
  check_shape "argmax with NaN" [||] result

(* ───── Argmin Tests ───── *)

let test_argmin_1d () =
  let t = Nx.create Nx.float32 [| 5 |] [| 3.; 1.; 4.; 1.; 5. |] in
  let result = Nx.argmin t in
  check_t "argmin 1D" [||] [| 1l |] result

let test_argmin_2d_axis0 () =
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let result = Nx.argmin ~axis:0 t in
  check_t "argmin 2D axis 0" [| 3 |] [| 0l; 0l; 0l |] result

let test_argmin_2d_axis1 () =
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let result = Nx.argmin ~axis:1 t in
  check_t "argmin 2D axis 1" [| 2 |] [| 0l; 0l |] result

let test_argmin_ties () =
  let t = Nx.create Nx.float32 [| 5 |] [| 3.; 1.; 2.; 1.; 3. |] in
  let result = Nx.argmin t in
  (* Should return first occurrence *)
  check_t "argmin ties" [||] [| 1l |] result

(* ───── Top-k Tests ───── *)

let test_top_k_1d () =
  let t = Nx.create Nx.float32 [| 5 |] [| 3.; 1.; 4.; 1.; 5. |] in
  let values, indices = Nx.top_k ~k:2 t in
  check_t "top_k 1D values" [| 2 |] [| 5.; 4. |] values;
  check_t "top_k 1D indices" [| 2 |] [| 4l; 2l |] indices

let test_top_k_axes () =
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 4.; 1.; 3.; 2.; 5.; 6. |] in
  let values, indices = Nx.top_k ~k:2 t in
  check_t "top_k last axis values" [| 2; 2 |] [| 4.; 3.; 6.; 5. |] values;
  check_t "top_k last axis indices" [| 2; 2 |] [| 0l; 2l; 2l; 1l |] indices;
  let values, indices = Nx.top_k ~k:1 ~axis:0 t in
  check_t "top_k axis 0 values" [| 1; 3 |] [| 4.; 5.; 6. |] values;
  check_t "top_k axis 0 indices" [| 1; 3 |] [| 0l; 1l; 1l |] indices

let test_top_k_ties () =
  let t = Nx.create Nx.float32 [| 6 |] [| 2.; 7.; 7.; 2.; 7.; 1. |] in
  let values, indices = Nx.top_k ~k:4 t in
  check_t "top_k ties values" [| 4 |] [| 7.; 7.; 7.; 2. |] values;
  check_t "top_k ties take the lowest position first" [| 4 |]
    [| 1l; 2l; 4l; 0l |] indices

(* An entry equal to the least value of its dtype is still an entry: it must not
   be confused with one already taken. *)
let test_top_k_least_values () =
  let ninf = Float.neg_infinity in
  let t = Nx.create Nx.float32 [| 4 |] [| ninf; 1.; ninf; ninf |] in
  let values, indices = Nx.top_k ~k:4 t in
  check_t "top_k -inf values" [| 4 |] [| 1.; ninf; ninf; ninf |] values;
  check_t "top_k -inf indices" [| 4 |] [| 1l; 0l; 2l; 3l |] indices;
  let t = Nx.create Nx.int32 [| 3 |] [| Int32.min_int; 5l; Int32.min_int |] in
  let values, indices = Nx.top_k ~k:3 t in
  check_t "top_k min_int values" [| 3 |]
    [| 5l; Int32.min_int; Int32.min_int |]
    values;
  check_t "top_k min_int indices" [| 3 |] [| 1l; 0l; 2l |] indices;
  let t = Nx.create Nx.uint8 [| 4 |] [| 0; 0; 9; 0 |] in
  let _, indices = Nx.top_k ~k:3 t in
  check_t "top_k uint8 zeros" [| 3 |] [| 2l; 0l; 1l |] indices

let test_top_k_nan () =
  let t = Nx.create Nx.float32 [| 5 |] [| 1.; Float.nan; 9.; Float.nan; 3. |] in
  let values, indices = Nx.top_k ~k:4 t in
  check_t "top_k NaN comes after every number" [| 4 |] [| 2l; 4l; 0l; 1l |]
    indices;
  let v = Nx.to_array values in
  equal ~msg:"top_k NaN values" bool true
    (v.(0) = 9. && v.(1) = 3. && v.(2) = 1. && Float.is_nan v.(3))

(* float8_e4m3 has no infinity: its least and greatest values are -448 and 448,
   which the selection rounds must still order as entries. *)
let test_top_k_float8_e4m3 () =
  let t =
    Nx.create Nx.float8_e4m3 [| 8 |]
      [| 1.; Float.nan; -448.; 448.; -2.; 448.; 0.5; -448. |]
  in
  let sorted_values, sorted_indices = Nx.sort ~descending:true t in
  List.iter
    (fun k ->
      let values, indices = Nx.top_k ~k t in
      let prefix a = Nx.to_array (Nx.slice [ Nx.R (0, k) ] a) in
      let msg what = Printf.sprintf "top %d %s" k what in
      equal ~msg:(msg "indices") (array int32) (prefix sorted_indices)
        (Nx.to_array indices);
      equal ~msg:(msg "values") (array float_exact) (prefix sorted_values)
        (Nx.to_array values))
    [ 1; 2; 4; 7; 8 ]

(* Both algorithms, either side of the switch between them, are the first [k]
   entries of a descending sort: duplicates, NaN and infinities included. *)
let test_top_k_is_a_sorted_prefix () =
  let n = 40 in
  let data =
    Array.init (3 * n) (fun i ->
        match i mod 11 with
        | 0 -> Float.nan
        | 1 -> Float.neg_infinity
        | 2 -> Float.infinity
        | _ -> float_of_int (i * 7919 mod 13))
  in
  let t = Nx.create Nx.float32 [| 3; n |] data in
  let sorted_indices = Nx.argsort ~descending:true t in
  List.iter
    (fun k ->
      let _, indices = Nx.top_k ~k t in
      let expected = Nx.slice [ Nx.A; Nx.R (0, k) ] sorted_indices in
      equal
        ~msg:(Printf.sprintf "top_k %d indices are argsort's first" k)
        bool true
        (Nx.to_array indices = Nx.to_array expected))
    [ 1; 5; 16; 17; n ]

(* The first [k] positions of a stable descending sort along [axis]. *)
let sorted_prefix ~k ~axis t =
  let bounds = Array.map (fun d -> (0, d)) (Nx.shape t) in
  bounds.(axis) <- (0, k);
  Nx.shrink bounds (Nx.argsort ~descending:true ~axis t)

let check_sorted_prefix ~msg ~k ?(axis = -1) t =
  let axis = if axis < 0 then axis + Nx.ndim t else axis in
  check_nx msg (sorted_prefix ~k ~axis t) (snd (Nx.top_k ~k ~axis t))

(* Rows long enough to pad the radix counts' chunks, of values drawn from a pool
   that repeats: every [k] cuts through a run of ties. *)
let scores dtype pool =
  let st = Random.State.make [| 7 |] in
  let pool = Array.of_list pool in
  Nx.cast dtype
    (Nx.init Nx.float64 [| 3; 2500 |] (fun _ ->
         if Random.State.int st 3 = 0 then
           pool.(Random.State.int st (Array.length pool))
         else Random.State.float st 8. -. 4.))

let check_every_k ~msg t =
  List.iter
    (fun k -> check_sorted_prefix ~msg:(Printf.sprintf "%s, k = %d" msg k) ~k t)
    [ 17; 100; 512; 513; 2500 ];
  (* A short axis is sorted, on the same keys. *)
  let short = Nx.slice [ Nx.A; Nx.R (0, 1000) ] t in
  List.iter
    (fun k ->
      check_sorted_prefix
        ~msg:(Printf.sprintf "%s, short, k = %d" msg k)
        ~k short)
    [ 17; 1000 ]

let test_top_k_radix_floats () =
  let special =
    [ Float.nan; -0.; 0.; Float.infinity; Float.neg_infinity; 1.; -1.; 2.5 ]
  in
  check_every_k ~msg:"float32" (scores Nx.float32 special);
  check_every_k ~msg:"float64" (scores Nx.float64 special);
  check_every_k ~msg:"float16" (scores Nx.float16 (6e-8 :: special));
  check_every_k ~msg:"bfloat16" (scores Nx.bfloat16 special);
  check_every_k ~msg:"float8_e4m3" (scores Nx.float8_e4m3 [ Float.nan; 448. ]);
  check_every_k ~msg:"float8_e5m2" (scores Nx.float8_e5m2 special);
  check_every_k ~msg:"float32 subnormals"
    (scores Nx.float32 [ 1e-45; -1e-45; 1e-40; -0.; Float.min_float ])

(* Integers draw from the whole range of their dtype, its ends included:
   unsigned ones get the upper half of the range from wrapped negatives. *)
let test_top_k_radix_ints () =
  let ints dtype lo hi =
    let st = Random.State.make [| 11 |] in
    Nx.cast dtype
      (Nx.init Nx.int64 [| 3; 2500 |] (fun _ ->
           match Random.State.int st 8 with
           | 0 -> lo
           | 1 -> hi
           | 2 -> Int64.of_int (Random.State.int st 5)
           | _ -> Random.State.int64 st hi))
  in
  let each_sign dtype width =
    let hi = Int64.pred (Int64.shift_left 1L (width - 1)) in
    ints dtype (Int64.neg (Int64.succ hi)) hi
  in
  check_every_k ~msg:"int8" (each_sign Nx.int8 8);
  check_every_k ~msg:"int16" (each_sign Nx.int16 16);
  check_every_k ~msg:"int32" (each_sign Nx.int32 32);
  check_every_k ~msg:"int64" (ints Nx.int64 Int64.min_int Int64.max_int);
  check_every_k ~msg:"uint8" (each_sign Nx.uint8 8);
  check_every_k ~msg:"uint16" (each_sign Nx.uint16 16);
  check_every_k ~msg:"uint32" (each_sign Nx.uint32 32);
  check_every_k ~msg:"uint64" (ints Nx.uint64 Int64.min_int Int64.max_int);
  check_every_k ~msg:"bool" (ints Nx.bool 0L 1L)

(* Fewer than [k] numbers: every number, then the NaN in position order. *)
let test_top_k_radix_mostly_nan () =
  let t =
    Nx.init Nx.float32 [| 2; 2600 |] (fun i ->
        if i.(1) * 7 mod 5 = 0 then float_of_int (i.(1) mod 3) else Float.nan)
  in
  check_sorted_prefix ~msg:"NaN fill the last places" ~k:1300 t;
  let t = Nx.full Nx.float32 [| 1; 40 |] Float.nan in
  check_sorted_prefix ~msg:"all NaN" ~k:20 t

let test_top_k_radix_ties () =
  check_sorted_prefix ~msg:"one value" ~k:17 (Nx.full Nx.int32 [| 2; 2300 |] 4l);
  let t =
    Nx.init Nx.float32 [| 2; 2300 |] (fun i ->
        if i.(1) mod 2 = 0 then 1. else 0.)
  in
  check_sorted_prefix ~msg:"two values" ~k:1200 t

(* Ties this many, taken this many times, count past int32; so many selected
   entries are sorted. *)
let test_top_k_radix_long_axis () =
  let t =
    Nx.init Nx.float32 [| 1; 50_000 |] (fun i ->
        if i.(1) mod 1000 = 0 then 1. else 0.)
  in
  check_sorted_prefix ~msg:"k = n = 50000" ~k:50_000 t

let test_top_k_radix_axes () =
  let st = Random.State.make [| 3 |] in
  let t =
    Nx.init Nx.float32 [| 4; 2100; 3 |] (fun _ ->
        float_of_int (Random.State.int st 50))
  in
  check_sorted_prefix ~msg:"middle axis" ~k:40 ~axis:1 t;
  check_sorted_prefix ~msg:"first axis" ~k:3 ~axis:0 (Nx.transpose t);
  let rows =
    Nx.init Nx.float32 [| 2; 4; 2500 |] (fun _ ->
        float_of_int (Random.State.int st 50))
  in
  check_sorted_prefix ~msg:"rows of a view" ~k:40
    (Nx.slice [ Nx.A; Nx.R (0, 3); Nx.A ] rows);
  let _, indices = Nx.top_k ~k:20 (Nx.zeros Nx.float32 [| 0; 40 |]) in
  check_shape "empty batch" [| 0; 20 |] indices

let test_top_k_invalid () =
  let t = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  check_invalid_arg "top_k k = 0" "top_k: k = 0 is outside [1, 3]" (fun () ->
      Nx.top_k ~k:0 t);
  check_invalid_arg "top_k k > n" "top_k: k = 4 is outside [1, 3]" (fun () ->
      Nx.top_k ~k:4 t);
  check_invalid_arg "top_k axis" "top_k: axis 1 out of bounds for 1D tensor"
    (fun () -> Nx.top_k ~k:1 ~axis:1 t);
  check_invalid_arg "top_k scalar" "top_k: requires at least one dimension"
    (fun () -> Nx.top_k ~k:1 (Nx.scalar Nx.float32 1.))

(* ───── Sort Regression Tests ───── *)

let test_sort_large_1d () =
  (* Regression: bitonic sort breaks for n >= 129. The sort produces duplicate
     values instead of a correct permutation. *)
  let n = 150 in
  let t = Nx.arange Nx.float32 0 n 1 in
  (* Reverse so it's not already sorted *)
  let t = Nx.flip ~axes:[ 0 ] t in
  let sorted_vals, sorted_indices = Nx.sort t in
  (* Check sorted values are 0, 1, 2, ..., n-1 *)
  let expected_vals = Nx.arange Nx.float32 0 n 1 in
  check_nx "sort large 1D values" expected_vals sorted_vals;
  (* Check indices map back to original positions *)
  let expected_indices = Nx.arange Nx.int32 (n - 1) (-1) (-1) in
  check_nx "sort large 1D indices" expected_indices sorted_indices

let test_sort_power_of_two () =
  (* n=256 is a power of two (no padding needed) but still breaks *)
  let n = 256 in
  let t = Nx.arange Nx.float32 0 n 1 in
  let t = Nx.flip ~axes:[ 0 ] t in
  let sorted_vals, _ = Nx.sort t in
  let expected_vals = Nx.arange Nx.float32 0 n 1 in
  check_nx "sort power-of-two values" expected_vals sorted_vals

let test_sort_128_boundary () =
  (* n=128 works, n=129 does not *)
  let t128 = Nx.flip ~axes:[ 0 ] (Nx.arange Nx.float32 0 128 1) in
  let sorted128, _ = Nx.sort t128 in
  check_nx "sort n=128 values" (Nx.arange Nx.float32 0 128 1) sorted128;
  let t129 = Nx.flip ~axes:[ 0 ] (Nx.arange Nx.float32 0 129 1) in
  let sorted129, _ = Nx.sort t129 in
  check_nx "sort n=129 values" (Nx.arange Nx.float32 0 129 1) sorted129

(* Test Suite Organization *)

let where_tests =
  [
    test "where 1D" test_where_1d;
    test "where broadcast" test_where_broadcast;
    test "where scalar inputs" test_where_scalar_inputs;
    test "where invalid shapes" test_where_invalid_shapes;
  ]

let sort_tests =
  [
    test "sort 2D axis 0" test_sort_2d_axis0;
    test "sort 2D axis 1" test_sort_2d_axis1;
    test "sort invalid axis" test_sort_invalid_axis;
    test "sort NaN handling" test_sort_nan_handling;
    test "sort stable" test_sort_stable;
  ]

let sort_regression_tests =
  [
    test "sort large 1D (n=150)" test_sort_large_1d;
    test "sort power of two (n=256)" test_sort_power_of_two;
    test "sort 128 boundary" test_sort_128_boundary;
  ]

let argsort_tests =
  [
    test "argsort 1D" test_argsort_1d;
    test "argsort 2D axis 0" test_argsort_2d_axis0;
    test "argsort 2D axis 1" test_argsort_2d_axis1;
    test "argsort empty" test_argsort_empty;
  ]

let top_k_tests =
  [
    test "top_k 1D" test_top_k_1d;
    test "top_k along each axis" test_top_k_axes;
    test "top_k ties" test_top_k_ties;
    test "top_k entries at the least value" test_top_k_least_values;
    test "top_k NaN" test_top_k_nan;
    test "top_k is a sorted prefix" test_top_k_is_a_sorted_prefix;
    test "top_k radix select, floats" test_top_k_radix_floats;
    test "top_k radix select, integers" test_top_k_radix_ints;
    test "top_k radix select, mostly NaN" test_top_k_radix_mostly_nan;
    test "top_k radix select, ties" test_top_k_radix_ties;
    test "top_k radix select, a long axis" test_top_k_radix_long_axis;
    test "top_k radix select, axes" test_top_k_radix_axes;
    test "top_k float8_e4m3" test_top_k_float8_e4m3;
    test "top_k invalid arguments" test_top_k_invalid;
  ]

let argmax_tests =
  [
    test "argmax 1D" test_argmax_1d;
    test "argmax 2D axis 0" test_argmax_2d_axis0;
    test "argmax 2D axis 1" test_argmax_2d_axis1;
    test "argmax keepdims" test_argmax_keepdims;
    test "argmax NaN" test_argmax_nan;
  ]

let argmin_tests =
  [
    test "argmin 1D" test_argmin_1d;
    test "argmin 2D axis 0" test_argmin_2d_axis0;
    test "argmin 2D axis 1" test_argmin_2d_axis1;
    test "argmin ties" test_argmin_ties;
  ]

let suite =
  [
    group "Where" where_tests;
    group "Sort" sort_tests;
    group "Sort Regression" sort_regression_tests;
    group "Argsort" argsort_tests;
    group "Argmax" argmax_tests;
    group "Argmin" argmin_tests;
    group "Top-k" top_k_tests;
  ]

let () = run "Nx Sorting" suite
