(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Comprehensive indexing and slicing tests for Nx *)

open Windtrap
open Test_nx_support

(* ───── Basic Slicing Tests (slice function) ───── *)

let test_slice_basic () =
  let t = Nx.create Nx.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
  let sliced = Nx.slice [ Nx.R (1, 4) ] t in
  check_t "slice [1:4]" [| 3 |] [| 2.; 3.; 4. |] sliced

let test_slice_with_step () =
  let t = Nx.create Nx.float32 [| 10 |] (Array.init 10 float_of_int) in
  let sliced = Nx.slice [ Nx.Rs (1, 8, 2) ] t in
  check_t "slice [1:8:2]" [| 4 |] [| 1.; 3.; 5.; 7. |] sliced

let test_slice_negative_indices () =
  let t = Nx.create Nx.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
  let sliced = Nx.slice [ Nx.R (-3, -1) ] t in
  check_t "slice [-3:-1]" [| 2 |] [| 3.; 4. |] sliced

let test_slice_2d () =
  let t =
    Nx.create Nx.float32 [| 3; 4 |]
      [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9.; 10.; 11.; 12. |]
  in
  let sliced = Nx.slice [ Nx.R (1, 3); Nx.R (1, 3) ] t in
  check_t "slice 2d [1:3, 1:3]" [| 2; 2 |] [| 6.; 7.; 10.; 11. |] sliced

let test_slice_empty () =
  let t = Nx.create Nx.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
  let sliced = Nx.slice [ Nx.R (3, 3) ] t in
  check_shape "empty slice" [| 0 |] sliced

(* ───── Advanced Indexing Tests (index function) ───── *)

let test_index_all () =
  let t = Nx.create Nx.float32 [| 3; 4 |] (Array.init 12 float_of_int) in
  let indexed = Nx.slice [ Nx.A; Nx.A ] t in
  check_t "index all" [| 3; 4 |] (Array.init 12 float_of_int) indexed

let test_index_at () =
  let t = Nx.create Nx.float32 [| 3; 4 |] (Array.init 12 float_of_int) in
  let indexed = Nx.slice [ Nx.I 1 ] t in
  check_t "index at" [| 4 |] [| 4.; 5.; 6.; 7. |] indexed

let test_index_at_negative () =
  let t = Nx.create Nx.float32 [| 3; 4 |] (Array.init 12 float_of_int) in
  let indexed = Nx.slice [ Nx.I (-1) ] t in
  check_t "index at negative" [| 4 |] [| 8.; 9.; 10.; 11. |] indexed

let test_index_rng () =
  let t = Nx.create Nx.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
  let indexed = Nx.slice [ Nx.R (1, 3) ] t in
  check_t "index rng" [| 2 |] [| 2.; 3. |] indexed

let test_index_rngs () =
  let t = Nx.create Nx.float32 [| 10 |] (Array.init 10 float_of_int) in
  let indexed = Nx.slice [ Nx.Rs (1, 8, 2) ] t in
  check_t "index rngs with step" [| 4 |] [| 1.; 3.; 5.; 7. |] indexed

let test_index_idx () =
  let t = Nx.create Nx.float32 [| 5 |] [| 10.; 20.; 30.; 40.; 50. |] in
  let indexed = Nx.slice [ Nx.L [ 0; 2; 4 ] ] t in
  check_t "index idx" [| 3 |] [| 10.; 30.; 50. |] indexed

let test_index_idx_repeated () =
  let t = Nx.create Nx.float32 [| 3 |] [| 10.; 20.; 30. |] in
  let indexed = Nx.slice [ Nx.L [ 0; 1; 1; 0; 2 ] ] t in
  check_t "index idx repeated" [| 5 |] [| 10.; 20.; 20.; 10.; 30. |] indexed

(* Regression test: fancy indexing should reorder even when length matches dim
   size *)
let test_index_idx_reorder () =
  let t = Nx.create Nx.float32 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  (* L [1; 2; 0] should reorder rows, not return unchanged *)
  let indexed = Nx.slice [ Nx.L [ 1; 2; 0 ]; Nx.A ] t in
  check_t "index idx reorder" [| 3; 2 |] [| 3.; 4.; 5.; 6.; 1.; 2. |] indexed

let test_index_mixed () =
  let t = Nx.create Nx.float32 [| 3; 4; 5 |] (Array.init 60 float_of_int) in
  (* Select row 1, columns 0 and 2, all in last dimension *)
  let indexed = Nx.slice [ Nx.I 1; Nx.L [ 0; 2 ]; Nx.A ] t in
  check_t "index mixed" [| 2; 5 |]
    [| 20.; 21.; 22.; 23.; 24.; 30.; 31.; 32.; 33.; 34. |]
    indexed

let test_index_new_axis () =
  let t = Nx.create Nx.float32 [| 3; 4 |] (Array.init 12 float_of_int) in
  let indexed = Nx.slice [ Nx.A; Nx.N; Nx.A ] t in
  check_shape "index new axis" [| 3; 1; 4 |] indexed

let test_index_mask () =
  let t = Nx.create Nx.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
  let mask = Nx.greater_s t 2.5 in
  let indexed = Nx.slice [ Nx.M mask ] t in
  check_t "index mask" [| 3 |] [| 3.; 4.; 5. |] indexed

let test_index_mask_rows () =
  let t =
    Nx.create Nx.float32 [| 4; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |]
  in
  let mask = Nx.create Nx.bool [| 4 |] [| true; false; true; false |] in
  let indexed = Nx.slice [ Nx.M mask ] t in
  check_t "index mask rows" [| 2; 2 |] [| 1.; 2.; 5.; 6. |] indexed

let test_index_mask_axis1 () =
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let mask = Nx.create Nx.bool [| 3 |] [| true; false; true |] in
  let indexed = Nx.slice [ Nx.A; Nx.M mask ] t in
  check_t "index mask axis 1" [| 2; 2 |] [| 1.; 3.; 4.; 6. |] indexed

let test_index_mask_all_false () =
  let t = Nx.create Nx.float32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
  let mask = Nx.create Nx.bool [| 4 |] [| false; false; false; false |] in
  let indexed = Nx.slice [ Nx.M mask ] t in
  check_shape "index mask all false" [| 0 |] indexed

let test_index_mask_length_mismatch () =
  let t = Nx.create Nx.float32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
  let mask = Nx.create Nx.bool [| 3 |] [| true; false; true |] in
  raises ~msg:"mask length mismatch"
    (Invalid_argument "slice: axis 0, boolean mask length 3, expected 4")
    (fun () -> ignore (Nx.slice [ Nx.M mask ] t))

let test_index_mask_rank () =
  let t = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let mask = Nx.create Nx.bool [| 2; 2 |] [| true; false; true; false |] in
  raises ~msg:"mask rank"
    (Invalid_argument
       "slice: axis 0, boolean mask must be rank 1 but has rank 2") (fun () ->
      ignore (Nx.slice [ Nx.M mask ] t))

let test_set_slice_mask () =
  let t = Nx.zeros Nx.float32 [| 4 |] in
  let mask = Nx.create Nx.bool [| 4 |] [| true; false; true; true |] in
  let value = Nx.create Nx.float32 [| 3 |] [| 10.; 20.; 30. |] in
  let t = Nx.set [ Nx.M mask ] value t in
  check_t "set mask" [| 4 |] [| 10.; 0.; 20.; 30. |] t

let test_set_slice_mask_broadcast () =
  let t = Nx.zeros Nx.float32 [| 4; 2 |] in
  let mask = Nx.create Nx.bool [| 4 |] [| true; false; true; false |] in
  let value = Nx.create Nx.float32 [| 2 |] [| 7.; 8. |] in
  let t = Nx.set [ Nx.M mask ] value t in
  check_t "set mask broadcast" [| 4; 2 |] [| 7.; 8.; 0.; 0.; 7.; 8.; 0.; 0. |] t

let test_set_new_axis_with_gather () =
  let t = Nx.zeros Nx.float32 [| 3; 2 |] in
  let value = Nx.ones Nx.float32 [| 2 |] in
  let t = Nx.set [ Nx.L [ 0; 2 ]; Nx.N ] value t in
  check_t "set new axis beside a gather" [| 3; 2 |]
    [| 1.; 1.; 0.; 0.; 1.; 1. |]
    t

let test_set_duplicate_list_raises () =
  let t = Nx.zeros Nx.float32 [| 4 |] in
  raises ~msg:"repeated position"
    (Invalid_argument "set: index 1 is listed twice") (fun () ->
      ignore (Nx.set [ Nx.L [ 1; 1 ] ] (Nx.scalar Nx.float32 1.) t))

let test_set_window_dynamic () =
  let t = Nx.zeros Nx.float32 [| 2; 6 |] in
  let v = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let at pos = Nx.set [ Nx.A; Nx.D (Nx.scalar Nx.int32 pos, 2) ] v t in
  check_t "window at 1" [| 2; 6 |]
    [| 0.; 1.; 2.; 0.; 0.; 0.; 0.; 3.; 4.; 0.; 0.; 0. |]
    (at 1l);
  (* the start is clamped so the window always fits *)
  check_t "window clamped at the end" [| 2; 6 |]
    [| 0.; 0.; 0.; 0.; 1.; 2.; 0.; 0.; 0.; 0.; 3.; 4. |]
    (at 9l);
  check_t "dynamic read" [| 2; 2 |] [| 1.; 2.; 3.; 4. |]
    (Nx.slice [ Nx.A; Nx.D (Nx.scalar Nx.int32 1l, 2) ] (at 1l));
  check_shape "empty window reads as empty" [| 2; 0 |]
    (Nx.slice [ Nx.A; Nx.D (Nx.scalar Nx.int32 1l, 0) ] t)

let test_set_reversed_range () =
  let t = Nx.zeros Nx.float32 [| 5 |] in
  let v = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  check_t "set through a step -1 range" [| 5 |] [| 0.; 3.; 2.; 1.; 0. |]
    (Nx.set [ Nx.Rs (3, 0, -1) ] v t)

(* ───── Set_slice Tests ───── *)

let test_set_slice_at () =
  let t = Nx.zeros Nx.float32 [| 3; 4 |] in
  let value = Nx.ones Nx.float32 [| 4 |] in
  let t = Nx.set [ Nx.I 1 ] value t in
  equal ~msg:"set at [1,2]" (float 1e-6) 1.0 (Nx.item [ 1; 2 ] t)

let test_set_slice_rng () =
  let t = Nx.zeros Nx.float32 [| 5 |] in
  let value = Nx.create Nx.float32 [| 2 |] [| 10.; 20. |] in
  let t = Nx.set [ Nx.R (1, 3) ] value t in
  check_t "set rng" [| 5 |] [| 0.; 10.; 20.; 0.; 0. |] t

let test_set_slice_idx () =
  let t = Nx.zeros Nx.float32 [| 5 |] in
  let value = Nx.create Nx.float32 [| 3 |] [| 10.; 20.; 30. |] in
  let t = Nx.set [ Nx.L [ 0; 2; 4 ] ] value t in
  check_t "set idx" [| 5 |] [| 10.; 0.; 20.; 0.; 30. |] t

(* ───── Item and Set_item Tests ───── *)

let test_item () =
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let value = Nx.item [ 1; 2 ] t in
  equal ~msg:"item [1,2]" (float 1e-6) 6.0 value

let test_item_negative_indices () =
  let t = Nx.create Nx.float32 [| 3; 3 |] (Array.init 9 float_of_int) in
  let value = Nx.item [ -1; -1 ] t in
  equal ~msg:"item negative indices" (float 1e-6) 8.0 value

let test_set_item () =
  let t = Nx.zeros Nx.float32 [| 2; 3 |] in
  let t = Nx.set [ Nx.I 1; Nx.I 2 ] (Nx.scalar Nx.float32 99.0) t in
  equal ~msg:"set element" (float 1e-6) 99.0 (Nx.item [ 1; 2 ] t)

(* ───── Take Tests ───── *)

let test_take_basic () =
  let t = Nx.create Nx.float32 [| 5 |] [| 10.; 20.; 30.; 40.; 50. |] in
  let indices = Nx.create Nx.int32 [| 3 |] [| 0l; 2l; 4l |] in
  let result = Nx.take ~indices t in
  check_t "take basic" [| 3 |] [| 10.; 30.; 50. |] result

let test_take_with_axis () =
  let t =
    Nx.create Nx.float32 [| 3; 4 |]
      [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9.; 10.; 11.; 12. |]
  in
  let indices = Nx.create Nx.int32 [| 2 |] [| 0l; 2l |] in
  let result = Nx.take ~axis:1 ~indices t in
  check_t "take with axis" [| 3; 2 |] [| 1.; 3.; 5.; 7.; 9.; 11. |] result

let test_take_out_of_range_raises () =
  let t = Nx.create Nx.float32 [| 3 |] [| 10.; 20.; 30. |] in
  raises ~msg:"past the end"
    (Invalid_argument
       "gather: index out of bounds for the gathered/scattered axis") (fun () ->
      ignore (Nx.take ~indices:(Nx.create Nx.int32 [| 1 |] [| 3l |]) t));
  raises ~msg:"negative"
    (Invalid_argument
       "gather: index out of bounds for the gathered/scattered axis") (fun () ->
      ignore (Nx.take ~indices:(Nx.create Nx.int32 [| 1 |] [| -1l |]) t))

(* ───── Take_along_axis Tests ───── *)

let test_take_along_axis_1d () =
  let t = Nx.create Nx.float32 [| 5 |] [| 3.; 1.; 4.; 1.; 5. |] in
  let indices = Nx.argsort ~axis:0 t in
  let sorted = Nx.take_along_axis ~axis:0 ~indices t in
  check_t "take_along_axis 1d" [| 5 |] [| 1.; 1.; 3.; 4.; 5. |] sorted

let test_take_along_axis_2d () =
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 4.; 1.; 2.; 3.; 5.; 6. |] in
  (* Get argmax along axis 1 *)
  let indices = Nx.argmax ~axis:1 ~keepdims:true t in
  let maxvals = Nx.take_along_axis ~axis:1 ~indices t in
  check_t "take_along_axis 2d" [| 2; 1 |] [| 4.; 6. |] maxvals

(* ───── Scatter Tests ───── *)

let test_scatter_set () =
  let t = Nx.zeros Nx.float32 [| 2; 3 |] in
  let indices = Nx.create Nx.int32 [| 2; 1 |] [| 1l; 0l |] in
  let values = Nx.create Nx.float32 [| 2; 1 |] [| 10.; 20. |] in
  let r = Nx.scatter ~axis:1 ~indices ~values t in
  check_t "scatter set" [| 2; 3 |] [| 0.; 10.; 0.; 20.; 0.; 0. |] r;
  check_t "scatter is pure" [| 2; 3 |] (Array.make 6 0.) t

let test_scatter_set_duplicates_last_wins () =
  let t = Nx.zeros Nx.float32 [| 4 |] in
  let indices = Nx.create Nx.int32 [| 3 |] [| 1l; 1l; 2l |] in
  let values = Nx.create Nx.float32 [| 3 |] [| 10.; 20.; 30. |] in
  let r = Nx.scatter ~axis:0 ~indices ~values t in
  check_t "scatter duplicate set" [| 4 |] [| 0.; 20.; 30.; 0. |] r

let test_scatter_add_accumulates () =
  let t = Nx.create Nx.float32 [| 4 |] [| 1.; 1.; 1.; 1. |] in
  let indices = Nx.create Nx.int32 [| 3 |] [| 1l; 1l; 3l |] in
  let values = Nx.create Nx.float32 [| 3 |] [| 10.; 20.; 30. |] in
  let r = Nx.scatter ~mode:`Add ~axis:0 ~indices ~values t in
  (* Duplicates accumulate; every slot keeps [t]'s value underneath. *)
  check_t "scatter add" [| 4 |] [| 1.; 31.; 1.; 31. |] r

let test_scatter_broadcast_values () =
  let t = Nx.zeros Nx.float32 [| 2; 3 |] in
  let indices = Nx.create Nx.int32 [| 2; 1 |] [| 2l; 0l |] in
  let values = Nx.scalar Nx.float32 5. in
  let r = Nx.scatter ~axis:(-1) ~indices ~values t in
  check_t "scatter broadcast" [| 2; 3 |] [| 0.; 0.; 5.; 5.; 0.; 0. |] r

let test_scatter_int_dtype () =
  let t = Nx.zeros Nx.int32 [| 3 |] in
  let indices = Nx.create Nx.int32 [| 2 |] [| 0l; 2l |] in
  let values = Nx.create Nx.int32 [| 2 |] [| 7l; 8l |] in
  let r = Nx.scatter ~axis:0 ~indices ~values t in
  check_t "scatter int32" [| 3 |] [| 7l; 0l; 8l |] r

let test_scatter_shape_mismatch () =
  let t = Nx.zeros Nx.float32 [| 2; 3 |] in
  let indices = Nx.create Nx.int32 [| 3; 1 |] [| 0l; 1l; 0l |] in
  let values = Nx.zeros Nx.float32 [| 3; 1 |] in
  raises ~msg:"scatter shape mismatch"
    (Invalid_argument
       "scatter: shape, dimension 0: indices has 3 but tensor has 2") (fun () ->
      ignore (Nx.scatter ~axis:1 ~indices ~values t))

(* ───── Compress Tests ───── *)

let test_compress_no_axis () =
  let t = Nx.create Nx.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
  let condition =
    Nx.create Nx.bool [| 5 |] [| true; false; true; false; true |]
  in
  let result = Nx.compress ~condition t in
  check_t "compress no axis" [| 3 |] [| 1.; 3.; 5. |] result

let test_compress_with_axis () =
  let t =
    Nx.create Nx.float32 [| 3; 4 |]
      [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9.; 10.; 11.; 12. |]
  in
  let condition = Nx.create Nx.bool [| 3 |] [| false; true; true |] in
  let result = Nx.compress ~axis:0 ~condition t in
  check_t "compress with axis" [| 2; 4 |]
    [| 5.; 6.; 7.; 8.; 9.; 10.; 11.; 12. |]
    result

let test_compress_empty_result () =
  let t = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let condition = Nx.create Nx.bool [| 3 |] [| false; false; false |] in
  let result = Nx.compress ~condition t in
  check_shape "compress empty result" [| 0 |] result

(* ───── Extract Tests ───── *)

let test_extract_basic () =
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let condition =
    Nx.create Nx.bool [| 2; 3 |] [| true; false; true; false; true; false |]
  in
  let result = Nx.extract ~condition t in
  check_t "extract basic" [| 3 |] [| 1.; 3.; 5. |] result

let test_extract_from_comparison () =
  let t = Nx.create Nx.float32 [| 3; 3 |] (Array.init 9 float_of_int) in
  let condition = Nx.greater_s t 4. in
  let result = Nx.extract ~condition t in
  check_t "extract from comparison" [| 4 |] [| 5.; 6.; 7.; 8. |] result

(* ───── Nonzero Tests ───── *)

let test_nonzero_1d () =
  let t = Nx.create Nx.float32 [| 5 |] [| 0.; 1.; 0.; 3.; 0. |] in
  let indices = Nx.nonzero t in
  equal ~msg:"nonzero 1d length" int 1 (Array.length indices);
  let expected = [| 1.; 3. |] in
  check_t "nonzero 1d indices" [| 2 |] expected (Nx.cast Nx.float32 indices.(0))

let test_nonzero_2d () =
  let t =
    Nx.create Nx.float32 [| 3; 3 |] [| 0.; 1.; 0.; 2.; 0.; 3.; 0.; 0.; 4. |]
  in
  let indices = Nx.nonzero t in
  equal ~msg:"nonzero 2d length" int 2 (Array.length indices);
  (* Row indices *)
  let expected_rows = [| 0.; 1.; 1.; 2. |] in
  check_t "nonzero 2d rows" [| 4 |] expected_rows
    (Nx.cast Nx.float32 indices.(0));
  (* Column indices *)
  let expected_cols = [| 1.; 0.; 2.; 2. |] in
  check_t "nonzero 2d cols" [| 4 |] expected_cols
    (Nx.cast Nx.float32 indices.(1))

let test_nonzero_scalar () =
  let indices = Nx.nonzero (Nx.scalar Nx.int32 5l) in
  equal ~msg:"a scalar has no axes" int 0 (Array.length indices);
  check_shape "argwhere of a scalar" [| 0; 0 |]
    (Nx.argwhere (Nx.scalar Nx.int32 5l))

let test_nonzero_empty () =
  let t = Nx.zeros Nx.float32 [| 3; 3 |] in
  let indices = Nx.nonzero t in
  equal ~msg:"nonzero empty length" int 2 (Array.length indices);
  Array.iter (fun idx -> check_shape "nonzero empty shape" [| 0 |] idx) indices

(* ───── Argwhere Tests ───── *)

let test_argwhere_basic () =
  let t =
    Nx.create Nx.float32 [| 3; 3 |] [| 0.; 1.; 0.; 2.; 0.; 3.; 0.; 0.; 4. |]
  in
  let coords = Nx.argwhere t in
  check_shape "argwhere shape" [| 4; 2 |] coords;
  let expected = [| 0.; 1.; 1.; 0.; 1.; 2.; 2.; 2. |] in
  check_t "argwhere coords" [| 4; 2 |] expected (Nx.cast Nx.float32 coords)

let test_argwhere_empty () =
  let t = Nx.zeros Nx.float32 [| 3; 3 |] in
  let coords = Nx.argwhere t in
  check_shape "argwhere empty" [| 0; 2 |] coords

let test_argwhere_1d () =
  let t = Nx.create Nx.float32 [| 5 |] [| 0.; 1.; 0.; 3.; 0. |] in
  let coords = Nx.argwhere t in
  check_shape "argwhere 1d shape" [| 2; 1 |] coords;
  let expected = [| 1.; 3. |] in
  check_t "argwhere 1d coords" [| 2; 1 |] expected (Nx.cast Nx.float32 coords)

(* ───── Edge Cases and Error Tests ───── *)

let test_item_wrong_indices () =
  let t = Nx.create Nx.float32 [| 2; 3 |] (Array.init 6 float_of_int) in
  raises ~msg:"item wrong number of indices"
    (Invalid_argument "item: need 2 indices for 2-d tensor, got 1") (fun () ->
      ignore (Nx.item [ 1 ] t))

let test_set_slice_broadcast () =
  let t = Nx.zeros Nx.float32 [| 3; 4 |] in
  let value = Nx.ones Nx.float32 [| 1 |] in
  let t = Nx.set [ Nx.R (1, 2) ] value t in
  (* Value should be broadcast to shape [1, 4] *)
  equal ~msg:"set broadcast" (float 1e-6) 1.0 (Nx.item [ 1; 2 ] t)

let test_index_chained () =
  let t = Nx.create Nx.float32 [| 4; 5; 6 |] (Array.init 120 float_of_int) in
  (* Chain multiple index operations *)
  let indexed1 = Nx.slice [ Nx.R (1, 3); Nx.A; Nx.A ] t in
  let indexed2 = Nx.slice [ Nx.A; Nx.L [ 0; 2; 4 ]; Nx.A ] indexed1 in
  let indexed3 = Nx.slice [ Nx.I 1; Nx.I 1; Nx.R (2, 5) ] indexed2 in
  check_shape "index chained shape" [| 3 |] indexed3

let test_take_empty_indices () =
  let t = Nx.create Nx.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
  let indices = Nx.create Nx.int32 [| 0 |] [||] in
  let result = Nx.take ~indices t in
  check_shape "take empty indices" [| 0 |] result

let test_compress_condition_mismatch () =
  let t = Nx.create Nx.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
  let condition = Nx.create Nx.bool [| 3 |] [| true; false; true |] in
  raises ~msg:"compress condition mismatch"
    (Invalid_argument "compress: length 3 doesn't match axis 0 size 5")
    (fun () -> ignore (Nx.compress ~axis:0 ~condition t))

let test_extract_shape_mismatch () =
  let t = Nx.create Nx.float32 [| 2; 3 |] (Array.init 6 float_of_int) in
  let condition = Nx.create Nx.bool [| 2; 2 |] [| true; false; true; false |] in
  raises ~msg:"extract shape mismatch"
    (Invalid_argument "extract: shape mismatch") (fun () ->
      ignore (Nx.extract ~condition t))

(* ───── Test Suite Organization ───── *)

let slice_tests =
  [
    test "slice basic" test_slice_basic;
    test "slice with step" test_slice_with_step;
    test "slice negative indices" test_slice_negative_indices;
    test "slice 2d" test_slice_2d;
    test "slice empty" test_slice_empty;
  ]

let index_tests =
  [
    test "index all" test_index_all;
    test "index at" test_index_at;
    test "index at negative" test_index_at_negative;
    test "index rng" test_index_rng;
    test "index rngs" test_index_rngs;
    test "index idx" test_index_idx;
    test "index idx repeated" test_index_idx_repeated;
    test "index idx reorder" test_index_idx_reorder;
    test "index mixed" test_index_mixed;
    test "set at" test_set_slice_at;
    test "set rng" test_set_slice_rng;
    test "set idx" test_set_slice_idx;
  ]

let item_tests =
  [
    test "item" test_item;
    test "item negative indices" test_item_negative_indices;
    test "set element" test_set_item;
    test "item wrong indices" test_item_wrong_indices;
  ]

let take_tests =
  [
    test "take basic" test_take_basic;
    test "take with axis" test_take_with_axis;
    test "take out of range raises" test_take_out_of_range_raises;
    test "take_along_axis 1d" test_take_along_axis_1d;
    test "take_along_axis 2d" test_take_along_axis_2d;
    test "take empty indices" test_take_empty_indices;
  ]

let put_tests = []

let scatter_tests =
  [
    test "scatter set" test_scatter_set;
    test "scatter set duplicates last wins"
      test_scatter_set_duplicates_last_wins;
    test "scatter add accumulates" test_scatter_add_accumulates;
    test "scatter broadcasts values" test_scatter_broadcast_values;
    test "scatter int dtype" test_scatter_int_dtype;
    test "scatter shape mismatch" test_scatter_shape_mismatch;
  ]

let compress_extract_tests =
  [
    test "compress no axis" test_compress_no_axis;
    test "compress with axis" test_compress_with_axis;
    test "compress empty result" test_compress_empty_result;
    test "extract basic" test_extract_basic;
    test "extract from comparison" test_extract_from_comparison;
    test "compress condition mismatch" test_compress_condition_mismatch;
    test "extract shape mismatch" test_extract_shape_mismatch;
  ]

let nonzero_argwhere_tests =
  [
    test "nonzero 1d" test_nonzero_1d;
    test "nonzero 2d" test_nonzero_2d;
    test "nonzero empty" test_nonzero_empty;
    test "nonzero scalar" test_nonzero_scalar;
    test "argwhere basic" test_argwhere_basic;
    test "argwhere empty" test_argwhere_empty;
    test "argwhere 1d" test_argwhere_1d;
  ]

let mask_tests =
  [
    test "index new axis" test_index_new_axis;
    test "index mask" test_index_mask;
    test "index mask rows" test_index_mask_rows;
    test "index mask axis 1" test_index_mask_axis1;
    test "index mask all false" test_index_mask_all_false;
    test "index mask length mismatch" test_index_mask_length_mismatch;
    test "index mask rank" test_index_mask_rank;
    test "set mask" test_set_slice_mask;
    test "set mask broadcast" test_set_slice_mask_broadcast;
    test "set new axis beside a gather" test_set_new_axis_with_gather;
    test "set duplicate list raises" test_set_duplicate_list_raises;
    test "set dynamic window" test_set_window_dynamic;
    test "set reversed range" test_set_reversed_range;
  ]

let edge_case_tests =
  [
    test "set broadcast" test_set_slice_broadcast;
    test "index chained" test_index_chained;
  ]

let suite =
  [
    group "slice" slice_tests;
    group "index" index_tests;
    group "item" item_tests;
    group "take" take_tests;
    group "put" put_tests;
    group "scatter" scatter_tests;
    group "compress/extract" compress_extract_tests;
    group "nonzero/argwhere" nonzero_argwhere_tests;
    group "mask/new-axis" mask_tests;
    group "edge cases" edge_case_tests;
  ]

let () = run "Nx Indexing" suite
