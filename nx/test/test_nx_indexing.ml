(* Comprehensive indexing and slicing tests for Nx *)

open Alcotest
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

let test_index_mixed () =
  let t = Nx.create Nx.float32 [| 3; 4; 5 |] (Array.init 60 float_of_int) in
  (* Select row 1, columns 0 and 2, all in last dimension *)
  let indexed = Nx.slice [ Nx.I 1; Nx.L [ 0; 2 ]; Nx.A ] t in
  check_t "index mixed" [| 2; 5 |]
    [| 20.; 21.; 22.; 23.; 24.; 30.; 31.; 32.; 33.; 34. |]
    indexed

(* Note: `new_ and `mask require implementation *)
(* let test_index_new_axis  () =
    let t = Nx.create  Nx.float32 [| 3; 4 |] (Array.init 12 float_of_int) in
    let indexed = Nx.slice [ Nx.A; Nx.N; Nx.A ] t in
    check_shape "index new axis" [| 3; 1; 4 |] indexed

  let test_index_mask  () =
    let t = Nx.create  Nx.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
    let mask = Nx.greater_s t 2.5 in
    let indexed = Nx.slice [ Nx.M mask ] t in
    check_t "index mask" [| 3 |] [| 3.; 4.; 5. |] indexed *)

(* ───── set_slice Tests ───── *)

let test_set_slice_at () =
  let t = Nx.zeros Nx.float32 [| 3; 4 |] in
  let value = Nx.ones Nx.float32 [| 4 |] in
  Nx.set_slice [ Nx.I 1 ] t value;
  check (float 1e-6) "set_slice at [1,2]" 1.0 (Nx.item [ 1; 2 ] t)

let test_set_slice_rng () =
  let t = Nx.zeros Nx.float32 [| 5 |] in
  let value = Nx.create Nx.float32 [| 2 |] [| 10.; 20. |] in
  Nx.set_slice [ Nx.R (1, 3) ] t value;
  check_t "set_slice rng" [| 5 |] [| 0.; 10.; 20.; 0.; 0. |] t

let test_set_slice_idx () =
  let t = Nx.zeros Nx.float32 [| 5 |] in
  let value = Nx.create Nx.float32 [| 3 |] [| 10.; 20.; 30. |] in
  Nx.set_slice [ Nx.L [ 0; 2; 4 ] ] t value;
  check_t "set_slice idx" [| 5 |] [| 10.; 0.; 20.; 0.; 30. |] t

(* ───── item and set_item Tests ───── *)

let test_item () =
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let value = Nx.item [ 1; 2 ] t in
  check (float 1e-6) "item [1,2]" 6.0 value

let test_item_negative_indices () =
  let t = Nx.create Nx.float32 [| 3; 3 |] (Array.init 9 float_of_int) in
  let value = Nx.item [ -1; -1 ] t in
  check (float 1e-6) "item negative indices" 8.0 value

let test_set_item () =
  let t = Nx.zeros Nx.float32 [| 2; 3 |] in
  Nx.set_item [ 1; 2 ] 99.0 t;
  check (float 1e-6) "set_item" 99.0 (Nx.item [ 1; 2 ] t)

(* ───── take Tests ───── *)

let test_take_basic () =
  let t = Nx.create Nx.float32 [| 5 |] [| 10.; 20.; 30.; 40.; 50. |] in
  let indices = Nx.create Nx.int32 [| 3 |] [| 0l; 2l; 4l |] in
  let result = Nx.take indices t in
  check_t "take basic" [| 3 |] [| 10.; 30.; 50. |] result

let test_take_with_axis () =
  let t =
    Nx.create Nx.float32 [| 3; 4 |]
      [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9.; 10.; 11.; 12. |]
  in
  let indices = Nx.create Nx.int32 [| 2 |] [| 0l; 2l |] in
  let result = Nx.take ~axis:1 indices t in
  check_t "take with axis" [| 3; 2 |] [| 1.; 3.; 5.; 7.; 9.; 11. |] result

let test_take_mode_wrap () =
  let t = Nx.create Nx.float32 [| 3 |] [| 10.; 20.; 30. |] in
  let indices = Nx.create Nx.int32 [| 4 |] [| 0l; 1l; 2l; 3l |] in
  let result = Nx.take ~mode:`wrap indices t in
  check_t "take mode wrap" [| 4 |] [| 10.; 20.; 30.; 10. |] result

let test_take_mode_clip () =
  let t = Nx.create Nx.float32 [| 3 |] [| 10.; 20.; 30. |] in
  let indices = Nx.create Nx.int32 [| 4 |] [| -1l; 0l; 2l; 5l |] in
  let result = Nx.take ~mode:`clip indices t in
  check_t "take mode clip" [| 4 |] [| 10.; 10.; 30.; 30. |] result

let test_take_negative_indices () =
  let t = Nx.create Nx.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
  let indices = Nx.create Nx.int32 [| 2 |] [| -1l; -2l |] in
  let result = Nx.take ~mode:`wrap indices t in
  check_t "take negative indices" [| 2 |] [| 5.; 4. |] result

(* ───── take_along_axis Tests ───── *)

let test_take_along_axis_1d () =
  let t = Nx.create Nx.float32 [| 5 |] [| 3.; 1.; 4.; 1.; 5. |] in
  let indices = Nx.argsort ~axis:0 t in
  let sorted = Nx.take_along_axis ~axis:0 indices t in
  check_t "take_along_axis 1d" [| 5 |] [| 1.; 1.; 3.; 4.; 5. |] sorted

let test_take_along_axis_2d () =
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 4.; 1.; 2.; 3.; 5.; 6. |] in
  (* Get argmax along axis 1 *)
  let indices = Nx.argmax ~axis:1 ~keepdims:true t in
  let maxvals = Nx.take_along_axis ~axis:1 indices t in
  check_t "take_along_axis 2d" [| 2; 1 |] [| 4.; 6. |] maxvals

(* ───── put Tests ───── *)

let test_put_basic () =
  let t = Nx.zeros Nx.float32 [| 5 |] in
  let indices = Nx.create Nx.int32 [| 3 |] [| 0l; 2l; 4l |] in
  let values = Nx.create Nx.float32 [| 3 |] [| 10.; 20.; 30. |] in
  Nx.put ~indices ~values t;
  check_t "put basic" [| 5 |] [| 10.; 0.; 20.; 0.; 30. |] t

let test_put_with_axis () =
  let t = Nx.zeros Nx.float32 [| 3; 4 |] in
  let indices = Nx.create Nx.int32 [| 3; 2 |] [| 0l; 2l; 0l; 2l; 0l; 2l |] in
  let values = Nx.ones Nx.float32 [| 3; 2 |] in
  Nx.put ~axis:1 ~indices ~values t;
  let expected = [| 1.; 0.; 1.; 0.; 1.; 0.; 1.; 0.; 1.; 0.; 1.; 0. |] in
  check_t "put with axis" [| 3; 4 |] expected t

let test_put_mode_wrap () =
  let t = Nx.zeros Nx.float32 [| 3 |] in
  let indices = Nx.create Nx.int32 [| 4 |] [| 0l; 1l; 2l; 3l |] in
  let values = Nx.create Nx.float32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
  Nx.put ~indices ~values ~mode:`wrap t;
  check_t "put mode wrap" [| 3 |] [| 4.; 2.; 3. |] t

let test_put_mode_clip () =
  let t = Nx.zeros Nx.float32 [| 3 |] in
  let indices = Nx.create Nx.int32 [| 4 |] [| -1l; 0l; 2l; 5l |] in
  let values = Nx.create Nx.float32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
  Nx.put ~indices ~values ~mode:`clip t;
  check_t "put mode clip" [| 3 |] [| 2.; 0.; 4. |] t

(* ───── put_along_axis Tests ───── *)

let test_put_along_axis () =
  let t = Nx.zeros Nx.float32 [| 2; 3 |] in
  let indices = Nx.create Nx.int32 [| 2; 1 |] [| 1l; 0l |] in
  let values = Nx.create Nx.float32 [| 2; 1 |] [| 10.; 20. |] in
  Nx.put_along_axis ~axis:1 ~indices ~values t;
  check_t "put_along_axis" [| 2; 3 |] [| 0.; 10.; 0.; 20.; 0.; 0. |] t

(* ───── compress Tests ───── *)

let test_compress_no_axis () =
  let t = Nx.create Nx.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
  let condition = Nx.create Nx.uint8 [| 5 |] [| 1; 0; 1; 0; 1 |] in
  let result = Nx.compress ~condition t in
  check_t "compress no axis" [| 3 |] [| 1.; 3.; 5. |] result

let test_compress_with_axis () =
  let t =
    Nx.create Nx.float32 [| 3; 4 |]
      [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9.; 10.; 11.; 12. |]
  in
  let condition = Nx.create Nx.uint8 [| 3 |] [| 0; 1; 1 |] in
  let result = Nx.compress ~axis:0 ~condition t in
  check_t "compress with axis" [| 2; 4 |]
    [| 5.; 6.; 7.; 8.; 9.; 10.; 11.; 12. |]
    result

let test_compress_empty_result () =
  let t = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let condition = Nx.zeros Nx.uint8 [| 3 |] in
  let result = Nx.compress ~condition t in
  check_shape "compress empty result" [| 0 |] result

(* ───── extract Tests ───── *)

let test_extract_basic () =
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let condition = Nx.create Nx.uint8 [| 2; 3 |] [| 1; 0; 1; 0; 1; 0 |] in
  let result = Nx.extract ~condition t in
  check_t "extract basic" [| 3 |] [| 1.; 3.; 5. |] result

let test_extract_from_comparison () =
  let t = Nx.create Nx.float32 [| 3; 3 |] (Array.init 9 float_of_int) in
  let condition = Nx.greater_s t 4. in
  let result = Nx.extract ~condition t in
  check_t "extract from comparison" [| 4 |] [| 5.; 6.; 7.; 8. |] result

(* ───── nonzero Tests ───── *)

let test_nonzero_1d () =
  let t = Nx.create Nx.float32 [| 5 |] [| 0.; 1.; 0.; 3.; 0. |] in
  let indices = Nx.nonzero t in
  (check int) "nonzero 1d length" 1 (Array.length indices);
  let expected = [| 1.; 3. |] in
  check_t "nonzero 1d indices" [| 2 |] expected
    (Nx.astype Nx.float32 indices.(0))

let test_nonzero_2d () =
  let t =
    Nx.create Nx.float32 [| 3; 3 |] [| 0.; 1.; 0.; 2.; 0.; 3.; 0.; 0.; 4. |]
  in
  let indices = Nx.nonzero t in
  (check int) "nonzero 2d length" 2 (Array.length indices);
  (* Row indices *)
  let expected_rows = [| 0.; 1.; 1.; 2. |] in
  check_t "nonzero 2d rows" [| 4 |] expected_rows
    (Nx.astype Nx.float32 indices.(0));
  (* Column indices *)
  let expected_cols = [| 1.; 0.; 2.; 2. |] in
  check_t "nonzero 2d cols" [| 4 |] expected_cols
    (Nx.astype Nx.float32 indices.(1))

let test_nonzero_empty () =
  let t = Nx.zeros Nx.float32 [| 3; 3 |] in
  let indices = Nx.nonzero t in
  (check int) "nonzero empty length" 2 (Array.length indices);
  Array.iter (fun idx -> check_shape "nonzero empty shape" [| 0 |] idx) indices

(* ───── argwhere Tests ───── *)

let test_argwhere_basic () =
  let t =
    Nx.create Nx.float32 [| 3; 3 |] [| 0.; 1.; 0.; 2.; 0.; 3.; 0.; 0.; 4. |]
  in
  let coords = Nx.argwhere t in
  check_shape "argwhere shape" [| 4; 2 |] coords;
  let expected = [| 0.; 1.; 1.; 0.; 1.; 2.; 2.; 2. |] in
  check_t "argwhere coords" [| 4; 2 |] expected (Nx.astype Nx.float32 coords)

let test_argwhere_empty () =
  let t = Nx.zeros Nx.float32 [| 3; 3 |] in
  let coords = Nx.argwhere t in
  check_shape "argwhere empty" [| 0; 2 |] coords

let test_argwhere_1d () =
  let t = Nx.create Nx.float32 [| 5 |] [| 0.; 1.; 0.; 3.; 0. |] in
  let coords = Nx.argwhere t in
  check_shape "argwhere 1d shape" [| 2; 1 |] coords;
  let expected = [| 1.; 3. |] in
  check_t "argwhere 1d coords" [| 2; 1 |] expected (Nx.astype Nx.float32 coords)

(* ───── Edge Cases and Error Tests ───── *)

let test_item_wrong_indices () =
  let t = Nx.create Nx.float32 [| 2; 3 |] (Array.init 6 float_of_int) in
  check_raises "item wrong number of indices"
    (Invalid_argument "item: need 2 indices for 2-d tensor, got 1") (fun () ->
      ignore (Nx.item [ 1 ] t))

let test_set_slice_broadcast () =
  let t = Nx.zeros Nx.float32 [| 3; 4 |] in
  let value = Nx.ones Nx.float32 [| 1 |] in
  Nx.set_slice [ Nx.R (1, 2) ] t value;
  (* Value should be broadcast to shape [1, 4] *)
  check (float 1e-6) "set_slice broadcast" 1.0 (Nx.item [ 1; 2 ] t)

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
  let result = Nx.take indices t in
  check_shape "take empty indices" [| 0 |] result

let test_compress_condition_mismatch () =
  let t = Nx.create Nx.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
  let condition = Nx.create Nx.uint8 [| 3 |] [| 1; 0; 1 |] in
  check_raises "compress condition mismatch"
    (Invalid_argument "compress: length 3 doesn't match axis 0 size 5")
    (fun () -> ignore (Nx.compress ~axis:0 ~condition t))

let test_extract_shape_mismatch () =
  let t = Nx.create Nx.float32 [| 2; 3 |] (Array.init 6 float_of_int) in
  let condition = Nx.create Nx.uint8 [| 2; 2 |] [| 1; 0; 1; 0 |] in
  check_raises "extract shape mismatch"
    (Invalid_argument "extract: shape mismatch") (fun () ->
      ignore (Nx.extract ~condition t))

(* ───── Test Suite Organization ───── *)

let slice_tests =
  [
    ("slice basic", `Quick, test_slice_basic);
    ("slice with step", `Quick, test_slice_with_step);
    ("slice negative indices", `Quick, test_slice_negative_indices);
    ("slice 2d", `Quick, test_slice_2d);
    ("slice empty", `Quick, test_slice_empty);
  ]

let index_tests =
  [
    ("index all", `Quick, test_index_all);
    ("index at", `Quick, test_index_at);
    ("index at negative", `Quick, test_index_at_negative);
    ("index rng", `Quick, test_index_rng);
    ("index rngs", `Quick, test_index_rngs);
    ("index idx", `Quick, test_index_idx);
    ("index idx repeated", `Quick, test_index_idx_repeated);
    ("index mixed", `Quick, test_index_mixed);
    ("set_slice at", `Quick, test_set_slice_at);
    ("set_slice rng", `Quick, test_set_slice_rng);
    ("set_slice idx", `Quick, test_set_slice_idx);
  ]

let item_tests =
  [
    ("item", `Quick, test_item);
    ("item negative indices", `Quick, test_item_negative_indices);
    ("set_item", `Quick, test_set_item);
    ("item wrong indices", `Quick, test_item_wrong_indices);
  ]

let take_tests =
  [
    ("take basic", `Quick, test_take_basic);
    ("take with axis", `Quick, test_take_with_axis);
    ("take mode wrap", `Quick, test_take_mode_wrap);
    ("take mode clip", `Quick, test_take_mode_clip);
    ("take negative indices", `Quick, test_take_negative_indices);
    ("take_along_axis 1d", `Quick, test_take_along_axis_1d);
    ("take_along_axis 2d", `Quick, test_take_along_axis_2d);
    ("take empty indices", `Quick, test_take_empty_indices);
  ]

let put_tests =
  [
    ("put basic", `Quick, test_put_basic);
    ("put with axis", `Quick, test_put_with_axis);
    ("put mode wrap", `Quick, test_put_mode_wrap);
    ("put mode clip", `Quick, test_put_mode_clip);
    ("put_along_axis", `Quick, test_put_along_axis);
  ]

let compress_extract_tests =
  [
    ("compress no axis", `Quick, test_compress_no_axis);
    ("compress with axis", `Quick, test_compress_with_axis);
    ("compress empty result", `Quick, test_compress_empty_result);
    ("extract basic", `Quick, test_extract_basic);
    ("extract from comparison", `Quick, test_extract_from_comparison);
    ("compress condition mismatch", `Quick, test_compress_condition_mismatch);
    ("extract shape mismatch", `Quick, test_extract_shape_mismatch);
  ]

let nonzero_argwhere_tests =
  [
    ("nonzero 1d", `Quick, test_nonzero_1d);
    ("nonzero 2d", `Quick, test_nonzero_2d);
    ("nonzero empty", `Quick, test_nonzero_empty);
    ("argwhere basic", `Quick, test_argwhere_basic);
    ("argwhere empty", `Quick, test_argwhere_empty);
    ("argwhere 1d", `Quick, test_argwhere_1d);
  ]

let edge_case_tests =
  [
    ("set_slice broadcast", `Quick, test_set_slice_broadcast);
    ("index chained", `Quick, test_index_chained);
  ]

let suite =
  [
    ("Indexing :: slice", slice_tests);
    ("Indexing :: index", index_tests);
    ("Indexing :: item", item_tests);
    ("Indexing :: take", take_tests);
    ("Indexing :: put", put_tests);
    ("Indexing :: compress/extract", compress_extract_tests);
    ("Indexing :: nonzero/argwhere", nonzero_argwhere_tests);
    ("Indexing :: edge cases", edge_case_tests);
  ]

let () = Alcotest.run "Nx Indexing" suite
