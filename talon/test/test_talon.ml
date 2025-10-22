open Talon

let check_int = Alcotest.(check int)
let check_float = Alcotest.(check (float 1e-6))
let check_bool = Alcotest.(check bool)
let check_string = Alcotest.(check string)
let check_option_float = Alcotest.(check (option (float 1e-6)))
let check_option_string = Alcotest.(check (option string))
let check_option_bool_array = Alcotest.(check (option (array bool)))

let mask_of_column df name =
  match get_column_exn df name with Col.P (_, _, mask) -> mask | _ -> None

(* Test column creation *)
let test_col_creation () =
  let df1 = create [ ("c", Col.float32 [| 1.0; 2.0; 3.0 |]) ] in
  check_int "float32 col rows" 3 (num_rows df1);
  let c1 = get_column_exn df1 "c" in
  check_bool "float32 no nulls" false (Col.has_nulls c1);

  let df2 = create [ ("c", Col.int32 [| 1l; 2l; 3l |]) ] in
  check_int "int32 col rows" 3 (num_rows df2);

  let df3 = create [ ("c", Col.string [| "a"; "b"; "c" |]) ] in
  check_int "string col rows" 3 (num_rows df3);

  let df4 = create [ ("c", Col.bool [| true; false; true |]) ] in
  check_int "bool col rows" 3 (num_rows df4)

let test_col_nulls () =
  let c1 = Col.float32_opt [| Some 1.0; None; Some 3.0 |] in
  check_bool "has nulls" true (Col.has_nulls c1);
  check_int "null count" 1 (Col.null_count c1);

  let c2 = Col.drop_nulls c1 in
  let df = create [ ("c", c2) ] in
  check_int "after drop_nulls" 2 (num_rows df);
  check_bool "no nulls after drop" false (Col.has_nulls c2)

let test_col_null_mask () =
  let col = Col.float32_opt [| Some 1.0; None |] in
  check_option_bool_array "mask kept"
    (Some [| false; true |])
    (Col.null_mask col);
  let plain = Col.float32 [| 1.0; 2.0 |] in
  check_option_bool_array "no mask" None (Col.null_mask plain)

let test_drop_nulls_preserves_data_with_mask () =
  let col = Col.int32_opt [| Some Int32.min_int; None |] in
  let dropped = Col.drop_nulls col in
  match dropped with
  | Col.P (Nx.Int32, tensor, _) ->
      let arr : int32 array = Nx.to_array tensor in
      check_int "length after drop" 1 (Array.length arr);
      check_bool "sentinel retained" true (arr.(0) = Int32.min_int);
      check_option_bool_array "mask cleared" None (Col.null_mask dropped)
  | _ -> Alcotest.fail "expected int32 column"

let test_fill_nulls_respects_mask () =
  let col = Col.int32_opt [| Some 42l; None |] in
  match Col.fill_nulls_int32 col ~value:0l with
  | Col.P (Nx.Int32, tensor, _) ->
      let arr : int32 array = Nx.to_array tensor in
      check_bool "first value untouched" true (arr.(0) = 42l);
      check_bool "null filled" true (arr.(1) = 0l)
  | _ -> Alcotest.fail "expected int32 column"

(* Test dataframe creation *)
let test_df_creation () =
  let df =
    create
      [
        ("a", Col.int32_list [ 1l; 2l; 3l ]);
        ("b", Col.float64_list [ 1.5; 2.5; 3.5 ]);
        ("c", Col.string_list [ "x"; "y"; "z" ]);
      ]
  in

  let rows, cols = shape df in
  check_int "rows" 3 rows;
  check_int "cols" 3 cols;
  check_bool "not empty" false (is_empty df);

  let names = column_names df in
  Alcotest.(check (list string)) "column names" [ "a"; "b"; "c" ] names

let test_df_empty () =
  let df = empty in
  let rows, cols = shape df in
  check_int "empty rows" 0 rows;
  check_int "empty cols" 0 cols;
  check_bool "is empty" true (is_empty df)

(* Test column operations *)
let test_column_access () =
  let df =
    create
      [
        ("x", Col.int32_list [ 1l; 2l; 3l ]);
        ("y", Col.float32_list [ 1.0; 2.0; 3.0 ]);
      ]
  in

  check_bool "has column x" true (has_column df "x");
  check_bool "has column y" true (has_column df "y");
  check_bool "no column z" false (has_column df "z");

  match get_column df "x" with
  | Some _col -> check_int "df has 3 rows" 3 (num_rows df)
  | None -> Alcotest.fail "column x should exist"

let test_column_add_drop () =
  let df = create [ ("a", Col.int32_list [ 1l; 2l ]) ] in

  let df2 = add_column df "b" (Col.float32_list [ 1.0; 2.0 ]) in
  check_int "cols after add" 2 (num_columns df2);
  check_bool "has new column" true (has_column df2 "b");

  let df3 = drop_column df2 "a" in
  check_int "cols after drop" 1 (num_columns df3);
  check_bool "column dropped" false (has_column df3 "a")

let test_rename_column () =
  let df = create [ ("old", Col.int32_list [ 1l; 2l ]) ] in
  let df2 = rename_column df ~old_name:"old" ~new_name:"new" in

  check_bool "old name gone" false (has_column df2 "old");
  check_bool "new name exists" true (has_column df2 "new")

let test_select () =
  let df =
    create
      [
        ("a", Col.int32_list [ 1l ]);
        ("b", Col.int32_list [ 2l ]);
        ("c", Col.int32_list [ 3l ]);
      ]
  in

  let df2 = select df [ "c"; "a" ] in
  Alcotest.(check (list string)) "selected cols" [ "c"; "a" ] (column_names df2);

  let df3 = select_loose df [ "a"; "missing"; "c" ] in
  Alcotest.(check (list string)) "loose select" [ "a"; "c" ] (column_names df3)

(* Test row operations *)
let test_head_tail () =
  let df = create [ ("x", Col.int32_list [ 1l; 2l; 3l; 4l; 5l ]) ] in

  let h = head ~n:2 df in
  check_int "head size" 2 (num_rows h);

  let t = tail ~n:2 df in
  check_int "tail size" 2 (num_rows t)

let test_slice () =
  let df = create [ ("x", Col.int32_list [ 0l; 1l; 2l; 3l; 4l ]) ] in
  let s = slice df ~start:1 ~stop:4 in
  check_int "slice size" 3 (num_rows s)

let test_filter () =
  let df =
    create
      [
        ("x", Col.int32_list [ 1l; 2l; 3l; 4l ]);
        ("y", Col.string_list [ "a"; "b"; "c"; "d" ]);
      ]
  in

  let mask = [| true; false; true; false |] in
  let filtered = filter df mask in
  check_int "filtered rows" 2 (num_rows filtered)

let test_mask_projection_ops () =
  let df =
    create [ ("x", Col.float32_opt [| Some 1.0; None; Some 3.0; None |]) ]
  in
  let head_df = head ~n:3 df in
  check_option_bool_array "head mask"
    (Some [| false; true; false |])
    (mask_of_column head_df "x");
  let slice_df = slice df ~start:1 ~stop:4 in
  check_option_bool_array "slice mask"
    (Some [| true; false; true |])
    (mask_of_column slice_df "x");
  let filtered = filter df [| false; true; false; true |] in
  check_option_bool_array "filter mask"
    (Some [| true; true |])
    (mask_of_column filtered "x")

let test_concat_mask_combines () =
  let df1 = create [ ("x", Col.float32_opt [| Some 1.0; None |]) ] in
  let df2 = create [ ("x", Col.float32_opt [| None; Some 4.0 |]) ] in
  let combined = concat ~axis:`Rows [ df1; df2 ] in
  check_option_bool_array "concat mask"
    (Some [| false; true; true; false |])
    (mask_of_column combined "x")

let test_cast_preserves_mask () =
  let df = create [ ("x", Col.float32_opt [| Some 1.0; None |]) ] in
  let df_cast = cast_column df "x" Nx.float64 in
  check_option_bool_array "cast mask"
    (Some [| false; true |])
    (mask_of_column df_cast "x")

let test_pct_change_has_no_mask () =
  let df = create [ ("x", Col.float32_opt [| Some 1.0; Some 2.0; None |]) ] in
  match Agg.pct_change df "x" () with
  | Col.P (_, _, mask) -> check_option_bool_array "pct_change mask" None mask
  | _ -> Alcotest.fail "expected numeric column"

let test_filter_by () =
  let df =
    create
      [
        ("x", Col.int32_list [ 1l; 2l; 3l; 4l ]);
        ("y", Col.float32_list [ 1.0; 2.0; 3.0; 4.0 ]);
      ]
  in

  let filtered = filter_by df Row.(map (int32 "x") ~f:(fun x -> x > 2l)) in
  check_int "filtered rows" 2 (num_rows filtered)

let test_drop_duplicates () =
  let df =
    create
      [
        ("x", Col.int32_list [ 1l; 1l; 2l; 2l; 3l ]);
        ("y", Col.string_list [ "a"; "a"; "b"; "b"; "c" ]);
      ]
  in

  let unique = drop_duplicates df in
  check_int "unique rows" 3 (num_rows unique)

(* Test concatenation *)
let test_concat_rows () =
  let df1 = create [ ("x", Col.int32_list [ 1l; 2l ]) ] in
  let df2 = create [ ("x", Col.int32_list [ 3l; 4l ]) ] in

  let combined = concat ~axis:`Rows [ df1; df2 ] in
  check_int "concat rows" 4 (num_rows combined)

let test_concat_cols () =
  let df1 = create [ ("a", Col.int32_list [ 1l; 2l ]) ] in
  let df2 = create [ ("b", Col.int32_list [ 3l; 4l ]) ] in

  let combined = concat ~axis:`Columns [ df1; df2 ] in
  check_int "concat cols" 2 (num_columns combined)

(* Test Row module *)
let test_row_accessors () =
  let df =
    create
      [
        ("i32", Col.int32_list [ 42l; 24l ]);
        ("f32", Col.float32_list [ 3.14; 2.71 ]);
        ("str", Col.string_list [ "hello"; "world" ]);
        ("bool", Col.bool_list [ true; false ]);
      ]
  in

  (* Test by filtering based on row values *)
  let filtered_i32 =
    filter_by df Row.(map (int32 "i32") ~f:(fun x -> x = 42l))
  in
  check_int "filtered by i32" 1 (num_rows filtered_i32);

  let filtered_str =
    filter_by df Row.(map (string "str") ~f:(fun s -> s = "hello"))
  in
  check_int "filtered by string" 1 (num_rows filtered_str);

  let filtered_bool = filter_by df Row.(bool "bool") in
  check_int "filtered by bool" 1 (num_rows filtered_bool)

let test_row_map () =
  let df = create [ ("x", Col.int32_list [ 1l; 2l; 3l ]) ] in

  (* Use with_column to create a new column *)
  let df2 =
    with_column df "doubled" Nx.int32
      Row.(map (int32 "x") ~f:(fun x -> Int32.mul x 2l))
  in

  match to_int32_array df2 "doubled" with
  | Some arr -> check_bool "mapped values" true (arr = [| 2l; 4l; 6l |])
  | None -> Alcotest.fail "doubled column should exist"

(* Test sorting *)
let test_sort () =
  let df =
    create
      [
        ("x", Col.int32_list [ 3l; 1l; 2l ]);
        ("y", Col.string_list [ "c"; "a"; "b" ]);
      ]
  in

  let sorted = sort_values df "x" in
  match to_int32_array sorted "x" with
  | Some arr -> check_bool "sorted" true (arr = [| 1l; 2l; 3l |])
  | None -> Alcotest.fail "column should exist"

let test_group_by () =
  let df =
    create
      [
        ("key", Col.string_list [ "a"; "b"; "a"; "b" ]);
        ("val", Col.int32_list [ 1l; 2l; 3l; 4l ]);
      ]
  in

  let groups = group_by_column df "key" in
  check_int "group count" 2 (List.length groups)

(* Test aggregations *)
let test_agg_float () =
  let df = create [ ("x", Col.float32_list [ 1.0; 2.0; 3.0; 4.0 ]) ] in

  check_float "sum" 10.0 (Agg.Float.sum df "x");
  check_float "mean" 2.5 (Agg.Float.mean df "x");
  check_float "std" 1.118034 (Agg.Float.std df "x");
  check_option_float "min" (Some 1.0) (Agg.Float.min df "x");
  check_option_float "max" (Some 4.0) (Agg.Float.max df "x");
  check_float "median" 2.5 (Agg.Float.median df "x")

let test_agg_int () =
  let df = create [ ("x", Col.int32_list [ 1l; 2l; 3l; 4l ]) ] in

  check_bool "sum" true (Agg.Int.sum df "x" = 10L);
  check_float "mean" 2.5 (Agg.Int.mean df "x");
  check_bool "min" true (Agg.Int.min df "x" = Some 1L);
  check_bool "max" true (Agg.Int.max df "x" = Some 4L)

let test_agg_string () =
  let df = create [ ("x", Col.string_list [ "b"; "a"; "c"; "b" ]) ] in

  check_option_string "min" (Some "a") (Agg.String.min df "x");
  check_option_string "max" (Some "c") (Agg.String.max df "x");
  check_string "concat" "bacb" (Agg.String.concat df "x" ());
  check_int "nunique" 3 (Agg.String.nunique df "x");
  check_option_string "mode" (Some "b") (Agg.String.mode df "x")

let test_agg_bool () =
  let df = create [ ("x", Col.bool_list [ true; false; true; true ]) ] in

  check_bool "all" false (Agg.Bool.all df "x");
  check_bool "any" true (Agg.Bool.any df "x");
  check_int "sum" 3 (Agg.Bool.sum df "x");
  check_float "mean" 0.75 (Agg.Bool.mean df "x")

let test_agg_cumulative () =
  let df = create [ ("x", Col.int32_list [ 1l; 2l; 3l ]) ] in

  let cumsum = Agg.cumsum df "x" in
  let df_cumsum = create [ ("cumsum", cumsum) ] in
  check_int "cumsum length" 3 (num_rows df_cumsum);

  let diff = Agg.diff df "x" () in
  let df_diff = create [ ("diff", diff) ] in
  check_int "diff length" 3 (num_rows df_diff)

let test_agg_nulls () =
  let df = create [ ("x", Col.float32_list [ 1.0; Float.nan; 3.0 ]) ] in

  let nulls = Agg.is_null df "x" in
  check_bool "null detection" true nulls.(1);
  check_bool "non-null" false nulls.(0);
  check_int "count non-null" 2 (Agg.count df "x")

(* Test type conversions *)
let test_to_arrays () =
  let df =
    create
      [
        ("f32", Col.float32_list [ 1.0; 2.0 ]);
        ("i32", Col.int32_list [ 1l; 2l ]);
        ("str", Col.string_list [ "a"; "b" ]);
        ("bool", Col.bool_list [ true; false ]);
      ]
  in

  (match to_float32_array df "f32" with
  | Some arr -> check_bool "float32 array" true (arr = [| 1.0; 2.0 |])
  | None -> Alcotest.fail "should get float32 array");

  (match to_int32_array df "i32" with
  | Some arr -> check_bool "int32 array" true (arr = [| 1l; 2l |])
  | None -> Alcotest.fail "should get int32 array");

  (match to_string_array df "str" with
  | Some arr -> check_bool "string array" true (arr = [| "a"; "b" |])
  | None -> Alcotest.fail "should get string array");

  match to_bool_array df "bool" with
  | Some arr -> check_bool "bool array" true (arr = [| true; false |])
  | None -> Alcotest.fail "should get bool array"

let test_to_nx () =
  let df =
    create
      [
        ("a", Col.float32_list [ 1.0; 2.0 ]);
        ("b", Col.float32_list [ 3.0; 4.0 ]);
      ]
  in

  let tensor = to_nx df in
  let shape = Nx.shape tensor in
  check_bool "to_nx shape" true (shape = [| 2; 2 |])

let test_from_nx () =
  (* Test basic from_nx functionality *)
  let tensor =
    Nx.create Nx.float32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]
  in
  let df = from_nx tensor in

  (* Just check that we get a dataframe with the right number of columns *)
  check_int "from_nx cols" 3 (num_columns df);
  check_bool "from_nx not empty" false (is_empty df)

(* Test edge cases *)
let test_empty_operations () =
  let df = empty in

  let df2 = head df in
  check_bool "head of empty" true (is_empty df2);

  let df3 = filter df [||] in
  check_bool "filter empty" true (is_empty df3);

  let df4 = concat ~axis:`Rows [] in
  check_bool "concat empty list" true (is_empty df4)

let test_single_row () =
  let df = create [ ("x", Col.int32_list [ 42l ]) ] in

  check_int "single row" 1 (num_rows df);

  let h = head ~n:10 df in
  check_int "head larger than df" 1 (num_rows h)

let test_cast_column () =
  let df = create [ ("x", Col.int32_list [ 1l; 2l; 3l ]) ] in
  let df2 = cast_column df "x" Nx.float32 in

  (* Check that we can extract as float array after casting *)
  match to_float32_array df2 "x" with
  | Some arr -> check_bool "cast to float32" true (Array.length arr = 3)
  | None -> Alcotest.fail "should be able to extract as float32 after cast"

(* Test suites *)
let col_tests =
  [
    ("creation", `Quick, test_col_creation);
    ("nulls", `Quick, test_col_nulls);
    ("null mask", `Quick, test_col_null_mask);
    ("drop_nulls mask", `Quick, test_drop_nulls_preserves_data_with_mask);
    ("fill_nulls mask", `Quick, test_fill_nulls_respects_mask);
  ]

let creation_tests =
  [ ("basic", `Quick, test_df_creation); ("empty", `Quick, test_df_empty) ]

let column_tests =
  [
    ("access", `Quick, test_column_access);
    ("add_drop", `Quick, test_column_add_drop);
    ("rename", `Quick, test_rename_column);
    ("select", `Quick, test_select);
  ]

(* Test option-based accessors *)
let test_row_opt_accessors () =
  let df =
    create
      [
        ("float_col", Col.float64_opt [| Some 1.0; None; Some 3.0 |]);
        ("int_col", Col.int32_opt [| Some 10l; None; Some 30l |]);
        ("string_col", Col.string_opt [| Some "a"; None; Some "c" |]);
        ("bool_col", Col.bool_opt [| Some true; None; Some false |]);
      ]
  in
  (* Test float64_opt *)
  let float_values =
    map df Nx.float64
      (Row.map (Row.float64_opt "float_col") ~f:(function
        | Some v -> v
        | None -> -1.0))
  in
  check_float "float_opt row 0" 1.0 (Nx.to_array float_values).(0);
  check_float "float_opt row 1 (null)" (-1.0) (Nx.to_array float_values).(1);
  check_float "float_opt row 2" 3.0 (Nx.to_array float_values).(2);
  (* Test int32_opt *)
  let int_values =
    map df Nx.int32
      (Row.map (Row.int32_opt "int_col") ~f:(function
        | Some v -> v
        | None -> -1l))
  in
  check_int "int_opt row 0" 10 (Int32.to_int (Nx.to_array int_values).(0));
  check_int "int_opt row 1 (null)" (-1)
    (Int32.to_int (Nx.to_array int_values).(1));
  check_int "int_opt row 2" 30 (Int32.to_int (Nx.to_array int_values).(2))

let test_to_options () =
  let df =
    create
      [
        ("float_col", Col.float64_opt [| Some 1.0; None; Some 3.0 |]);
        ("int_col", Col.int32_opt [| Some 10l; None; Some 30l |]);
      ]
  in
  (* Test to_float64_options *)
  (match to_float64_options df "float_col" with
  | Some arr ->
      check_option_float "float opt 0" (Some 1.0) arr.(0);
      check_option_float "float opt 1 (null)" None arr.(1);
      check_option_float "float opt 2" (Some 3.0) arr.(2)
  | None -> Alcotest.fail "Expected Some array");
  (* Test to_int32_options *)
  match to_int32_options df "int_col" with
  | Some arr ->
      Alcotest.(check (option int32)) "int opt 0" (Some 10l) arr.(0);
      Alcotest.(check (option int32)) "int opt 1 (null)" None arr.(1);
      Alcotest.(check (option int32)) "int opt 2" (Some 30l) arr.(2)
  | None -> Alcotest.fail "Expected Some array"

let test_drop_nulls_helper () =
  let df =
    create
      [
        ("a", Col.float64_opt [| Some 1.0; None; Some 3.0; Some 4.0 |]);
        ("b", Col.int32 [| 10l; 20l; 30l; 40l |]);
      ]
  in
  (* Drop rows with any nulls *)
  let cleaned = drop_nulls df in
  check_int "drop_nulls all" 3 (num_rows cleaned);
  (* Drop only checking column "b" (which has no nulls) *)
  let partial = drop_nulls df ~subset:[ "b" ] in
  check_int "drop_nulls subset" 4 (num_rows partial)

let test_fill_missing_helper () =
  let df = create [ ("x", Col.float64_opt [| Some 1.0; None; Some 3.0 |]) ] in
  let filled = fill_missing df "x" ~with_value:(`Float 0.0) in
  match to_float64_array filled "x" with
  | Some arr ->
      check_float "filled 0" 1.0 arr.(0);
      check_float "filled 1 (was null)" 0.0 arr.(1);
      check_float "filled 2" 3.0 arr.(2)
  | None -> Alcotest.fail "Expected float array"

let test_fillna_replaces_nulls () =
  let df =
    create
      [
        ("a", Col.float32_opt [| Some 1.0; None; Some 3.0 |]);
        ("b", Col.int32_opt [| Some 10l; None; Some 30l |]);
      ]
  in
  let filled_a = Agg.fillna df "a" ~value:(Col.float32 [| 0.0 |]) in
  (match filled_a with
  | Col.P (Nx.Float32, tensor, mask_opt) ->
      let arr : float array = Nx.to_array tensor in
      check_float "filled a[0]" 1.0 arr.(0);
      check_float "filled a[1]" 0.0 arr.(1);
      check_float "filled a[2]" 3.0 arr.(2);
      check_option_bool_array "mask cleared for a" None mask_opt
  | _ -> Alcotest.fail "Expected float32 column");
  let filled_b =
    Agg.fillna df "b" ~value:(Col.int32_list [ 0l; 99l; 0l ])
  in
  match filled_b with
  | Col.P (Nx.Int32, tensor, mask_opt) ->
      let arr : int32 array = Nx.to_array tensor in
      check_int "filled b[0]" 10 (Int32.to_int arr.(0));
      check_int "filled b[1]" 99 (Int32.to_int arr.(1));
      check_int "filled b[2]" 30 (Int32.to_int arr.(2));
      check_option_bool_array "mask cleared for b" None mask_opt
  | _ -> Alcotest.fail "Expected int32 column"

let test_null_count_helper () =
  let df =
    create
      [
        ("a", Col.float64_opt [| Some 1.0; None; Some 3.0 |]);
        ("b", Col.int32 [| 10l; 20l; 30l |]);
      ]
  in
  check_int "null_count a" 1 (null_count df "a");
  check_int "null_count b" 0 (null_count df "b");
  check_bool "has_nulls a" true (has_nulls df "a");
  check_bool "has_nulls b" false (has_nulls df "b")

let test_mask_aware_aggregations () =
  let df =
    create
      [ ("values", Col.float64_opt [| Some 1.0; None; Some 3.0; Some 5.0 |]) ]
  in
  (* Sum should skip the null *)
  let sum_result = Agg.Float.sum df "values" in
  check_float "masked sum" 9.0 sum_result;
  (* Mean should compute over non-null values only *)
  let mean_result = Agg.Float.mean df "values" in
  check_float "masked mean" 3.0 mean_result;
  (* Min/max should skip nulls *)
  (match Agg.Float.min df "values" with
  | Some v -> check_float "masked min" 1.0 v
  | None -> Alcotest.fail "Expected Some min");
  match Agg.Float.max df "values" with
  | Some v -> check_float "masked max" 5.0 v
  | None -> Alcotest.fail "Expected Some max"

let option_tests =
  [
    ("row_opt_accessors", `Quick, test_row_opt_accessors);
    ("to_options", `Quick, test_to_options);
    ("drop_nulls", `Quick, test_drop_nulls_helper);
    ("fill_missing", `Quick, test_fill_missing_helper);
    ("fillna", `Quick, test_fillna_replaces_nulls);
    ("null_count", `Quick, test_null_count_helper);
    ("mask_aware_agg", `Quick, test_mask_aware_aggregations);
  ]

let mask_tests =
  [
    ("projection", `Quick, test_mask_projection_ops);
    ("concat", `Quick, test_concat_mask_combines);
    ("cast", `Quick, test_cast_preserves_mask);
    ("pct_change", `Quick, test_pct_change_has_no_mask);
  ]

let row_tests =
  [
    ("head_tail", `Quick, test_head_tail);
    ("slice", `Quick, test_slice);
    ("filter", `Quick, test_filter);
    ("filter_by", `Quick, test_filter_by);
    ("drop_duplicates", `Quick, test_drop_duplicates);
  ]

let concat_tests =
  [ ("rows", `Quick, test_concat_rows); ("columns", `Quick, test_concat_cols) ]

let row_module_tests =
  [ ("accessors", `Quick, test_row_accessors); ("map", `Quick, test_row_map) ]

let sort_group_tests =
  [ ("sort", `Quick, test_sort); ("group_by", `Quick, test_group_by) ]

let agg_tests =
  [
    ("float", `Quick, test_agg_float);
    ("int", `Quick, test_agg_int);
    ("string", `Quick, test_agg_string);
    ("bool", `Quick, test_agg_bool);
    ("cumulative", `Quick, test_agg_cumulative);
    ("nulls", `Quick, test_agg_nulls);
  ]

let conversion_tests =
  [
    ("to_arrays", `Quick, test_to_arrays);
    ("to_nx", `Quick, test_to_nx);
    ("from_nx", `Quick, test_from_nx);
    ("cast_column", `Quick, test_cast_column);
  ]

let edge_tests =
  [
    ("empty_operations", `Quick, test_empty_operations);
    ("single_row", `Quick, test_single_row);
  ]

(* Test wide operations with mapHomo *)
let test_map_list_product () =
  (* Create a dataframe with 6 int32 columns *)
  let df =
    create
      [
        ("A", Col.int32_list [ 2l; 3l; 4l ]);
        ("B", Col.int32_list [ 1l; 2l; 3l ]);
        ("C", Col.int32_list [ 3l; 1l; 2l ]);
        ("D", Col.int32_list [ 1l; 1l; 1l ]);
        ("E", Col.int32_list [ 2l; 2l; 2l ]);
        ("F", Col.int32_list [ 1l; 3l; 1l ]);
      ]
  in

  (* Compute product of all 6 columns using map_list *)
  let df2 =
    with_column df "allMul" Nx.int32
      Row.(
        map_list
          (int32s [ "A"; "B"; "C"; "D"; "E"; "F" ])
          ~f:(List.fold_left Int32.mul 1l))
  in

  (* Check the results *)
  match to_int32_array df2 "allMul" with
  | Some arr ->
      (* Row 1: 2*1*3*1*2*1 = 12 *)
      (* Row 2: 3*2*1*1*2*3 = 36 *)
      (* Row 3: 4*3*2*1*2*1 = 48 *)
      check_bool "Product of 6 columns" true (arr = [| 12l; 36l; 48l |])
  | None -> Alcotest.fail "allMul column should exist"

let test_sequence_sum () =
  (* Create a dataframe with float columns *)
  let df =
    create
      [
        ("A", Col.float64_list [ 1.0; 2.0; 3.0 ]);
        ("B", Col.float64_list [ 2.0; 3.0; 4.0 ]);
        ("C", Col.float64_list [ 3.0; 4.0; 5.0 ]);
      ]
  in

  (* Sum all columns using sequence + map *)
  let df2 =
    with_column df "total" Nx.float64
      Row.(
        map
          (sequence (float64s [ "A"; "B"; "C" ]))
          ~f:(List.fold_left ( +. ) 0.))
  in

  match to_float64_array df2 "total" with
  | Some arr ->
      check_float "Row 1 sum" 6.0 arr.(0);
      check_float "Row 2 sum" 9.0 arr.(1);
      check_float "Row 3 sum" 12.0 arr.(2)
  | None -> Alcotest.fail "total column should exist"

let test_weighted_sum () =
  (* Test weighted sum example *)
  let df =
    create
      [
        ("A", Col.float64_list [ 1.0; 2.0; 3.0 ]);
        ("B", Col.float64_list [ 2.0; 3.0; 4.0 ]);
        ("C", Col.float64_list [ 3.0; 4.0; 5.0 ]);
        ("D", Col.float64_list [ 4.0; 5.0; 6.0 ]);
        ("E", Col.float64_list [ 5.0; 6.0; 7.0 ]);
        ("F", Col.float64_list [ 6.0; 7.0; 8.0 ]);
      ]
  in

  let feats = [ "A"; "B"; "C"; "D"; "E"; "F" ] in
  let weights = [ 0.2; 0.3; 0.1; 0.1; 0.1; 0.2 ] in

  let df' =
    with_column df "score" Nx.float64
      Row.(
        map_list (float64s feats) ~f:(fun xs ->
            List.fold_left2 (fun acc wi xi -> acc +. (wi *. xi)) 0. weights xs))
  in

  match to_float64_array df' "score" with
  | Some scores ->
      (* First row: 0.2*1 + 0.3*2 + 0.1*3 + 0.1*4 + 0.1*5 + 0.2*6 = 3.2 *)
      check_float "First weighted sum" 3.2 scores.(0);
      check_float "Second weighted sum" 4.2 scores.(1);
      check_float "Third weighted sum" 5.2 scores.(2)
  | None -> Alcotest.fail "score column should exist"

let test_numeric_column_names () =
  let df =
    create
      [
        ("name", Col.string_list [ "Alice"; "Bob"; "Charlie" ]);
        ("age", Col.int32_list [ 25l; 30l; 35l ]);
        ("score", Col.float64_list [ 85.5; 92.0; 78.5 ]);
        ("active", Col.bool_list [ true; false; true ]);
        ("height", Col.float32_list [ 1.75; 1.80; 1.70 ]);
        ("id", Col.int64_list [ 1L; 2L; 3L ]);
      ]
  in

  let numeric_cols = Cols.numeric df in
  let expected = [ "age"; "score"; "height"; "id" ] in

  (* Sort both lists for comparison since order might vary *)
  let sorted_numeric = List.sort String.compare numeric_cols in
  let sorted_expected = List.sort String.compare expected in

  Alcotest.(check (list string))
    "Numeric column names" sorted_expected sorted_numeric

let test_row_helpers () =
  let df =
    create
      [
        ("a", Col.int32_list [ 1l; 2l ]);
        ("b", Col.int32_list [ 3l; 4l ]);
        ("c", Col.float64_list [ 5.0; 6.0 ]);
      ]
  in

  (* Test int32s helper *)
  let int_cols = Row.int32s [ "a"; "b" ] in
  check_int "int32s creates 2 accessors" 2 (List.length int_cols);

  (* Test float64s helper *)
  let float_cols = Row.float64s [ "c" ] in
  check_int "float64s creates 1 accessor" 1 (List.length float_cols);

  (* Test using the helpers with all (alias for sequence) *)
  let df2 =
    with_column df "sum" Nx.float64
      Row.(
        map
          (all (float64s [ "c" ]))
          ~f:(fun xs ->
            match xs with
            | [ x ] -> x *. 2.0
            | _ -> failwith "Expected exactly one column"))
  in

  match to_float64_array df2 "sum" with
  | Some arr ->
      check_float "First doubled" 10.0 arr.(0);
      check_float "Second doubled" 12.0 arr.(1)
  | None -> Alcotest.fail "sum column should exist"

let test_sequence_all_equivalence () =
  (* Test that sequence and all are equivalent *)
  let df =
    create
      [
        ("x", Col.int32_list [ 1l; 2l ]);
        ("y", Col.int32_list [ 10l; 20l ]);
        ("z", Col.int32_list [ 100l; 200l ]);
      ]
  in

  (* Using sequence *)
  let df_seq =
    with_column df "sum_seq" Nx.int32
      Row.(
        map
          (sequence (int32s [ "x"; "y"; "z" ]))
          ~f:(List.fold_left Int32.add 0l))
  in

  (* Using all (should be identical) *)
  let df_all =
    with_column df "sum_all" Nx.int32
      Row.(
        map (all (int32s [ "x"; "y"; "z" ])) ~f:(List.fold_left Int32.add 0l))
  in

  match (to_int32_array df_seq "sum_seq", to_int32_array df_all "sum_all") with
  | Some seq_arr, Some all_arr ->
      check_bool "sequence and all produce same results" true
        (seq_arr = all_arr && seq_arr = [| 111l; 222l |])
  | _ -> Alcotest.fail "columns should exist"

let wide_tests =
  [
    ("map_list product", `Quick, test_map_list_product);
    ("sequence sum", `Quick, test_sequence_sum);
    ("weighted sum", `Quick, test_weighted_sum);
    ("numeric_column_names", `Quick, test_numeric_column_names);
    ("row helpers", `Quick, test_row_helpers);
    ("sequence/all equivalence", `Quick, test_sequence_all_equivalence);
  ]

(* Test ergonomic APIs *)
let test_with_columns () =
  let df =
    create
      [
        ("x", Col.float64_list [ 1.0; 2.0; 3.0 ]);
        ("y", Col.float64_list [ 4.0; 5.0; 6.0 ]);
      ]
  in

  let df2 =
    with_columns df
      [
        ("z", Col.float64_list [ 7.0; 8.0; 9.0 ]);
        ("sum", Col.float64_list [ 5.0; 7.0; 9.0 ]);
      ]
  in

  check_int "columns added" 4 (num_columns df2);
  check_bool "has z" true (has_column df2 "z");
  check_bool "has sum" true (has_column df2 "sum")

let test_column_selectors () =
  let df =
    create
      [
        ("feat_1", Col.float64_list [ 1.0 ]);
        ("feat_2", Col.float64_list [ 2.0 ]);
        ("id", Col.int32_list [ 1l ]);
        ("name", Col.string_list [ "test" ]);
        ("score_a", Col.float64_list [ 3.0 ]);
        ("score_b", Col.float64_list [ 4.0 ]);
      ]
  in

  (* Test columns_with_prefix *)
  let feat_cols = Cols.with_prefix df "feat_" in
  Alcotest.(check (list string))
    "feat_ prefix" [ "feat_1"; "feat_2" ]
    (List.sort String.compare feat_cols);

  (* Test columns_with_suffix *)
  let score_cols = Cols.with_suffix df "_a" in
  Alcotest.(check (list string)) "suffix _a" [ "score_a" ] score_cols;

  (* Test select_dtypes *)
  let numeric = Cols.select_dtypes df [ `Numeric ] in
  check_int "numeric columns" 5 (List.length numeric);

  let strings = Cols.select_dtypes df [ `String ] in
  Alcotest.(check (list string)) "string columns" [ "name" ] strings

let test_rowagg_sum () =
  let df =
    create
      [
        ("a", Col.float64_list [ 1.0; 2.0; Float.nan ]);
        ("b", Col.float64_list [ 3.0; 4.0; 5.0 ]);
        ("c", Col.int32_list [ 5l; 6l; 7l ]);
      ]
  in

  (* Test sum with skipna=true (default) *)
  let sum_col = Row.Agg.sum df ~names:[ "a"; "b"; "c" ] in
  let df2 = add_column df "row_sum" sum_col in

  match to_float64_array df2 "row_sum" with
  | Some arr ->
      check_float "Row 0 sum" 9.0 arr.(0);
      (* 1 + 3 + 5 *)
      check_float "Row 1 sum" 12.0 arr.(1);
      (* 2 + 4 + 6 *)
      check_float "Row 2 sum" 12.0 arr.(2)
      (* NaN + 5 + 7, NaN skipped *)
  | None -> Alcotest.fail "row_sum should exist"

let test_row_number () =
  let df =
    create
      [
        ("i32", Col.int32_list [ 1l; 2l; 3l ]);
        ("i64", Col.int64_list [ 10L; 20L; 30L ]);
        ("f32", Col.float32_list [ 1.5; 2.5; 3.5 ]);
        ("f64", Col.float64_list [ 100.0; 200.0; 300.0 ]);
      ]
  in

  (* Test Row.number coerces all numeric types to float *)
  let df2 = with_column df "i32_as_float" Nx.float64 Row.(number "i32") in
  match to_float64_array df2 "i32_as_float" with
  | Some arr ->
      check_float "Int32 as float" 1.0 arr.(0);
      check_float "Int32 as float" 2.0 arr.(1);
      check_float "Int32 as float" 3.0 arr.(2)
  | None -> Alcotest.fail "i32_as_float should exist"

let test_row_fold_list () =
  let df =
    create
      [
        ("a", Col.float64_list [ 1.0; 2.0; 3.0 ]);
        ("b", Col.float64_list [ 10.0; 20.0; 30.0 ]);
        ("c", Col.float64_list [ 100.0; 200.0; 300.0 ]);
      ]
  in

  (* Test fold_list to compute sum without intermediate list *)
  let df2 =
    with_column df "sum" Nx.float64
      Row.(fold_list (numbers [ "a"; "b"; "c" ]) ~init:0. ~f:( +. ))
  in

  match to_float64_array df2 "sum" with
  | Some arr ->
      check_float "Row 0 sum" 111.0 arr.(0);
      check_float "Row 1 sum" 222.0 arr.(1);
      check_float "Row 2 sum" 333.0 arr.(2)
  | None -> Alcotest.fail "sum should exist"

let test_with_columns_map () =
  let df =
    create
      [
        ("x", Col.float64_list [ 1.0; 2.0; 3.0 ]);
        ("y", Col.float64_list [ 4.0; 5.0; 6.0 ]);
      ]
  in

  let df2 =
    with_columns_map df
      [
        ( "sum",
          Nx.float64,
          Row.map2 (Row.float64 "x") (Row.float64 "y") ~f:( +. ) );
        ( "diff",
          Nx.float64,
          Row.map2 (Row.float64 "x") (Row.float64 "y") ~f:( -. ) );
        ( "prod",
          Nx.float64,
          Row.map2 (Row.float64 "x") (Row.float64 "y") ~f:( *. ) );
      ]
  in

  check_int "columns added" 5 (num_columns df2);

  match
    ( to_float64_array df2 "sum",
      to_float64_array df2 "diff",
      to_float64_array df2 "prod" )
  with
  | Some sum, Some diff, Some prod ->
      check_float "Sum row 0" 5.0 sum.(0);
      check_float "Diff row 0" (-3.0) diff.(0);
      check_float "Prod row 0" 4.0 prod.(0)
  | _ -> Alcotest.fail "computed columns should exist"

let test_columns_except () =
  let df =
    create
      [
        ("keep1", Col.int32_list [ 1l ]);
        ("drop1", Col.int32_list [ 2l ]);
        ("keep2", Col.int32_list [ 3l ]);
        ("drop2", Col.int32_list [ 4l ]);
        ("keep3", Col.int32_list [ 5l ]);
      ]
  in

  let kept = Cols.except df [ "drop1"; "drop2" ] in
  Alcotest.(check (list string))
    "columns except"
    [ "keep1"; "keep2"; "keep3" ]
    (List.sort String.compare kept)

let test_rowagg_dot () =
  let df =
    create
      [
        ("x", Col.float64_list [ 1.0; 2.0; 3.0 ]);
        ("y", Col.float64_list [ 4.0; 5.0; 6.0 ]);
        ("z", Col.float64_list [ 7.0; 8.0; 9.0 ]);
      ]
  in

  let weights = [| 0.2; 0.3; 0.5 |] in
  let score = Row.Agg.dot df ~names:[ "x"; "y"; "z" ] ~weights in
  let df2 = add_column df "score" score in

  match to_float64_array df2 "score" with
  | Some arr ->
      check_float "Row 0 weighted" 4.9 arr.(0);
      (* 0.2*1 + 0.3*4 + 0.5*7 = 0.2 + 1.2 + 3.5 = 4.9 *)
      check_float "Row 1 weighted" 5.9 arr.(1);
      (* 0.2*2 + 0.3*5 + 0.5*8 = 0.4 + 1.5 + 4.0 = 5.9 *)
      check_float "Row 2 weighted" 6.9 arr.(2)
      (* 0.2*3 + 0.3*6 + 0.5*9 = 0.6 + 1.8 + 4.5 = 6.9 *)
  | None -> Alcotest.fail "score should exist"

let test_join_inner () =
  let df1 =
    create
      [
        ("id", Col.int32_list [ 1l; 2l; 3l ]);
        ("name", Col.string_list [ "Alice"; "Bob"; "Charlie" ]);
      ]
  in
  let df2 =
    create
      [
        ("id", Col.int32_list [ 2l; 3l; 4l ]);
        ("score", Col.float64_list [ 85.0; 90.0; 95.0 ]);
      ]
  in

  let result = join df1 df2 ~on:"id" ~how:`Inner () in
  check_int "inner join rows" 2 (num_rows result);
  check_bool "has name column" true (has_column result "name");
  check_bool "has score column" true (has_column result "score")

let test_join_left () =
  let df1 =
    create
      [
        ("key", Col.string_list [ "a"; "b"; "c" ]);
        ("val1", Col.int32_list [ 1l; 2l; 3l ]);
      ]
  in
  let df2 =
    create
      [
        ("key", Col.string_list [ "b"; "c"; "d" ]);
        ("val2", Col.int32_list [ 20l; 30l; 40l ]);
      ]
  in

  let result = join df1 df2 ~on:"key" ~how:`Left () in
  check_int "left join rows" 3 (num_rows result);

  (* Check that all left keys are present *)
  match to_string_array result "key" with
  | Some arr ->
      check_string "first key" "a" arr.(0);
      check_string "second key" "b" arr.(1);
      check_string "third key" "c" arr.(2)
  | None -> Alcotest.fail "key column should exist"

let test_merge () =
  let df1 =
    create
      [
        ("id", Col.int32_list [ 1l; 2l ]); ("x", Col.float64_list [ 10.0; 20.0 ]);
      ]
  in
  let df2 =
    create
      [
        ("code", Col.int32_list [ 1l; 2l ]);
        ("y", Col.float64_list [ 100.0; 200.0 ]);
      ]
  in

  let result = merge df1 df2 ~left_on:"id" ~right_on:"code" ~how:`Inner () in
  check_int "merge rows" 2 (num_rows result);
  check_bool "has x column" true (has_column result "x");
  check_bool "has y column" true (has_column result "y")

let test_pivot () =
  let df =
    create
      [
        ("date", Col.string_list [ "2024-01"; "2024-01"; "2024-02"; "2024-02" ]);
        ("product", Col.string_list [ "A"; "B"; "A"; "B" ]);
        ("sales", Col.float64_list [ 100.0; 150.0; 120.0; 180.0 ]);
      ]
  in

  let pivoted =
    pivot df ~index:"date" ~columns:"product" ~values:"sales" ~agg_func:`Sum ()
  in
  check_int "pivot rows" 2 (num_rows pivoted);
  check_bool "has A column" true (has_column pivoted "A");
  check_bool "has B column" true (has_column pivoted "B");

  (* Check aggregated values *)
  match (to_float64_array pivoted "A", to_float64_array pivoted "B") with
  | Some a_vals, Some b_vals ->
      check_float "Jan A sales" 100.0 a_vals.(0);
      check_float "Jan B sales" 150.0 b_vals.(0);
      check_float "Feb A sales" 120.0 a_vals.(1);
      check_float "Feb B sales" 180.0 b_vals.(1)
  | _ -> Alcotest.fail "pivot columns should exist"

let test_melt () =
  let df =
    create
      [
        ("id", Col.int32_list [ 1l; 2l ]);
        ("A", Col.float64_list [ 10.0; 20.0 ]);
        ("B", Col.float64_list [ 30.0; 40.0 ]);
        ("C", Col.float64_list [ 50.0; 60.0 ]);
      ]
  in

  let melted = melt df ~id_vars:[ "id" ] ~value_vars:[ "A"; "B"; "C" ] () in
  check_int "melt rows" 6 (num_rows melted);
  (* 2 rows * 3 columns = 6 *)
  check_bool "has id column" true (has_column melted "id");
  check_bool "has variable column" true (has_column melted "variable");
  check_bool "has value column" true (has_column melted "value");

  (* Check melted structure *)
  match to_string_array melted "variable" with
  | Some vars ->
      check_string "first var" "A" vars.(0);
      check_string "second var" "B" vars.(1);
      check_string "third var" "C" vars.(2);
      check_string "fourth var" "A" vars.(3);
      check_string "fifth var" "B" vars.(4);
      check_string "sixth var" "C" vars.(5)
  | None -> Alcotest.fail "variable column should exist"

let test_join_with_suffixes () =
  let df1 =
    create
      [
        ("id", Col.int32_list [ 1l; 2l ]);
        ("value", Col.float64_list [ 10.0; 20.0 ]);
      ]
  in
  let df2 =
    create
      [
        ("id", Col.int32_list [ 1l; 2l ]);
        ("value", Col.float64_list [ 100.0; 200.0 ]);
      ]
  in

  let result =
    join df1 df2 ~on:"id" ~how:`Inner ~suffixes:("_left", "_right") ()
  in
  check_bool "has value_left column" true (has_column result "value_left");
  check_bool "has value_right column" true (has_column result "value_right");

  match
    (to_float64_array result "value_left", to_float64_array result "value_right")
  with
  | Some left, Some right ->
      check_float "left value 1" 10.0 left.(0);
      check_float "right value 1" 100.0 right.(0);
      check_float "left value 2" 20.0 left.(1);
      check_float "right value 2" 200.0 right.(1)
  | _ -> Alcotest.fail "value columns should exist"

let join_reshape_tests =
  [
    ("join inner", `Quick, test_join_inner);
    ("join left", `Quick, test_join_left);
    ("merge", `Quick, test_merge);
    ("pivot", `Quick, test_pivot);
    ("melt", `Quick, test_melt);
    ("join with suffixes", `Quick, test_join_with_suffixes);
  ]

let ergonomic_tests =
  [
    ("with_columns", `Quick, test_with_columns);
    ("with_columns_map", `Quick, test_with_columns_map);
    ("column_selectors", `Quick, test_column_selectors);
    ("columns_except", `Quick, test_columns_except);
    ("Row.number", `Quick, test_row_number);
    ("Row.fold_list", `Quick, test_row_fold_list);
    ("Row.Agg.sum", `Quick, test_rowagg_sum);
    ("Row.Agg.dot", `Quick, test_rowagg_dot);
  ]

let () =
  Alcotest.run "Talon"
    [
      ("Col", col_tests);
      ("Creation", creation_tests);
      ("Columns", column_tests);
      ("Null masks", mask_tests);
      ("Option accessors", option_tests);
      ("Rows", row_tests);
      ("Concatenation", concat_tests);
      ("Row module", row_module_tests);
      ("Sort & Group", sort_group_tests);
      ("Aggregations", agg_tests);
      ("Conversions", conversion_tests);
      ("Edge cases", edge_tests);
      ("Wide operations", wide_tests);
      ("Ergonomic APIs", ergonomic_tests);
      ("Join & Reshape", join_reshape_tests);
    ]
