open Talon

let check_int = Alcotest.(check int)
let check_float = Alcotest.(check (float 1e-6))
let check_bool = Alcotest.(check bool)
let check_string = Alcotest.(check string)
let check_option_float = Alcotest.(check (option (float 1e-6)))
let check_option_string = Alcotest.(check (option string))

(* Test column creation *)
let test_col_creation () =
  let df1 = create [("c", Col.float32 [| 1.0; 2.0; 3.0 |])] in
  check_int "float32 col rows" 3 (num_rows df1);
  let c1 = get_column_exn df1 "c" in
  check_bool "float32 no nulls" false (Col.has_nulls c1);
  
  let df2 = create [("c", Col.int32 [| 1l; 2l; 3l |])] in
  check_int "int32 col rows" 3 (num_rows df2);
  
  let df3 = create [("c", Col.string [| "a"; "b"; "c" |])] in
  check_int "string col rows" 3 (num_rows df3);
  
  let df4 = create [("c", Col.bool [| true; false; true |])] in
  check_int "bool col rows" 3 (num_rows df4)

let test_col_nulls () =
  let c1 = Col.float32_opt [| Some 1.0; None; Some 3.0 |] in
  check_bool "has nulls" true (Col.has_nulls c1);
  check_int "null count" 1 (Col.null_count c1);
  
  let c2 = Col.drop_nulls c1 in
  let df = create [("c", c2)] in
  check_int "after drop_nulls" 2 (num_rows df);
  check_bool "no nulls after drop" false (Col.has_nulls c2)

(* Test dataframe creation *)
let test_df_creation () =
  let df = create [
    ("a", Col.int32_list [1l; 2l; 3l]);
    ("b", Col.float64_list [1.5; 2.5; 3.5]);
    ("c", Col.string_list ["x"; "y"; "z"])
  ] in
  
  let rows, cols = shape df in
  check_int "rows" 3 rows;
  check_int "cols" 3 cols;
  check_bool "not empty" false (is_empty df);
  
  let names = column_names df in
  Alcotest.(check (list string)) "column names" ["a"; "b"; "c"] names

let test_df_empty () =
  let df = empty in
  let rows, cols = shape df in
  check_int "empty rows" 0 rows;
  check_int "empty cols" 0 cols;
  check_bool "is empty" true (is_empty df)

(* Test column operations *)
let test_column_access () =
  let df = create [
    ("x", Col.int32_list [1l; 2l; 3l]);
    ("y", Col.float32_list [1.0; 2.0; 3.0])
  ] in
  
  check_bool "has column x" true (has_column df "x");
  check_bool "has column y" true (has_column df "y");
  check_bool "no column z" false (has_column df "z");
  
  match get_column df "x" with
  | Some _col -> check_int "df has 3 rows" 3 (num_rows df)
  | None -> Alcotest.fail "column x should exist"

let test_column_add_drop () =
  let df = create [("a", Col.int32_list [1l; 2l])] in
  
  let df2 = add_column df "b" (Col.float32_list [1.0; 2.0]) in
  check_int "cols after add" 2 (num_columns df2);
  check_bool "has new column" true (has_column df2 "b");
  
  let df3 = drop_column df2 "a" in
  check_int "cols after drop" 1 (num_columns df3);
  check_bool "column dropped" false (has_column df3 "a")

let test_rename_column () =
  let df = create [("old", Col.int32_list [1l; 2l])] in
  let df2 = rename_column df ~old_name:"old" ~new_name:"new" in
  
  check_bool "old name gone" false (has_column df2 "old");
  check_bool "new name exists" true (has_column df2 "new")

let test_select () =
  let df = create [
    ("a", Col.int32_list [1l]);
    ("b", Col.int32_list [2l]);
    ("c", Col.int32_list [3l])
  ] in
  
  let df2 = select df ["c"; "a"] in
  Alcotest.(check (list string)) "selected cols" ["c"; "a"] (column_names df2);
  
  let df3 = select_loose df ["a"; "missing"; "c"] in
  Alcotest.(check (list string)) "loose select" ["a"; "c"] (column_names df3)

(* Test row operations *)
let test_head_tail () =
  let df = create [("x", Col.int32_list [1l; 2l; 3l; 4l; 5l])] in
  
  let h = head ~n:2 df in
  check_int "head size" 2 (num_rows h);
  
  let t = tail ~n:2 df in
  check_int "tail size" 2 (num_rows t)

let test_slice () =
  let df = create [("x", Col.int32_list [0l; 1l; 2l; 3l; 4l])] in
  let s = slice df ~start:1 ~stop:4 in
  check_int "slice size" 3 (num_rows s)

let test_filter () =
  let df = create [
    ("x", Col.int32_list [1l; 2l; 3l; 4l]);
    ("y", Col.string_list ["a"; "b"; "c"; "d"])
  ] in
  
  let mask = [| true; false; true; false |] in
  let filtered = filter df mask in
  check_int "filtered rows" 2 (num_rows filtered)

let test_filter_by () =
  let df = create [
    ("x", Col.int32_list [1l; 2l; 3l; 4l]);
    ("y", Col.float32_list [1.0; 2.0; 3.0; 4.0])
  ] in
  
  let filtered = filter_by df Row.(map (int32 "x") ~f:(fun x -> x > 2l)) in
  check_int "filtered rows" 2 (num_rows filtered)

let test_drop_duplicates () =
  let df = create [
    ("x", Col.int32_list [1l; 1l; 2l; 2l; 3l]);
    ("y", Col.string_list ["a"; "a"; "b"; "b"; "c"])
  ] in
  
  let unique = drop_duplicates df in
  check_int "unique rows" 3 (num_rows unique)

(* Test concatenation *)
let test_concat_rows () =
  let df1 = create [("x", Col.int32_list [1l; 2l])] in
  let df2 = create [("x", Col.int32_list [3l; 4l])] in
  
  let combined = concat ~axis:`Rows [df1; df2] in
  check_int "concat rows" 4 (num_rows combined)

let test_concat_cols () =
  let df1 = create [("a", Col.int32_list [1l; 2l])] in
  let df2 = create [("b", Col.int32_list [3l; 4l])] in
  
  let combined = concat ~axis:`Columns [df1; df2] in
  check_int "concat cols" 2 (num_columns combined)

(* Test Row module *)
let test_row_accessors () =
  let df = create [
    ("i32", Col.int32_list [42l; 24l]);
    ("f32", Col.float32_list [3.14; 2.71]);
    ("str", Col.string_list ["hello"; "world"]);
    ("bool", Col.bool_list [true; false])
  ] in
  
  (* Test by filtering based on row values *)
  let filtered_i32 = filter_by df Row.(map (int32 "i32") ~f:(fun x -> x = 42l)) in
  check_int "filtered by i32" 1 (num_rows filtered_i32);
  
  let filtered_str = filter_by df Row.(map (string "str") ~f:(fun s -> s = "hello")) in
  check_int "filtered by string" 1 (num_rows filtered_str);
  
  let filtered_bool = filter_by df Row.(bool "bool") in
  check_int "filtered by bool" 1 (num_rows filtered_bool)

let test_row_map () =
  let df = create [("x", Col.int32_list [1l; 2l; 3l])] in
  
  (* Use map_column to create a new column *)
  let df2 = map_column df "doubled" Nx.int32 
    Row.(map (int32 "x") ~f:(fun x -> Int32.mul x 2l)) in
  
  match to_int32_array df2 "doubled" with
  | Some arr -> check_bool "mapped values" true (arr = [|2l; 4l; 6l|])
  | None -> Alcotest.fail "doubled column should exist"

(* Test sorting *)
let test_sort () =
  let df = create [
    ("x", Col.int32_list [3l; 1l; 2l]);
    ("y", Col.string_list ["c"; "a"; "b"])
  ] in
  
  let sorted = sort_by_column df "x" in
  match to_int32_array sorted "x" with
  | Some arr -> 
      check_bool "sorted" true (arr = [|1l; 2l; 3l|])
  | None -> Alcotest.fail "column should exist"

let test_group_by () =
  let df = create [
    ("key", Col.string_list ["a"; "b"; "a"; "b"]);
    ("val", Col.int32_list [1l; 2l; 3l; 4l])
  ] in
  
  let groups = group_by_column df "key" in
  check_int "group count" 2 (List.length groups)

(* Test aggregations *)
let test_agg_float () =
  let df = create [("x", Col.float32_list [1.0; 2.0; 3.0; 4.0])] in
  
  check_float "sum" 10.0 (Agg.Float.sum df "x");
  check_float "mean" 2.5 (Agg.Float.mean df "x");
  check_float "std" 1.118034 (Agg.Float.std df "x");
  check_option_float "min" (Some 1.0) (Agg.Float.min df "x");
  check_option_float "max" (Some 4.0) (Agg.Float.max df "x");
  check_float "median" 2.5 (Agg.Float.median df "x")

let test_agg_int () =
  let df = create [("x", Col.int32_list [1l; 2l; 3l; 4l])] in
  
  check_bool "sum" true (Agg.Int.sum df "x" = 10L);
  check_float "mean" 2.5 (Agg.Int.mean df "x");
  check_bool "min" true (Agg.Int.min df "x" = Some 1L);
  check_bool "max" true (Agg.Int.max df "x" = Some 4L)

let test_agg_string () =
  let df = create [("x", Col.string_list ["b"; "a"; "c"; "b"])] in
  
  check_option_string "min" (Some "a") (Agg.String.min df "x");
  check_option_string "max" (Some "c") (Agg.String.max df "x");
  check_string "concat" "bacb" (Agg.String.concat df "x" ());
  check_int "nunique" 3 (Agg.String.nunique df "x");
  check_option_string "mode" (Some "b") (Agg.String.mode df "x")

let test_agg_bool () =
  let df = create [("x", Col.bool_list [true; false; true; true])] in
  
  check_bool "all" false (Agg.Bool.all df "x");
  check_bool "any" true (Agg.Bool.any df "x");
  check_int "sum" 3 (Agg.Bool.sum df "x");
  check_float "mean" 0.75 (Agg.Bool.mean df "x")

let test_agg_cumulative () =
  let df = create [("x", Col.int32_list [1l; 2l; 3l])] in
  
  let cumsum = Agg.cumsum df "x" in
  let df_cumsum = create [("cumsum", cumsum)] in
  check_int "cumsum length" 3 (num_rows df_cumsum);
  
  let diff = Agg.diff df "x" () in
  let df_diff = create [("diff", diff)] in
  check_int "diff length" 3 (num_rows df_diff)

let test_agg_nulls () =
  let df = create [("x", Col.float32_list [1.0; Float.nan; 3.0])] in
  
  let nulls = Agg.is_null df "x" in
  check_bool "null detection" true (nulls.(1));
  check_bool "non-null" false (nulls.(0));
  check_int "count non-null" 2 (Agg.count df "x")

(* Test type conversions *)
let test_to_arrays () =
  let df = create [
    ("f32", Col.float32_list [1.0; 2.0]);
    ("i32", Col.int32_list [1l; 2l]);
    ("str", Col.string_list ["a"; "b"]);
    ("bool", Col.bool_list [true; false])
  ] in
  
  begin match to_float32_array df "f32" with
  | Some arr -> check_bool "float32 array" true (arr = [|1.0; 2.0|])
  | None -> Alcotest.fail "should get float32 array"
  end;
  
  begin match to_int32_array df "i32" with
  | Some arr -> check_bool "int32 array" true (arr = [|1l; 2l|])
  | None -> Alcotest.fail "should get int32 array"
  end;
  
  begin match to_string_array df "str" with
  | Some arr -> check_bool "string array" true (arr = [|"a"; "b"|])
  | None -> Alcotest.fail "should get string array"
  end;
  
  begin match to_bool_array df "bool" with
  | Some arr -> check_bool "bool array" true (arr = [|true; false|])
  | None -> Alcotest.fail "should get bool array"
  end

let test_to_nx () =
  let df = create [
    ("a", Col.float32_list [1.0; 2.0]);
    ("b", Col.float32_list [3.0; 4.0])
  ] in
  
  let tensor = to_nx df in
  let shape = Nx.shape tensor in
  check_bool "to_nx shape" true (shape = [|2; 2|])

let test_from_nx () =
  (* Test basic from_nx functionality *)
  let tensor = Nx.create Nx.float32 [|2; 3|] [|1.0; 2.0; 3.0; 4.0; 5.0; 6.0|] in
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
  let df = create [("x", Col.int32_list [42l])] in
  
  check_int "single row" 1 (num_rows df);
  
  let h = head ~n:10 df in
  check_int "head larger than df" 1 (num_rows h)

let test_cast_column () =
  let df = create [("x", Col.int32_list [1l; 2l; 3l])] in
  let df2 = cast_column df "x" Nx.float32 in
  
  (* Check that we can extract as float array after casting *)
  match to_float32_array df2 "x" with
  | Some arr -> check_bool "cast to float32" true (Array.length arr = 3)
  | None -> Alcotest.fail "should be able to extract as float32 after cast"

(* Test suites *)
let col_tests = [
  "creation", `Quick, test_col_creation;
  "nulls", `Quick, test_col_nulls;
]

let creation_tests = [
  "basic", `Quick, test_df_creation;
  "empty", `Quick, test_df_empty;
]

let column_tests = [
  "access", `Quick, test_column_access;
  "add_drop", `Quick, test_column_add_drop;
  "rename", `Quick, test_rename_column;
  "select", `Quick, test_select;
]

let row_tests = [
  "head_tail", `Quick, test_head_tail;
  "slice", `Quick, test_slice;
  "filter", `Quick, test_filter;
  "filter_by", `Quick, test_filter_by;
  "drop_duplicates", `Quick, test_drop_duplicates;
]

let concat_tests = [
  "rows", `Quick, test_concat_rows;
  "columns", `Quick, test_concat_cols;
]

let row_module_tests = [
  "accessors", `Quick, test_row_accessors;
  "map", `Quick, test_row_map;
]

let sort_group_tests = [
  "sort", `Quick, test_sort;
  "group_by", `Quick, test_group_by;
]

let agg_tests = [
  "float", `Quick, test_agg_float;
  "int", `Quick, test_agg_int;
  "string", `Quick, test_agg_string;
  "bool", `Quick, test_agg_bool;
  "cumulative", `Quick, test_agg_cumulative;
  "nulls", `Quick, test_agg_nulls;
]

let conversion_tests = [
  "to_arrays", `Quick, test_to_arrays;
  "to_nx", `Quick, test_to_nx;
  "from_nx", `Quick, test_from_nx;
  "cast_column", `Quick, test_cast_column;
]

let edge_tests = [
  "empty_operations", `Quick, test_empty_operations;
  "single_row", `Quick, test_single_row;
]

(* Test wide operations with mapHomo *)
let test_map_list_product () =
  (* Create a dataframe with 6 int32 columns *)
  let df = create [
    ("A", Col.int32_list [2l; 3l; 4l]);
    ("B", Col.int32_list [1l; 2l; 3l]);
    ("C", Col.int32_list [3l; 1l; 2l]);
    ("D", Col.int32_list [1l; 1l; 1l]);
    ("E", Col.int32_list [2l; 2l; 2l]);
    ("F", Col.int32_list [1l; 3l; 1l]);
  ] in
  
  (* Compute product of all 6 columns using map_list *)
  let df2 = map_column df "allMul" Nx.int32
    Row.(map_list (int32s ["A";"B";"C";"D";"E";"F"])
           ~f:(List.fold_left Int32.mul 1l)) in
  
  (* Check the results *)
  match to_int32_array df2 "allMul" with
  | Some arr -> 
      (* Row 1: 2*1*3*1*2*1 = 12 *)
      (* Row 2: 3*2*1*1*2*3 = 36 *)
      (* Row 3: 4*3*2*1*2*1 = 48 *)
      check_bool "Product of 6 columns" true 
        (arr = [|12l; 36l; 48l|])
  | None -> Alcotest.fail "allMul column should exist"

let test_sequence_sum () =
  (* Create a dataframe with float columns *)
  let df = create [
    ("A", Col.float64_list [1.0; 2.0; 3.0]);
    ("B", Col.float64_list [2.0; 3.0; 4.0]);
    ("C", Col.float64_list [3.0; 4.0; 5.0]);
  ] in
  
  (* Sum all columns using sequence + map *)
  let df2 = map_column df "total" Nx.float64
    Row.(map (sequence (float64s ["A";"B";"C"]))
           ~f:(List.fold_left ( +. ) 0.)) in
  
  match to_float64_array df2 "total" with
  | Some arr -> 
      check_float "Row 1 sum" 6.0 arr.(0);
      check_float "Row 2 sum" 9.0 arr.(1);
      check_float "Row 3 sum" 12.0 arr.(2)
  | None -> Alcotest.fail "total column should exist"

let test_weighted_sum () =
  (* Test weighted sum example *)
  let df = create [
    ("A", Col.float64_list [1.0; 2.0; 3.0]);
    ("B", Col.float64_list [2.0; 3.0; 4.0]);
    ("C", Col.float64_list [3.0; 4.0; 5.0]);
    ("D", Col.float64_list [4.0; 5.0; 6.0]);
    ("E", Col.float64_list [5.0; 6.0; 7.0]);
    ("F", Col.float64_list [6.0; 7.0; 8.0]);
  ] in
  
  let feats = ["A";"B";"C";"D";"E";"F"] in
  let weights = [0.2; 0.3; 0.1; 0.1; 0.1; 0.2] in
  
  let df' = map_column df "score" Nx.float64
    Row.(map_list (float64s feats)
          ~f:(fun xs -> List.fold_left2 
                (fun acc wi xi -> acc +. wi *. xi) 0. weights xs)) in
  
  match to_float64_array df' "score" with
  | Some scores ->
      (* First row: 0.2*1 + 0.3*2 + 0.1*3 + 0.1*4 + 0.1*5 + 0.2*6 = 3.2 *)
      check_float "First weighted sum" 3.2 scores.(0);
      check_float "Second weighted sum" 4.2 scores.(1);
      check_float "Third weighted sum" 5.2 scores.(2)
  | None -> Alcotest.fail "score column should exist"

let test_numeric_column_names () =
  let df = create [
    ("name", Col.string_list ["Alice"; "Bob"; "Charlie"]);
    ("age", Col.int32_list [25l; 30l; 35l]);
    ("score", Col.float64_list [85.5; 92.0; 78.5]);
    ("active", Col.bool_list [true; false; true]);
    ("height", Col.float32_list [1.75; 1.80; 1.70]);
    ("id", Col.int64_list [1L; 2L; 3L]);
  ] in
  
  let numeric_cols = numeric_column_names df in
  let expected = ["age"; "score"; "height"; "id"] in
  
  (* Sort both lists for comparison since order might vary *)
  let sorted_numeric = List.sort String.compare numeric_cols in
  let sorted_expected = List.sort String.compare expected in
  
  Alcotest.(check (list string)) "Numeric column names" 
    sorted_expected sorted_numeric

let test_row_helpers () =
  let df = create [
    ("a", Col.int32_list [1l; 2l]);
    ("b", Col.int32_list [3l; 4l]);
    ("c", Col.float64_list [5.0; 6.0]);
  ] in
  
  (* Test int32s helper *)
  let int_cols = Row.int32s ["a"; "b"] in
  check_int "int32s creates 2 accessors" 2 (List.length int_cols);
  
  (* Test float64s helper *)
  let float_cols = Row.float64s ["c"] in
  check_int "float64s creates 1 accessor" 1 (List.length float_cols);
  
  (* Test using the helpers with all (alias for sequence) *)
  let df2 = map_column df "sum" Nx.float64
    Row.(map (all (float64s ["c"])) ~f:(fun xs -> 
      match xs with 
      | [x] -> x *. 2.0
      | _ -> failwith "Expected exactly one column")) in
  
  match to_float64_array df2 "sum" with
  | Some arr ->
      check_float "First doubled" 10.0 arr.(0);
      check_float "Second doubled" 12.0 arr.(1)
  | None -> Alcotest.fail "sum column should exist"

let test_sequence_all_equivalence () =
  (* Test that sequence and all are equivalent *)
  let df = create [
    ("x", Col.int32_list [1l; 2l]);
    ("y", Col.int32_list [10l; 20l]);
    ("z", Col.int32_list [100l; 200l]);
  ] in
  
  (* Using sequence *)
  let df_seq = map_column df "sum_seq" Nx.int32
    Row.(map (sequence (int32s ["x"; "y"; "z"]))
           ~f:(List.fold_left Int32.add 0l)) in
  
  (* Using all (should be identical) *)
  let df_all = map_column df "sum_all" Nx.int32
    Row.(map (all (int32s ["x"; "y"; "z"]))
           ~f:(List.fold_left Int32.add 0l)) in
  
  match to_int32_array df_seq "sum_seq", to_int32_array df_all "sum_all" with
  | Some seq_arr, Some all_arr ->
      check_bool "sequence and all produce same results" true
        (seq_arr = all_arr && seq_arr = [|111l; 222l|])
  | _ -> Alcotest.fail "columns should exist"

let wide_tests = [
  "map_list product", `Quick, test_map_list_product;
  "sequence sum", `Quick, test_sequence_sum;
  "weighted sum", `Quick, test_weighted_sum;
  "numeric_column_names", `Quick, test_numeric_column_names;
  "row helpers", `Quick, test_row_helpers;
  "sequence/all equivalence", `Quick, test_sequence_all_equivalence;
]

(* Test ergonomic APIs *)
let test_with_columns () =
  let df = create [
    ("x", Col.float64_list [1.0; 2.0; 3.0]);
    ("y", Col.float64_list [4.0; 5.0; 6.0]);
  ] in
  
  let df2 = with_columns df [
    ("z", Col.float64_list [7.0; 8.0; 9.0]);
    ("sum", Col.float64_list [5.0; 7.0; 9.0]);
  ] in
  
  check_int "columns added" 4 (num_columns df2);
  check_bool "has z" true (has_column df2 "z");
  check_bool "has sum" true (has_column df2 "sum")

let test_column_selectors () =
  let df = create [
    ("feat_1", Col.float64_list [1.0]);
    ("feat_2", Col.float64_list [2.0]);
    ("id", Col.int32_list [1l]);
    ("name", Col.string_list ["test"]);
    ("score_a", Col.float64_list [3.0]);
    ("score_b", Col.float64_list [4.0]);
  ] in
  
  (* Test columns_with_prefix *)
  let feat_cols = columns_with_prefix df "feat_" in
  Alcotest.(check (list string)) "feat_ prefix" ["feat_1"; "feat_2"] 
    (List.sort String.compare feat_cols);
  
  (* Test columns_with_suffix *)
  let score_cols = columns_with_suffix df "_a" in
  Alcotest.(check (list string)) "suffix _a" ["score_a"] score_cols;
  
  (* Test select_dtypes *)
  let numeric = select_dtypes df [`Numeric] in
  check_int "numeric columns" 5 (List.length numeric);
  
  let strings = select_dtypes df [`String] in
  Alcotest.(check (list string)) "string columns" ["name"] strings

let test_rowagg_sum () =
  let df = create [
    ("a", Col.float64_list [1.0; 2.0; Float.nan]);
    ("b", Col.float64_list [3.0; 4.0; 5.0]);
    ("c", Col.int32_list [5l; 6l; 7l]);
  ] in
  
  (* Test sum with skipna=true (default) *)
  let sum_col = Row_agg.sum df ~names:["a"; "b"; "c"] in
  let df2 = add_column df "row_sum" sum_col in
  
  match to_float64_array df2 "row_sum" with
  | Some arr ->
      check_float "Row 0 sum" 9.0 arr.(0);  (* 1 + 3 + 5 *)
      check_float "Row 1 sum" 12.0 arr.(1); (* 2 + 4 + 6 *)
      check_float "Row 2 sum" 12.0 arr.(2)  (* NaN + 5 + 7, NaN skipped *)
  | None -> Alcotest.fail "row_sum should exist"

let test_rowagg_dot () =
  let df = create [
    ("x", Col.float64_list [1.0; 2.0; 3.0]);
    ("y", Col.float64_list [4.0; 5.0; 6.0]);
    ("z", Col.float64_list [7.0; 8.0; 9.0]);
  ] in
  
  let weights = [|0.2; 0.3; 0.5|] in
  let score = Row_agg.dot df ~names:["x"; "y"; "z"] ~weights in
  let df2 = add_column df "score" score in
  
  match to_float64_array df2 "score" with
  | Some arr ->
      check_float "Row 0 weighted" 4.9 arr.(0);  (* 0.2*1 + 0.3*4 + 0.5*7 = 0.2 + 1.2 + 3.5 = 4.9 *)
      check_float "Row 1 weighted" 5.9 arr.(1);  (* 0.2*2 + 0.3*5 + 0.5*8 = 0.4 + 1.5 + 4.0 = 5.9 *)
      check_float "Row 2 weighted" 6.9 arr.(2)   (* 0.2*3 + 0.3*6 + 0.5*9 = 0.6 + 1.8 + 4.5 = 6.9 *)
  | None -> Alcotest.fail "score should exist"

let ergonomic_tests = [
  "with_columns", `Quick, test_with_columns;
  "column_selectors", `Quick, test_column_selectors;
  "Row_agg.sum", `Quick, test_rowagg_sum;
  "Row_agg.dot", `Quick, test_rowagg_dot;
]

let () =
  Alcotest.run "Talon" [
    "Col", col_tests;
    "Creation", creation_tests;
    "Columns", column_tests;
    "Rows", row_tests;
    "Concatenation", concat_tests;
    "Row module", row_module_tests;
    "Sort & Group", sort_group_tests;
    "Aggregations", agg_tests;
    "Conversions", conversion_tests;
    "Edge cases", edge_tests;
    "Wide operations", wide_tests;
    "Ergonomic APIs", ergonomic_tests;
  ]