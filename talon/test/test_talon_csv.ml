(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon

let check_int = Alcotest.(check int)
let check_bool = Alcotest.(check bool)

(* Test reading CSV from string *)
let test_from_string_basic () =
  let csv = "name,age,score\nAlice,25,85.5\nBob,30,92.0\nCharlie,35,78.5" in
  let df = Talon_csv.from_string csv in

  check_int "rows" 3 (num_rows df);
  check_int "cols" 3 (num_columns df);

  let names = column_names df in
  Alcotest.(check (list string)) "column names" [ "name"; "age"; "score" ] names

let test_from_string_no_header () =
  let csv = "Alice,25,85.5\nBob,30,92.0" in
  let df = Talon_csv.from_string ~header:false csv in

  check_int "rows" 2 (num_rows df);
  check_int "cols" 3 (num_columns df);

  let names = column_names df in
  Alcotest.(check (list string))
    "default names" [ "col0"; "col1"; "col2" ] names

let test_from_string_custom_sep () =
  let csv = "name;age\nAlice;25\nBob;30" in
  let df = Talon_csv.from_string ~sep:';' csv in

  check_int "rows" 2 (num_rows df);
  check_int "cols" 2 (num_columns df)

let test_from_string_with_nulls () =
  let csv = "name,value\nAlice,1.5\nBob,NA\nCharlie,2.5" in
  let df = Talon_csv.from_string csv in

  check_int "rows" 3 (num_rows df);

  let col = get_column_exn df "value" in
  check_bool "has nulls" true (Col.has_nulls col);
  check_int "null count" 1 (Col.null_count col)

let test_from_string_dtype_spec () =
  let csv = "id,flag\n1,true\n2,false\n3,true" in
  let dtype_spec = [ ("id", `Int32); ("flag", `Bool) ] in
  let df = Talon_csv.from_string ~dtype_spec csv in

  check_int "rows" 3 (num_rows df);

  (* Check that boolean column was parsed correctly *)
  match to_bool_array df "flag" with
  | Some arr -> check_bool "bool values" true (arr = [| true; false; true |])
  | None -> Alcotest.fail "flag column should be bool"

let test_from_string_empty () =
  let csv = "col1,col2,col3" in
  let df = Talon_csv.from_string csv in
  check_int "empty df rows" 0 (num_rows df);
  check_int "empty df cols" 3 (num_columns df)

(* Test writing CSV to string *)
let test_to_string_basic () =
  let df =
    create
      [
        ("name", Col.string_list [ "Alice"; "Bob" ]);
        ("age", Col.int32_list [ 25l; 30l ]);
      ]
  in

  let csv = Talon_csv.to_string df in

  (* Check header is present *)
  check_bool "has header" true (String.contains csv 'n');
  check_bool "has name" true (String.contains csv 'A')

let test_to_string_no_header () =
  let df =
    create
      [ ("name", Col.string_list [ "Alice" ]); ("age", Col.int32_list [ 25l ]) ]
  in

  let csv = Talon_csv.to_string ~header:false df in

  (* Should not contain column names *)
  check_bool "no name header" false (String.starts_with ~prefix:"name" csv);
  check_bool "has data" true (String.contains csv 'A')

let test_to_string_custom_sep () =
  let df =
    create
      [ ("a", Col.int32_list [ 1l; 2l ]); ("b", Col.int32_list [ 3l; 4l ]) ]
  in

  let csv = Talon_csv.to_string ~sep:';' df in
  check_bool "has semicolon" true (String.contains csv ';');
  check_bool "no comma" false (String.contains csv ',')

let test_to_string_with_nulls () =
  let df =
    create [ ("values", Col.float32_opt [| Some 1.0; None; Some 3.0 |]) ]
  in

  let csv = Talon_csv.to_string ~na_repr:"NULL" df in
  check_bool "has NULL" true (String.contains csv 'N')

(* Test round-trip *)
let test_round_trip () =
  let df1 =
    create
      [
        ("id", Col.int32_list [ 1l; 2l; 3l ]);
        ("value", Col.float32_list [ 1.5; 2.5; 3.5 ]);
        ("label", Col.string_list [ "a"; "b"; "c" ]);
      ]
  in

  let csv = Talon_csv.to_string df1 in
  let df2 = Talon_csv.from_string csv in

  check_int "same rows" (num_rows df1) (num_rows df2);
  check_int "same cols" (num_columns df1) (num_columns df2);

  let names1 = column_names df1 in
  let names2 = column_names df2 in
  Alcotest.(check (list string)) "same names" names1 names2

let test_auto_detect_dtypes () =
  (* Test that type detection works correctly *)
  let csv =
    "int_col,float_col,bool_col,str_col\n\
     42,3.14,true,hello\n\
     100,2.71,false,world"
  in
  let df = Talon_csv.from_string csv in

  check_int "rows" 2 (num_rows df);

  (* Check int column *)
  match to_int32_array df "int_col" with
  | Some arr -> check_bool "int values" true (arr = [| 42l; 100l |])
  | None -> Alcotest.fail "int_col should be int32"

let test_mixed_nulls () =
  let csv = "a,b,c\n1,2.5,foo\n,NA,\n3,4.5,bar" in
  let df = Talon_csv.from_string csv in

  check_int "rows" 3 (num_rows df);

  let col_a = get_column_exn df "a" in
  let col_b = get_column_exn df "b" in
  let col_c = get_column_exn df "c" in

  check_bool "col a has nulls" true (Col.has_nulls col_a);
  check_bool "col b has nulls" true (Col.has_nulls col_b);
  check_bool "col c has nulls" true (Col.has_nulls col_c)

let test_big_int_detection () =
  (* Test that large int are detected as Int64 *)
  let csv = "id\n9223372036854775806" in
  let df = Talon_csv.from_string csv in

  check_int "rows" 1 (num_rows df);

  (* Check that the column is detected as Int64 *)
  let col = get_column_exn df "id" in
  let is_int64 = match col with Col.P (Nx.Int64, _, _) -> true | _ -> false in
  check_bool "big int detected as Int64" true is_int64;

  match to_int64_array df "id" with
  | Some arr ->
      check_int "array length" 1 (Array.length arr);
      check_bool "correct value" true (arr.(0) = 9223372036854775806L)
  | None -> Alcotest.fail "to_int64_array should return Some for Int64 column"

(* Test suites *)
let reading_tests =
  [
    ("basic", `Quick, test_from_string_basic);
    ("no_header", `Quick, test_from_string_no_header);
    ("custom_sep", `Quick, test_from_string_custom_sep);
    ("with_nulls", `Quick, test_from_string_with_nulls);
    ("dtype_spec", `Quick, test_from_string_dtype_spec);
    ("empty", `Quick, test_from_string_empty);
    ("auto_detect", `Quick, test_auto_detect_dtypes);
    ("mixed_nulls", `Quick, test_mixed_nulls);
    ("big_int_detection", `Quick, test_big_int_detection);
  ]

let writing_tests =
  [
    ("basic", `Quick, test_to_string_basic);
    ("no_header", `Quick, test_to_string_no_header);
    ("custom_sep", `Quick, test_to_string_custom_sep);
    ("with_nulls", `Quick, test_to_string_with_nulls);
  ]

let integration_tests = [ ("round_trip", `Quick, test_round_trip) ]

let () =
  Alcotest.run "Talon CSV"
    [
      ("Reading", reading_tests);
      ("Writing", writing_tests);
      ("Integration", integration_tests);
    ]
