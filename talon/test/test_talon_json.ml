(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
open Windtrap

let check_int msg = equal ~msg int
let check_string msg = equal ~msg string
let check_bool msg = equal ~msg bool

let json_of_string s =
  match Jsont_bytesrw.decode_string Jsont.json s with
  | Ok v -> v
  | Error e -> failwith e

let json_assoc = function
  | Jsont.Object (mems, _) -> List.map (fun ((n, _), v) -> (n, v)) mems
  | _ -> []

(* Test JSON records orientation *)
let test_to_string_records () =
  let df =
    create
      [
        ("name", Col.string_list [ "Alice"; "Bob" ]);
        ("age", Col.int32_list [ 25l; 30l ]);
        ("active", Col.bool_list [ true; false ]);
      ]
  in

  let json = Talon_json.to_string ~orient:`Records df in

  (* Parse back to verify structure *)
  let parsed = json_of_string json in
  match parsed with
  | Jsont.Array (records, _) -> (
      check_int "record count" 2 (List.length records);
      (* Check first record has expected fields *)
      match List.hd records with
      | Jsont.Object (mems, _) ->
          let fields = List.map (fun ((n, _), _) -> n) mems in
          check_bool "has name" true (List.mem "name" fields);
          check_bool "has age" true (List.mem "age" fields);
          check_bool "has active" true (List.mem "active" fields)
      | _ -> fail "Expected object in records")
  | _ -> fail "Expected array for records orientation"

let test_to_string_columns () =
  let df =
    create
      [
        ("x", Col.int32_list [ 1l; 2l; 3l ]);
        ("y", Col.float32_list [ 1.5; 2.5; 3.5 ]);
      ]
  in

  let json = Talon_json.to_string ~orient:`Columns df in

  let parsed = json_of_string json in
  match parsed with
  | Jsont.Object (mems, _) -> (
      let fields = List.map (fun ((n, _), _) -> n) mems in
      check_bool "has x column" true (List.mem "x" fields);
      check_bool "has y column" true (List.mem "y" fields);
      (* Check column is array *)
      let assoc = json_assoc parsed in
      match List.assoc "x" assoc with
      | Jsont.Array (vals, _) -> check_int "x length" 3 (List.length vals)
      | _ -> fail "Column should be array")
  | _ -> fail "Expected object for columns orientation"

let test_from_string_records () =
  let json =
    {|[
    {"name": "Alice", "age": 25, "score": 85.5},
    {"name": "Bob", "age": 30, "score": 92.0}
  ]|}
  in

  let df = Talon_json.from_string ~orient:`Records json in

  check_int "rows" 2 (num_rows df);
  check_int "cols" 3 (num_columns df);

  let names = column_names df in
  check_bool "has name" true (List.mem "name" names);
  check_bool "has age" true (List.mem "age" names);
  check_bool "has score" true (List.mem "score" names)

let test_from_string_columns () =
  let json =
    {|{
    "x": [1, 2, 3],
    "y": [4.5, 5.5, 6.5],
    "z": ["a", "b", "c"]
  }|}
  in

  let df = Talon_json.from_string ~orient:`Columns json in

  check_int "rows" 3 (num_rows df);
  check_int "cols" 3 (num_columns df);

  (* Check string column *)
  match to_string_array df "z" with
  | Some arr -> check_bool "string values" true (arr = [| "a"; "b"; "c" |])
  | None -> fail "z should be string column"

let test_with_nulls () =
  let df =
    create
      [
        ("values", Col.float32_opt [| Some 1.0; None; Some 3.0 |]);
        ("labels", Col.string_opt [| Some "a"; None; Some "c" |]);
      ]
  in

  let json = Talon_json.to_string ~orient:`Records df in

  (* Check that nulls are represented as JSON null *)
  check_bool "has null" true (String.contains json 'n')

let test_int_null_masks_are_serialized () =
  let df = create [ ("ints", Col.int32_opt [| Some 1l; None |]) ] in
  let json = Talon_json.to_string ~orient:`Records df in
  match json_of_string json with
  | Jsont.Array ([ first; second ], _) -> (
      let first_fields = json_assoc first in
      let second_fields = json_assoc second in
      match (List.assoc "ints" first_fields, List.assoc "ints" second_fields)
      with
      | Jsont.Number (f, _), Jsont.Null _ when int_of_float f = 1 -> ()
      | _ -> fail "expected null for masked integer value")
  | _ -> fail "unexpected JSON structure for int mask test"

let test_round_trip_records () =
  let df1 =
    create
      [
        ("id", Col.int32_list [ 1l; 2l; 3l ]);
        ("value", Col.float32_list [ 1.5; 2.5; 3.5 ]);
        ("flag", Col.bool_list [ true; false; true ]);
        ("label", Col.string_list [ "a"; "b"; "c" ]);
      ]
  in

  let json = Talon_json.to_string ~orient:`Records df1 in
  let df2 = Talon_json.from_string ~orient:`Records json in

  check_int "same rows" (num_rows df1) (num_rows df2);
  check_int "same cols" (num_columns df1) (num_columns df2)

let test_round_trip_columns () =
  let df1 =
    create
      [
        ("x", Col.int32_list [ 10l; 20l ]); ("y", Col.float32_list [ 1.1; 2.2 ]);
      ]
  in

  let json = Talon_json.to_string ~orient:`Columns df1 in
  let df2 = Talon_json.from_string ~orient:`Columns json in

  check_int "same rows" (num_rows df1) (num_rows df2);
  check_int "same cols" (num_columns df1) (num_columns df2)

let test_empty_dataframe () =
  let df = empty in

  let json_records = Talon_json.to_string ~orient:`Records df in
  let json_columns = Talon_json.to_string ~orient:`Columns df in

  check_string "empty records" "[]" json_records;
  check_string "empty columns" "{}" json_columns

let test_single_row () =
  let df = create [ ("x", Col.int32_list [ 42l ]) ] in

  let json = Talon_json.to_string ~orient:`Records df in
  let parsed = json_of_string json in

  match parsed with
  | Jsont.Array ([ record ], _) -> (
      match record with
      | Jsont.Object (mems, _) -> (
          let fields = List.map (fun ((n, _), v) -> (n, v)) mems in
          match List.assoc "x" fields with
          | Jsont.Number (f, _) -> check_int "single value" 42 (int_of_float f)
          | _ -> fail "Expected int")
      | _ -> fail "Expected object")
  | _ -> fail "Expected single record"

let test_complex_types () =
  let df =
    create
      [
        ( "complex",
          Col.of_tensor
            (Nx.create Nx.complex64 [| 2 |]
               [|
                 { Complex.re = 1.0; im = 2.0 }; { Complex.re = 3.0; im = 4.0 };
               |]) );
      ]
  in

  let json = Talon_json.to_string ~orient:`Records df in

  (* Complex numbers should be serialized as strings *)
  check_bool "has complex string" true (String.contains json '+')

let test_large_ints () =
  let df =
    create [ ("big_int", Col.int64_list [ Int64.max_int; Int64.min_int ]) ]
  in

  let json = Talon_json.to_string ~orient:`Records df in

  (* Large int64 values should be preserved as strings *)
  check_bool "has large int" true (String.length json > 20)

let test_bool_parsing () =
  let json =
    {|[
    {"flag": true},
    {"flag": false},
    {"flag": true}
  ]|}
  in

  let df = Talon_json.from_string ~orient:`Records json in

  match to_bool_array df "flag" with
  | Some arr -> check_bool "bool values" true (arr = [| true; false; true |])
  | None -> fail "flag should be bool column"

let test_mixed_types () =
  (* Test that mixed types default to string *)
  let json = {|{
    "mixed": [1, "two", 3.0, true]
  }|} in

  let df = Talon_json.from_string ~orient:`Columns json in

  match to_string_array df "mixed" with
  | Some arr ->
      check_int "mixed length" 4 (Array.length arr);
      check_string "int as string" "1" arr.(0);
      check_string "string" "two" arr.(1)
  | None -> fail "mixed should be string column"

let test_force_all_null_to_float64 () =
  let json = {| [{"a": null}, {"a": null}] |} in
  let df = Talon_json.from_string ~dtype_spec:[ ("a", `Float64) ] json in

  check_int "rows" 2 (num_rows df);

  let col = get_column_exn df "a" in
  check_bool "has nulls" true (Col.has_nulls col);
  check_int "null count" 2 (Col.null_count col)

let test_force_string_id_to_int64 () =
  let json = {| [{"id": "123456789123456"}, {"id": "42"}] |} in
  let dtype_spec = [ ("id", `Int64) ] in
  let df = Talon_json.from_string ~dtype_spec json in
  check_int "is int" 2 (num_rows df);
  check_bool "int values" true
    (to_int64_array df "id" = Some [| 123456789123456L; 42L |])

let dtype_spec_tests =
  [
    test "Force all-null to Float64" test_force_all_null_to_float64;
    test "Force string ID to Int64" test_force_string_id_to_int64;
  ]

(* Test suites *)
let serialization_tests =
  [
    test "to_string_records" test_to_string_records;
    test "to_string_columns" test_to_string_columns;
    test "with_nulls" test_with_nulls;
    test "int null masks" test_int_null_masks_are_serialized;
    test "empty_dataframe" test_empty_dataframe;
    test "single_row" test_single_row;
    test "complex_types" test_complex_types;
    test "large_ints" test_large_ints;
  ]

let deserialization_tests =
  [
    test "from_string_records" test_from_string_records;
    test "from_string_columns" test_from_string_columns;
    test "bool_parsing" test_bool_parsing;
    test "mixed_types" test_mixed_types;
  ]

let integration_tests =
  [
    test "round_trip_records" test_round_trip_records;
    test "round_trip_columns" test_round_trip_columns;
  ]

let () =
  run "Talon JSON"
    [
      group "Dtype Spec" dtype_spec_tests;
      group "Serialization" serialization_tests;
      group "Deserialization" deserialization_tests;
      group "Integration" integration_tests;
    ]
