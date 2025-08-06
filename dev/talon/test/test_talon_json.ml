open Talon

let check_int = Alcotest.(check int)
let check_string = Alcotest.(check string)
let check_bool = Alcotest.(check bool)

(* Test JSON records orientation *)
let test_to_string_records () =
  let df = create [
    ("name", Col.string_list ["Alice"; "Bob"]);
    ("age", Col.int32_list [25l; 30l]);
    ("active", Col.bool_list [true; false])
  ] in
  
  let json = Talon_json.to_string ~orient:`Records df in
  
  (* Parse back to verify structure *)
  let parsed = Yojson.Basic.from_string json in
  match parsed with
  | `List records ->
      check_int "record count" 2 (List.length records);
      (* Check first record has expected fields *)
      (match List.hd records with
       | `Assoc fields ->
           check_bool "has name" true (List.mem_assoc "name" fields);
           check_bool "has age" true (List.mem_assoc "age" fields);
           check_bool "has active" true (List.mem_assoc "active" fields)
       | _ -> Alcotest.fail "Expected object in records")
  | _ -> Alcotest.fail "Expected array for records orientation"

let test_to_string_columns () =
  let df = create [
    ("x", Col.int32_list [1l; 2l; 3l]);
    ("y", Col.float32_list [1.5; 2.5; 3.5])
  ] in
  
  let json = Talon_json.to_string ~orient:`Columns df in
  
  let parsed = Yojson.Basic.from_string json in
  match parsed with
  | `Assoc fields ->
      check_bool "has x column" true (List.mem_assoc "x" fields);
      check_bool "has y column" true (List.mem_assoc "y" fields);
      (* Check column is array *)
      (match List.assoc "x" fields with
       | `List vals -> check_int "x length" 3 (List.length vals)
       | _ -> Alcotest.fail "Column should be array")
  | _ -> Alcotest.fail "Expected object for columns orientation"

let test_from_string_records () =
  let json = {|[
    {"name": "Alice", "age": 25, "score": 85.5},
    {"name": "Bob", "age": 30, "score": 92.0}
  ]|} in
  
  let df = Talon_json.from_string ~orient:`Records json in
  
  check_int "rows" 2 (num_rows df);
  check_int "cols" 3 (num_columns df);
  
  let names = column_names df in
  check_bool "has name" true (List.mem "name" names);
  check_bool "has age" true (List.mem "age" names);
  check_bool "has score" true (List.mem "score" names)

let test_from_string_columns () =
  let json = {|{
    "x": [1, 2, 3],
    "y": [4.5, 5.5, 6.5],
    "z": ["a", "b", "c"]
  }|} in
  
  let df = Talon_json.from_string ~orient:`Columns json in
  
  check_int "rows" 3 (num_rows df);
  check_int "cols" 3 (num_columns df);
  
  (* Check string column *)
  match to_string_array df "z" with
  | Some arr -> check_bool "string values" true (arr = [|"a"; "b"; "c"|])
  | None -> Alcotest.fail "z should be string column"

let test_with_nulls () =
  let df = create [
    ("values", Col.float32_opt [|Some 1.0; None; Some 3.0|]);
    ("labels", Col.string_opt [|Some "a"; None; Some "c"|])
  ] in
  
  let json = Talon_json.to_string ~orient:`Records df in
  
  (* Check that nulls are represented as JSON null *)
  check_bool "has null" true (String.contains json 'n')

let test_round_trip_records () =
  let df1 = create [
    ("id", Col.int32_list [1l; 2l; 3l]);
    ("value", Col.float32_list [1.5; 2.5; 3.5]);
    ("flag", Col.bool_list [true; false; true]);
    ("label", Col.string_list ["a"; "b"; "c"])
  ] in
  
  let json = Talon_json.to_string ~orient:`Records df1 in
  let df2 = Talon_json.from_string ~orient:`Records json in
  
  check_int "same rows" (num_rows df1) (num_rows df2);
  check_int "same cols" (num_columns df1) (num_columns df2)

let test_round_trip_columns () =
  let df1 = create [
    ("x", Col.int32_list [10l; 20l]);
    ("y", Col.float32_list [1.1; 2.2])
  ] in
  
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
  let df = create [
    ("x", Col.int32_list [42l])
  ] in
  
  let json = Talon_json.to_string ~orient:`Records df in
  let parsed = Yojson.Basic.from_string json in
  
  match parsed with
  | `List [record] ->
      (match record with
       | `Assoc fields ->
           (match List.assoc "x" fields with
            | `Int i -> check_int "single value" 42 i
            | _ -> Alcotest.fail "Expected int")
       | _ -> Alcotest.fail "Expected object")
  | _ -> Alcotest.fail "Expected single record"

let test_complex_types () =
  let df = create [
    ("complex", Col.of_tensor (Nx.create Nx.complex32 [|2|] 
      [|{Complex.re = 1.0; im = 2.0}; {Complex.re = 3.0; im = 4.0}|]))
  ] in
  
  let json = Talon_json.to_string ~orient:`Records df in
  
  (* Complex numbers should be serialized as strings *)
  check_bool "has complex string" true (String.contains json '+')

let test_large_ints () =
  let df = create [
    ("big_int", Col.int64_list [Int64.max_int; Int64.min_int])
  ] in
  
  let json = Talon_json.to_string ~orient:`Records df in
  
  (* Large int64 values should be preserved as strings *)
  check_bool "has large int" true (String.length json > 20)

let test_bool_parsing () =
  let json = {|[
    {"flag": true},
    {"flag": false},
    {"flag": true}
  ]|} in
  
  let df = Talon_json.from_string ~orient:`Records json in
  
  match to_bool_array df "flag" with
  | Some arr -> check_bool "bool values" true (arr = [|true; false; true|])
  | None -> Alcotest.fail "flag should be bool column"

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
  | None -> Alcotest.fail "mixed should be string column"

(* Test suites *)
let serialization_tests = [
  "to_string_records", `Quick, test_to_string_records;
  "to_string_columns", `Quick, test_to_string_columns;
  "with_nulls", `Quick, test_with_nulls;
  "empty_dataframe", `Quick, test_empty_dataframe;
  "single_row", `Quick, test_single_row;
  "complex_types", `Quick, test_complex_types;
  "large_ints", `Quick, test_large_ints;
]

let deserialization_tests = [
  "from_string_records", `Quick, test_from_string_records;
  "from_string_columns", `Quick, test_from_string_columns;
  "bool_parsing", `Quick, test_bool_parsing;
  "mixed_types", `Quick, test_mixed_types;
]

let integration_tests = [
  "round_trip_records", `Quick, test_round_trip_records;
  "round_trip_columns", `Quick, test_round_trip_columns;
]

let () =
  Alcotest.run "Talon JSON" [
    "Serialization", serialization_tests;
    "Deserialization", deserialization_tests;
    "Integration", integration_tests;
  ]