open Fehu
open Windtrap

let value = testable ~pp:Value.pp ~equal:Value.equal ()

(* Equality *)

let test_null_equal () = equal ~msg:"null = null" value Value.Null Value.Null

let test_bool_equal () =
  equal ~msg:"true = true" value (Bool true) (Bool true);
  not_equal ~msg:"true <> false" value (Bool true) (Bool false)

let test_int_equal () =
  equal ~msg:"1 = 1" value (Int 1) (Int 1);
  not_equal ~msg:"1 <> 2" value (Int 1) (Int 2)

let test_float_equal () = equal ~msg:"1.0 = 1.0" value (Float 1.0) (Float 1.0)

let test_string_equal () =
  equal ~msg:"a = a" value (String "a") (String "a");
  not_equal ~msg:"a <> b" value (String "a") (String "b")

let test_int_array_equal () =
  equal ~msg:"[|1;2|] = [|1;2|]" value
    (Int_array [| 1; 2 |])
    (Int_array [| 1; 2 |])

let test_float_array_equal () =
  equal ~msg:"[|1.0|] = [|1.0|]" value (Float_array [| 1.0 |])
    (Float_array [| 1.0 |])

let test_bool_array_equal () =
  equal ~msg:"[|true|] = [|true|]" value (Bool_array [| true |])
    (Bool_array [| true |])

let test_list_equal () =
  equal ~msg:"[Int 1] = [Int 1]" value (List [ Int 1 ]) (List [ Int 1 ])

let test_dict_equal () =
  equal ~msg:"dict equal" value (Dict [ ("k", Int 1) ]) (Dict [ ("k", Int 1) ])

let test_cross_type_inequality () =
  not_equal ~msg:"Int 1 <> Float 1.0" value (Int 1) (Float 1.0);
  not_equal ~msg:"Null <> Int 0" value Null (Int 0)

(* Formatting *)

let test_to_string_null () =
  equal ~msg:"null" string "null" (Value.to_string Null)

let test_to_string_bool () =
  equal ~msg:"bool true" string "true" (Value.to_string (Bool true))

let test_to_string_int () =
  equal ~msg:"int 42" string "42" (Value.to_string (Int 42))

let test_to_string_float () =
  let s = Value.to_string (Float 3.14) in
  is_true ~msg:"float non-empty" (String.length s > 0)

let test_to_string_string () =
  let s = Value.to_string (String "hello") in
  is_true ~msg:"string non-empty" (String.length s > 0)

let test_to_string_arrays () =
  is_true ~msg:"int_array non-empty"
    (String.length (Value.to_string (Int_array [| 1 |])) > 0);
  is_true ~msg:"float_array non-empty"
    (String.length (Value.to_string (Float_array [| 1.0 |])) > 0);
  is_true ~msg:"bool_array non-empty"
    (String.length (Value.to_string (Bool_array [| true |])) > 0)

let test_to_string_list () =
  let s = Value.to_string (List [ Int 1; Int 2 ]) in
  is_true ~msg:"list non-empty" (String.length s > 0)

let test_to_string_dict () =
  let s = Value.to_string (Dict [ ("k", Int 1) ]) in
  is_true ~msg:"dict non-empty" (String.length s > 0)

let () =
  run "Fehu.Value"
    [
      group "equality"
        [
          test "null" test_null_equal;
          test "bool" test_bool_equal;
          test "int" test_int_equal;
          test "float" test_float_equal;
          test "string" test_string_equal;
          test "int_array" test_int_array_equal;
          test "float_array" test_float_array_equal;
          test "bool_array" test_bool_array_equal;
          test "list" test_list_equal;
          test "dict" test_dict_equal;
          test "cross-type inequality" test_cross_type_inequality;
        ];
      group "formatting"
        [
          test "to_string null" test_to_string_null;
          test "to_string bool" test_to_string_bool;
          test "to_string int" test_to_string_int;
          test "to_string float" test_to_string_float;
          test "to_string string" test_to_string_string;
          test "to_string arrays" test_to_string_arrays;
          test "to_string list" test_to_string_list;
          test "to_string dict" test_to_string_dict;
        ];
    ]
