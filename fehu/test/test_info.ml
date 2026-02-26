open Fehu
open Windtrap

let value = testable ~pp:Value.pp ~equal:Value.equal ()

(* Operations *)

let test_empty_is_empty () =
  equal ~msg:"empty is_empty" bool true (Info.is_empty Info.empty)

let test_set_then_find () =
  let info = Info.set "k" (Value.Int 1) Info.empty in
  equal ~msg:"find after set" (option value) (Some (Value.Int 1))
    (Info.find "k" info)

let test_find_missing () =
  let info = Info.set "k" (Value.Int 1) Info.empty in
  equal ~msg:"find missing" (option value) None (Info.find "other" info)

let test_find_exn_existing () =
  let info = Info.set "k" (Value.Int 42) Info.empty in
  equal ~msg:"find_exn existing" value (Value.Int 42) (Info.find_exn "k" info)

let test_remove () =
  let info = Info.set "k" (Value.Int 1) Info.empty in
  let info = Info.remove "k" info in
  equal ~msg:"find after remove" (option value) None (Info.find "k" info)

let test_merge_right_biased () =
  let a = Info.set "k" (Value.Int 1) Info.empty in
  let b = Info.set "k" (Value.Int 2) Info.empty in
  let merged = Info.merge a b in
  equal ~msg:"merge right wins" value (Value.Int 2) (Info.find_exn "k" merged)

let test_of_list_to_list_round_trip () =
  let kvs = [ ("a", Value.Int 1); ("c", Value.Int 3); ("b", Value.Int 2) ] in
  let info = Info.of_list kvs in
  let result = Info.to_list info in
  equal ~msg:"round-trip keys sorted" (list string) [ "a"; "b"; "c" ]
    (List.map fst result);
  equal ~msg:"round-trip values" (list value)
    [ Value.Int 1; Value.Int 2; Value.Int 3 ]
    (List.map snd result)

(* Errors *)

let test_find_exn_missing () =
  raises_invalid_arg "Info.find_exn: key \"missing\" not present" (fun () ->
      ignore (Info.find_exn "missing" Info.empty))

(* Convenience *)

let test_int_convenience () =
  equal ~msg:"Info.int" value (Value.Int 42) (Info.int 42)

let test_float_convenience () =
  equal ~msg:"Info.float" value (Value.Float 1.0) (Info.float 1.0)

let test_bool_convenience () =
  equal ~msg:"Info.bool" value (Value.Bool true) (Info.bool true)

let test_string_convenience () =
  equal ~msg:"Info.string" value (Value.String "hi") (Info.string "hi")

let test_null_convenience () = equal ~msg:"Info.null" value Value.Null Info.null

let () =
  run "Fehu.Info"
    [
      group "operations"
        [
          test "empty is_empty" test_empty_is_empty;
          test "set then find" test_set_then_find;
          test "find missing key" test_find_missing;
          test "find_exn existing" test_find_exn_existing;
          test "remove" test_remove;
          test "merge right-biased" test_merge_right_biased;
          test "of_list/to_list round-trip" test_of_list_to_list_round_trip;
        ];
      group "errors" [ test "find_exn missing raises" test_find_exn_missing ];
      group "convenience"
        [
          test "int" test_int_convenience;
          test "float" test_float_convenience;
          test "bool" test_bool_convenience;
          test "string" test_string_convenience;
          test "null" test_null_convenience;
        ];
    ]
