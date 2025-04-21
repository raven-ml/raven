open Alcotest

let string_contains needle haystack =
  let open String in
  let rec loop i =
    if i + length needle > length haystack then false
    else if sub haystack i (length needle) = needle then true
    else loop (i + 1)
  in
  loop 0

let check_status expected actual =
  match actual with
  | `Success when expected = `Success -> ()
  | `Error when expected = `Error -> ()
  | _ ->
      fail
        (Printf.sprintf "Expected status %s but got %s"
           (match expected with `Success -> "Success" | `Error -> "Error")
           (match actual with `Success -> "Success" | `Error -> "Error"))

let test_eval_success () =
  let result = Quill_top.eval ~id:"test1" "2" in
  check string "output" "2" (String.trim result.output);
  check (option string) "error" None result.error;
  check_status `Success result.status

let test_eval_print () =
  let result = Quill_top.eval ~id:"test2" "print_endline \"hello\"" in
  check string "output" "hello" (String.trim result.output);
  check (option string) "error" None result.error;
  check_status `Success result.status

let test_eval_type_error () =
  let result = Quill_top.eval ~id:"test3" "1 + true" in
  check string "output" "" (String.trim result.output);
  check (option string) "error"
    (Some
       "Line 1, characters 4-8:\n\
        Error: The constructor true has type bool\n\
       \       but an expression was expected of type int\n")
    result.error;
  check_status `Error result.status

let test_eval_runtime_error () =
  let result = Quill_top.eval ~id:"test4" "failwith \"error\"" in
  check string "output" "" (String.trim result.output);
  check (option string) "error" (Some "Failure(\"error\")") result.error;
  check_status `Error result.status

let test_isolation () =
  let _ = Quill_top.eval ~id:"test5" "let x = 1" in
  let result1 = Quill_top.eval ~id:"test5" "x" in
  check string "output" "1" (String.trim result1.output);
  let result2 = Quill_top.eval ~id:"test6" "x" in
  (match result2.error with
  | Some err ->
      check bool "error contains 'Unbound'" true (string_contains "Unbound" err)
  | None -> fail "Expected an error message");
  check_status `Error result2.status

let test_multiple_evals () =
  let _ = Quill_top.eval ~id:"test7" "let x = 1" in
  let _ = Quill_top.eval ~id:"test7" "let y = x + 1" in
  let result = Quill_top.eval ~id:"test7" "y" in
  check string "output" "2" (String.trim result.output);
  check (option string) "error" None result.error;
  check_status `Success result.status

let test_non_interference () =
  let _ = Quill_top.eval ~id:"test8" "let x = 1" in
  let _ = Quill_top.eval ~id:"test9" "let x = 2" in
  let result = Quill_top.eval ~id:"test8" "x" in
  check string "output" "1" (String.trim result.output);
  check (option string) "error" None result.error;
  check_status `Success result.status

let suite =
  [
    ("eval success", `Quick, test_eval_success);
    ("eval print", `Quick, test_eval_print);
    ("eval type error", `Quick, test_eval_type_error);
    ("eval runtime error", `Quick, test_eval_runtime_error);
    ("isolation", `Quick, test_isolation);
    ("multiple evals", `Quick, test_multiple_evals);
    ("non-interference", `Quick, test_non_interference);
  ]

let () = run "Quill_top tests" [ ("tests", suite) ]
