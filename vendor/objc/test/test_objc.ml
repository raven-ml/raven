(* Helper for checking if a class is the one we expect by name *)
let check_class_name msg expected_name class_obj =
  Alcotest.(check string) msg expected_name (Objc.get_class_name class_obj)

(* Helper for checking if a selector is the one we expect by name *)
let check_selector_name msg expected_name selector_obj =
  Alcotest.(check string)
    msg expected_name
    (Objc.get_selector_name selector_obj)

(* Class Tests *)

let test_get_known_class () =
  let ns_object_class = Objc.get_class "NSObject" in
  Alcotest.(check bool)
    "NSObject class is not null" true
    (not (Objc.Class.is_null ns_object_class));
  check_class_name "NSObject class name verification" "NSObject" ns_object_class

let test_get_another_known_class () =
  let ns_string_class = Objc.get_class "NSString" in
  Alcotest.(check bool)
    "NSString class is not null" true
    (not (Objc.Class.is_null ns_string_class));
  check_class_name "NSString class name verification" "NSString" ns_string_class

let test_get_non_existent_class () =
  let non_existent_class = Objc.get_class "ThisClassDoesNotExistAndNeverWill" in
  Alcotest.(check bool)
    "Non-existent class is null" true
    (Objc.Class.is_null non_existent_class)

let test_get_class_name_on_null_class () =
  Alcotest.check_raises "get_class_name on Class.null raises Failure"
    (Failure "get_class_name: class is null") (fun () ->
      ignore (Objc.get_class_name Objc.Class.null))

let test_get_class_with_empty_name () =
  Alcotest.check_raises "get_class with empty name raises Invalid_argument"
    (Invalid_argument "get_class: class name cannot be empty") (fun () ->
      ignore (Objc.get_class ""))

(* Selector Tests *)

let test_register_and_get_selector () =
  let description_sel = Objc.register_selector "description" in
  Alcotest.(check bool)
    "description selector is not null" true
    (not (Objc.Sel.is_null description_sel));
  check_selector_name "description selector name verification" "description"
    description_sel;
  (* Registering the same selector again should return the same SEL *)
  let description_sel_again = Objc.register_selector "description" in
  Alcotest.(check bool)
    "description selector (again) is not null" true
    (not (Objc.Sel.is_null description_sel_again));
  (* While the pointers should be identical, a name check is sufficient for
     functionality. Direct pointer equality for SELs is usually true but not
     strictly guaranteed by all runtimes to be a new pointer vs a cached one,
     though name mapping is unique. *)
  check_selector_name "description selector name (again)" "description"
    description_sel_again

let test_register_another_selector () =
  let alloc_sel = Objc.register_selector "alloc" in
  Alcotest.(check bool)
    "alloc selector is not null" true
    (not (Objc.Sel.is_null alloc_sel));
  check_selector_name "alloc selector name verification" "alloc" alloc_sel

let test_get_selector_name_on_null_selector () =
  Alcotest.check_raises "get_selector_name on Sel.null raises Failure"
    (Failure "get_selector_name: selector is null") (fun () ->
      ignore (Objc.get_selector_name Objc.Sel.null))

let test_register_selector_with_empty_name () =
  Alcotest.check_raises
    "register_selector with empty name raises Invalid_argument"
    (Invalid_argument "register_selector: selector name cannot be empty")
    (fun () -> ignore (Objc.register_selector ""))

(* Object Tests (Limited for Part 1) *)

let test_get_object_class_on_nil_object () =
  let class_of_nil = Objc.get_object_class Objc.Id.null in
  (* object_getClass(nil) returns Nil (which is a null Class pointer) *)
  Alcotest.(check bool)
    "Class of Id.null (nil object) is Class.null" true
    (Objc.Class.is_null class_of_nil)

(* Test Suite Setup *)

let class_tests =
  [
    Alcotest.test_case "Get known class (NSObject)" `Quick test_get_known_class;
    Alcotest.test_case "Get another known class (NSString)" `Quick
      test_get_another_known_class;
    Alcotest.test_case "Get non-existent class" `Quick
      test_get_non_existent_class;
    Alcotest.test_case "get_class_name on Class.null" `Quick
      test_get_class_name_on_null_class;
    Alcotest.test_case "get_class with empty name" `Quick
      test_get_class_with_empty_name;
  ]

let selector_tests =
  [
    Alcotest.test_case "Register and get selector (description)" `Quick
      test_register_and_get_selector;
    Alcotest.test_case "Register another selector (alloc)" `Quick
      test_register_another_selector;
    Alcotest.test_case "get_selector_name on Sel.null" `Quick
      test_get_selector_name_on_null_selector;
    Alcotest.test_case "register_selector with empty name" `Quick
      test_register_selector_with_empty_name;
  ]

let object_tests =
  [
    Alcotest.test_case "Get class of nil object" `Quick
      test_get_object_class_on_nil_object;
  ]

let () =
  Alcotest.run "Objective-C Runtime Bindings (Part 1)"
    [
      ("Class Functions", class_tests);
      ("Selector Functions", selector_tests);
      ("Object Functions (Basic)", object_tests);
    ]
