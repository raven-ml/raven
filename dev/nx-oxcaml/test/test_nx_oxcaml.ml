(* Unit tests for Nx_oxcaml backend operations *)

module Float_u = Stdlib_upstream_compatible.Float_u
module Float32_u = Stdlib_stable.Float32_u
module Int32_u = Stdlib_upstream_compatible.Int32_u
module Int64_u = Stdlib_upstream_compatible.Int64_u

let failed = ref 0
let passed = ref 0

let check name cond =
  if cond then incr passed
  else (
    incr failed;
    Printf.printf "FAIL: %s\n%!" name)

let check_float name ~eps expected actual =
  check name (Float.abs (expected -. actual) < eps)

let check_float32 name ~eps expected actual =
  check name (Float.abs (expected -. actual) < eps)

let check_int32 name expected actual = check name (expected = actual)
let check_int64 name expected actual = check name (expected = actual)

(* Helper to extract buffer values *)
let get_float64 (Nx_oxcaml.Float64 arr) i = Float_u.to_float (Array.get arr i)
let get_float32 (Nx_oxcaml.Float32 arr) i = Float32_u.to_float (Array.get arr i)
let get_int32 (Nx_oxcaml.Int32 arr) i = Int32_u.to_int32 (Array.get arr i)
let get_int64 (Nx_oxcaml.Int64 arr) i = Int64_u.to_int64 (Array.get arr i)

let set_float64 (Nx_oxcaml.Float64 arr) i v =
  Array.set arr i (Float_u.of_float v)

let set_float32 (Nx_oxcaml.Float32 arr) i v =
  Array.set arr i (Float32_u.of_float v)

let set_int32 (Nx_oxcaml.Int32 arr) i v = Array.set arr i (Int32_u.of_int32 v)
let set_int64 (Nx_oxcaml.Int64 arr) i v = Array.set arr i (Int64_u.of_int64 v)

(* op_buffer tests *)

let test_buffer_float64 () =
  let ctx = Nx_oxcaml.create_context () in
  let t = Nx_oxcaml.op_buffer ctx Dtype.Float64 5 in
  check "buffer_float64: dtype" (Nx_oxcaml.dtype t = Dtype.Float64);
  check "buffer_float64: size" (View.numel (Nx_oxcaml.view t) = 5)

let test_buffer_float32 () =
  let ctx = Nx_oxcaml.create_context () in
  let t = Nx_oxcaml.op_buffer ctx Dtype.Float32 3 in
  check "buffer_float32: dtype" (Nx_oxcaml.dtype t = Dtype.Float32);
  check "buffer_float32: size" (View.numel (Nx_oxcaml.view t) = 3)

let test_buffer_int32 () =
  let ctx = Nx_oxcaml.create_context () in
  let t = Nx_oxcaml.op_buffer ctx Dtype.Int32 4 in
  check "buffer_int32: dtype" (Nx_oxcaml.dtype t = Dtype.Int32);
  check "buffer_int32: size" (View.numel (Nx_oxcaml.view t) = 4)

let test_buffer_int64 () =
  let ctx = Nx_oxcaml.create_context () in
  let t = Nx_oxcaml.op_buffer ctx Dtype.Int64 2 in
  check "buffer_int64: dtype" (Nx_oxcaml.dtype t = Dtype.Int64);
  check "buffer_int64: size" (View.numel (Nx_oxcaml.view t) = 2)

(* op_add tests *)

let test_add_float64 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.op_buffer ctx Dtype.Float64 3 in
  let b = Nx_oxcaml.op_buffer ctx Dtype.Float64 3 in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Float64 3 in
  set_float64 (Nx_oxcaml.data a) 0 1.0;
  set_float64 (Nx_oxcaml.data a) 1 2.0;
  set_float64 (Nx_oxcaml.data a) 2 3.0;
  set_float64 (Nx_oxcaml.data b) 0 10.0;
  set_float64 (Nx_oxcaml.data b) 1 20.0;
  set_float64 (Nx_oxcaml.data b) 2 30.0;
  Nx_oxcaml.op_add ~out a b;
  check_float "add_float64[0]" ~eps:1e-9 11.0 (get_float64 (Nx_oxcaml.data out) 0);
  check_float "add_float64[1]" ~eps:1e-9 22.0 (get_float64 (Nx_oxcaml.data out) 1);
  check_float "add_float64[2]" ~eps:1e-9 33.0 (get_float64 (Nx_oxcaml.data out) 2)

let test_add_float32 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.op_buffer ctx Dtype.Float32 3 in
  let b = Nx_oxcaml.op_buffer ctx Dtype.Float32 3 in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Float32 3 in
  set_float32 (Nx_oxcaml.data a) 0 1.5;
  set_float32 (Nx_oxcaml.data a) 1 2.5;
  set_float32 (Nx_oxcaml.data a) 2 3.5;
  set_float32 (Nx_oxcaml.data b) 0 0.5;
  set_float32 (Nx_oxcaml.data b) 1 0.5;
  set_float32 (Nx_oxcaml.data b) 2 0.5;
  Nx_oxcaml.op_add ~out a b;
  check_float32 "add_float32[0]" ~eps:1e-6 2.0 (get_float32 (Nx_oxcaml.data out) 0);
  check_float32 "add_float32[1]" ~eps:1e-6 3.0 (get_float32 (Nx_oxcaml.data out) 1);
  check_float32 "add_float32[2]" ~eps:1e-6 4.0 (get_float32 (Nx_oxcaml.data out) 2)

let test_add_int32 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.op_buffer ctx Dtype.Int32 3 in
  let b = Nx_oxcaml.op_buffer ctx Dtype.Int32 3 in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Int32 3 in
  set_int32 (Nx_oxcaml.data a) 0 1l;
  set_int32 (Nx_oxcaml.data a) 1 2l;
  set_int32 (Nx_oxcaml.data a) 2 3l;
  set_int32 (Nx_oxcaml.data b) 0 100l;
  set_int32 (Nx_oxcaml.data b) 1 200l;
  set_int32 (Nx_oxcaml.data b) 2 300l;
  Nx_oxcaml.op_add ~out a b;
  check_int32 "add_int32[0]" 101l (get_int32 (Nx_oxcaml.data out) 0);
  check_int32 "add_int32[1]" 202l (get_int32 (Nx_oxcaml.data out) 1);
  check_int32 "add_int32[2]" 303l (get_int32 (Nx_oxcaml.data out) 2)

let test_add_int64 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.op_buffer ctx Dtype.Int64 3 in
  let b = Nx_oxcaml.op_buffer ctx Dtype.Int64 3 in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Int64 3 in
  set_int64 (Nx_oxcaml.data a) 0 1000L;
  set_int64 (Nx_oxcaml.data a) 1 2000L;
  set_int64 (Nx_oxcaml.data a) 2 3000L;
  set_int64 (Nx_oxcaml.data b) 0 1L;
  set_int64 (Nx_oxcaml.data b) 1 2L;
  set_int64 (Nx_oxcaml.data b) 2 3L;
  Nx_oxcaml.op_add ~out a b;
  check_int64 "add_int64[0]" 1001L (get_int64 (Nx_oxcaml.data out) 0);
  check_int64 "add_int64[1]" 2002L (get_int64 (Nx_oxcaml.data out) 1);
  check_int64 "add_int64[2]" 3003L (get_int64 (Nx_oxcaml.data out) 2)

(* op_sub tests *)

let test_sub_float64 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.op_buffer ctx Dtype.Float64 3 in
  let b = Nx_oxcaml.op_buffer ctx Dtype.Float64 3 in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Float64 3 in
  set_float64 (Nx_oxcaml.data a) 0 10.0;
  set_float64 (Nx_oxcaml.data a) 1 20.0;
  set_float64 (Nx_oxcaml.data a) 2 30.0;
  set_float64 (Nx_oxcaml.data b) 0 1.0;
  set_float64 (Nx_oxcaml.data b) 1 2.0;
  set_float64 (Nx_oxcaml.data b) 2 3.0;
  Nx_oxcaml.op_sub ~out a b;
  check_float "sub_float64[0]" ~eps:1e-9 9.0 (get_float64 (Nx_oxcaml.data out) 0);
  check_float "sub_float64[1]" ~eps:1e-9 18.0 (get_float64 (Nx_oxcaml.data out) 1);
  check_float "sub_float64[2]" ~eps:1e-9 27.0 (get_float64 (Nx_oxcaml.data out) 2)

let test_sub_float32 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.op_buffer ctx Dtype.Float32 3 in
  let b = Nx_oxcaml.op_buffer ctx Dtype.Float32 3 in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Float32 3 in
  set_float32 (Nx_oxcaml.data a) 0 5.0;
  set_float32 (Nx_oxcaml.data a) 1 10.0;
  set_float32 (Nx_oxcaml.data a) 2 15.0;
  set_float32 (Nx_oxcaml.data b) 0 1.0;
  set_float32 (Nx_oxcaml.data b) 1 2.0;
  set_float32 (Nx_oxcaml.data b) 2 3.0;
  Nx_oxcaml.op_sub ~out a b;
  check_float32 "sub_float32[0]" ~eps:1e-6 4.0 (get_float32 (Nx_oxcaml.data out) 0);
  check_float32 "sub_float32[1]" ~eps:1e-6 8.0 (get_float32 (Nx_oxcaml.data out) 1);
  check_float32 "sub_float32[2]" ~eps:1e-6 12.0 (get_float32 (Nx_oxcaml.data out) 2)

let test_sub_int32 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.op_buffer ctx Dtype.Int32 3 in
  let b = Nx_oxcaml.op_buffer ctx Dtype.Int32 3 in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Int32 3 in
  set_int32 (Nx_oxcaml.data a) 0 100l;
  set_int32 (Nx_oxcaml.data a) 1 200l;
  set_int32 (Nx_oxcaml.data a) 2 300l;
  set_int32 (Nx_oxcaml.data b) 0 1l;
  set_int32 (Nx_oxcaml.data b) 1 2l;
  set_int32 (Nx_oxcaml.data b) 2 3l;
  Nx_oxcaml.op_sub ~out a b;
  check_int32 "sub_int32[0]" 99l (get_int32 (Nx_oxcaml.data out) 0);
  check_int32 "sub_int32[1]" 198l (get_int32 (Nx_oxcaml.data out) 1);
  check_int32 "sub_int32[2]" 297l (get_int32 (Nx_oxcaml.data out) 2)

let test_sub_int64 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.op_buffer ctx Dtype.Int64 3 in
  let b = Nx_oxcaml.op_buffer ctx Dtype.Int64 3 in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Int64 3 in
  set_int64 (Nx_oxcaml.data a) 0 1000L;
  set_int64 (Nx_oxcaml.data a) 1 2000L;
  set_int64 (Nx_oxcaml.data a) 2 3000L;
  set_int64 (Nx_oxcaml.data b) 0 1L;
  set_int64 (Nx_oxcaml.data b) 1 2L;
  set_int64 (Nx_oxcaml.data b) 2 3L;
  Nx_oxcaml.op_sub ~out a b;
  check_int64 "sub_int64[0]" 999L (get_int64 (Nx_oxcaml.data out) 0);
  check_int64 "sub_int64[1]" 1998L (get_int64 (Nx_oxcaml.data out) 1);
  check_int64 "sub_int64[2]" 2997L (get_int64 (Nx_oxcaml.data out) 2)

(* Edge cases *)

let test_add_single_element () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.op_buffer ctx Dtype.Float64 1 in
  let b = Nx_oxcaml.op_buffer ctx Dtype.Float64 1 in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Float64 1 in
  set_float64 (Nx_oxcaml.data a) 0 42.0;
  set_float64 (Nx_oxcaml.data b) 0 8.0;
  Nx_oxcaml.op_add ~out a b;
  check_float "add_single[0]" ~eps:1e-9 50.0 (get_float64 (Nx_oxcaml.data out) 0)

let test_add_negative_values () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.op_buffer ctx Dtype.Float64 2 in
  let b = Nx_oxcaml.op_buffer ctx Dtype.Float64 2 in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Float64 2 in
  set_float64 (Nx_oxcaml.data a) 0 (-5.0);
  set_float64 (Nx_oxcaml.data a) 1 10.0;
  set_float64 (Nx_oxcaml.data b) 0 (-3.0);
  set_float64 (Nx_oxcaml.data b) 1 (-7.0);
  Nx_oxcaml.op_add ~out a b;
  check_float "add_neg[0]" ~eps:1e-9 (-8.0) (get_float64 (Nx_oxcaml.data out) 0);
  check_float "add_neg[1]" ~eps:1e-9 3.0 (get_float64 (Nx_oxcaml.data out) 1)

let test_sub_to_zero () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.op_buffer ctx Dtype.Int32 2 in
  let b = Nx_oxcaml.op_buffer ctx Dtype.Int32 2 in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Int32 2 in
  set_int32 (Nx_oxcaml.data a) 0 5l;
  set_int32 (Nx_oxcaml.data a) 1 10l;
  set_int32 (Nx_oxcaml.data b) 0 5l;
  set_int32 (Nx_oxcaml.data b) 1 10l;
  Nx_oxcaml.op_sub ~out a b;
  check_int32 "sub_zero[0]" 0l (get_int32 (Nx_oxcaml.data out) 0);
  check_int32 "sub_zero[1]" 0l (get_int32 (Nx_oxcaml.data out) 1)

let test_in_place_add () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.op_buffer ctx Dtype.Float64 3 in
  let b = Nx_oxcaml.op_buffer ctx Dtype.Float64 3 in
  set_float64 (Nx_oxcaml.data a) 0 1.0;
  set_float64 (Nx_oxcaml.data a) 1 2.0;
  set_float64 (Nx_oxcaml.data a) 2 3.0;
  set_float64 (Nx_oxcaml.data b) 0 10.0;
  set_float64 (Nx_oxcaml.data b) 1 20.0;
  set_float64 (Nx_oxcaml.data b) 2 30.0;
  Nx_oxcaml.op_add ~out:a a b;
  check_float "inplace_add[0]" ~eps:1e-9 11.0 (get_float64 (Nx_oxcaml.data a) 0);
  check_float "inplace_add[1]" ~eps:1e-9 22.0 (get_float64 (Nx_oxcaml.data a) 1);
  check_float "inplace_add[2]" ~eps:1e-9 33.0 (get_float64 (Nx_oxcaml.data a) 2)

let () =
  print_endline "Running Nx_oxcaml backend tests...";
  (* op_buffer *)
  test_buffer_float64 ();
  test_buffer_float32 ();
  test_buffer_int32 ();
  test_buffer_int64 ();
  (* op_add *)
  test_add_float64 ();
  test_add_float32 ();
  test_add_int32 ();
  test_add_int64 ();
  (* op_sub *)
  test_sub_float64 ();
  test_sub_float32 ();
  test_sub_int32 ();
  test_sub_int64 ();
  (* edge cases *)
  test_add_single_element ();
  test_add_negative_values ();
  test_sub_to_zero ();
  test_in_place_add ();
  (* summary *)
  Printf.printf "\nResults: %d passed, %d failed\n" !passed !failed;
  if !failed > 0 then exit 1
