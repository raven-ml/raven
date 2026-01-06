(* Unit tests for Nx_oxcaml backend operations *)

module Dtype = Nx_core.Dtype
module View = Nx_core.View
module Symbolic_shape = Nx_core.Symbolic_shape
module Float_u = Stdlib_upstream_compatible.Float_u
module Float32_u = Stdlib_stable.Float32_u
module Int32_u = Stdlib_upstream_compatible.Int32_u
module Int64_u = Stdlib_upstream_compatible.Int64_u

external array_get : ('a : any mod non_null separable). 'a array -> int -> 'a
  = "%array_safe_get"
[@@layout_poly]

let failed = ref 0
let passed = ref 0

let check name cond =
  if cond then incr passed
  else (
    incr failed;
    Printf.printf "FAIL: %s\n%!" name)

let check_float64 name ~eps exp act =
  check name (Float_u.to_float (Float_u.abs (Float_u.sub exp act)) < eps)

let check_float32 name ~eps exp act =
  check name
    (Float_u.to_float
       (Float32_u.to_float (Float32_u.abs (Float32_u.sub exp act)))
    < eps)

let check_int32 name exp act = check name (Int32_u.equal exp act)
let check_int64 name exp act = check name (Int64_u.equal exp act)

let numel v =
  match Symbolic_shape.eval_dim (View.numel v) with
  | Some n -> n
  | None -> failwith "symbolic numel not evaluable"

let get64 (Nx_oxcaml.Float64 a) i = array_get a i
let get32 (Nx_oxcaml.Float32 a) i = array_get a i
let geti32 (Nx_oxcaml.Int32 a) i = array_get a i
let geti64 (Nx_oxcaml.Int64 a) i = array_get a i

let test_buffer_float64 () =
  let t = Nx_oxcaml.op_buffer (Nx_oxcaml.create_context ()) Dtype.Float64 5 in
  check "buffer_float64: dtype" (Nx_oxcaml.dtype t = Dtype.Float64);
  check "buffer_float64: size" (numel (Nx_oxcaml.view t) = 5)

let test_buffer_float32 () =
  let t = Nx_oxcaml.op_buffer (Nx_oxcaml.create_context ()) Dtype.Float32 3 in
  check "buffer_float32: dtype" (Nx_oxcaml.dtype t = Dtype.Float32);
  check "buffer_float32: size" (numel (Nx_oxcaml.view t) = 3)

let test_buffer_int32 () =
  let t = Nx_oxcaml.op_buffer (Nx_oxcaml.create_context ()) Dtype.Int32 4 in
  check "buffer_int32: dtype" (Nx_oxcaml.dtype t = Dtype.Int32);
  check "buffer_int32: size" (numel (Nx_oxcaml.view t) = 4)

let test_buffer_int64 () =
  let t = Nx_oxcaml.op_buffer (Nx_oxcaml.create_context ()) Dtype.Int64 2 in
  check "buffer_int64: dtype" (Nx_oxcaml.dtype t = Dtype.Int64);
  check "buffer_int64: size" (numel (Nx_oxcaml.view t) = 2)

let test_add_float64 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_float64 ctx [| #1.0; #2.0; #3.0 |] in
  let b = Nx_oxcaml.of_float64 ctx [| #10.0; #20.0; #30.0 |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Float64 3 in
  Nx_oxcaml.op_add ~out a b;
  let d = Nx_oxcaml.data_array out in
  check_float64 "add_float64[0]" ~eps:1e-9 #11.0 (get64 d 0);
  check_float64 "add_float64[1]" ~eps:1e-9 #22.0 (get64 d 1);
  check_float64 "add_float64[2]" ~eps:1e-9 #33.0 (get64 d 2)

let test_add_float32 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_float32 ctx [| #1.5s; #2.5s; #3.5s |] in
  let b = Nx_oxcaml.of_float32 ctx [| #0.5s; #0.5s; #0.5s |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Float32 3 in
  Nx_oxcaml.op_add ~out a b;
  let d = Nx_oxcaml.data_array out in
  check_float32 "add_float32[0]" ~eps:1e-6 #2.0s (get32 d 0);
  check_float32 "add_float32[1]" ~eps:1e-6 #3.0s (get32 d 1);
  check_float32 "add_float32[2]" ~eps:1e-6 #4.0s (get32 d 2)

let test_add_int32 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_int32 ctx [| #1l; #2l; #3l |] in
  let b = Nx_oxcaml.of_int32 ctx [| #100l; #200l; #300l |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Int32 3 in
  Nx_oxcaml.op_add ~out a b;
  let d = Nx_oxcaml.data_array out in
  check_int32 "add_int32[0]" #101l (geti32 d 0);
  check_int32 "add_int32[1]" #202l (geti32 d 1);
  check_int32 "add_int32[2]" #303l (geti32 d 2)

let test_add_int64 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_int64 ctx [| #1000L; #2000L; #3000L |] in
  let b = Nx_oxcaml.of_int64 ctx [| #1L; #2L; #3L |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Int64 3 in
  Nx_oxcaml.op_add ~out a b;
  let d = Nx_oxcaml.data_array out in
  check_int64 "add_int64[0]" #1001L (geti64 d 0);
  check_int64 "add_int64[1]" #2002L (geti64 d 1);
  check_int64 "add_int64[2]" #3003L (geti64 d 2)

let test_sub_float64 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_float64 ctx [| #10.0; #20.0; #30.0 |] in
  let b = Nx_oxcaml.of_float64 ctx [| #1.0; #2.0; #3.0 |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Float64 3 in
  Nx_oxcaml.op_sub ~out a b;
  let d = Nx_oxcaml.data_array out in
  check_float64 "sub_float64[0]" ~eps:1e-9 #9.0 (get64 d 0);
  check_float64 "sub_float64[1]" ~eps:1e-9 #18.0 (get64 d 1);
  check_float64 "sub_float64[2]" ~eps:1e-9 #27.0 (get64 d 2)

let test_sub_float32 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_float32 ctx [| #5.0s; #10.0s; #15.0s |] in
  let b = Nx_oxcaml.of_float32 ctx [| #1.0s; #2.0s; #3.0s |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Float32 3 in
  Nx_oxcaml.op_sub ~out a b;
  let d = Nx_oxcaml.data_array out in
  check_float32 "sub_float32[0]" ~eps:1e-6 #4.0s (get32 d 0);
  check_float32 "sub_float32[1]" ~eps:1e-6 #8.0s (get32 d 1);
  check_float32 "sub_float32[2]" ~eps:1e-6 #12.0s (get32 d 2)

let test_sub_int32 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_int32 ctx [| #100l; #200l; #300l |] in
  let b = Nx_oxcaml.of_int32 ctx [| #1l; #2l; #3l |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Int32 3 in
  Nx_oxcaml.op_sub ~out a b;
  let d = Nx_oxcaml.data_array out in
  check_int32 "sub_int32[0]" #99l (geti32 d 0);
  check_int32 "sub_int32[1]" #198l (geti32 d 1);
  check_int32 "sub_int32[2]" #297l (geti32 d 2)

let test_sub_int64 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_int64 ctx [| #1000L; #2000L; #3000L |] in
  let b = Nx_oxcaml.of_int64 ctx [| #1L; #2L; #3L |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Int64 3 in
  Nx_oxcaml.op_sub ~out a b;
  let d = Nx_oxcaml.data_array out in
  check_int64 "sub_int64[0]" #999L (geti64 d 0);
  check_int64 "sub_int64[1]" #1998L (geti64 d 1);
  check_int64 "sub_int64[2]" #2997L (geti64 d 2)

let test_add_single_element () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_float64 ctx [| #42.0 |] in
  let b = Nx_oxcaml.of_float64 ctx [| #8.0 |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Float64 1 in
  Nx_oxcaml.op_add ~out a b;
  check_float64 "add_single[0]" ~eps:1e-9 #50.0
    (get64 (Nx_oxcaml.data_array out) 0)

let test_add_negative_values () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_float64 ctx [| -#5.0; #10.0 |] in
  let b = Nx_oxcaml.of_float64 ctx [| -#3.0; -#7.0 |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Float64 2 in
  Nx_oxcaml.op_add ~out a b;
  let d = Nx_oxcaml.data_array out in
  check_float64 "add_neg[0]" ~eps:1e-9 (-#8.0) (get64 d 0);
  check_float64 "add_neg[1]" ~eps:1e-9 #3.0 (get64 d 1)

let test_sub_to_zero () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_int32 ctx [| #5l; #10l |] in
  let b = Nx_oxcaml.of_int32 ctx [| #5l; #10l |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Int32 2 in
  Nx_oxcaml.op_sub ~out a b;
  let d = Nx_oxcaml.data_array out in
  check_int32 "sub_zero[0]" #0l (geti32 d 0);
  check_int32 "sub_zero[1]" #0l (geti32 d 1)

let test_in_place_add () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_float64 ctx [| #1.0; #2.0; #3.0 |] in
  let b = Nx_oxcaml.of_float64 ctx [| #10.0; #20.0; #30.0 |] in
  Nx_oxcaml.op_add ~out:a a b;
  let d = Nx_oxcaml.data_array a in
  check_float64 "inplace_add[0]" ~eps:1e-9 #11.0 (get64 d 0);
  check_float64 "inplace_add[1]" ~eps:1e-9 #22.0 (get64 d 1);
  check_float64 "inplace_add[2]" ~eps:1e-9 #33.0 (get64 d 2)

  let test_mul_float64 () =
    let ctx = Nx_oxcaml.create_context () in
    let a = Nx_oxcaml.of_float64 ctx [| #1.0; #2.0; #3.0 |] in
    let b = Nx_oxcaml.of_float64 ctx [| #10.0; #20.0; #30.0 |] in
    let out = Nx_oxcaml.op_buffer ctx Dtype.Float64 3 in
    Nx_oxcaml.op_mul ~out a b;
    let d = Nx_oxcaml.data_array out in
    check_float64 "mul_float64[0]" ~eps:1e-9 #10.0 (get64 d 0);
    check_float64 "mul_float64[1]" ~eps:1e-9 #40.0 (get64 d 1);
    check_float64 "mul_float64[2]" ~eps:1e-9 #90.0 (get64 d 2)
  
  let test_mul_float32 () =
    let ctx = Nx_oxcaml.create_context () in
    let a = Nx_oxcaml.of_float32 ctx [| #1.5s; #2.5s; #3.5s |] in
    let b = Nx_oxcaml.of_float32 ctx [| #0.5s; #0.5s; #2.0s |] in
    let out = Nx_oxcaml.op_buffer ctx Dtype.Float32 3 in
    Nx_oxcaml.op_mul ~out a b;
    let d = Nx_oxcaml.data_array out in
    check_float32 "mul_float32[0]" ~eps:1e-6 #0.75s (get32 d 0);
    check_float32 "mul_float32[1]" ~eps:1e-6 #1.25s (get32 d 1);
    check_float32 "mul_float32[2]" ~eps:1e-6 #7.0s (get32 d 2)
  
  let test_mul_int32 () =
    let ctx = Nx_oxcaml.create_context () in
    let a = Nx_oxcaml.of_int32 ctx [| #1l; #2l; #3l |] in
    let b = Nx_oxcaml.of_int32 ctx [| #100l; #200l; #300l |] in
    let out = Nx_oxcaml.op_buffer ctx Dtype.Int32 3 in
    Nx_oxcaml.op_mul ~out a b;
    let d = Nx_oxcaml.data_array out in
    check_int32 "mul_int32[0]" #100l (geti32 d 0);
    check_int32 "mul_int32[1]" #400l (geti32 d 1);
    check_int32 "mul_int32[2]" #900l (geti32 d 2)
  
  let test_mul_int64 () =
    let ctx = Nx_oxcaml.create_context () in
    let a = Nx_oxcaml.of_int64 ctx [| #1000L; #2000L; #3000L |] in
    let b = Nx_oxcaml.of_int64 ctx [| #1L; #2L; #3L |] in
    let out = Nx_oxcaml.op_buffer ctx Dtype.Int64 3 in
    Nx_oxcaml.op_mul ~out a b;
    let d = Nx_oxcaml.data_array out in
    check_int64 "mul_int64[0]" #1000L (geti64 d 0);
    check_int64 "mul_int64[1]" #4000L (geti64 d 1);
    check_int64 "mul_int64[2]" #9000L (geti64 d 2)

  let test_fdiv_float64 () =
    let ctx = Nx_oxcaml.create_context () in
    let a = Nx_oxcaml.of_float64 ctx [| #1.0; #2.0; #2.0 |] in
    let b = Nx_oxcaml.of_float64 ctx [| #0.0; #2.0; #3.0 |] in
    let out = Nx_oxcaml.op_buffer ctx Dtype.Float64 3 in
    Nx_oxcaml.op_fdiv ~out b a;
    let d = Nx_oxcaml.data_array out in
    check_float64 "fdiv_float64[0]" ~eps:1e-9 #0.0 (get64 d 0);
    check_float64 "fdiv_float64[1]" ~eps:1e-9 #1.0 (get64 d 1);
    check_float64 "fdiv_float64[2]" ~eps:1e-9 #1.5 (get64 d 2)
  
  let test_fdiv_float32 () =
    let ctx = Nx_oxcaml.create_context () in
    let a = Nx_oxcaml.of_float32 ctx [| #1.5s; #2.5s; #7.0s |] in
    let b = Nx_oxcaml.of_float32 ctx [| #0.5s; #0.5s; #2.0s |] in
    let out = Nx_oxcaml.op_buffer ctx Dtype.Float32 3 in
    Nx_oxcaml.op_fdiv ~out a b;
    let d = Nx_oxcaml.data_array out in
    check_float32 "fdiv_float32[0]" ~eps:1e-6 #3.0s (get32 d 0);
    check_float32 "fdiv_float32[1]" ~eps:1e-6 #5.0s (get32 d 1);
    check_float32 "fdiv_float32[2]" ~eps:1e-6 #3.5s (get32 d 2)
  
  let test_fdiv_int32 () =
    let ctx = Nx_oxcaml.create_context () in
    let a = Nx_oxcaml.of_int32 ctx [| #1l; #2l; #3l |] in
    let b = Nx_oxcaml.of_int32 ctx [| #100l; #1l; #2l |] in
    let out = Nx_oxcaml.op_buffer ctx Dtype.Int32 3 in
    Nx_oxcaml.op_fdiv ~out a b;
    let d = Nx_oxcaml.data_array out in
    check_int32 "fdiv_int32[0]" #0l (geti32 d 0);
    check_int32 "fdiv_int32[1]" #2l (geti32 d 1);
    check_int32 "fdiv_int32[2]" #1l (geti32 d 2)
  
  let test_fdiv_int64 () =
    let ctx = Nx_oxcaml.create_context () in
    let a = Nx_oxcaml.of_int64 ctx [| #1000L; #2000L; #3000L |] in
    let b = Nx_oxcaml.of_int64 ctx [| #1L; #2L; #3L |] in
    let out = Nx_oxcaml.op_buffer ctx Dtype.Int64 3 in
    Nx_oxcaml.op_fdiv ~out a b;
    let d = Nx_oxcaml.data_array out in
    check_int64 "fdiv_int64[0]" #1000L (geti64 d 0);
    check_int64 "fdiv_int64[1]" #1000L (geti64 d 1);
    check_int64 "fdiv_int64[2]" #1000L (geti64 d 2)

let test_idiv_float64 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_float64 ctx [| #1.0; #2.0; #2.0 |] in
  let b = Nx_oxcaml.of_float64 ctx [| #0.0; #2.0; #3.0 |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Float64 3 in
  Nx_oxcaml.op_idiv ~out b a;
  let d = Nx_oxcaml.data_array out in
  check_float64 "idiv_float64[0]" ~eps:1e-9 #0.0 (get64 d 0);
  check_float64 "idiv_float64[1]" ~eps:1e-9 #1.0 (get64 d 1);
  check_float64 "idiv_float64[2]" ~eps:1e-9 #1.0 (get64 d 2)

let test_idiv_float32 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_float32 ctx [| #1.5s; #2.5s; #7.0s |] in
  let b = Nx_oxcaml.of_float32 ctx [| #0.5s; #0.5s; #2.0s |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Float32 3 in
  Nx_oxcaml.op_idiv ~out a b;
  let d = Nx_oxcaml.data_array out in
  check_float32 "idiv_float32[0]" ~eps:1e-6 #3.0s (get32 d 0);
  check_float32 "idiv_float32[1]" ~eps:1e-6 #5.0s (get32 d 1);
  check_float32 "idiv_float32[2]" ~eps:1e-6 #3.0s (get32 d 2)

let test_idiv_int32 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_int32 ctx [| #1l; #2l; #3l |] in
  let b = Nx_oxcaml.of_int32 ctx [| #100l; #1l; #2l |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Int32 3 in
  Nx_oxcaml.op_idiv ~out a b;
  let d = Nx_oxcaml.data_array out in
  check_int32 "idiv_int32[0]" #0l (geti32 d 0);
  check_int32 "idiv_int32[1]" #2l (geti32 d 1);
  check_int32 "idiv_int32[2]" #1l (geti32 d 2)

let test_idiv_int64 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_int64 ctx [| #1000L; #2000L; #3000L |] in
  let b = Nx_oxcaml.of_int64 ctx [| #1L; #2L; #3L |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Int64 3 in
  Nx_oxcaml.op_idiv ~out a b;
  let d = Nx_oxcaml.data_array out in
  check_int64 "idiv_int64[0]" #1000L (geti64 d 0);
  check_int64 "idiv_int64[1]" #1000L (geti64 d 1);
  check_int64 "idiv_int64[2]" #1000L (geti64 d 2)
let () =
  print_endline "Running Nx_oxcaml backend tests...";
  test_buffer_float64 ();
  test_buffer_float32 ();
  test_buffer_int32 ();
  test_buffer_int64 ();
  test_add_float64 ();
  test_add_float32 ();
  test_add_int32 ();
  test_add_int64 ();
  test_sub_float64 ();
  test_sub_float32 ();
  test_sub_int32 ();
  test_sub_int64 ();
  test_add_single_element ();
  test_add_negative_values ();
  test_sub_to_zero ();
  test_in_place_add ();
  test_mul_float64 ();
  test_mul_float32 ();
  test_mul_int64 ();
  test_mul_int32 ();
  test_fdiv_float64 ();
  test_fdiv_float32 ();
  test_fdiv_int64 ();
  test_fdiv_int32 ();
  test_idiv_float64 ();
  test_idiv_float32 ();
  test_idiv_int64 ();
  test_idiv_int32 ();
  Printf.printf "\nResults: %d passed, %d failed\n" !passed !failed;
  if !failed > 0 then exit 1
