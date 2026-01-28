(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Unit tests for Nx_oxcaml backend operations *)

module Dtype = Nx_core.Dtype
module View = Nx_core.View
module Nx_ox = Nx_core.Make_frontend (Nx_oxcaml)
module Symbolic_shape = Nx_core.Symbolic_shape
module Float_u = Stdlib_upstream_compatible.Float_u
module Float32_u = Stdlib_stable.Float32_u
module Int8_u = Stdlib_stable.Int8_u
module Int16_u = Stdlib_stable.Int16_u
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

let check_int8 name exp act = check name (Int8_u.equal exp act)
let check_int16 name exp act = check name (Int16_u.equal exp act)
let check_int32 name exp act = check name (Int32_u.equal exp act)
let check_int64 name exp act = check name (Int64_u.equal exp act)
let check_bool name exp act = check name (exp == act)

let numel v =
  match Symbolic_shape.eval_dim (View.numel v) with
  | Some n -> n
  | None -> failwith "symbolic numel not evaluable"

let get64 (Nx_oxcaml.Float64 a) i = array_get a i
let get32 (Nx_oxcaml.Float32 a) i = array_get a i
let geti8 (Nx_oxcaml.Int8 a) i = array_get a i
let geti16 (Nx_oxcaml.Int16 a) i = array_get a i
let geti32 (Nx_oxcaml.Int32 a) i = array_get a i
let geti64 (Nx_oxcaml.Int64 a) i = array_get a i
let getbool (Nx_oxcaml.Bool a) i = array_get a i

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

let test_mod_float64 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_float64 ctx [| #1.0; #2.0; #2.0 |] in
  let b = Nx_oxcaml.of_float64 ctx [| #0.0; #2.0; #3.0 |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Float64 3 in
  Nx_oxcaml.op_mod ~out b a;
  let d = Nx_oxcaml.data_array out in
  check_float64 "mod_float64[0]" ~eps:1e-9 #0.0 (get64 d 0);
  (* 0 mod 1 = 0 *)
  check_float64 "mod_float64[1]" ~eps:1e-9 #0.0 (get64 d 1);
  (* 2 mod 2 = 0 *)
  check_float64 "mod_float64[2]" ~eps:1e-9 #1.0 (get64 d 2)
(* 3 mod 2 = 1 *)

let test_mod_float32 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_float32 ctx [| #1.5s; #2.5s; #7.0s |] in
  let b = Nx_oxcaml.of_float32 ctx [| #0.5s; #0.5s; #2.0s |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Float32 3 in
  Nx_oxcaml.op_mod ~out a b;
  let d = Nx_oxcaml.data_array out in
  check_float32 "mod_float32[0]" ~eps:1e-6 #0.0s (get32 d 0);
  (* 1.5 mod 0.5 = 0 *)
  check_float32 "mod_float32[1]" ~eps:1e-6 #0.0s (get32 d 1);
  (* 2.5 mod 0.5 = 0 *)
  check_float32 "mod_float32[2]" ~eps:1e-6 #1.0s (get32 d 2)
(* 7 mod 2 = 1 *)

let test_mod_int32 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_int32 ctx [| #1l; #2l; #3l |] in
  let b = Nx_oxcaml.of_int32 ctx [| #100l; #1l; #2l |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Int32 3 in
  Nx_oxcaml.op_mod ~out a b;
  let d = Nx_oxcaml.data_array out in
  check_int32 "mod_int32[0]" #1l (geti32 d 0);
  (* 1 mod 100 = 1 *)
  check_int32 "mod_int32[1]" #0l (geti32 d 1);
  (* 2 mod 1 = 0 *)
  check_int32 "mod_int32[2]" #1l (geti32 d 2)
(* 3 mod 2 = 1 *)

let test_mod_int64 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_int64 ctx [| #1000L; #2000L; #3000L |] in
  let b = Nx_oxcaml.of_int64 ctx [| #1L; #2L; #3L |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Int64 3 in
  Nx_oxcaml.op_mod ~out a b;
  let d = Nx_oxcaml.data_array out in
  check_int64 "mod_int64[0]" #0L (geti64 d 0);
  (* 1000 mod 1 = 0 *)
  check_int64 "mod_int64[1]" #0L (geti64 d 1);
  (* 2000 mod 2 = 0 *)
  check_int64 "mod_int64[2]" #0L (geti64 d 2)
(* 3000 mod 3 = 0 *)

(*let test_max_float64 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_float64 ctx [| #1.0; #2.0; #2.0 |] in
  let b = Nx_oxcaml.of_float64 ctx [| #0.0; #2.5; #1.5 |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Float64 3 in
  Nx_oxcaml.op_max ~out a b;
  let d = Nx_oxcaml.data_array out in
  check_float64 "max_float64[0]" ~eps:1e-9 #1.0 (get64 d 0);
  check_float64 "max_float64[1]" ~eps:1e-9 #2.5 (get64 d 1);
  check_float64 "max_float64[2]" ~eps:1e-9 #2.0 (get64 d 2)

let test_max_float32 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_float32 ctx [| #1.5s; #2.5s; #7.0s |] in
  let b = Nx_oxcaml.of_float32 ctx [| #2.0s; #2.0s; #3.0s |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Float32 3 in
  Nx_oxcaml.op_max ~out a b;
  let d = Nx_oxcaml.data_array out in
  check_float32 "max_float32[0]" ~eps:1e-6 #2.0s (get32 d 0);
  check_float32 "max_float32[1]" ~eps:1e-6 #2.5s (get32 d 1);
  check_float32 "max_float32[2]" ~eps:1e-6 #7.0s (get32 d 2)

let test_max_int32 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_int32 ctx [| #1l; #2l; #3l |] in
  let b = Nx_oxcaml.of_int32 ctx [| #0l; #3l; #2l |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Int32 3 in
  Nx_oxcaml.op_max ~out a b;
  let d = Nx_oxcaml.data_array out in
  check_int32 "max_int32[0]" #1l (geti32 d 0);
  check_int32 "max_int32[1]" #3l (geti32 d 1);
  check_int32 "max_int32[2]" #3l (geti32 d 2)

 let test_max_int64 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_int64 ctx [| #1000L; #2000L; #3000L |] in
  let b = Nx_oxcaml.of_int64 ctx [| #1500L; #1500L; #1000L |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Int64 3 in
  Nx_oxcaml.op_max ~out a b;
  let d = Nx_oxcaml.data_array out in
  check_int64 "max_int64[0]" #1500L (geti64 d 0);
  check_int64 "max_int64[1]" #2000L (geti64 d 1);
  check_int64 "max_int64[2]" #3000L (geti64 d 2) *)

let test_min_float64 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_float64 ctx [| #1.0; #2.0; #2.0 |] in
  let b = Nx_oxcaml.of_float64 ctx [| #0.0; #2.5; #1.5 |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Float64 3 in
  Nx_oxcaml.op_min ~out a b;
  let d = Nx_oxcaml.data_array out in
  check_float64 "min_float64[0]" ~eps:1e-9 #0.0 (get64 d 0);
  check_float64 "min_float64[1]" ~eps:1e-9 #2.0 (get64 d 1);
  check_float64 "min_float64[2]" ~eps:1e-9 #1.5 (get64 d 2)

let test_min_float32 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_float32 ctx [| #1.5s; #2.5s; #7.0s |] in
  let b = Nx_oxcaml.of_float32 ctx [| #2.0s; #2.0s; #3.0s |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Float32 3 in
  Nx_oxcaml.op_min ~out a b;
  let d = Nx_oxcaml.data_array out in
  check_float32 "min_float32[0]" ~eps:1e-6 #1.5s (get32 d 0);
  check_float32 "min_float32[1]" ~eps:1e-6 #2.0s (get32 d 1);
  check_float32 "min_float32[2]" ~eps:1e-6 #3.0s (get32 d 2)

let test_min_int32 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_int32 ctx [| #1l; #2l; #3l |] in
  let b = Nx_oxcaml.of_int32 ctx [| #0l; #3l; #2l |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Int32 3 in
  Nx_oxcaml.op_min ~out a b;
  let d = Nx_oxcaml.data_array out in
  check_int32 "min_int32[0]" #0l (geti32 d 0);
  check_int32 "min_int32[1]" #2l (geti32 d 1);
  check_int32 "min_int32[2]" #2l (geti32 d 2)

(* let test_min_int64 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_int64 ctx [| #1000L; #2000L; #3000L |] in
  let b = Nx_oxcaml.of_int64 ctx [| #1500L; #1500L; #1000L |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Int64 3 in
  Nx_oxcaml.op_min ~out a b;
  let d = Nx_oxcaml.data_array out in
  check_int64 "min_int64[0]" #1000L (geti64 d 0);
  check_int64 "min_int64[1]" #1500L (geti64 d 1);
  check_int64 "min_int64[2]" #1000L (geti64 d 2) *)

let test_pow_float64 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_float64 ctx [| #2.0; #3.0; #4.0 |] in
  let b = Nx_oxcaml.of_float64 ctx [| #3.0; #2.0; #0.5 |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Float64 3 in
  Nx_oxcaml.op_pow ~out a b;
  let d = Nx_oxcaml.data_array out in
  check_float64 "pow_float64[0]" ~eps:1e-9 #8.0 (get64 d 0);
  (* 2^3 *)
  check_float64 "pow_float64[1]" ~eps:1e-9 #9.0 (get64 d 1);
  (* 3^2 *)
  check_float64 "pow_float64[2]" ~eps:1e-9 #2.0 (get64 d 2)
(* 4^0.5 *)

let test_pow_float32 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_float32 ctx [| #2.0s; #5.0s; #9.0s |] in
  let b = Nx_oxcaml.of_float32 ctx [| #3.0s; #1.0s; #0.5s |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Float32 3 in
  Nx_oxcaml.op_pow ~out a b;
  let d = Nx_oxcaml.data_array out in
  check_float32 "pow_float32[0]" ~eps:1e-6 #8.0s (get32 d 0);
  (* 2^3 *)
  check_float32 "pow_float32[1]" ~eps:1e-6 #5.0s (get32 d 1);
  (* 5^1 *)
  check_float32 "pow_float32[2]" ~eps:1e-6 #3.0s (get32 d 2)
(* 9^0.5 *)

let test_and_int32 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_int32 ctx [| #0b1101l; #0b1010l; #0b1111l |] in
  let b = Nx_oxcaml.of_int32 ctx [| #0b1011l; #0b1100l; #0b0101l |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Int32 3 in
  Nx_oxcaml.op_and ~out a b;
  let d = Nx_oxcaml.data_array out in
  check_int32 "and_int32[0]" #0b1001l (geti32 d 0);
  (* 1101 & 1011 *)
  check_int32 "and_int32[1]" #0b1000l (geti32 d 1);
  (* 1010 & 1100 *)
  check_int32 "and_int32[2]" #0b0101l (geti32 d 2)
(* 1111 & 0101 *)

let test_and_int64 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_int64 ctx [| #0b1101L; #0b1010L; #0b1111L |] in
  let b = Nx_oxcaml.of_int64 ctx [| #0b1011L; #0b1100L; #0b0101L |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Int64 3 in
  Nx_oxcaml.op_and ~out a b;
  let d = Nx_oxcaml.data_array out in
  check_int64 "and_int64[0]" #0b1001L (geti64 d 0);
  check_int64 "and_int64[1]" #0b1000L (geti64 d 1);
  check_int64 "and_int64[2]" #0b0101L (geti64 d 2)

let test_or_int32 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_int32 ctx [| #0b1101l; #0b1010l; #0b1111l |] in
  let b = Nx_oxcaml.of_int32 ctx [| #0b1011l; #0b1100l; #0b0101l |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Int32 3 in
  Nx_oxcaml.op_or ~out a b;
  let d = Nx_oxcaml.data_array out in
  check_int32 "or_int32[0]" #0b1111l (geti32 d 0);
  (* 1101 | 1011 *)
  check_int32 "or_int32[1]" #0b1110l (geti32 d 1);
  (* 1010 | 1100 *)
  check_int32 "or_int32[2]" #0b1111l (geti32 d 2)
(* 1111 | 0101 *)

let test_or_int64 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_int64 ctx [| #0b1101L; #0b1010L; #0b1111L |] in
  let b = Nx_oxcaml.of_int64 ctx [| #0b1011L; #0b1100L; #0b0101L |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Int64 3 in
  Nx_oxcaml.op_or ~out a b;
  let d = Nx_oxcaml.data_array out in
  check_int64 "or_int64[0]" #0b1111L (geti64 d 0);
  check_int64 "or_int64[1]" #0b1110L (geti64 d 1);
  check_int64 "or_int64[2]" #0b1111L (geti64 d 2)

let test_xor_int32 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_int32 ctx [| #0b1101l; #0b1010l; #0b1111l |] in
  let b = Nx_oxcaml.of_int32 ctx [| #0b1011l; #0b1100l; #0b0101l |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Int32 3 in
  Nx_oxcaml.op_xor ~out a b;
  let d = Nx_oxcaml.data_array out in
  check_int32 "xor_int32[0]" #0b0110l (geti32 d 0);
  (* 1101 ^ 1011 *)
  check_int32 "xor_int32[1]" #0b0110l (geti32 d 1);
  (* 1010 ^ 1100 *)
  check_int32 "xor_int32[2]" #0b1010l (geti32 d 2)
(* 1111 ^ 0101 *)

let test_xor_int64 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_int64 ctx [| #0b1101L; #0b1010L; #0b1111L |] in
  let b = Nx_oxcaml.of_int64 ctx [| #0b1011L; #0b1100L; #0b0101L |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Int64 3 in
  Nx_oxcaml.op_xor ~out a b;
  let d = Nx_oxcaml.data_array out in
  check_int64 "xor_int64[0]" #0b0110L (geti64 d 0);
  check_int64 "xor_int64[1]" #0b0110L (geti64 d 1);
  check_int64 "xor_int64[2]" #0b1010L (geti64 d 2)

let test_neg_float64 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_float64 ctx [| #1.0; -#2.5; #0.0 |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Float64 3 in
  Nx_oxcaml.op_neg ~out a;
  let d = Nx_oxcaml.data_array out in
  check_float64 "neg_float64[0]" ~eps:1e-9 (-#1.0) (get64 d 0);
  check_float64 "neg_float64[1]" ~eps:1e-9 #2.5 (get64 d 1);
  check_float64 "neg_float64[2]" ~eps:1e-9 #0.0 (get64 d 2)

let test_neg_float32 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_float32 ctx [| #1.5s; -#3.0s; #0.0s |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Float32 3 in
  Nx_oxcaml.op_neg ~out a;
  let d = Nx_oxcaml.data_array out in
  check_float32 "neg_float32[0]" ~eps:1e-6 (-#1.5s) (get32 d 0);
  check_float32 "neg_float32[1]" ~eps:1e-6 #3.0s (get32 d 1);
  check_float32 "neg_float32[2]" ~eps:1e-6 #0.0s (get32 d 2)

let test_neg_int32 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_int32 ctx [| #1l; -#2l; #0l |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Int32 3 in
  Nx_oxcaml.op_neg ~out a;
  let d = Nx_oxcaml.data_array out in
  check_int32 "neg_int32[0]" (-#1l) (geti32 d 0);
  check_int32 "neg_int32[1]" #2l (geti32 d 1);
  check_int32 "neg_int32[2]" #0l (geti32 d 2)

let test_neg_int64 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_int64 ctx [| #10L; -#20L; #0L |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Int64 3 in
  Nx_oxcaml.op_neg ~out a;
  let d = Nx_oxcaml.data_array out in
  check_int64 "neg_int64[0]" (-#10L) (geti64 d 0);
  check_int64 "neg_int64[1]" #20L (geti64 d 1);
  check_int64 "neg_int64[2]" #0L (geti64 d 2)

let test_abs_float64 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_float64 ctx [| -#1.0; #2.5; -#0.0 |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Float64 3 in
  Nx_oxcaml.op_abs ~out a;
  let d = Nx_oxcaml.data_array out in
  check_float64 "abs_float64[0]" ~eps:1e-9 #1.0 (get64 d 0);
  check_float64 "abs_float64[1]" ~eps:1e-9 #2.5 (get64 d 1);
  check_float64 "abs_float64[2]" ~eps:1e-9 #0.0 (get64 d 2)

let test_abs_float32 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_float32 ctx [| -#1.5s; #3.0s; #0.0s |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Float32 3 in
  Nx_oxcaml.op_abs ~out a;
  let d = Nx_oxcaml.data_array out in
  check_float32 "abs_float32[0]" ~eps:1e-6 #1.5s (get32 d 0);
  check_float32 "abs_float32[1]" ~eps:1e-6 #3.0s (get32 d 1);
  check_float32 "abs_float32[2]" ~eps:1e-6 #0.0s (get32 d 2)

let test_abs_int32 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_int32 ctx [| -#1l; #2l; #0l |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Int32 3 in
  Nx_oxcaml.op_abs ~out a;
  let d = Nx_oxcaml.data_array out in
  check_int32 "abs_int32[0]" #1l (geti32 d 0);
  check_int32 "abs_int32[1]" #2l (geti32 d 1);
  check_int32 "abs_int32[2]" #0l (geti32 d 2)

let test_abs_int64 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_int64 ctx [| -#10L; #20L; #0L |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Int64 3 in
  Nx_oxcaml.op_abs ~out a;
  let d = Nx_oxcaml.data_array out in
  check_int64 "abs_int64[0]" #10L (geti64 d 0);
  check_int64 "abs_int64[1]" #20L (geti64 d 1);
  check_int64 "abs_int64[2]" #0L (geti64 d 2)

let test_log_float64 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_float64 ctx [| #1.0; #2.718281828459045; #10.0 |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Float64 3 in
  Nx_oxcaml.op_log ~out a;
  let d = Nx_oxcaml.data_array out in
  check_float64 "log_float64[0]" ~eps:1e-9 #0.0 (get64 d 0);
  check_float64 "log_float64[1]" ~eps:1e-9 #1.0 (get64 d 1);
  check_float64 "log_float64[2]" ~eps:1e-9 #2.302585092994046 (get64 d 2)

let test_log_float32 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_float32 ctx [| #1.0s; #2.7182817s; #10.0s |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Float32 3 in
  Nx_oxcaml.op_log ~out a;
  let d = Nx_oxcaml.data_array out in
  check_float32 "log_float32[0]" ~eps:1e-6 #0.0s (get32 d 0);
  check_float32 "log_float32[1]" ~eps:1e-6 #1.0s (get32 d 1);
  check_float32 "log_float32[2]" ~eps:1e-6 #2.3025851s (get32 d 2)

let test_exp_float64 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_float64 ctx [| #0.0; #1.0; #2.0 |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Float64 3 in
  Nx_oxcaml.op_exp ~out a;
  let d = Nx_oxcaml.data_array out in
  check_float64 "exp_float64[0]" ~eps:1e-9 #1.0 (get64 d 0);
  check_float64 "exp_float64[1]" ~eps:1e-9 #2.718281828459045 (get64 d 1);
  check_float64 "exp_float64[2]" ~eps:1e-9 #7.38905609893065 (get64 d 2)

let test_exp_float32 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_float32 ctx [| #0.0s; #1.0s; #2.0s |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Float32 3 in
  Nx_oxcaml.op_exp ~out a;
  let d = Nx_oxcaml.data_array out in
  check_float32 "exp_float32[0]" ~eps:1e-6 #1.0s (get32 d 0);
  check_float32 "exp_float32[1]" ~eps:1e-6 #2.7182817s (get32 d 1);
  check_float32 "exp_float32[2]" ~eps:1e-6 #7.389056s (get32 d 2)

let test_sin_float64 () =
  let ctx = Nx_oxcaml.create_context () in
  let a =
    Nx_oxcaml.of_float64 ctx [| #0.0; #1.5707963267948966; #3.141592653589793 |]
  in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Float64 3 in
  Nx_oxcaml.op_sin ~out a;
  let d = Nx_oxcaml.data_array out in
  check_float64 "sin_float64[0]" ~eps:1e-9 #0.0 (get64 d 0);
  check_float64 "sin_float64[1]" ~eps:1e-9 #1.0 (get64 d 1);
  check_float64 "sin_float64[2]" ~eps:1e-9 #0.0 (get64 d 2)

let test_sin_float32 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_float32 ctx [| #0.0s; #1.5707964s; #3.1415927s |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Float32 3 in
  Nx_oxcaml.op_sin ~out a;
  let d = Nx_oxcaml.data_array out in
  check_float32 "sin_float32[0]" ~eps:1e-6 #0.0s (get32 d 0);
  check_float32 "sin_float32[1]" ~eps:1e-6 #1.0s (get32 d 1);
  check_float32 "sin_float32[2]" ~eps:1e-6 #0.0s (get32 d 2)

let test_cos_float64 () =
  let ctx = Nx_oxcaml.create_context () in
  let a =
    Nx_oxcaml.of_float64 ctx [| #0.0; #1.5707963267948966; #3.141592653589793 |]
  in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Float64 3 in
  Nx_oxcaml.op_cos ~out a;
  let d = Nx_oxcaml.data_array out in
  check_float64 "cos_float64[0]" ~eps:1e-9 #1.0 (get64 d 0);
  check_float64 "cos_float64[1]" ~eps:1e-9 #0.0 (get64 d 1);
  check_float64 "cos_float64[2]" ~eps:1e-9 (-#1.0) (get64 d 2)

let test_cos_float32 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_float32 ctx [| #0.0s; #1.5707964s; #3.1415927s |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Float32 3 in
  Nx_oxcaml.op_cos ~out a;
  let d = Nx_oxcaml.data_array out in
  check_float32 "cos_float32[0]" ~eps:1e-6 #1.0s (get32 d 0);
  check_float32 "cos_float32[1]" ~eps:1e-6 #0.0s (get32 d 1);
  check_float32 "cos_float32[2]" ~eps:1e-6 (-#1.0s) (get32 d 2)

let test_sqrt_float64 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_float64 ctx [| #0.0; #4.0; #9.0 |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Float64 3 in
  Nx_oxcaml.op_sqrt ~out a;
  let d = Nx_oxcaml.data_array out in
  check_float64 "sqrt_float64[0]" ~eps:1e-9 #0.0 (get64 d 0);
  check_float64 "sqrt_float64[1]" ~eps:1e-9 #2.0 (get64 d 1);
  check_float64 "sqrt_float64[2]" ~eps:1e-9 #3.0 (get64 d 2)

let test_sqrt_float32 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_float32 ctx [| #0.0s; #4.0s; #9.0s |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Float32 3 in
  Nx_oxcaml.op_sqrt ~out a;
  let d = Nx_oxcaml.data_array out in
  check_float32 "sqrt_float32[0]" ~eps:1e-6 #0.0s (get32 d 0);
  check_float32 "sqrt_float32[1]" ~eps:1e-6 #2.0s (get32 d 1);
  check_float32 "sqrt_float32[2]" ~eps:1e-6 #3.0s (get32 d 2)

let test_recip_float64 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_float64 ctx [| #0.5; #0.25; #0.125 |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Float64 3 in
  Nx_oxcaml.op_recip ~out a;
  let d = Nx_oxcaml.data_array out in
  check_float64 "recip_float64[0]" ~eps:1e-9 #2.0 (get64 d 0);
  check_float64 "recip_float64[1]" ~eps:1e-9 #4.0 (get64 d 1);
  check_float64 "recip_float64[2]" ~eps:1e-9 #8.0 (get64 d 2)

let test_cmpeq_int64 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_int64 ctx [| #1L; #2L; #3L |] in
  let b = Nx_oxcaml.of_int64 ctx [| #1L; #2L; #4L |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Bool 3 in
  Nx_oxcaml.op_cmpeq ~out a b;
  let d = Nx_oxcaml.data_array out in
  check_bool "cmpeq_bool[0]" true (getbool d 0);
  check_bool "cmpeq_bool[1]" true (getbool d 1);
  check_bool "cmpeq_bool[2]" false (getbool d 2)

let test_cmpeq_float64 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_float64 ctx [| #1.0; #2.0; #3.0 |] in
  let b = Nx_oxcaml.of_float64 ctx [| #1.0; #2.0; #4.0 |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Bool 3 in
  Nx_oxcaml.op_cmpeq ~out a b;
  let d = Nx_oxcaml.data_array out in
  check_bool "cmpeq_bool[0]" true (getbool d 0);
  check_bool "cmpeq_bool[1]" true (getbool d 1);
  check_bool "cmpeq_bool[2]" false (getbool d 2)

  let test_cmpne_int64 () =
    let ctx = Nx_oxcaml.create_context () in
    let a = Nx_oxcaml.of_int64 ctx [| #1L; #2L; #3L |] in
    let b = Nx_oxcaml.of_int64 ctx [| #1L; #2L; #4L |] in
    let out = Nx_oxcaml.op_buffer ctx Dtype.Bool 3 in
    Nx_oxcaml.op_cmpne ~out a b;
    let d = Nx_oxcaml.data_array out in
    check_bool "cmpne_bool[0]" false (getbool d 0);
    check_bool "cmpne_bool[1]" false (getbool d 1);
    check_bool "cmpne_bool[2]" true (getbool d 2)
  
  let test_cmpne_float64 () =
    let ctx = Nx_oxcaml.create_context () in
    let a = Nx_oxcaml.of_float64 ctx [| #1.0; #2.0; #3.0 |] in
    let b = Nx_oxcaml.of_float64 ctx [| #1.0; #2.0; #4.0 |] in
    let out = Nx_oxcaml.op_buffer ctx Dtype.Bool 3 in
    Nx_oxcaml.op_cmpne ~out a b;
    let d = Nx_oxcaml.data_array out in
    check_bool "cmpne_bool[0]" false (getbool d 0);
    check_bool "cmpne_bool[1]" false (getbool d 1);
    check_bool "cmpne_bool[2]" true (getbool d 2)
  
  let test_cmplt_float64 () =
    let ctx = Nx_oxcaml.create_context () in
    let a = Nx_oxcaml.of_float64 ctx [| #0.5; #1.0; #2.0 |] in
    let b = Nx_oxcaml.of_float64 ctx [| #1.0; #1.0; #1.0 |] in
    let out = Nx_oxcaml.op_buffer ctx Dtype.Bool 3 in
    Nx_oxcaml.op_cmplt ~out a b;
    let d = Nx_oxcaml.data_array out in
    check_bool "cmplt_bool[0]" true (getbool d 0);
    check_bool "cmplt_bool[1]" false (getbool d 1);
    check_bool "cmplt_bool[2]" false (getbool d 2)

    let test_cmplt_int64 () =
      let ctx = Nx_oxcaml.create_context () in
      let a = Nx_oxcaml.of_int64 ctx [| #0L; #1L; #2L |] in
      let b = Nx_oxcaml.of_int64 ctx [| #1L; #1L; #1L |] in
      let out = Nx_oxcaml.op_buffer ctx Dtype.Bool 3 in
      Nx_oxcaml.op_cmplt ~out a b;
      let d = Nx_oxcaml.data_array out in
      check_bool "cmplt_bool[0]" true (getbool d 0);
      check_bool "cmplt_bool[1]" false (getbool d 1);
      check_bool "cmplt_bool[2]" false (getbool d 2)

let test_cmple_float64 () =
  let ctx = Nx_oxcaml.create_context () in
  let a = Nx_oxcaml.of_float64 ctx [| #0.5; #1.0; #2.0 |] in
  let b = Nx_oxcaml.of_float64 ctx [| #1.0; #1.0; #1.0 |] in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Bool 3 in
  Nx_oxcaml.op_cmple ~out a b;
  let d = Nx_oxcaml.data_array out in
  check_bool "cmple_bool[0]" true (getbool d 0);
  check_bool "cmple_bool[1]" true (getbool d 1);
  check_bool "cmple_bool[2]" false (getbool d 2)

  let test_cmple_int64 () =
    let ctx = Nx_oxcaml.create_context () in
    let a = Nx_oxcaml.of_int64 ctx [| #0L; #1L; #2L |] in
    let b = Nx_oxcaml.of_int64 ctx [| #1L; #1L; #1L |] in
    let out = Nx_oxcaml.op_buffer ctx Dtype.Bool 3 in
    Nx_oxcaml.op_cmple ~out a b;
    let d = Nx_oxcaml.data_array out in
    check_bool "cmple_bool[0]" true (getbool d 0);
    check_bool "cmple_bool[1]" true (getbool d 1);
    check_bool "cmple_bool[2]" false (getbool d 2)

let test_where_float64_basic () =
  let ctx = Nx_oxcaml.create_context () in
  let cond =
    Nx_oxcaml.of_bool
      ctx
      [| true; false; true; false |]
  in
  let if_true =
    Nx_oxcaml.of_float64
      ctx
      [| #1.0; #2.0; #3.0; #4.0 |]
  in
  let if_false =
    Nx_oxcaml.of_float64
      ctx
      [| #10.0; #20.0; #30.0; #40.0 |]
  in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Float64 4 in
  Nx_oxcaml.op_where ~out cond if_true if_false;
  let d = Nx_oxcaml.data_array out in
  check_float64 "where_basic[0]" ~eps:1e-9 #1.0 (get64 d 0);
  check_float64 "where_basic[1]" ~eps:1e-9 #20.0 (get64 d 1);
  check_float64 "where_basic[2]" ~eps:1e-9 #3.0 (get64 d 2);
  check_float64 "where_basic[3]" ~eps:1e-9 #40.0 (get64 d 3)

  let test_where_float32_basic () =
    let ctx = Nx_oxcaml.create_context () in
    let cond =
      Nx_oxcaml.of_bool ctx
        [| true; false; true; false |]
    in
    let if_true =
      Nx_oxcaml.of_float32 ctx
        [| #1.0s; #2.0s; #3.0s; #4.0s |]
    in
    let if_false =
      Nx_oxcaml.of_float32 ctx
        [| #10.0s; #20.0s; #30.0s; #40.0s |]
    in
    let out = Nx_oxcaml.op_buffer ctx Dtype.Float32 4 in
    Nx_oxcaml.op_where ~out cond if_true if_false;
    let d = Nx_oxcaml.data_array out in
    check_float32 "where_float32[0]" ~eps:1e-6 #1.0s (get32 d 0);
    check_float32 "where_float32[1]" ~eps:1e-6 #20.0s (get32 d 1);
    check_float32 "where_float32[2]" ~eps:1e-6 #3.0s (get32 d 2);
    check_float32 "where_float32[3]" ~eps:1e-6 #40.0s (get32 d 3)      

let test_where_int32_basic () =
  let ctx = Nx_oxcaml.create_context () in
  let cond =
    Nx_oxcaml.of_bool ctx
      [| true; false; true; false |]
  in
  let if_true =
    Nx_oxcaml.of_int32 ctx
      [| #1l; #2l; #3l; #4l |]
  in
  let if_false =
    Nx_oxcaml.of_int32 ctx
      [| #10l; #20l; #30l; #40l |]
  in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Int32 4 in
  Nx_oxcaml.op_where ~out cond if_true if_false;
  let d = Nx_oxcaml.data_array out in
  check_int32 "where_int32[0]" #1l (geti32 d 0);
  check_int32 "where_int32[1]" #20l (geti32 d 1);
  check_int32 "where_int32[2]" #3l (geti32 d 2);
  check_int32 "where_int32[3]" #40l (geti32 d 3)

let test_where_int32_zero_negative () =
  let ctx = Nx_oxcaml.create_context () in
  let cond =
    Nx_oxcaml.of_bool ctx
      [| true; false; false; true |]
  in
  let if_true =
    Nx_oxcaml.of_int32 ctx
      [| #0l; -#1l; -#2l; #3l |]
  in
  let if_false =
    Nx_oxcaml.of_int32 ctx
      [| #5l; #6l; #7l; #8l |]
  in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Int32 4 in
  Nx_oxcaml.op_where ~out cond if_true if_false;
  let d = Nx_oxcaml.data_array out in
  check_int32 "where_int32_zero_neg[0]" #0l (geti32 d 0);
  check_int32 "where_int32_zero_neg[1]" #6l (geti32 d 1);
  check_int32 "where_int32_zero_neg[2]" #7l (geti32 d 2);
  check_int32 "where_int32_zero_neg[3]" #3l (geti32 d 3)

let test_where_int64_zero_negative () =
  let ctx = Nx_oxcaml.create_context () in
  let cond =
    Nx_oxcaml.of_bool ctx
      [| true; false; false; true |]
  in
  let if_true =
    Nx_oxcaml.of_int64 ctx
      [| #0L; -#1L; -#2L; #3L |]
  in
  let if_false =
    Nx_oxcaml.of_int64 ctx
      [| #5L; #6L; #7L; #8L |]
  in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Int64 4 in
  Nx_oxcaml.op_where ~out cond if_true if_false;
  let d = Nx_oxcaml.data_array out in
  check_int64 "where_int64_zero_neg[0]" #0L (geti64 d 0);
  check_int64 "where_int64_zero_neg[1]" #6L (geti64 d 1);
  check_int64 "where_int64_zero_neg[2]" #7L (geti64 d 2);
  check_int64 "where_int64_zero_neg[3]" #3L (geti64 d 3)

let test_where_int8_basic () =
  let ctx = Nx_oxcaml.create_context () in
  let cond =
    Nx_oxcaml.of_bool ctx
      [| true; false; true; false |]
  in
  let if_true =
    Nx_oxcaml.of_int8 ctx
      [| #1s; #2s; #3s; #4s |]
  in
  let if_false =
    Nx_oxcaml.of_int8 ctx
      [| #10s; #20s; #30s; #40s |]
  in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Int8 4 in
  Nx_oxcaml.op_where ~out cond if_true if_false;
  let d = Nx_oxcaml.data_array out in
  check_int8 "where_int8[0]" #1s (geti8 d 0);
  check_int8 "where_int8[1]" #20s (geti8 d 1);
  check_int8 "where_int8[2]" #3s (geti8 d 2);
  check_int8 "where_int8[3]" #40s (geti8 d 3)

let test_where_int16_zero_negative () =
  let ctx = Nx_oxcaml.create_context () in
  let cond =
    Nx_oxcaml.of_bool ctx
      [| true; false; false; true |]
  in
  let if_true =
    Nx_oxcaml.of_int16 ctx
      [| #0S; -#1S; -#2S; #3S |]
  in
  let if_false =
    Nx_oxcaml.of_int16 ctx
      [| #5S; #6S; #7S; #8S |]
  in
  let out = Nx_oxcaml.op_buffer ctx Dtype.Int16 4 in
  Nx_oxcaml.op_where ~out cond if_true if_false;
  let d = Nx_oxcaml.data_array out in
  check_int16 "where_int16_zero_neg[0]" #0S (geti16 d 0);
  check_int16 "where_int16_zero_neg[1]" #6S (geti16 d 1);
  check_int16 "where_int16_zero_neg[2]" #7S (geti16 d 2);
  check_int16 "where_int16_zero_neg[3]" #3S (geti16 d 3)

  let test_matmul_2d () =
    let ctx = Nx_oxcaml.create_context () in
    let a = Nx_oxcaml.of_float64_multidim ctx [|#1.; #1.; #1.; #1.|] [|2; 2|] in
    let b = Nx_oxcaml.of_float64_multidim ctx [|#1.; #1.; #1.; #1.|] [|2; 2|] in
    let out = Nx_ox.empty ctx Dtype.Float64 [|2; 2|] in
    Nx_oxcaml.op_matmul ~out a b;
    let d = Nx_oxcaml.data_array out in
    check_float64 "mm[0,0]" ~eps:1e-9 #2.0 (get64 d 0);
    check_float64 "mm[1,1]" ~eps:1e-9 #2.0 (get64 d 1);
    check_float64 "mm[2,2]" ~eps:1e-9 #2.0 (get64 d 2);
    check_float64 "mm[0,1]" ~eps:1e-9 #2.0 (get64 d 3)
    
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
  test_mod_float64 ();
  test_mod_float32 ();
  test_mod_int64 ();
  test_mod_int32 ();
  test_min_float64 ();
  test_min_float32 ();
  (* test_min_int64 (); *)
  test_min_int32 ();
  (* test_max_float64 ();
  test_max_float32 (); *)
  (* test_max_int64 (); *)
  (* test_max_int32 (); *)
  test_pow_float64 ();
  test_pow_float32 ();
  test_xor_int64 ();
  test_xor_int32 ();
  test_or_int64 ();
  test_or_int32 ();
  test_and_int64 ();
  test_and_int32 ();
  test_neg_float64 ();
  test_neg_float32 ();
  test_neg_int64 ();
  test_neg_int32 ();
  test_abs_float64 ();
  test_abs_float32 ();
  test_abs_int64 ();
  test_abs_int32 ();
  test_log_float64 ();
  test_log_float32 ();
  test_exp_float64 ();
  test_exp_float32 ();
  test_sin_float64 ();
  test_sin_float32 ();
  test_cos_float64 ();
  test_cos_float32 ();
  test_sqrt_float64 ();
  test_sqrt_float32 ();
  test_cmpeq_int64 ();
  test_cmpeq_float64 ();
  test_cmpne_int64 ();
  test_cmpne_float64 ();
  test_cmplt_float64 ();
  test_cmplt_int64 ();
  test_cmple_float64 ();
  test_cmple_int64 ();
  test_recip_float64 ();
  test_where_float64_basic ();
  test_where_float32_basic ();
  test_where_int32_basic ();
  test_where_int32_zero_negative ();
  test_where_int64_zero_negative ();
  test_where_int8_basic ();
  test_where_int16_zero_negative ();
  test_matmul_2d ();
  Printf.printf "\nResults: %d passed, %d failed\n" !passed !failed;
  if !failed > 0 then exit 1
