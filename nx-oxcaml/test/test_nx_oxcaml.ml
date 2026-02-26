(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Unit tests for Nx_backend backend operations *)

module Dtype = Nx_core.Dtype
module View = Nx_core.View
module Nx_ox = Nx_core.Make_frontend (Nx_backend)
module Symbolic_shape = Nx_core.Symbolic_shape

let failed = ref 0
let passed = ref 0

let check name cond =
  if cond then incr passed
  else (
    incr failed;
    Printf.printf "FAIL: %s\n%!" name)

let check_float name ~eps exp act =
  check name (Float.abs (exp -. act) < eps)

let check_int32 name exp act = check name (Int32.equal exp act)
let check_int64 name exp act = check name (Int64.equal exp act)
let check_int name exp act = check name (exp = act)
let check_bool name exp act = check name (exp = act)

let numel v =
  match Symbolic_shape.eval_dim (View.numel v) with
  | Some n -> n
  | None -> failwith "symbolic numel not evaluable"

let test_buffer_float64 () =
  let t = Nx_ox.empty (Nx_backend.create_context ()) Dtype.Float64 [| 5 |] in
  check "buffer_float64: dtype" (Nx_backend.dtype t = Dtype.Float64);
  check "buffer_float64: size" (numel (Nx_backend.view t) = 5)

let test_buffer_float32 () =
  let t = Nx_ox.empty (Nx_backend.create_context ()) Dtype.Float32 [| 3 |] in
  check "buffer_float32: dtype" (Nx_backend.dtype t = Dtype.Float32);
  check "buffer_float32: size" (numel (Nx_backend.view t) = 3)

let test_buffer_int32 () =
  let t = Nx_ox.empty (Nx_backend.create_context ()) Dtype.Int32 [| 4 |] in
  check "buffer_int32: dtype" (Nx_backend.dtype t = Dtype.Int32);
  check "buffer_int32: size" (numel (Nx_backend.view t) = 4)

let test_buffer_int64 () =
  let t = Nx_ox.empty (Nx_backend.create_context ()) Dtype.Int64 [| 2 |] in
  check "buffer_int64: dtype" (Nx_backend.dtype t = Dtype.Int64);
  check "buffer_int64: size" (numel (Nx_backend.view t) = 2)

let test_add_float64 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Float64 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let b = Nx_ox.create ctx Dtype.Float64 [| 3 |] [| 10.0; 20.0; 30.0 |] in
  let out = Nx_ox.empty ctx Dtype.Float64 [| 3 |] in
  Nx_backend.add ~out a b;
  let d = Nx_ox.to_array out in
  check_float "add_float64[0]" ~eps:1e-9 11.0 d.(0);
  check_float "add_float64[1]" ~eps:1e-9 22.0 d.(1);
  check_float "add_float64[2]" ~eps:1e-9 33.0 d.(2)

let test_add_float32 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Float32 [| 3 |] [| 1.5; 2.5; 3.5 |] in
  let b = Nx_ox.create ctx Dtype.Float32 [| 3 |] [| 0.5; 0.5; 0.5 |] in
  let out = Nx_ox.empty ctx Dtype.Float32 [| 3 |] in
  Nx_backend.add ~out a b;
  let d = Nx_ox.to_array out in
  check_float "add_float32[0]" ~eps:1e-6 2.0 d.(0);
  check_float "add_float32[1]" ~eps:1e-6 3.0 d.(1);
  check_float "add_float32[2]" ~eps:1e-6 4.0 d.(2)

let test_add_int32 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Int32 [| 3 |] [| 1l; 2l; 3l |] in
  let b = Nx_ox.create ctx Dtype.Int32 [| 3 |] [| 100l; 200l; 300l |] in
  let out = Nx_ox.empty ctx Dtype.Int32 [| 3 |] in
  Nx_backend.add ~out a b;
  let d = Nx_ox.to_array out in
  check_int32 "add_int32[0]" 101l d.(0);
  check_int32 "add_int32[1]" 202l d.(1);
  check_int32 "add_int32[2]" 303l d.(2)

let test_add_int64 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Int64 [| 3 |] [| 1000L; 2000L; 3000L |] in
  let b = Nx_ox.create ctx Dtype.Int64 [| 3 |] [| 1L; 2L; 3L |] in
  let out = Nx_ox.empty ctx Dtype.Int64 [| 3 |] in
  Nx_backend.add ~out a b;
  let d = Nx_ox.to_array out in
  check_int64 "add_int64[0]" 1001L d.(0);
  check_int64 "add_int64[1]" 2002L d.(1);
  check_int64 "add_int64[2]" 3003L d.(2)

let test_sub_float64 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Float64 [| 3 |] [| 10.0; 20.0; 30.0 |] in
  let b = Nx_ox.create ctx Dtype.Float64 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let out = Nx_ox.empty ctx Dtype.Float64 [| 3 |] in
  Nx_backend.sub ~out a b;
  let d = Nx_ox.to_array out in
  check_float "sub_float64[0]" ~eps:1e-9 9.0 d.(0);
  check_float "sub_float64[1]" ~eps:1e-9 18.0 d.(1);
  check_float "sub_float64[2]" ~eps:1e-9 27.0 d.(2)

let test_sub_float32 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Float32 [| 3 |] [| 5.0; 10.0; 15.0 |] in
  let b = Nx_ox.create ctx Dtype.Float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let out = Nx_ox.empty ctx Dtype.Float32 [| 3 |] in
  Nx_backend.sub ~out a b;
  let d = Nx_ox.to_array out in
  check_float "sub_float32[0]" ~eps:1e-6 4.0 d.(0);
  check_float "sub_float32[1]" ~eps:1e-6 8.0 d.(1);
  check_float "sub_float32[2]" ~eps:1e-6 12.0 d.(2)

let test_sub_int32 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Int32 [| 3 |] [| 100l; 200l; 300l |] in
  let b = Nx_ox.create ctx Dtype.Int32 [| 3 |] [| 1l; 2l; 3l |] in
  let out = Nx_ox.empty ctx Dtype.Int32 [| 3 |] in
  Nx_backend.sub ~out a b;
  let d = Nx_ox.to_array out in
  check_int32 "sub_int32[0]" 99l d.(0);
  check_int32 "sub_int32[1]" 198l d.(1);
  check_int32 "sub_int32[2]" 297l d.(2)

let test_sub_int64 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Int64 [| 3 |] [| 1000L; 2000L; 3000L |] in
  let b = Nx_ox.create ctx Dtype.Int64 [| 3 |] [| 1L; 2L; 3L |] in
  let out = Nx_ox.empty ctx Dtype.Int64 [| 3 |] in
  Nx_backend.sub ~out a b;
  let d = Nx_ox.to_array out in
  check_int64 "sub_int64[0]" 999L d.(0);
  check_int64 "sub_int64[1]" 1998L d.(1);
  check_int64 "sub_int64[2]" 2997L d.(2)

let test_add_single_element () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Float64 [| 1 |] [| 42.0 |] in
  let b = Nx_ox.create ctx Dtype.Float64 [| 1 |] [| 8.0 |] in
  let out = Nx_ox.empty ctx Dtype.Float64 [| 1 |] in
  Nx_backend.add ~out a b;
  let d = Nx_ox.to_array out in
  check_float "add_single[0]" ~eps:1e-9 50.0 d.(0)

let test_add_negative_values () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Float64 [| 2 |] [| -5.0; 10.0 |] in
  let b = Nx_ox.create ctx Dtype.Float64 [| 2 |] [| -3.0; -7.0 |] in
  let out = Nx_ox.empty ctx Dtype.Float64 [| 2 |] in
  Nx_backend.add ~out a b;
  let d = Nx_ox.to_array out in
  check_float "add_neg[0]" ~eps:1e-9 (-8.0) d.(0);
  check_float "add_neg[1]" ~eps:1e-9 3.0 d.(1)

let test_sub_to_zero () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Int32 [| 2 |] [| 5l; 10l |] in
  let b = Nx_ox.create ctx Dtype.Int32 [| 2 |] [| 5l; 10l |] in
  let out = Nx_ox.empty ctx Dtype.Int32 [| 2 |] in
  Nx_backend.sub ~out a b;
  let d = Nx_ox.to_array out in
  check_int32 "sub_zero[0]" 0l d.(0);
  check_int32 "sub_zero[1]" 0l d.(1)

let test_in_place_add () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Float64 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let b = Nx_ox.create ctx Dtype.Float64 [| 3 |] [| 10.0; 20.0; 30.0 |] in
  Nx_backend.add ~out:a a b;
  let d = Nx_ox.to_array a in
  check_float "inplace_add[0]" ~eps:1e-9 11.0 d.(0);
  check_float "inplace_add[1]" ~eps:1e-9 22.0 d.(1);
  check_float "inplace_add[2]" ~eps:1e-9 33.0 d.(2)

let test_mul_float64 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Float64 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let b = Nx_ox.create ctx Dtype.Float64 [| 3 |] [| 10.0; 20.0; 30.0 |] in
  let out = Nx_ox.empty ctx Dtype.Float64 [| 3 |] in
  Nx_backend.mul ~out a b;
  let d = Nx_ox.to_array out in
  check_float "mul_float64[0]" ~eps:1e-9 10.0 d.(0);
  check_float "mul_float64[1]" ~eps:1e-9 40.0 d.(1);
  check_float "mul_float64[2]" ~eps:1e-9 90.0 d.(2)

let test_mul_float32 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Float32 [| 3 |] [| 1.5; 2.5; 3.5 |] in
  let b = Nx_ox.create ctx Dtype.Float32 [| 3 |] [| 0.5; 0.5; 2.0 |] in
  let out = Nx_ox.empty ctx Dtype.Float32 [| 3 |] in
  Nx_backend.mul ~out a b;
  let d = Nx_ox.to_array out in
  check_float "mul_float32[0]" ~eps:1e-6 0.75 d.(0);
  check_float "mul_float32[1]" ~eps:1e-6 1.25 d.(1);
  check_float "mul_float32[2]" ~eps:1e-6 7.0 d.(2)

let test_mul_int32 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Int32 [| 3 |] [| 1l; 2l; 3l |] in
  let b = Nx_ox.create ctx Dtype.Int32 [| 3 |] [| 100l; 200l; 300l |] in
  let out = Nx_ox.empty ctx Dtype.Int32 [| 3 |] in
  Nx_backend.mul ~out a b;
  let d = Nx_ox.to_array out in
  check_int32 "mul_int32[0]" 100l d.(0);
  check_int32 "mul_int32[1]" 400l d.(1);
  check_int32 "mul_int32[2]" 900l d.(2)

let test_mul_int64 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Int64 [| 3 |] [| 1000L; 2000L; 3000L |] in
  let b = Nx_ox.create ctx Dtype.Int64 [| 3 |] [| 1L; 2L; 3L |] in
  let out = Nx_ox.empty ctx Dtype.Int64 [| 3 |] in
  Nx_backend.mul ~out a b;
  let d = Nx_ox.to_array out in
  check_int64 "mul_int64[0]" 1000L d.(0);
  check_int64 "mul_int64[1]" 4000L d.(1);
  check_int64 "mul_int64[2]" 9000L d.(2)

let test_fdiv_float64 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Float64 [| 3 |] [| 1.0; 2.0; 2.0 |] in
  let b = Nx_ox.create ctx Dtype.Float64 [| 3 |] [| 0.0; 2.0; 3.0 |] in
  let out = Nx_ox.empty ctx Dtype.Float64 [| 3 |] in
  Nx_backend.div ~out b a;
  let d = Nx_ox.to_array out in
  check_float "fdiv_float64[0]" ~eps:1e-9 0.0 d.(0);
  check_float "fdiv_float64[1]" ~eps:1e-9 1.0 d.(1);
  check_float "fdiv_float64[2]" ~eps:1e-9 1.5 d.(2)

let test_fdiv_float32 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Float32 [| 3 |] [| 1.5; 2.5; 7.0 |] in
  let b = Nx_ox.create ctx Dtype.Float32 [| 3 |] [| 0.5; 0.5; 2.0 |] in
  let out = Nx_ox.empty ctx Dtype.Float32 [| 3 |] in
  Nx_backend.div ~out a b;
  let d = Nx_ox.to_array out in
  check_float "fdiv_float32[0]" ~eps:1e-6 3.0 d.(0);
  check_float "fdiv_float32[1]" ~eps:1e-6 5.0 d.(1);
  check_float "fdiv_float32[2]" ~eps:1e-6 3.5 d.(2)

let test_fdiv_int32 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Int32 [| 3 |] [| 1l; 2l; 3l |] in
  let b = Nx_ox.create ctx Dtype.Int32 [| 3 |] [| 100l; 1l; 2l |] in
  let out = Nx_ox.empty ctx Dtype.Int32 [| 3 |] in
  Nx_backend.div ~out a b;
  let d = Nx_ox.to_array out in
  check_int32 "fdiv_int32[0]" 0l d.(0);
  check_int32 "fdiv_int32[1]" 2l d.(1);
  check_int32 "fdiv_int32[2]" 1l d.(2)

let test_fdiv_int64 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Int64 [| 3 |] [| 1000L; 2000L; 3000L |] in
  let b = Nx_ox.create ctx Dtype.Int64 [| 3 |] [| 1L; 2L; 3L |] in
  let out = Nx_ox.empty ctx Dtype.Int64 [| 3 |] in
  Nx_backend.div ~out a b;
  let d = Nx_ox.to_array out in
  check_int64 "fdiv_int64[0]" 1000L d.(0);
  check_int64 "fdiv_int64[1]" 1000L d.(1);
  check_int64 "fdiv_int64[2]" 1000L d.(2)

let test_idiv_int32 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Int32 [| 3 |] [| 1l; 2l; 3l |] in
  let b = Nx_ox.create ctx Dtype.Int32 [| 3 |] [| 100l; 1l; 2l |] in
  let out = Nx_ox.empty ctx Dtype.Int32 [| 3 |] in
  Nx_backend.div ~out a b;
  let d = Nx_ox.to_array out in
  check_int32 "idiv_int32[0]" 0l d.(0);
  check_int32 "idiv_int32[1]" 2l d.(1);
  check_int32 "idiv_int32[2]" 1l d.(2)

let test_idiv_int64 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Int64 [| 3 |] [| 1000L; 2000L; 3000L |] in
  let b = Nx_ox.create ctx Dtype.Int64 [| 3 |] [| 1L; 2L; 3L |] in
  let out = Nx_ox.empty ctx Dtype.Int64 [| 3 |] in
  Nx_backend.div ~out a b;
  let d = Nx_ox.to_array out in
  check_int64 "idiv_int64[0]" 1000L d.(0);
  check_int64 "idiv_int64[1]" 1000L d.(1);
  check_int64 "idiv_int64[2]" 1000L d.(2)

let test_mod_float64 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Float64 [| 3 |] [| 1.0; 2.0; 2.0 |] in
  let b = Nx_ox.create ctx Dtype.Float64 [| 3 |] [| 0.0; 2.0; 3.0 |] in
  let out = Nx_ox.empty ctx Dtype.Float64 [| 3 |] in
  Nx_backend.mod_ ~out b a;
  let d = Nx_ox.to_array out in
  check_float "mod_float64[0]" ~eps:1e-9 0.0 d.(0);
  (* 0 mod 1 = 0 *)
  check_float "mod_float64[1]" ~eps:1e-9 0.0 d.(1);
  (* 2 mod 2 = 0 *)
  check_float "mod_float64[2]" ~eps:1e-9 1.0 d.(2)
(* 3 mod 2 = 1 *)

let test_mod_float32 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Float32 [| 3 |] [| 1.5; 2.5; 7.0 |] in
  let b = Nx_ox.create ctx Dtype.Float32 [| 3 |] [| 0.5; 0.5; 2.0 |] in
  let out = Nx_ox.empty ctx Dtype.Float32 [| 3 |] in
  Nx_backend.mod_ ~out a b;
  let d = Nx_ox.to_array out in
  check_float "mod_float32[0]" ~eps:1e-6 0.0 d.(0);
  (* 1.5 mod 0.5 = 0 *)
  check_float "mod_float32[1]" ~eps:1e-6 0.0 d.(1);
  (* 2.5 mod 0.5 = 0 *)
  check_float "mod_float32[2]" ~eps:1e-6 1.0 d.(2)
(* 7 mod 2 = 1 *)

let test_mod_int32 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Int32 [| 3 |] [| 1l; 2l; 3l |] in
  let b = Nx_ox.create ctx Dtype.Int32 [| 3 |] [| 100l; 1l; 2l |] in
  let out = Nx_ox.empty ctx Dtype.Int32 [| 3 |] in
  Nx_backend.mod_ ~out a b;
  let d = Nx_ox.to_array out in
  check_int32 "mod_int32[0]" 1l d.(0);
  (* 1 mod 100 = 1 *)
  check_int32 "mod_int32[1]" 0l d.(1);
  (* 2 mod 1 = 0 *)
  check_int32 "mod_int32[2]" 1l d.(2)
(* 3 mod 2 = 1 *)

let test_mod_int64 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Int64 [| 3 |] [| 1000L; 2000L; 3000L |] in
  let b = Nx_ox.create ctx Dtype.Int64 [| 3 |] [| 1L; 2L; 3L |] in
  let out = Nx_ox.empty ctx Dtype.Int64 [| 3 |] in
  Nx_backend.mod_ ~out a b;
  let d = Nx_ox.to_array out in
  check_int64 "mod_int64[0]" 0L d.(0);
  (* 1000 mod 1 = 0 *)
  check_int64 "mod_int64[1]" 0L d.(1);
  (* 2000 mod 2 = 0 *)
  check_int64 "mod_int64[2]" 0L d.(2)
(* 3000 mod 3 = 0 *)

let test_max_float64 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Float64 [| 3 |] [| 1.0; 2.0; 2.0 |] in
  let b = Nx_ox.create ctx Dtype.Float64 [| 3 |] [| 0.0; 2.5; 1.5 |] in
  let out = Nx_ox.empty ctx Dtype.Float64 [| 3 |] in
  Nx_backend.max ~out a b;
  let d = Nx_ox.to_array out in
  check_float "max_float64[0]" ~eps:1e-9 1.0 d.(0);
  check_float "max_float64[1]" ~eps:1e-9 2.5 d.(1);
  check_float "max_float64[2]" ~eps:1e-9 2.0 d.(2)

let test_max_float32 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Float32 [| 3 |] [| 1.5; 2.5; 7.0 |] in
  let b = Nx_ox.create ctx Dtype.Float32 [| 3 |] [| 2.0; 2.0; 3.0 |] in
  let out = Nx_ox.empty ctx Dtype.Float32 [| 3 |] in
  Nx_backend.max ~out a b;
  let d = Nx_ox.to_array out in
  check_float "max_float32[0]" ~eps:1e-6 2.0 d.(0);
  check_float "max_float32[1]" ~eps:1e-6 2.5 d.(1);
  check_float "max_float32[2]" ~eps:1e-6 7.0 d.(2)

let test_max_int32 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Int32 [| 3 |] [| 1l; 2l; 3l |] in
  let b = Nx_ox.create ctx Dtype.Int32 [| 3 |] [| 0l; 3l; 2l |] in
  let out = Nx_ox.empty ctx Dtype.Int32 [| 3 |] in
  Nx_backend.max ~out a b;
  let d = Nx_ox.to_array out in
  check_int32 "max_int32[0]" 1l d.(0);
  check_int32 "max_int32[1]" 3l d.(1);
  check_int32 "max_int32[2]" 3l d.(2)

let test_max_int64 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Int64 [| 3 |] [| 1000L; 2000L; 3000L |] in
  let b = Nx_ox.create ctx Dtype.Int64 [| 3 |] [| 1500L; 1500L; 1000L |] in
  let out = Nx_ox.empty ctx Dtype.Int64 [| 3 |] in
  Nx_backend.max ~out a b;
  let d = Nx_ox.to_array out in
  check_int64 "max_int64[0]" 1500L d.(0);
  check_int64 "max_int64[1]" 2000L d.(1);
  check_int64 "max_int64[2]" 3000L d.(2)

let test_min_float64 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Float64 [| 3 |] [| 1.0; 2.0; 2.0 |] in
  let b = Nx_ox.create ctx Dtype.Float64 [| 3 |] [| 0.0; 2.5; 1.5 |] in
  let out = Nx_ox.empty ctx Dtype.Float64 [| 3 |] in
  Nx_backend.min ~out a b;
  let d = Nx_ox.to_array out in
  check_float "min_float64[0]" ~eps:1e-9 0.0 d.(0);
  check_float "min_float64[1]" ~eps:1e-9 2.0 d.(1);
  check_float "min_float64[2]" ~eps:1e-9 1.5 d.(2)

let test_min_float32 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Float32 [| 3 |] [| 1.5; 2.5; 7.0 |] in
  let b = Nx_ox.create ctx Dtype.Float32 [| 3 |] [| 2.0; 2.0; 3.0 |] in
  let out = Nx_ox.empty ctx Dtype.Float32 [| 3 |] in
  Nx_backend.min ~out a b;
  let d = Nx_ox.to_array out in
  check_float "min_float32[0]" ~eps:1e-6 1.5 d.(0);
  check_float "min_float32[1]" ~eps:1e-6 2.0 d.(1);
  check_float "min_float32[2]" ~eps:1e-6 3.0 d.(2)

let test_min_int32 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Int32 [| 3 |] [| 1l; 2l; 3l |] in
  let b = Nx_ox.create ctx Dtype.Int32 [| 3 |] [| 0l; 3l; 2l |] in
  let out = Nx_ox.empty ctx Dtype.Int32 [| 3 |] in
  Nx_backend.min ~out a b;
  let d = Nx_ox.to_array out in
  check_int32 "min_int32[0]" 0l d.(0);
  check_int32 "min_int32[1]" 2l d.(1);
  check_int32 "min_int32[2]" 2l d.(2)

let test_min_int64 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Int64 [| 3 |] [| 1000L; 2000L; 3000L |] in
  let b = Nx_ox.create ctx Dtype.Int64 [| 3 |] [| 1500L; 1500L; 1000L |] in
  let out = Nx_ox.empty ctx Dtype.Int64 [| 3 |] in
  Nx_backend.min ~out a b;
  let d = Nx_ox.to_array out in
  check_int64 "min_int64[0]" 1000L d.(0);
  check_int64 "min_int64[1]" 1500L d.(1);
  check_int64 "min_int64[2]" 1000L d.(2)

let test_pow_float64 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Float64 [| 3 |] [| 2.0; 3.0; 4.0 |] in
  let b = Nx_ox.create ctx Dtype.Float64 [| 3 |] [| 3.0; 2.0; 0.5 |] in
  let out = Nx_ox.empty ctx Dtype.Float64 [| 3 |] in
  Nx_backend.pow ~out a b;
  let d = Nx_ox.to_array out in
  check_float "pow_float64[0]" ~eps:1e-9 8.0 d.(0);
  (* 2^3 *)
  check_float "pow_float64[1]" ~eps:1e-9 9.0 d.(1);
  (* 3^2 *)
  check_float "pow_float64[2]" ~eps:1e-9 2.0 d.(2)
(* 4^0.5 *)

let test_pow_float32 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Float32 [| 3 |] [| 2.0; 5.0; 9.0 |] in
  let b = Nx_ox.create ctx Dtype.Float32 [| 3 |] [| 3.0; 1.0; 0.5 |] in
  let out = Nx_ox.empty ctx Dtype.Float32 [| 3 |] in
  Nx_backend.pow ~out a b;
  let d = Nx_ox.to_array out in
  check_float "pow_float32[0]" ~eps:1e-6 8.0 d.(0);
  (* 2^3 *)
  check_float "pow_float32[1]" ~eps:1e-6 5.0 d.(1);
  (* 5^1 *)
  check_float "pow_float32[2]" ~eps:1e-6 3.0 d.(2)
(* 9^0.5 *)

let test_and_int32 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Int32 [| 3 |] [| 0b1101l; 0b1010l; 0b1111l |] in
  let b = Nx_ox.create ctx Dtype.Int32 [| 3 |] [| 0b1011l; 0b1100l; 0b0101l |] in
  let out = Nx_ox.empty ctx Dtype.Int32 [| 3 |] in
  Nx_backend.and_ ~out a b;
  let d = Nx_ox.to_array out in
  check_int32 "and_int32[0]" 0b1001l d.(0);
  (* 1101 & 1011 *)
  check_int32 "and_int32[1]" 0b1000l d.(1);
  (* 1010 & 1100 *)
  check_int32 "and_int32[2]" 0b0101l d.(2)
(* 1111 & 0101 *)

let test_and_int64 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Int64 [| 3 |] [| 0b1101L; 0b1010L; 0b1111L |] in
  let b = Nx_ox.create ctx Dtype.Int64 [| 3 |] [| 0b1011L; 0b1100L; 0b0101L |] in
  let out = Nx_ox.empty ctx Dtype.Int64 [| 3 |] in
  Nx_backend.and_ ~out a b;
  let d = Nx_ox.to_array out in
  check_int64 "and_int64[0]" 0b1001L d.(0);
  check_int64 "and_int64[1]" 0b1000L d.(1);
  check_int64 "and_int64[2]" 0b0101L d.(2)

let test_or_int32 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Int32 [| 3 |] [| 0b1101l; 0b1010l; 0b1111l |] in
  let b = Nx_ox.create ctx Dtype.Int32 [| 3 |] [| 0b1011l; 0b1100l; 0b0101l |] in
  let out = Nx_ox.empty ctx Dtype.Int32 [| 3 |] in
  Nx_backend.or_ ~out a b;
  let d = Nx_ox.to_array out in
  check_int32 "or_int32[0]" 0b1111l d.(0);
  (* 1101 | 1011 *)
  check_int32 "or_int32[1]" 0b1110l d.(1);
  (* 1010 | 1100 *)
  check_int32 "or_int32[2]" 0b1111l d.(2)
(* 1111 | 0101 *)

let test_or_int64 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Int64 [| 3 |] [| 0b1101L; 0b1010L; 0b1111L |] in
  let b = Nx_ox.create ctx Dtype.Int64 [| 3 |] [| 0b1011L; 0b1100L; 0b0101L |] in
  let out = Nx_ox.empty ctx Dtype.Int64 [| 3 |] in
  Nx_backend.or_ ~out a b;
  let d = Nx_ox.to_array out in
  check_int64 "or_int64[0]" 0b1111L d.(0);
  check_int64 "or_int64[1]" 0b1110L d.(1);
  check_int64 "or_int64[2]" 0b1111L d.(2)

let test_xor_int32 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Int32 [| 3 |] [| 0b1101l; 0b1010l; 0b1111l |] in
  let b = Nx_ox.create ctx Dtype.Int32 [| 3 |] [| 0b1011l; 0b1100l; 0b0101l |] in
  let out = Nx_ox.empty ctx Dtype.Int32 [| 3 |] in
  Nx_backend.xor ~out a b;
  let d = Nx_ox.to_array out in
  check_int32 "xor_int32[0]" 0b0110l d.(0);
  (* 1101 ^ 1011 *)
  check_int32 "xor_int32[1]" 0b0110l d.(1);
  (* 1010 ^ 1100 *)
  check_int32 "xor_int32[2]" 0b1010l d.(2)
(* 1111 ^ 0101 *)

let test_xor_int64 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Int64 [| 3 |] [| 0b1101L; 0b1010L; 0b1111L |] in
  let b = Nx_ox.create ctx Dtype.Int64 [| 3 |] [| 0b1011L; 0b1100L; 0b0101L |] in
  let out = Nx_ox.empty ctx Dtype.Int64 [| 3 |] in
  Nx_backend.xor ~out a b;
  let d = Nx_ox.to_array out in
  check_int64 "xor_int64[0]" 0b0110L d.(0);
  check_int64 "xor_int64[1]" 0b0110L d.(1);
  check_int64 "xor_int64[2]" 0b1010L d.(2)

let test_neg_float64 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Float64 [| 3 |] [| 1.0; -2.5; 0.0 |] in
  let out = Nx_ox.empty ctx Dtype.Float64 [| 3 |] in
  Nx_backend.neg ~out a;
  let d = Nx_ox.to_array out in
  check_float "neg_float64[0]" ~eps:1e-9 (-1.0) d.(0);
  check_float "neg_float64[1]" ~eps:1e-9 2.5 d.(1);
  check_float "neg_float64[2]" ~eps:1e-9 0.0 d.(2)

let test_neg_float32 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Float32 [| 3 |] [| 1.5; -3.0; 0.0 |] in
  let out = Nx_ox.empty ctx Dtype.Float32 [| 3 |] in
  Nx_backend.neg ~out a;
  let d = Nx_ox.to_array out in
  check_float "neg_float32[0]" ~eps:1e-6 (-1.5) d.(0);
  check_float "neg_float32[1]" ~eps:1e-6 3.0 d.(1);
  check_float "neg_float32[2]" ~eps:1e-6 0.0 d.(2)

let test_neg_int32 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Int32 [| 3 |] [| 1l; (-2l); 0l |] in
  let out = Nx_ox.empty ctx Dtype.Int32 [| 3 |] in
  Nx_backend.neg ~out a;
  let d = Nx_ox.to_array out in
  check_int32 "neg_int32[0]" (-1l) d.(0);
  check_int32 "neg_int32[1]" 2l d.(1);
  check_int32 "neg_int32[2]" 0l d.(2)

let test_neg_int64 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Int64 [| 3 |] [| 10L; (-20L); 0L |] in
  let out = Nx_ox.empty ctx Dtype.Int64 [| 3 |] in
  Nx_backend.neg ~out a;
  let d = Nx_ox.to_array out in
  check_int64 "neg_int64[0]" (-10L) d.(0);
  check_int64 "neg_int64[1]" 20L d.(1);
  check_int64 "neg_int64[2]" 0L d.(2)

let test_abs_float64 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Float64 [| 3 |] [| -1.0; 2.5; -0.0 |] in
  let out = Nx_ox.empty ctx Dtype.Float64 [| 3 |] in
  Nx_backend.abs ~out a;
  let d = Nx_ox.to_array out in
  check_float "abs_float64[0]" ~eps:1e-9 1.0 d.(0);
  check_float "abs_float64[1]" ~eps:1e-9 2.5 d.(1);
  check_float "abs_float64[2]" ~eps:1e-9 0.0 d.(2)

let test_abs_float32 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Float32 [| 3 |] [| -1.5; 3.0; 0.0 |] in
  let out = Nx_ox.empty ctx Dtype.Float32 [| 3 |] in
  Nx_backend.abs ~out a;
  let d = Nx_ox.to_array out in
  check_float "abs_float32[0]" ~eps:1e-6 1.5 d.(0);
  check_float "abs_float32[1]" ~eps:1e-6 3.0 d.(1);
  check_float "abs_float32[2]" ~eps:1e-6 0.0 d.(2)

let test_abs_int32 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Int32 [| 3 |] [| (-1l); 2l; 0l |] in
  let out = Nx_ox.empty ctx Dtype.Int32 [| 3 |] in
  Nx_backend.abs ~out a;
  let d = Nx_ox.to_array out in
  check_int32 "abs_int32[0]" 1l d.(0);
  check_int32 "abs_int32[1]" 2l d.(1);
  check_int32 "abs_int32[2]" 0l d.(2)

let test_abs_int64 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Int64 [| 3 |] [| (-10L); 20L; 0L |] in
  let out = Nx_ox.empty ctx Dtype.Int64 [| 3 |] in
  Nx_backend.abs ~out a;
  let d = Nx_ox.to_array out in
  check_int64 "abs_int64[0]" 10L d.(0);
  check_int64 "abs_int64[1]" 20L d.(1);
  check_int64 "abs_int64[2]" 0L d.(2)

let test_log_float64 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Float64 [| 3 |] [| 1.0; 2.718281828459045; 10.0 |] in
  let out = Nx_ox.empty ctx Dtype.Float64 [| 3 |] in
  Nx_backend.log ~out a;
  let d = Nx_ox.to_array out in
  check_float "log_float64[0]" ~eps:1e-9 0.0 d.(0);
  check_float "log_float64[1]" ~eps:1e-9 1.0 d.(1);
  check_float "log_float64[2]" ~eps:1e-9 2.302585092994046 d.(2)

let test_log_float32 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Float32 [| 3 |] [| 1.0; 2.7182817; 10.0 |] in
  let out = Nx_ox.empty ctx Dtype.Float32 [| 3 |] in
  Nx_backend.log ~out a;
  let d = Nx_ox.to_array out in
  check_float "log_float32[0]" ~eps:1e-6 0.0 d.(0);
  check_float "log_float32[1]" ~eps:1e-6 1.0 d.(1);
  check_float "log_float32[2]" ~eps:1e-6 2.3025851 d.(2)

let test_exp_float64 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Float64 [| 3 |] [| 0.0; 1.0; 2.0 |] in
  let out = Nx_ox.empty ctx Dtype.Float64 [| 3 |] in
  Nx_backend.exp ~out a;
  let d = Nx_ox.to_array out in
  check_float "exp_float64[0]" ~eps:1e-9 1.0 d.(0);
  check_float "exp_float64[1]" ~eps:1e-9 2.718281828459045 d.(1);
  check_float "exp_float64[2]" ~eps:1e-9 7.38905609893065 d.(2)

let test_exp_float32 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Float32 [| 3 |] [| 0.0; 1.0; 2.0 |] in
  let out = Nx_ox.empty ctx Dtype.Float32 [| 3 |] in
  Nx_backend.exp ~out a;
  let d = Nx_ox.to_array out in
  check_float "exp_float32[0]" ~eps:1e-6 1.0 d.(0);
  check_float "exp_float32[1]" ~eps:1e-6 2.7182817 d.(1);
  check_float "exp_float32[2]" ~eps:1e-6 7.389056 d.(2)

let test_sin_float64 () =
  let ctx = Nx_backend.create_context () in
  let a =
    Nx_ox.create ctx Dtype.Float64 [| 3 |]
      [| 0.0; 1.5707963267948966; 3.141592653589793 |]
  in
  let out = Nx_ox.empty ctx Dtype.Float64 [| 3 |] in
  Nx_backend.sin ~out a;
  let d = Nx_ox.to_array out in
  check_float "sin_float64[0]" ~eps:1e-9 0.0 d.(0);
  check_float "sin_float64[1]" ~eps:1e-9 1.0 d.(1);
  check_float "sin_float64[2]" ~eps:1e-9 0.0 d.(2)

let test_sin_float32 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Float32 [| 3 |] [| 0.0; 1.5707964; 3.1415927 |] in
  let out = Nx_ox.empty ctx Dtype.Float32 [| 3 |] in
  Nx_backend.sin ~out a;
  let d = Nx_ox.to_array out in
  check_float "sin_float32[0]" ~eps:1e-6 0.0 d.(0);
  check_float "sin_float32[1]" ~eps:1e-6 1.0 d.(1);
  check_float "sin_float32[2]" ~eps:1e-6 0.0 d.(2)

let test_cos_float64 () =
  let ctx = Nx_backend.create_context () in
  let a =
    Nx_ox.create ctx Dtype.Float64 [| 3 |]
      [| 0.0; 1.5707963267948966; 3.141592653589793 |]
  in
  let out = Nx_ox.empty ctx Dtype.Float64 [| 3 |] in
  Nx_backend.cos ~out a;
  let d = Nx_ox.to_array out in
  check_float "cos_float64[0]" ~eps:1e-9 1.0 d.(0);
  check_float "cos_float64[1]" ~eps:1e-9 0.0 d.(1);
  check_float "cos_float64[2]" ~eps:1e-9 (-1.0) d.(2)

let test_cos_float32 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Float32 [| 3 |] [| 0.0; 1.5707964; 3.1415927 |] in
  let out = Nx_ox.empty ctx Dtype.Float32 [| 3 |] in
  Nx_backend.cos ~out a;
  let d = Nx_ox.to_array out in
  check_float "cos_float32[0]" ~eps:1e-6 1.0 d.(0);
  check_float "cos_float32[1]" ~eps:1e-6 0.0 d.(1);
  check_float "cos_float32[2]" ~eps:1e-6 (-1.0) d.(2)

let test_sqrt_float64 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Float64 [| 3 |] [| 0.0; 4.0; 9.0 |] in
  let out = Nx_ox.empty ctx Dtype.Float64 [| 3 |] in
  Nx_backend.sqrt ~out a;
  let d = Nx_ox.to_array out in
  check_float "sqrt_float64[0]" ~eps:1e-9 0.0 d.(0);
  check_float "sqrt_float64[1]" ~eps:1e-9 2.0 d.(1);
  check_float "sqrt_float64[2]" ~eps:1e-9 3.0 d.(2)

let test_sqrt_float32 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Float32 [| 3 |] [| 0.0; 4.0; 9.0 |] in
  let out = Nx_ox.empty ctx Dtype.Float32 [| 3 |] in
  Nx_backend.sqrt ~out a;
  let d = Nx_ox.to_array out in
  check_float "sqrt_float32[0]" ~eps:1e-6 0.0 d.(0);
  check_float "sqrt_float32[1]" ~eps:1e-6 2.0 d.(1);
  check_float "sqrt_float32[2]" ~eps:1e-6 3.0 d.(2)

let test_recip_float64 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Float64 [| 3 |] [| 0.5; 0.25; 0.125 |] in
  let out = Nx_ox.empty ctx Dtype.Float64 [| 3 |] in
  Nx_backend.recip ~out a;
  let d = Nx_ox.to_array out in
  check_float "recip_float64[0]" ~eps:1e-9 2.0 d.(0);
  check_float "recip_float64[1]" ~eps:1e-9 4.0 d.(1);
  check_float "recip_float64[2]" ~eps:1e-9 8.0 d.(2)

let test_cmpeq_int64 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Int64 [| 3 |] [| 1L; 2L; 3L |] in
  let b = Nx_ox.create ctx Dtype.Int64 [| 3 |] [| 1L; 2L; 4L |] in
  let out = Nx_ox.empty ctx Dtype.Bool [| 3 |] in
  Nx_backend.cmpeq ~out a b;
  let d = Nx_ox.to_array out in
  check_bool "cmpeq_bool[0]" true d.(0);
  check_bool "cmpeq_bool[1]" true d.(1);
  check_bool "cmpeq_bool[2]" false d.(2)

let test_cmpeq_float64 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Float64 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let b = Nx_ox.create ctx Dtype.Float64 [| 3 |] [| 1.0; 2.0; 4.0 |] in
  let out = Nx_ox.empty ctx Dtype.Bool [| 3 |] in
  Nx_backend.cmpeq ~out a b;
  let d = Nx_ox.to_array out in
  check_bool "cmpeq_bool[0]" true d.(0);
  check_bool "cmpeq_bool[1]" true d.(1);
  check_bool "cmpeq_bool[2]" false d.(2)

let test_cmpne_int64 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Int64 [| 3 |] [| 1L; 2L; 3L |] in
  let b = Nx_ox.create ctx Dtype.Int64 [| 3 |] [| 1L; 2L; 4L |] in
  let out = Nx_ox.empty ctx Dtype.Bool [| 3 |] in
  Nx_backend.cmpne ~out a b;
  let d = Nx_ox.to_array out in
  check_bool "cmpne_bool[0]" false d.(0);
  check_bool "cmpne_bool[1]" false d.(1);
  check_bool "cmpne_bool[2]" true d.(2)

let test_cmpne_float64 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Float64 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let b = Nx_ox.create ctx Dtype.Float64 [| 3 |] [| 1.0; 2.0; 4.0 |] in
  let out = Nx_ox.empty ctx Dtype.Bool [| 3 |] in
  Nx_backend.cmpne ~out a b;
  let d = Nx_ox.to_array out in
  check_bool "cmpne_bool[0]" false d.(0);
  check_bool "cmpne_bool[1]" false d.(1);
  check_bool "cmpne_bool[2]" true d.(2)

let test_cmplt_float64 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Float64 [| 3 |] [| 0.5; 1.0; 2.0 |] in
  let b = Nx_ox.create ctx Dtype.Float64 [| 3 |] [| 1.0; 1.0; 1.0 |] in
  let out = Nx_ox.empty ctx Dtype.Bool [| 3 |] in
  Nx_backend.cmplt ~out a b;
  let d = Nx_ox.to_array out in
  check_bool "cmplt_bool[0]" true d.(0);
  check_bool "cmplt_bool[1]" false d.(1);
  check_bool "cmplt_bool[2]" false d.(2)

let test_cmplt_int64 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Int64 [| 3 |] [| 0L; 1L; 2L |] in
  let b = Nx_ox.create ctx Dtype.Int64 [| 3 |] [| 1L; 1L; 1L |] in
  let out = Nx_ox.empty ctx Dtype.Bool [| 3 |] in
  Nx_backend.cmplt ~out a b;
  let d = Nx_ox.to_array out in
  check_bool "cmplt_bool[0]" true d.(0);
  check_bool "cmplt_bool[1]" false d.(1);
  check_bool "cmplt_bool[2]" false d.(2)

let test_cmple_float64 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Float64 [| 3 |] [| 0.5; 1.0; 2.0 |] in
  let b = Nx_ox.create ctx Dtype.Float64 [| 3 |] [| 1.0; 1.0; 1.0 |] in
  let out = Nx_ox.empty ctx Dtype.Bool [| 3 |] in
  Nx_backend.cmple ~out a b;
  let d = Nx_ox.to_array out in
  check_bool "cmple_bool[0]" true d.(0);
  check_bool "cmple_bool[1]" true d.(1);
  check_bool "cmple_bool[2]" false d.(2)

let test_cmple_int64 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Int64 [| 3 |] [| 0L; 1L; 2L |] in
  let b = Nx_ox.create ctx Dtype.Int64 [| 3 |] [| 1L; 1L; 1L |] in
  let out = Nx_ox.empty ctx Dtype.Bool [| 3 |] in
  Nx_backend.cmple ~out a b;
  let d = Nx_ox.to_array out in
  check_bool "cmple_bool[0]" true d.(0);
  check_bool "cmple_bool[1]" true d.(1);
  check_bool "cmple_bool[2]" false d.(2)

let test_where_float64_basic () =
  let ctx = Nx_backend.create_context () in
  let cond = Nx_ox.create ctx Dtype.Bool [| 4 |] [| true; false; true; false |] in
  let if_true = Nx_ox.create ctx Dtype.Float64 [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let if_false =
    Nx_ox.create ctx Dtype.Float64 [| 4 |] [| 10.0; 20.0; 30.0; 40.0 |]
  in
  let out = Nx_ox.empty ctx Dtype.Float64 [| 4 |] in
  Nx_backend.where ~out cond if_true if_false;
  let d = Nx_ox.to_array out in
  check_float "where_basic[0]" ~eps:1e-9 1.0 d.(0);
  check_float "where_basic[1]" ~eps:1e-9 20.0 d.(1);
  check_float "where_basic[2]" ~eps:1e-9 3.0 d.(2);
  check_float "where_basic[3]" ~eps:1e-9 40.0 d.(3)

let test_where_float32_basic () =
  let ctx = Nx_backend.create_context () in
  let cond = Nx_ox.create ctx Dtype.Bool [| 4 |] [| true; false; true; false |] in
  let if_true = Nx_ox.create ctx Dtype.Float32 [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let if_false =
    Nx_ox.create ctx Dtype.Float32 [| 4 |] [| 10.0; 20.0; 30.0; 40.0 |]
  in
  let out = Nx_ox.empty ctx Dtype.Float32 [| 4 |] in
  Nx_backend.where ~out cond if_true if_false;
  let d = Nx_ox.to_array out in
  check_float "where_float32[0]" ~eps:1e-6 1.0 d.(0);
  check_float "where_float32[1]" ~eps:1e-6 20.0 d.(1);
  check_float "where_float32[2]" ~eps:1e-6 3.0 d.(2);
  check_float "where_float32[3]" ~eps:1e-6 40.0 d.(3)

let test_where_int32_basic () =
  let ctx = Nx_backend.create_context () in
  let cond = Nx_ox.create ctx Dtype.Bool [| 4 |] [| true; false; true; false |] in
  let if_true = Nx_ox.create ctx Dtype.Int32 [| 4 |] [| 1l; 2l; 3l; 4l |] in
  let if_false = Nx_ox.create ctx Dtype.Int32 [| 4 |] [| 10l; 20l; 30l; 40l |] in
  let out = Nx_ox.empty ctx Dtype.Int32 [| 4 |] in
  Nx_backend.where ~out cond if_true if_false;
  let d = Nx_ox.to_array out in
  check_int32 "where_int32[0]" 1l d.(0);
  check_int32 "where_int32[1]" 20l d.(1);
  check_int32 "where_int32[2]" 3l d.(2);
  check_int32 "where_int32[3]" 40l d.(3)

let test_where_int32_zero_negative () =
  let ctx = Nx_backend.create_context () in
  let cond = Nx_ox.create ctx Dtype.Bool [| 4 |] [| true; false; false; true |] in
  let if_true = Nx_ox.create ctx Dtype.Int32 [| 4 |] [| 0l; (-1l); (-2l); 3l |] in
  let if_false = Nx_ox.create ctx Dtype.Int32 [| 4 |] [| 5l; 6l; 7l; 8l |] in
  let out = Nx_ox.empty ctx Dtype.Int32 [| 4 |] in
  Nx_backend.where ~out cond if_true if_false;
  let d = Nx_ox.to_array out in
  check_int32 "where_int32_zero_neg[0]" 0l d.(0);
  check_int32 "where_int32_zero_neg[1]" 6l d.(1);
  check_int32 "where_int32_zero_neg[2]" 7l d.(2);
  check_int32 "where_int32_zero_neg[3]" 3l d.(3)

let test_where_int64_zero_negative () =
  let ctx = Nx_backend.create_context () in
  let cond = Nx_ox.create ctx Dtype.Bool [| 4 |] [| true; false; false; true |] in
  let if_true = Nx_ox.create ctx Dtype.Int64 [| 4 |] [| 0L; (-1L); (-2L); 3L |] in
  let if_false = Nx_ox.create ctx Dtype.Int64 [| 4 |] [| 5L; 6L; 7L; 8L |] in
  let out = Nx_ox.empty ctx Dtype.Int64 [| 4 |] in
  Nx_backend.where ~out cond if_true if_false;
  let d = Nx_ox.to_array out in
  check_int64 "where_int64_zero_neg[0]" 0L d.(0);
  check_int64 "where_int64_zero_neg[1]" 6L d.(1);
  check_int64 "where_int64_zero_neg[2]" 7L d.(2);
  check_int64 "where_int64_zero_neg[3]" 3L d.(3)

let test_where_int8_basic () =
  let ctx = Nx_backend.create_context () in
  let cond = Nx_ox.create ctx Dtype.Bool [| 4 |] [| true; false; true; false |] in
  let if_true = Nx_ox.create ctx Dtype.Int8 [| 4 |] [| 1; 2; 3; 4 |] in
  let if_false = Nx_ox.create ctx Dtype.Int8 [| 4 |] [| 10; 20; 30; 40 |] in
  let out = Nx_ox.empty ctx Dtype.Int8 [| 4 |] in
  Nx_backend.where ~out cond if_true if_false;
  let d = Nx_ox.to_array out in
  check_int "where_int8[0]" 1 d.(0);
  check_int "where_int8[1]" 20 d.(1);
  check_int "where_int8[2]" 3 d.(2);
  check_int "where_int8[3]" 40 d.(3)

let test_where_int16_zero_negative () =
  let ctx = Nx_backend.create_context () in
  let cond = Nx_ox.create ctx Dtype.Bool [| 4 |] [| true; false; false; true |] in
  let if_true = Nx_ox.create ctx Dtype.Int16 [| 4 |] [| 0; (-1); (-2); 3 |] in
  let if_false = Nx_ox.create ctx Dtype.Int16 [| 4 |] [| 5; 6; 7; 8 |] in
  let out = Nx_ox.empty ctx Dtype.Int16 [| 4 |] in
  Nx_backend.where ~out cond if_true if_false;
  let d = Nx_ox.to_array out in
  check_int "where_int16_zero_neg[0]" 0 d.(0);
  check_int "where_int16_zero_neg[1]" 6 d.(1);
  check_int "where_int16_zero_neg[2]" 7 d.(2);
  check_int "where_int16_zero_neg[3]" 3 d.(3)

let test_matmul_2d () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Float64 [| 2; 2 |] [| 1.; 1.; 1.; 1. |] in
  let b = Nx_ox.create ctx Dtype.Float64 [| 2; 2 |] [| 1.; 1.; 1.; 1. |] in
  let out = Nx_ox.empty ctx Dtype.Float64 [| 2; 2 |] in
  Nx_backend.matmul ~out a b;
  let d = Nx_ox.to_array out in
  check_float "mm[0,0]" ~eps:1e-9 2.0 d.(0);
  check_float "mm[1,1]" ~eps:1e-9 2.0 d.(1);
  check_float "mm[2,2]" ~eps:1e-9 2.0 d.(2);
  check_float "mm[0,1]" ~eps:1e-9 2.0 d.(3)

let test_matmul_identity () =
  let ctx = Nx_backend.create_context () in
  let a =
    Nx_ox.create ctx Dtype.Float64 [| 2; 3 |]
      [| 1.; 2.; 3.; 4.; 5.; 6. |]
  in
  let id =
    Nx_ox.create ctx Dtype.Float64 [| 3; 3 |]
      [| 1.; 0.; 0.; 0.; 1.; 0.; 0.; 0.; 1. |]
  in
  let out = Nx_ox.empty ctx Dtype.Float64 [| 2; 3 |] in
  Nx_backend.matmul ~out a id;
  let d = Nx_ox.to_array out in
  check_float "id@0" ~eps:1e-9 1. d.(0);
  check_float "id@1" ~eps:1e-9 2. d.(1);
  check_float "id@2" ~eps:1e-9 3. d.(2);
  check_float "id@3" ~eps:1e-9 4. d.(3);
  check_float "id@4" ~eps:1e-9 5. d.(4);
  check_float "id@5" ~eps:1e-9 6. d.(5)

let test_matmul_rectangular () =
  let ctx = Nx_backend.create_context () in
  let a =
    Nx_ox.create ctx Dtype.Float64 [| 2; 3 |]
      [| 1.; 2.; 3.; 4.; 5.; 6. |]
  in
  let b =
    Nx_ox.create ctx Dtype.Float64 [| 3; 4 |]
      [| 7.; 8.; 9.; 10.; 11.; 12.; 13.; 14.; 15.; 16.; 17.; 18. |]
  in
  let out = Nx_ox.empty ctx Dtype.Float64 [| 2; 4 |] in
  Nx_backend.matmul ~out a b;
  let d = Nx_ox.to_array out in
  (* row 0 *)
  check_float "rect[0,0]" ~eps:1e-9 74. d.(0);
  check_float "rect[0,1]" ~eps:1e-9 80. d.(1);
  check_float "rect[0,2]" ~eps:1e-9 86. d.(2);
  check_float "rect[0,3]" ~eps:1e-9 92. d.(3);
  (* row 1 *)
  check_float "rect[1,0]" ~eps:1e-9 173. d.(4);
  check_float "rect[1,1]" ~eps:1e-9 188. d.(5);
  check_float "rect[1,2]" ~eps:1e-9 203. d.(6);
  check_float "rect[1,3]" ~eps:1e-9 218. d.(7)

let test_matmul_batched () =
  let ctx = Nx_backend.create_context () in
  let a =
    Nx_ox.create ctx Dtype.Float64 [| 2; 2; 2 |]
      [| 1.; 0.; 0.; 1.; 2.; 0.; 0.; 2. |]
  in
  let b =
    Nx_ox.create ctx Dtype.Float64 [| 2; 2; 2 |]
      [| 3.; 4.; 5.; 6.; 1.; 1.; 1.; 1. |]
  in
  let out = Nx_ox.empty ctx Dtype.Float64 [| 2; 2; 2 |] in
  Nx_backend.matmul ~out a b;
  let d = Nx_ox.to_array out in
  (* batch 0 *)
  check_float "bat0[0,0]" ~eps:1e-9 3. d.(0);
  check_float "bat0[0,1]" ~eps:1e-9 4. d.(1);
  check_float "bat0[1,0]" ~eps:1e-9 5. d.(2);
  check_float "bat0[1,1]" ~eps:1e-9 6. d.(3);
  (* batch 1 *)
  check_float "bat1[0,0]" ~eps:1e-9 2. d.(4);
  check_float "bat1[0,1]" ~eps:1e-9 2. d.(5);
  check_float "bat1[1,0]" ~eps:1e-9 2. d.(6);
  check_float "bat1[1,1]" ~eps:1e-9 2. d.(7)

let test_matmul_dot_product () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Float64 [| 1; 3 |] [| 1.; 2.; 3. |] in
  let b = Nx_ox.create ctx Dtype.Float64 [| 3; 1 |] [| 4.; 5.; 6. |] in
  let out = Nx_ox.empty ctx Dtype.Float64 [| 1; 1 |] in
  Nx_backend.matmul ~out a b;
  let d = Nx_ox.to_array out in
  check_float "dot" ~eps:1e-9 32. d.(0)

let test_matmul_rectangular_f32 () =
  let ctx = Nx_backend.create_context () in
  let a =
    Nx_ox.create ctx Dtype.Float32 [| 2; 3 |]
      [| 1.; 2.; 3.; 4.; 5.; 6. |]
  in
  let b =
    Nx_ox.create ctx Dtype.Float32 [| 3; 4 |]
      [| 7.; 8.; 9.; 10.; 11.; 12.; 13.; 14.; 15.; 16.; 17.; 18. |]
  in
  let out = Nx_ox.empty ctx Dtype.Float32 [| 2; 4 |] in
  Nx_backend.matmul ~out a b;
  let d = Nx_ox.to_array out in
  (* row 0 *)
  check_float "rect[0,0]" ~eps:1e-9 74. d.(0);
  check_float "rect[0,1]" ~eps:1e-9 80. d.(1);
  check_float "rect[0,2]" ~eps:1e-9 86. d.(2);
  check_float "rect[0,3]" ~eps:1e-9 92. d.(3);
  (* row 1 *)
  check_float "rect[1,0]" ~eps:1e-9 173. d.(4);
  check_float "rect[1,1]" ~eps:1e-9 188. d.(5);
  check_float "rect[1,2]" ~eps:1e-9 203. d.(6);
  check_float "rect[1,3]" ~eps:1e-9 218. d.(7)

let test_matmul_batched_f32 () =
  let ctx = Nx_backend.create_context () in
  let a =
    Nx_ox.create ctx Dtype.Float32 [| 2; 2; 2 |]
      [| 1.; 0.; 0.; 1.; 2.; 0.; 0.; 2. |]
  in
  let b =
    Nx_ox.create ctx Dtype.Float32 [| 2; 2; 2 |]
      [| 3.; 4.; 5.; 6.; 1.; 1.; 1.; 1. |]
  in
  let out = Nx_ox.empty ctx Dtype.Float32 [| 2; 2; 2 |] in
  Nx_backend.matmul ~out a b;
  let d = Nx_ox.to_array out in
  (* batch 0 *)
  check_float "bat0[0,0]" ~eps:1e-9 3. d.(0);
  check_float "bat0[0,1]" ~eps:1e-9 4. d.(1);
  check_float "bat0[1,0]" ~eps:1e-9 5. d.(2);
  check_float "bat0[1,1]" ~eps:1e-9 6. d.(3);
  (* batch 1 *)
  check_float "bat1[0,0]" ~eps:1e-9 2. d.(4);
  check_float "bat1[0,1]" ~eps:1e-9 2. d.(5);
  check_float "bat1[1,0]" ~eps:1e-9 2. d.(6);
  check_float "bat1[1,1]" ~eps:1e-9 2. d.(7)

let test_pad_int32_1d () =
  let ctx = Nx_backend.create_context () in
  let x = Nx_ox.create ctx Dtype.Int32 [| 3 |] [| 10l; 20l; 30l |] in
  let y = Nx_backend.pad x [| (2, 1) |] (-7l) in
  check "pad_int32_1d: dtype" (Nx_backend.dtype y = Dtype.Int32);
  check "pad_int32_1d: size" (numel (Nx_backend.view y) = 6);
  let d = Nx_ox.to_array y in
  check_int32 "pad_int32_1d[0]" (-7l) d.(0);
  check_int32 "pad_int32_1d[1]" (-7l) d.(1);
  check_int32 "pad_int32_1d[2]" 10l d.(2);
  check_int32 "pad_int32_1d[3]" 20l d.(3);
  check_int32 "pad_int32_1d[4]" 30l d.(4);
  check_int32 "pad_int32_1d[5]" (-7l) d.(5)

let test_pad_float64_2d () =
  let ctx = Nx_backend.create_context () in
  let x = Nx_ox.create ctx Dtype.Float64 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let y = Nx_backend.pad x [| (1, 2); (2, 1) |] (-1.0) in
  let shape_y =
    match Symbolic_shape.eval (View.shape (Nx_backend.view y)) with
    | Some s -> s
    | None -> failwith "shape not evaluable"
  in
  check "pad_float64_2d: shape0" (shape_y.(0) = 5);
  check "pad_float64_2d: shape1" (shape_y.(1) = 5);
  let d = Nx_ox.to_array y in
  check_float "pad_float64_2d[0,0]" ~eps:1e-9 (-1.0) d.(0);
  check_float "pad_float64_2d[1,2]" ~eps:1e-9 1.0 d.(7);
  check_float "pad_float64_2d[1,3]" ~eps:1e-9 2.0 d.(8);
  check_float "pad_float64_2d[2,2]" ~eps:1e-9 3.0 d.(12);
  check_float "pad_float64_2d[2,3]" ~eps:1e-9 4.0 d.(13);
  check_float "pad_float64_2d[4,4]" ~eps:1e-9 (-1.0) d.(24)

let test_pad_float64_permuted_view () =
  let ctx = Nx_backend.create_context () in
  let base =
    Nx_ox.create ctx Dtype.Float64 [| 2; 3 |]
      [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]
  in
  let x = Nx_backend.permute base [| 1; 0 |] in
  let y = Nx_backend.pad x [| (1, 0); (0, 1) |] 0.0 in
  let shape_y =
    match Symbolic_shape.eval (View.shape (Nx_backend.view y)) with
    | Some s -> s
    | None -> failwith "shape not evaluable"
  in
  check "pad_float64_perm: shape0" (shape_y.(0) = 4);
  check "pad_float64_perm: shape1" (shape_y.(1) = 3);
  let d = Nx_ox.to_array y in
  check_float "pad_float64_perm[0,0]" ~eps:1e-9 0.0 d.(0);
  check_float "pad_float64_perm[1,0]" ~eps:1e-9 1.0 d.(3);
  check_float "pad_float64_perm[1,1]" ~eps:1e-9 4.0 d.(4);
  check_float "pad_float64_perm[2,0]" ~eps:1e-9 2.0 d.(6);
  check_float "pad_float64_perm[2,1]" ~eps:1e-9 5.0 d.(7);
  check_float "pad_float64_perm[3,0]" ~eps:1e-9 3.0 d.(9);
  check_float "pad_float64_perm[3,1]" ~eps:1e-9 6.0 d.(10);
  check_float "pad_float64_perm[3,2]" ~eps:1e-9 0.0 d.(11)

let test_shrink_int32_view () =
  let ctx = Nx_backend.create_context () in
  let x = Nx_ox.create ctx Dtype.Int32 [| 2; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l |] in
  let y = Nx_backend.shrink x [| (0, 2); (1, 3) |] in
  let zeros = Nx_ox.create ctx Dtype.Int32 [| 2; 2 |] [| 0l; 0l; 0l; 0l |] in
  let out = Nx_ox.empty ctx Dtype.Int32 [| 2; 2 |] in
  Nx_backend.add ~out y zeros;
  let d = Nx_ox.to_array out in
  check_int32 "shrink_int32_view[0]" 2l d.(0);
  check_int32 "shrink_int32_view[1]" 3l d.(1);
  check_int32 "shrink_int32_view[2]" 5l d.(2);
  check_int32 "shrink_int32_view[3]" 6l d.(3)

let test_flip_int32_view () =
  let ctx = Nx_backend.create_context () in
  let x = Nx_ox.create ctx Dtype.Int32 [| 2; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l |] in
  let y = Nx_backend.flip x [| true; false |] in
  let zeros =
    Nx_ox.create ctx Dtype.Int32 [| 2; 3 |] [| 0l; 0l; 0l; 0l; 0l; 0l |]
  in
  let out = Nx_ox.empty ctx Dtype.Int32 [| 2; 3 |] in
  Nx_backend.add ~out y zeros;
  let d = Nx_ox.to_array out in
  check_int32 "flip_int32_view[0]" 4l d.(0);
  check_int32 "flip_int32_view[1]" 5l d.(1);
  check_int32 "flip_int32_view[2]" 6l d.(2);
  check_int32 "flip_int32_view[3]" 1l d.(3);
  check_int32 "flip_int32_view[4]" 2l d.(4);
  check_int32 "flip_int32_view[5]" 3l d.(5)

let test_cat_int32_axis1 () =
  let ctx = Nx_backend.create_context () in
  let a = Nx_ox.create ctx Dtype.Int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |] in
  let b = Nx_ox.create ctx Dtype.Int32 [| 2; 2 |] [| 5l; 6l; 7l; 8l |] in
  let out = Nx_ox.empty ctx Dtype.Int32 [| 2; 4 |] in
  Nx_backend.cat ~out [ a; b ] ~axis:1;
  let d = Nx_ox.to_array out in
  check_int32 "cat_int32_axis1[0]" 1l d.(0);
  check_int32 "cat_int32_axis1[1]" 2l d.(1);
  check_int32 "cat_int32_axis1[2]" 5l d.(2);
  check_int32 "cat_int32_axis1[3]" 6l d.(3);
  check_int32 "cat_int32_axis1[4]" 3l d.(4);
  check_int32 "cat_int32_axis1[5]" 4l d.(5);
  check_int32 "cat_int32_axis1[6]" 7l d.(6);
  check_int32 "cat_int32_axis1[7]" 8l d.(7)

let test_gather_int32_axis1 () =
  let ctx = Nx_backend.create_context () in
  let data =
    Nx_ox.create ctx  Dtype.Int32
    [| 2; 4 |]
      [| 10l; 11l; 12l; 13l; 20l; 21l; 22l; 23l |]
  in
  let indices =
    Nx_ox.create ctx Dtype.Int32 [| 2; 3 |] [| 3l; 1l; 0l; 0l; 2l; 2l |] 
  in
  let out = Nx_ox.empty ctx Dtype.Int32 [| 2; 3 |] in
  Nx_backend.gather ~out:out data indices ~axis:1;
  let d = Nx_ox.to_array out in
  check_int32 "gather_int32_axis1[0]" 13l d.(0);
  check_int32 "gather_int32_axis1[1]" 11l d.(1);
  check_int32 "gather_int32_axis1[2]" 10l d.(2);
  check_int32 "gather_int32_axis1[3]" 20l d.(3);
  check_int32 "gather_int32_axis1[4]" 22l d.(4);
  check_int32 "gather_int32_axis1[5]" 22l d.(5)

let test_gather_float32_axis0_contiguous () =
  let ctx = Nx_backend.create_context () in
  let data = Nx_ox.create ctx Dtype.Float32 [| 8 |] [| 0.5; 1.5; 2.5; 3.5; 4.5; 5.5; 6.5; 7.5 |] in
  let indices = Nx_ox.create ctx Dtype.Int32 [| 8 |] [| 7l; 0l; 6l; 1l; 5l; 2l; 4l; 3l |] in
  let out = Nx_ox.empty ctx Dtype.Float32 [| 8 |] in
  Nx_backend.gather ~out data indices ~axis:0;
  let d = Nx_ox.to_array out in
  check_float "gather_float32_axis0_contiguous[0]" ~eps:1e-6 7.5 d.(0);
  check_float "gather_float32_axis0_contiguous[1]" ~eps:1e-6 0.5 d.(1);
  check_float "gather_float32_axis0_contiguous[2]" ~eps:1e-6 6.5 d.(2);
  check_float "gather_float32_axis0_contiguous[3]" ~eps:1e-6 1.5 d.(3);
  check_float "gather_float32_axis0_contiguous[4]" ~eps:1e-6 5.5 d.(4);
  check_float "gather_float32_axis0_contiguous[5]" ~eps:1e-6 2.5 d.(5);
  check_float "gather_float32_axis0_contiguous[6]" ~eps:1e-6 4.5 d.(6);
  check_float "gather_float32_axis0_contiguous[7]" ~eps:1e-6 3.5 d.(7)

let test_scatter_int32_set_axis1 () =
  let ctx = Nx_backend.create_context () in
  let template =
    Nx_ox.create ctx Dtype.Int32
    [| 2; 4 |]
      [| 0l; 0l; 0l; 0l; 0l; 0l; 0l; 0l |]
  in
  let indices =
    Nx_ox.create ctx Dtype.Int32 [| 2; 3 |] [| 3l; 1l; 0l; 0l; 2l; 2l |]
  in
  let updates =
    Nx_ox.create ctx Dtype.Int32 [| 2; 3 |] [| 9l; 8l; 7l; 6l; 5l; 4l |]
  in
  let y = Nx_backend.scatter template ~indices ~updates ~axis:1 in
  let d = Nx_ox.to_array y in
  check_int32 "scatter_int32_set_axis1[0]" 7l d.(0);
  check_int32 "scatter_int32_set_axis1[1]" 8l d.(1);
  check_int32 "scatter_int32_set_axis1[2]" 0l d.(2);
  check_int32 "scatter_int32_set_axis1[3]" 9l d.(3);
  check_int32 "scatter_int32_set_axis1[4]" 6l d.(4);
  check_int32 "scatter_int32_set_axis1[5]" 0l d.(5);
  check_int32 "scatter_int32_set_axis1[6]" 4l d.(6);
  check_int32 "scatter_int32_set_axis1[7]" 0l d.(7)

let test_scatter_int32_add_axis1 () =
  let ctx = Nx_backend.create_context () in
  let template =
    Nx_ox.create ctx Dtype.Int32
      [| 2; 4 |]
      [| 100l; 100l; 100l; 100l;
          100l; 100l; 100l; 100l |]
  in
  let indices =
    Nx_ox.create ctx Dtype.Int32
      [| 2; 3 |]
      [| 3l; 1l; 0l;
          0l; 2l; 2l |]
  in
  let updates =
    Nx_ox.create ctx Dtype.Int32
      [| 2; 3 |]
      [| 9l; 8l; 7l;
          6l; 5l; 4l |]
  in
  let y = Nx_backend.scatter ~mode:`Add template ~indices ~updates ~axis:1 in
  let d = Nx_ox.to_array y in

  check_int32 "scatter_int32_add_axis1[0]" 107l d.(0);
  check_int32 "scatter_int32_add_axis1[1]" 108l d.(1);
  check_int32 "scatter_int32_add_axis1[2]" 100l d.(2);
  check_int32 "scatter_int32_add_axis1[3]" 109l d.(3);
  check_int32 "scatter_int32_add_axis1[4]" 106l d.(4);
  check_int32 "scatter_int32_add_axis1[5]" 100l d.(5);
  check_int32 "scatter_int32_add_axis1[6]" 109l d.(6);
  check_int32 "scatter_int32_add_axis1[7]" 100l d.(7)

(* Gather: float64 1D contiguous — exercises the Float64x2 SIMD path *)
let test_gather_float64_axis0_contiguous () =
  let ctx = Nx_backend.create_context () in
  let data =
    Nx_ox.create ctx Dtype.Float64 [| 6 |]
      [| 10.0; 20.0; 30.0; 40.0; 50.0; 60.0 |]
  in
  let indices =
    Nx_ox.create ctx Dtype.Int32 [| 6 |] [| 5l; 3l; 1l; 0l; 4l; 2l |]
  in
  let out = Nx_ox.empty ctx Dtype.Float64 [| 6 |] in
  Nx_backend.gather ~out data indices ~axis:0;
  let d = Nx_ox.to_array out in
  check_float "gather_f64_contiguous[0]" ~eps:1e-12 60.0 d.(0);
  check_float "gather_f64_contiguous[1]" ~eps:1e-12 40.0 d.(1);
  check_float "gather_f64_contiguous[2]" ~eps:1e-12 20.0 d.(2);
  check_float "gather_f64_contiguous[3]" ~eps:1e-12 10.0 d.(3);
  check_float "gather_f64_contiguous[4]" ~eps:1e-12 50.0 d.(4);
  check_float "gather_f64_contiguous[5]" ~eps:1e-12 30.0 d.(5)

(* Gather: axis=0 with 2D tensor — general multi-dim path *)
let test_gather_float64_axis0_2d () =
  let ctx = Nx_backend.create_context () in
  (* 3x2 data, gather rows 2, 0 *)
  let data =
    Nx_ox.create ctx Dtype.Float64 [| 3; 2 |]
      [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]
  in
  let indices =
    Nx_ox.create ctx Dtype.Int32 [| 2; 2 |] [| 2l; 0l; 1l; 2l |]
  in
  let out = Nx_ox.empty ctx Dtype.Float64 [| 2; 2 |] in
  Nx_backend.gather ~out data indices ~axis:0;
  let d = Nx_ox.to_array out in
  check_float "gather_f64_axis0_2d[0]" ~eps:1e-12 5.0 d.(0);
  check_float "gather_f64_axis0_2d[1]" ~eps:1e-12 2.0 d.(1);
  check_float "gather_f64_axis0_2d[2]" ~eps:1e-12 3.0 d.(2);
  check_float "gather_f64_axis0_2d[3]" ~eps:1e-12 6.0 d.(3)

(* Gather: int32 1D contiguous — exercises the Int32x4 SIMD path *)
let test_gather_int32_axis0_contiguous () =
  let ctx = Nx_backend.create_context () in
  let data =
    Nx_ox.create ctx Dtype.Int32 [| 8 |]
      [| 10l; 20l; 30l; 40l; 50l; 60l; 70l; 80l |]
  in
  let indices =
    Nx_ox.create ctx Dtype.Int32 [| 8 |]
      [| 7l; 5l; 3l; 1l; 6l; 4l; 2l; 0l |]
  in
  let out = Nx_ox.empty ctx Dtype.Int32 [| 8 |] in
  Nx_backend.gather ~out data indices ~axis:0;
  let d = Nx_ox.to_array out in
  check_int32 "gather_i32_contiguous[0]" 80l d.(0);
  check_int32 "gather_i32_contiguous[1]" 60l d.(1);
  check_int32 "gather_i32_contiguous[2]" 40l d.(2);
  check_int32 "gather_i32_contiguous[3]" 20l d.(3);
  check_int32 "gather_i32_contiguous[4]" 70l d.(4);
  check_int32 "gather_i32_contiguous[5]" 50l d.(5);
  check_int32 "gather_i32_contiguous[6]" 30l d.(6);
  check_int32 "gather_i32_contiguous[7]" 10l d.(7)

(* Gather: int64 1D contiguous — exercises the Int64x2 SIMD path *)
let test_gather_int64_axis0_contiguous () =
  let ctx = Nx_backend.create_context () in
  let data =
    Nx_ox.create ctx Dtype.Int64 [| 6 |]
      [| 100L; 200L; 300L; 400L; 500L; 600L |]
  in
  let indices =
    Nx_ox.create ctx Dtype.Int32 [| 6 |] [| 4l; 2l; 0l; 5l; 3l; 1l |]
  in
  let out = Nx_ox.empty ctx Dtype.Int64 [| 6 |] in
  Nx_backend.gather ~out data indices ~axis:0;
  let d = Nx_ox.to_array out in
  check_int64 "gather_i64_contiguous[0]" 500L d.(0);
  check_int64 "gather_i64_contiguous[1]" 300L d.(1);
  check_int64 "gather_i64_contiguous[2]" 100L d.(2);
  check_int64 "gather_i64_contiguous[3]" 600L d.(3);
  check_int64 "gather_i64_contiguous[4]" 400L d.(4);
  check_int64 "gather_i64_contiguous[5]" 200L d.(5)

(* Gather: single element *)
let test_gather_single_element () =
  let ctx = Nx_backend.create_context () in
  let data = Nx_ox.create ctx Dtype.Float64 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let indices = Nx_ox.create ctx Dtype.Int32 [| 1 |] [| 2l |] in
  let out = Nx_ox.empty ctx Dtype.Float64 [| 1 |] in
  Nx_backend.gather ~out data indices ~axis:0;
  let d = Nx_ox.to_array out in
  check_float "gather_single[0]" ~eps:1e-12 3.0 d.(0)

(* Gather: negative axis *)
let test_gather_negative_axis () =
  let ctx = Nx_backend.create_context () in
  let data =
    Nx_ox.create ctx Dtype.Int32 [| 2; 4 |]
      [| 10l; 11l; 12l; 13l; 20l; 21l; 22l; 23l |]
  in
  let indices =
    Nx_ox.create ctx Dtype.Int32 [| 2; 2 |] [| 3l; 0l; 1l; 2l |]
  in
  let out = Nx_ox.empty ctx Dtype.Int32 [| 2; 2 |] in
  Nx_backend.gather ~out data indices ~axis:(-1);
  let d = Nx_ox.to_array out in
  check_int32 "gather_neg_axis[0]" 13l d.(0);
  check_int32 "gather_neg_axis[1]" 10l d.(1);
  check_int32 "gather_neg_axis[2]" 21l d.(2);
  check_int32 "gather_neg_axis[3]" 22l d.(3)

(* Scatter: float64 set *)
let test_scatter_float64_set () =
  let ctx = Nx_backend.create_context () in
  let template =
    Nx_ox.create ctx Dtype.Float64 [| 5 |] [| 0.0; 0.0; 0.0; 0.0; 0.0 |]
  in
  let indices = Nx_ox.create ctx Dtype.Int32 [| 3 |] [| 4l; 1l; 0l |] in
  let updates =
    Nx_ox.create ctx Dtype.Float64 [| 3 |] [| 9.0; 8.0; 7.0 |]
  in
  let y = Nx_backend.scatter template ~indices ~updates ~axis:0 in
  let d = Nx_ox.to_array y in
  check_float "scatter_f64_set[0]" ~eps:1e-12 7.0 d.(0);
  check_float "scatter_f64_set[1]" ~eps:1e-12 8.0 d.(1);
  check_float "scatter_f64_set[2]" ~eps:1e-12 0.0 d.(2);
  check_float "scatter_f64_set[3]" ~eps:1e-12 0.0 d.(3);
  check_float "scatter_f64_set[4]" ~eps:1e-12 9.0 d.(4)

(* Scatter: duplicate indices with Add mode — accumulation *)
let test_scatter_float64_add_duplicates () =
  let ctx = Nx_backend.create_context () in
  let template =
    Nx_ox.create ctx Dtype.Float64 [| 4 |] [| 0.0; 0.0; 0.0; 0.0 |]
  in
  let indices =
    Nx_ox.create ctx Dtype.Int32 [| 5 |] [| 0l; 1l; 0l; 2l; 0l |]
  in
  let updates =
    Nx_ox.create ctx Dtype.Float64 [| 5 |] [| 1.0; 2.0; 3.0; 4.0; 5.0 |]
  in
  let y = Nx_backend.scatter ~mode:`Add template ~indices ~updates ~axis:0 in
  let d = Nx_ox.to_array y in
  check_float "scatter_f64_add_dup[0]" ~eps:1e-12 9.0 d.(0);
  check_float "scatter_f64_add_dup[1]" ~eps:1e-12 2.0 d.(1);
  check_float "scatter_f64_add_dup[2]" ~eps:1e-12 4.0 d.(2);
  check_float "scatter_f64_add_dup[3]" ~eps:1e-12 0.0 d.(3)

(* Scatter: bool dtype *)
let test_scatter_bool_set () =
  let ctx = Nx_backend.create_context () in
  let template =
    Nx_ox.create ctx Dtype.Bool [| 4 |] [| false; false; false; false |]
  in
  let indices = Nx_ox.create ctx Dtype.Int32 [| 2 |] [| 1l; 3l |] in
  let updates = Nx_ox.create ctx Dtype.Bool [| 2 |] [| true; true |] in
  let y = Nx_backend.scatter template ~indices ~updates ~axis:0 in
  let d = Nx_ox.to_array y in
  check_bool "scatter_bool_set[0]" false d.(0);
  check_bool "scatter_bool_set[1]" true d.(1);
  check_bool "scatter_bool_set[2]" false d.(2);
  check_bool "scatter_bool_set[3]" true d.(3)

(* Scatter: preserves template values for untouched indices *)
let test_scatter_preserves_template () =
  let ctx = Nx_backend.create_context () in
  let template =
    Nx_ox.create ctx Dtype.Float64 [| 4 |] [| 10.0; 20.0; 30.0; 40.0 |]
  in
  let indices = Nx_ox.create ctx Dtype.Int32 [| 1 |] [| 2l |] in
  let updates = Nx_ox.create ctx Dtype.Float64 [| 1 |] [| 99.0 |] in
  let y = Nx_backend.scatter template ~indices ~updates ~axis:0 in
  let d = Nx_ox.to_array y in
  check_float "scatter_preserve[0]" ~eps:1e-12 10.0 d.(0);
  check_float "scatter_preserve[1]" ~eps:1e-12 20.0 d.(1);
  check_float "scatter_preserve[2]" ~eps:1e-12 99.0 d.(2);
  check_float "scatter_preserve[3]" ~eps:1e-12 40.0 d.(3)

let test_fold_int32_1d_overlap () =
  let ctx = Nx_backend.create_context () in
  (* Shape [N=1, C*K=2, L=2] where C=1, K=2 *)
  let x_flat = Nx_ox.create ctx Dtype.Int32 [|4|] [| 1l; 3l; 2l; 4l |] in
  let x = Nx_backend.reshape x_flat (Symbolic_shape.of_ints [| 1; 2; 2 |]) in
  let y =
    Nx_backend.fold x
      ~output_size:[| 3 |]
      ~kernel_size:[| 2 |]
      ~stride:[| 1 |]
      ~dilation:[| 1 |]
      ~padding:[| (0, 0) |]
  in
  let shape_y =
    match Symbolic_shape.eval (View.shape (Nx_backend.view y)) with
    | Some s -> s
    | None -> failwith "shape not evaluable"
  in
  check "fold_int32_1d_overlap: shape0" (shape_y.(0) = 1);
  check "fold_int32_1d_overlap: shape1" (shape_y.(1) = 1);
  check "fold_int32_1d_overlap: shape2" (shape_y.(2) = 3);
  let d = Nx_ox.to_array y in
  check_int32 "fold_int32_1d_overlap[0]" 1l d.(0);
  check_int32 "fold_int32_1d_overlap[1]" 5l d.(1);
  check_int32 "fold_int32_1d_overlap[2]" 4l d.(2)

let test_fold_int32_1d_padding_stride () =
  let ctx = Nx_backend.create_context () in
  (* Shape [N=1, C*K=3, L=2] where C=1, K=3 *)
  let x_flat = Nx_ox.create ctx Dtype.Int32 [|6|] [| 10l; 20l; 30l; 40l; 50l; 60l |] in
  let x = Nx_backend.reshape x_flat (Symbolic_shape.of_ints [| 1; 3; 2 |]) in
  let y =
    Nx_backend.fold x
      ~output_size:[| 4 |]
      ~kernel_size:[| 3 |]
      ~stride:[| 2 |]
      ~dilation:[| 1 |]
      ~padding:[| (1, 1) |]
  in
  let d = Nx_ox.to_array y in
  check_int32 "fold_int32_1d_padding_stride[0]" 30l d.(0);
  check_int32 "fold_int32_1d_padding_stride[1]" 70l d.(1);
  check_int32 "fold_int32_1d_padding_stride[2]" 40l d.(2);
  check_int32 "fold_int32_1d_padding_stride[3]" 60l d.(3)

let test_unfold_int32_1d_basic () =
  let ctx = Nx_backend.create_context () in
  let x_flat = Nx_ox.create ctx Dtype.Int32 [|4|] [| 1l; 2l; 3l; 4l |] in
  let x = Nx_backend.reshape x_flat (Symbolic_shape.of_ints [| 1; 1; 4 |]) in
  let y =
    Nx_backend.unfold x
      ~kernel_size:[| 2 |]
      ~stride:[| 1 |]
      ~dilation:[| 1 |]
      ~padding:[| (0, 0) |]
  in
  let shape_y =
    match Symbolic_shape.eval (View.shape (Nx_backend.view y)) with
    | Some s -> s
    | None -> failwith "shape not evaluable"
  in
  check "unfold_int32_1d_basic: shape0" (shape_y.(0) = 1);
  check "unfold_int32_1d_basic: shape1" (shape_y.(1) = 2);
  check "unfold_int32_1d_basic: shape2" (shape_y.(2) = 3);
  let d = Nx_ox.to_array y in
  check_int32 "unfold_int32_1d_basic[0]" 1l d.(0);
  check_int32 "unfold_int32_1d_basic[1]" 2l d.(1);
  check_int32 "unfold_int32_1d_basic[2]" 3l d.(2);
  check_int32 "unfold_int32_1d_basic[3]" 2l d.(3);
  check_int32 "unfold_int32_1d_basic[4]" 3l d.(4);
  check_int32 "unfold_int32_1d_basic[5]" 4l d.(5)

let test_unfold_int32_1d_padding_stride () =
  let ctx = Nx_backend.create_context () in
  let x_flat = Nx_ox.create ctx Dtype.Int32 [|4|] [| 1l; 2l; 3l; 4l |] in
  let x = Nx_backend.reshape x_flat (Symbolic_shape.of_ints [| 1; 1; 4 |]) in
  let y =
    Nx_backend.unfold x
      ~kernel_size:[| 3 |]
      ~stride:[| 2 |]
      ~dilation:[| 1 |]
      ~padding:[| (1, 1) |]
  in
  let shape_y =
    match Symbolic_shape.eval (View.shape (Nx_backend.view y)) with
    | Some s -> s
    | None -> failwith "shape not evaluable"
  in
  check "unfold_int32_1d_padding_stride: shape0" (shape_y.(0) = 1);
  check "unfold_int32_1d_padding_stride: shape1" (shape_y.(1) = 3);
  check "unfold_int32_1d_padding_stride: shape2" (shape_y.(2) = 2);
  let d = Nx_ox.to_array y in
  check_int32 "unfold_int32_1d_padding_stride[0]" 0l d.(0);
  check_int32 "unfold_int32_1d_padding_stride[1]" 2l d.(1);
  check_int32 "unfold_int32_1d_padding_stride[2]" 1l d.(2);
  check_int32 "unfold_int32_1d_padding_stride[3]" 3l d.(3);
  check_int32 "unfold_int32_1d_padding_stride[4]" 2l d.(4);
  check_int32 "unfold_int32_1d_padding_stride[5]" 4l d.(5)

let test_unfold_int64_1d_identity () =
  let ctx = Nx_backend.create_context () in
  let x_flat = Nx_ox.create ctx Dtype.Int64 [| 4 |] [| 11L; 22L; 33L; 44L |] in
  let x = Nx_backend.reshape x_flat (Symbolic_shape.of_ints [| 1; 1; 4 |]) in
  let y =
    Nx_backend.unfold x
      ~kernel_size:[| 1 |]
      ~stride:[| 1 |]
      ~dilation:[| 1 |]
      ~padding:[| (0, 0) |]
  in
  let d = Nx_ox.to_array y in
  check_int64 "unfold_int64_1d_identity[0]" 11L d.(0);
  check_int64 "unfold_int64_1d_identity[1]" 22L d.(1);
  check_int64 "unfold_int64_1d_identity[2]" 33L d.(2);
  check_int64 "unfold_int64_1d_identity[3]" 44L d.(3)

let test_unfold_float32_1d_identity () =
  let ctx = Nx_backend.create_context () in
  let x_flat = Nx_ox.create ctx Dtype.Float32 [| 4 |] [| 1.5; 2.5; 3.5; 4.5 |] in
  let x = Nx_backend.reshape x_flat (Symbolic_shape.of_ints [| 1; 1; 4 |]) in
  let y =
    Nx_backend.unfold x
      ~kernel_size:[| 1 |]
      ~stride:[| 1 |]
      ~dilation:[| 1 |]
      ~padding:[| (0, 0) |]
  in
  let d = Nx_ox.to_array y in
  check_float "unfold_float32_1d_identity[0]" ~eps:1e-6 1.5 d.(0);
  check_float "unfold_float32_1d_identity[1]" ~eps:1e-6 2.5 d.(1);
  check_float "unfold_float32_1d_identity[2]" ~eps:1e-6 3.5 d.(2);
  check_float "unfold_float32_1d_identity[3]" ~eps:1e-6 4.5 d.(3)

let test_unfold_float64_1d_identity () =
  let ctx = Nx_backend.create_context () in
  let x_flat = Nx_ox.create ctx Dtype.Float64 [| 4 |] [| 1.25; 2.25; 3.25; 4.25 |] in
  let x = Nx_backend.reshape x_flat (Symbolic_shape.of_ints [| 1; 1; 4 |]) in
  let y =
    Nx_backend.unfold x
      ~kernel_size:[| 1 |]
      ~stride:[| 1 |]
      ~dilation:[| 1 |]
      ~padding:[| (0, 0) |]
  in
  let d = Nx_ox.to_array y in
  check_float "unfold_float64_1d_identity[0]" ~eps:1e-9 1.25 d.(0);
  check_float "unfold_float64_1d_identity[1]" ~eps:1e-9 2.25 d.(1);
  check_float "unfold_float64_1d_identity[2]" ~eps:1e-9 3.25 d.(2);
  check_float "unfold_float64_1d_identity[3]" ~eps:1e-9 4.25 d.(3)

let test_unfold_int8_1d_identity () =
  let ctx = Nx_backend.create_context () in
  let x_flat = Nx_ox.create ctx Dtype.Int8 [| 4 |] [| 1; 2; 3; 4 |] in
  let x = Nx_backend.reshape x_flat (Symbolic_shape.of_ints [| 1; 1; 4 |]) in
  let y =
    Nx_backend.unfold x
      ~kernel_size:[| 1 |]
      ~stride:[| 1 |]
      ~dilation:[| 1 |]
      ~padding:[| (0, 0) |]
  in
  let d = Nx_ox.to_array y in
  check_int "unfold_int8_1d_identity[0]" 1 d.(0);
  check_int "unfold_int8_1d_identity[1]" 2 d.(1);
  check_int "unfold_int8_1d_identity[2]" 3 d.(2);
  check_int "unfold_int8_1d_identity[3]" 4 d.(3)

let test_unfold_int16_1d_identity () =
  let ctx = Nx_backend.create_context () in
  let x_flat = Nx_ox.create ctx Dtype.Int16 [| 4 |] [| 10; 20; 30; 40 |] in
  let x = Nx_backend.reshape x_flat (Symbolic_shape.of_ints [| 1; 1; 4 |]) in
  let y =
    Nx_backend.unfold x
      ~kernel_size:[| 1 |]
      ~stride:[| 1 |]
      ~dilation:[| 1 |]
      ~padding:[| (0, 0) |]
  in
  let d = Nx_ox.to_array y in
  check_int "unfold_int16_1d_identity[0]" 10 d.(0);
  check_int "unfold_int16_1d_identity[1]" 20 d.(1);
  check_int "unfold_int16_1d_identity[2]" 30 d.(2);
  check_int "unfold_int16_1d_identity[3]" 40 d.(3)

let test_unfold_bool_1d_identity () =
  let ctx = Nx_backend.create_context () in
  let x_flat =
    Nx_ox.create ctx Dtype.Bool [| 4 |] [| true; false; true; false |]
  in
  let x = Nx_backend.reshape x_flat (Symbolic_shape.of_ints [| 1; 1; 4 |]) in
  let y =
    Nx_backend.unfold x
      ~kernel_size:[| 1 |]
      ~stride:[| 1 |]
      ~dilation:[| 1 |]
      ~padding:[| (0, 0) |]
  in
  let d = Nx_ox.to_array y in
  check_bool "unfold_bool_1d_identity[0]" true d.(0);
  check_bool "unfold_bool_1d_identity[1]" false d.(1);
  check_bool "unfold_bool_1d_identity[2]" true d.(2);
  check_bool "unfold_bool_1d_identity[3]" false d.(3)

let test_fold_int64_1d_identity () =
  let ctx = Nx_backend.create_context () in
  let x_flat = Nx_ox.create ctx Dtype.Int64 [| 4 |] [| 9L; 8L; 7L; 6L |] in
  let x = Nx_backend.reshape x_flat (Symbolic_shape.of_ints [| 1; 1; 4 |]) in
  let y =
    Nx_backend.fold x
      ~output_size:[| 4 |]
      ~kernel_size:[| 1 |]
      ~stride:[| 1 |]
      ~dilation:[| 1 |]
      ~padding:[| (0, 0) |]
  in
  let d = Nx_ox.to_array y in
  check_int64 "fold_int64_1d_identity[0]" 9L d.(0);
  check_int64 "fold_int64_1d_identity[1]" 8L d.(1);
  check_int64 "fold_int64_1d_identity[2]" 7L d.(2);
  check_int64 "fold_int64_1d_identity[3]" 6L d.(3)

let test_fold_float32_1d_identity () =
  let ctx = Nx_backend.create_context () in
  let x_flat = Nx_ox.create ctx Dtype.Float32 [| 4 |] [| 0.5; 1.5; 2.5; 3.5 |] in
  let x = Nx_backend.reshape x_flat (Symbolic_shape.of_ints [| 1; 1; 4 |]) in
  let y =
    Nx_backend.fold x
      ~output_size:[| 4 |]
      ~kernel_size:[| 1 |]
      ~stride:[| 1 |]
      ~dilation:[| 1 |]
      ~padding:[| (0, 0) |]
  in
  let d = Nx_ox.to_array y in
  check_float "fold_float32_1d_identity[0]" ~eps:1e-6 0.5 d.(0);
  check_float "fold_float32_1d_identity[1]" ~eps:1e-6 1.5 d.(1);
  check_float "fold_float32_1d_identity[2]" ~eps:1e-6 2.5 d.(2);
  check_float "fold_float32_1d_identity[3]" ~eps:1e-6 3.5 d.(3)

let test_fold_float64_1d_identity () =
  let ctx = Nx_backend.create_context () in
  let x_flat =
    Nx_ox.create ctx Dtype.Float64 [| 4 |] [| 10.25; 11.25; 12.25; 13.25 |]
  in
  let x = Nx_backend.reshape x_flat (Symbolic_shape.of_ints [| 1; 1; 4 |]) in
  let y =
    Nx_backend.fold x
      ~output_size:[| 4 |]
      ~kernel_size:[| 1 |]
      ~stride:[| 1 |]
      ~dilation:[| 1 |]
      ~padding:[| (0, 0) |]
  in
  let d = Nx_ox.to_array y in
  check_float "fold_float64_1d_identity[0]" ~eps:1e-9 10.25 d.(0);
  check_float "fold_float64_1d_identity[1]" ~eps:1e-9 11.25 d.(1);
  check_float "fold_float64_1d_identity[2]" ~eps:1e-9 12.25 d.(2);
  check_float "fold_float64_1d_identity[3]" ~eps:1e-9 13.25 d.(3)

let test_fold_int8_1d_identity () =
  let ctx = Nx_backend.create_context () in
  let x_flat = Nx_ox.create ctx Dtype.Int8 [| 4 |] [| 1; 3; 5; 7 |] in
  let x = Nx_backend.reshape x_flat (Symbolic_shape.of_ints [| 1; 1; 4 |]) in
  let y =
    Nx_backend.fold x
      ~output_size:[| 4 |]
      ~kernel_size:[| 1 |]
      ~stride:[| 1 |]
      ~dilation:[| 1 |]
      ~padding:[| (0, 0) |]
  in
  let d = Nx_ox.to_array y in
  check_int "fold_int8_1d_identity[0]" 1 d.(0);
  check_int "fold_int8_1d_identity[1]" 3 d.(1);
  check_int "fold_int8_1d_identity[2]" 5 d.(2);
  check_int "fold_int8_1d_identity[3]" 7 d.(3)

let test_fold_int16_1d_identity () =
  let ctx = Nx_backend.create_context () in
  let x_flat = Nx_ox.create ctx Dtype.Int16 [| 4 |] [| 2; 4; 6; 8 |] in
  let x = Nx_backend.reshape x_flat (Symbolic_shape.of_ints [| 1; 1; 4 |]) in
  let y =
    Nx_backend.fold x
      ~output_size:[| 4 |]
      ~kernel_size:[| 1 |]
      ~stride:[| 1 |]
      ~dilation:[| 1 |]
      ~padding:[| (0, 0) |]
  in
  let d = Nx_ox.to_array y in
  check_int "fold_int16_1d_identity[0]" 2 d.(0);
  check_int "fold_int16_1d_identity[1]" 4 d.(1);
  check_int "fold_int16_1d_identity[2]" 6 d.(2);
  check_int "fold_int16_1d_identity[3]" 8 d.(3)

let test_associative_scan_sum_int32_axis1 () =
  let ctx = Nx_backend.create_context () in
  let x =
    Nx_ox.create ctx Dtype.Int32 [| 2; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l |]
  in
  let out = Nx_ox.empty ctx Dtype.Int32 [| 2; 3 |] in
  Nx_backend.associative_scan ~out ~axis:1 ~op:`Sum x;
  let d = Nx_ox.to_array out in
  check_int32 "associative_scan_sum_int32_axis1[0]" 1l d.(0);
  check_int32 "associative_scan_sum_int32_axis1[1]" 3l d.(1);
  check_int32 "associative_scan_sum_int32_axis1[2]" 6l d.(2);
  check_int32 "associative_scan_sum_int32_axis1[3]" 4l d.(3);
  check_int32 "associative_scan_sum_int32_axis1[4]" 9l d.(4);
  check_int32 "associative_scan_sum_int32_axis1[5]" 15l d.(5)

let test_associative_scan_prod_int64_axis0 () =
  let ctx = Nx_backend.create_context () in
  let x =
    Nx_ox.create ctx Dtype.Int64 [| 2; 3 |] [| 1L; 2L; 3L; 4L; 5L; 6L |]
  in
  let out = Nx_ox.empty ctx Dtype.Int64 [| 2; 3 |] in
  Nx_backend.associative_scan ~out ~axis:0 ~op:`Prod x;
  let d = Nx_ox.to_array out in
  check_int64 "associative_scan_prod_int64_axis0[0]" 1L d.(0);
  check_int64 "associative_scan_prod_int64_axis0[1]" 2L d.(1);
  check_int64 "associative_scan_prod_int64_axis0[2]" 3L d.(2);
  check_int64 "associative_scan_prod_int64_axis0[3]" 4L d.(3);
  check_int64 "associative_scan_prod_int64_axis0[4]" 10L d.(4);
  check_int64 "associative_scan_prod_int64_axis0[5]" 18L d.(5)

let test_associative_scan_sum_int32_permuted_view () =
  let ctx = Nx_backend.create_context () in
  let x =
    Nx_ox.create ctx Dtype.Int32 [| 2; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l |]
  in
  let x_permuted = Nx_backend.permute x [| 1; 0 |] in
  let out = Nx_ox.empty ctx Dtype.Int32 [| 3; 2 |] in
  Nx_backend.associative_scan ~out ~axis:1 ~op:`Sum x_permuted;
  let d = Nx_ox.to_array out in
  check_int32 "associative_scan_sum_int32_permuted_view[0]" 1l d.(0);
  check_int32 "associative_scan_sum_int32_permuted_view[1]" 5l d.(1);
  check_int32 "associative_scan_sum_int32_permuted_view[2]" 2l d.(2);
  check_int32 "associative_scan_sum_int32_permuted_view[3]" 7l d.(3);
  check_int32 "associative_scan_sum_int32_permuted_view[4]" 3l d.(4);
  check_int32 "associative_scan_sum_int32_permuted_view[5]" 9l d.(5)

let test_associative_scan_zero_axis_length () =
  let ctx = Nx_backend.create_context () in
  let x = Nx_ox.empty ctx Dtype.Float32 [| 0; 3 |] in
  let out = Nx_ox.empty ctx Dtype.Float32 [| 0; 3 |] in
  Nx_backend.associative_scan ~out ~axis:0 ~op:`Max x;
  check_int "associative_scan_zero_axis_length:numel"
    (numel (Nx_backend.view out)) 0

let test_threefry_strided_view_matches_contiguous () =
  let ctx = Nx_backend.create_context () in
  let key_base =
    Nx_ox.create ctx Dtype.Int32
      [| 2; 2 |]
      [| 1l; 2l; -1l; 0l |]
  in
  let ctr_base =
    Nx_ox.create ctx Dtype.Int32
      [| 2; 2 |]
      [| 3l; 4l; 123l; 456l |]
  in
  let key_perm = Nx_backend.permute key_base [| 1; 0 |] in
  let ctr_perm = Nx_backend.permute ctr_base [| 1; 0 |] in
  let out_perm = Nx_ox.empty ctx Dtype.Int32 [| 2; 2 |] in
  Nx_backend.threefry ~out:out_perm key_perm ctr_perm;
  let key_contig = Nx_backend.contiguous key_perm in
  let ctr_contig = Nx_backend.contiguous ctr_perm in
  let out_contig = Nx_ox.empty ctx Dtype.Int32 [| 2; 2 |] in
  Nx_backend.threefry ~out:out_contig key_contig ctr_contig;
  let perm_data = Nx_ox.to_array out_perm in
  let contig_data = Nx_ox.to_array out_contig in
  for i = 0 to Array.length perm_data - 1 do
    check_int32 (Printf.sprintf "threefry_strided_view_matches_contiguous[%d]" i)
      contig_data.(i) perm_data.(i)
  done

let test_argmax_float64_1d () =
  let ctx = Nx_backend.create_context () in
  let x = Nx_ox.create ctx Dtype.Float64 [| 5 |] [| 1.0; 5.0; 3.0; 2.0; 4.0 |] in
  let out = Nx_ox.empty ctx Dtype.Int32 [| 1 |] in
  Nx_backend.argmax ~out ~axis:0 ~keepdims:true x;
  let d = Nx_ox.to_array out in
  check_int32 "argmax_float64_1d" 1l d.(0)

let test_argmax_float64_2d_axis0 () =
  let ctx = Nx_backend.create_context () in
  (* [[1, 4], [3, 2]] -> axis 0 -> [1, 0] *)
  let x = Nx_ox.create ctx Dtype.Float64 [| 2; 2 |] [| 1.0; 4.0; 3.0; 2.0 |] in
  let out = Nx_ox.empty ctx Dtype.Int32 [| 2 |] in
  Nx_backend.argmax ~out ~axis:0 ~keepdims:false x;
  let d = Nx_ox.to_array out in
  check_int32 "argmax_float64_2d_axis0[0]" 1l d.(0);
  check_int32 "argmax_float64_2d_axis0[1]" 0l d.(1)

let test_argmax_float64_2d_axis1 () =
  let ctx = Nx_backend.create_context () in
  (* [[1, 4], [3, 2]] -> axis 1 -> [1, 0] *)
  let x = Nx_ox.create ctx Dtype.Float64 [| 2; 2 |] [| 1.0; 4.0; 3.0; 2.0 |] in
  let out = Nx_ox.empty ctx Dtype.Int32 [| 2 |] in
  Nx_backend.argmax ~out ~axis:1 ~keepdims:false x;
  let d = Nx_ox.to_array out in
  check_int32 "argmax_float64_2d_axis1[0]" 1l d.(0);
  check_int32 "argmax_float64_2d_axis1[1]" 0l d.(1)

let test_argmin_float64_1d () =
  let ctx = Nx_backend.create_context () in
  let x = Nx_ox.create ctx Dtype.Float64 [| 5 |] [| 3.0; 1.0; 5.0; 2.0; 4.0 |] in
  let out = Nx_ox.empty ctx Dtype.Int32 [| 1 |] in
  Nx_backend.argmin ~out ~axis:0 ~keepdims:true x;
  let d = Nx_ox.to_array out in
  check_int32 "argmin_float64_1d" 1l d.(0)

let test_argmax_int32 () =
  let ctx = Nx_backend.create_context () in
  let x = Nx_ox.create ctx Dtype.Int32 [| 4 |] [| 10l; 30l; 20l; 5l |] in
  let out = Nx_ox.empty ctx Dtype.Int32 [| 1 |] in
  Nx_backend.argmax ~out ~axis:0 ~keepdims:true x;
  let d = Nx_ox.to_array out in
  check_int32 "argmax_int32" 1l d.(0)

let test_argmin_int64 () =
  let ctx = Nx_backend.create_context () in
  let x = Nx_ox.create ctx Dtype.Int64 [| 4 |] [| 10L; 30L; 5L; 20L |] in
  let out = Nx_ox.empty ctx Dtype.Int32 [| 1 |] in
  Nx_backend.argmin ~out ~axis:0 ~keepdims:true x;
  let d = Nx_ox.to_array out in
  check_int32 "argmin_int64" 2l d.(0)

let test_atan2_float64 () =
  let ctx = Nx_backend.create_context () in
  let y = Nx_ox.create ctx Dtype.Float64 [| 4 |] [| 1.0; -1.0; 1.0; 0.0 |] in
  let x = Nx_ox.create ctx Dtype.Float64 [| 4 |] [| 1.0; 1.0; -1.0; 1.0 |] in
  let out = Nx_ox.empty ctx Dtype.Float64 [| 4 |] in
  Nx_backend.atan2 ~out y x;
  let data = Nx_ox.to_array out in
  check_float "atan2_float64[0]" ~eps:1e-10 (Float.atan2 1.0 1.0) data.(0);
  check_float "atan2_float64[1]" ~eps:1e-10 (Float.atan2 (-1.0) 1.0) data.(1);
  check_float "atan2_float64[2]" ~eps:1e-10 (Float.atan2 1.0 (-1.0)) data.(2);
  check_float "atan2_float64[3]" ~eps:1e-10 (Float.atan2 0.0 1.0) data.(3)

let test_atan2_float32 () =
  let ctx = Nx_backend.create_context () in
  let y = Nx_ox.create ctx Dtype.Float32 [| 3 |] [| 1.0; 0.0; -1.0 |] in
  let x = Nx_ox.create ctx Dtype.Float32 [| 3 |] [| 0.0; 1.0; -1.0 |] in
  let out = Nx_ox.empty ctx Dtype.Float32 [| 3 |] in
  Nx_backend.atan2 ~out y x;
  let data = Nx_ox.to_array out in
  check_float "atan2_float32[0]" ~eps:1e-5 (Float.atan2 1.0 0.0) data.(0);
  check_float "atan2_float32[1]" ~eps:1e-5 (Float.atan2 0.0 1.0) data.(1);
  check_float "atan2_float32[2]" ~eps:1e-5 (Float.atan2 (-1.0) (-1.0)) data.(2)

let () =
  print_endline "Running Nx_backend backend tests...";
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
  test_idiv_int64 ();
  test_idiv_int32 ();
  test_mod_float64 ();
  test_mod_float32 ();
  test_mod_int64 ();
  test_mod_int32 ();
  test_min_float64 ();
  test_min_float32 ();
  test_min_int64 ();
  test_min_int32 ();
  test_max_float64 ();
  test_max_float32 ();
  test_max_int64 ();
  test_max_int32 ();
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
  test_matmul_identity ();
  test_matmul_rectangular ();
  test_matmul_batched ();
  test_matmul_dot_product ();
  test_matmul_rectangular_f32 ();
  test_matmul_batched_f32 ();
  test_pad_int32_1d ();
  test_pad_float64_2d ();
  test_pad_float64_permuted_view ();
  test_shrink_int32_view ();
  test_flip_int32_view ();
  test_cat_int32_axis1 ();
  test_gather_int32_axis1 ();
  test_gather_float32_axis0_contiguous ();
  test_gather_float64_axis0_contiguous ();
  test_gather_float64_axis0_2d ();
  test_gather_int32_axis0_contiguous ();
  test_gather_int64_axis0_contiguous ();
  test_gather_single_element ();
  test_gather_negative_axis ();
  test_scatter_int32_set_axis1 ();
  test_scatter_int32_add_axis1 ();
  test_scatter_float64_set ();
  test_scatter_float64_add_duplicates ();
  test_scatter_bool_set ();
  test_scatter_preserves_template ();
  test_unfold_int32_1d_basic ();
  test_unfold_int32_1d_padding_stride ();
  test_unfold_int64_1d_identity ();
  test_unfold_float32_1d_identity ();
  test_unfold_float64_1d_identity ();
  test_unfold_int8_1d_identity ();
  test_unfold_int16_1d_identity ();
  test_unfold_bool_1d_identity ();
  test_fold_int32_1d_overlap ();
  test_fold_int32_1d_padding_stride ();
  test_fold_int64_1d_identity ();
  test_fold_float32_1d_identity ();
  test_fold_float64_1d_identity ();
  test_fold_int8_1d_identity ();
  test_fold_int16_1d_identity ();
  test_associative_scan_sum_int32_axis1 ();
  test_associative_scan_prod_int64_axis0 ();
  test_associative_scan_sum_int32_permuted_view ();
  test_associative_scan_zero_axis_length ();
  test_threefry_strided_view_matches_contiguous ();
  test_atan2_float64 ();
  test_atan2_float32 ();
  test_argmax_float64_1d ();
  test_argmax_float64_2d_axis0 ();
  test_argmax_float64_2d_axis1 ();
  test_argmin_float64_1d ();
  test_argmax_int32 ();
  test_argmin_int64 ();
  Printf.printf "\nResults: %d passed, %d failed\n" !passed !failed;
  if !failed > 0 then exit 1
