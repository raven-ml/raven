(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Generates .actual files for expect tests. Each file contains tolk's rendered
   output for a specific backend + test case, matching the programs in
   generate_expected.py. Dune diff rules compare .actual against .expected
   (generated from tinygrad). *)

open Tolk
module P = Ir.Program

let global_ptr dt = Dtype.Ptr.create dt ~addrspace:Global ()
let local_ptr dt = Dtype.Ptr.create dt ~addrspace:Local ()

(* IR program builders — must match generate_expected.py exactly. *)

let make_simple_add_f32 () =
  let dt = Dtype.float32 in
  let ptr = global_ptr dt in
  [|
    (* 0 *) P.Param { idx = 0; dtype = ptr };
    (* 1 *) P.Param { idx = 1; dtype = ptr };
    (* 2 *) P.Param { idx = 2; dtype = ptr };
    (* 3 *) P.Const { value = Int 0; dtype = Dtype.int32 };
    (* 4 *) P.Index { ptr = 0; idxs = [ 3 ]; gate = None; dtype = ptr };
    (* 5 *) P.Index { ptr = 1; idxs = [ 3 ]; gate = None; dtype = ptr };
    (* 6 *) P.Load { src = 4; alt = None; dtype = dt };
    (* 7 *) P.Load { src = 5; alt = None; dtype = dt };
    (* 8 *) P.Add { lhs = 6; rhs = 7; dtype = dt };
    (* 9 *) P.Index { ptr = 2; idxs = [ 3 ]; gate = None; dtype = ptr };
    (* 10 *) P.Store { dst = 9; value = 8 };
  |]

let make_simple_mul_i32 () =
  let dt = Dtype.int32 in
  let ptr = global_ptr dt in
  [|
    (* 0 *) P.Param { idx = 0; dtype = ptr };
    (* 1 *) P.Param { idx = 1; dtype = ptr };
    (* 2 *) P.Param { idx = 2; dtype = ptr };
    (* 3 *) P.Const { value = Int 0; dtype = Dtype.int32 };
    (* 4 *) P.Index { ptr = 0; idxs = [ 3 ]; gate = None; dtype = ptr };
    (* 5 *) P.Index { ptr = 1; idxs = [ 3 ]; gate = None; dtype = ptr };
    (* 6 *) P.Load { src = 4; alt = None; dtype = dt };
    (* 7 *) P.Load { src = 5; alt = None; dtype = dt };
    (* 8 *) P.Mul { lhs = 6; rhs = 7; dtype = dt };
    (* 9 *) P.Index { ptr = 2; idxs = [ 3 ]; gate = None; dtype = ptr };
    (* 10 *) P.Store { dst = 9; value = 8 };
  |]

let make_loop () =
  let dt = Dtype.float32 in
  let ptr = global_ptr dt in
  [|
    (* 0 *) P.Param { idx = 0; dtype = ptr };
    (* 1 *) P.Const { value = Int 10; dtype = Dtype.int32 };
    (* 2 *)
    P.Range { size = 1; dtype = Dtype.int32; axis = 0; kind = Ir.Loop };
    (* 3 *) P.Index { ptr = 0; idxs = [ 2 ]; gate = None; dtype = ptr };
    (* 4 *) P.Load { src = 3; alt = None; dtype = dt };
    (* 5 *) P.Index { ptr = 0; idxs = [ 2 ]; gate = None; dtype = ptr };
    (* 6 *) P.Store { dst = 5; value = 4 };
    (* 7 *) P.End_range { range = 2 };
  |]

let make_gated_load () =
  let dt = Dtype.float32 in
  let ptr = global_ptr dt in
  [|
    (* 0 *) P.Param { idx = 0; dtype = ptr };
    (* 1 *) P.Param { idx = 1; dtype = ptr };
    (* 2 *) P.Const { value = Int 0; dtype = Dtype.int32 };
    (* 3 *) P.Const { value = Bool true; dtype = Dtype.bool };
    (* 4 *)
    P.Index { ptr = 0; idxs = [ 2 ]; gate = Some 3; dtype = ptr };
    (* 5 *) P.Const { value = Float 0.0; dtype = dt };
    (* 6 *) P.Load { src = 4; alt = Some 5; dtype = dt };
    (* 7 *) P.Index { ptr = 1; idxs = [ 2 ]; gate = None; dtype = ptr };
    (* 8 *) P.Store { dst = 7; value = 6 };
  |]

let make_shared_memory () =
  let dt = Dtype.float32 in
  let gptr = global_ptr dt in
  let lptr = local_ptr dt in
  [|
    (* 0 *) P.Param { idx = 0; dtype = gptr };
    (* 1 *) P.Define_local { size = 256; dtype = lptr };
    (* 2 *) P.Const { value = Int 0; dtype = Dtype.int32 };
    (* 3 *) P.Index { ptr = 1; idxs = [ 2 ]; gate = None; dtype = lptr };
    (* 4 *) P.Const { value = Float 0.0; dtype = dt };
    (* 5 *) P.Store { dst = 3; value = 4 };
    (* 6 *) P.Barrier;
    (* 7 *) P.Load { src = 3; alt = None; dtype = dt };
    (* 8 *) P.Index { ptr = 0; idxs = [ 2 ]; gate = None; dtype = gptr };
    (* 9 *) P.Store { dst = 8; value = 7 };
  |]

let make_where_select () =
  let dt = Dtype.float32 in
  let ptr = global_ptr dt in
  [|
    (* 0 *) P.Param { idx = 0; dtype = ptr };
    (* 1 *) P.Param { idx = 1; dtype = ptr };
    (* 2 *) P.Param { idx = 2; dtype = ptr };
    (* 3 *) P.Const { value = Int 0; dtype = Dtype.int32 };
    (* 4 *) P.Index { ptr = 0; idxs = [ 3 ]; gate = None; dtype = ptr };
    (* 5 *) P.Index { ptr = 1; idxs = [ 3 ]; gate = None; dtype = ptr };
    (* 6 *) P.Load { src = 4; alt = None; dtype = dt };
    (* 7 *) P.Load { src = 5; alt = None; dtype = dt };
    (* 8 *) P.Const { value = Bool true; dtype = Dtype.bool };
    (* 9 *) P.Where { cond = 8; a = 6; b = 7; dtype = dt };
    (* 10 *) P.Index { ptr = 2; idxs = [ 3 ]; gate = None; dtype = ptr };
    (* 11 *) P.Store { dst = 10; value = 9 };
  |]

let make_cast_f16_to_f32 () =
  let from_dt = Dtype.float16 in
  let to_dt = Dtype.float32 in
  let from_ptr = global_ptr from_dt in
  let to_ptr = global_ptr to_dt in
  [|
    (* 0 *) P.Param { idx = 0; dtype = from_ptr };
    (* 1 *) P.Param { idx = 1; dtype = to_ptr };
    (* 2 *) P.Const { value = Int 0; dtype = Dtype.int32 };
    (* 3 *)
    P.Index { ptr = 0; idxs = [ 2 ]; gate = None; dtype = from_ptr };
    (* 4 *) P.Index { ptr = 1; idxs = [ 2 ]; gate = None; dtype = to_ptr };
    (* 5 *) P.Load { src = 3; alt = None; dtype = from_dt };
    (* 6 *) P.Cast { src = 5; dtype = to_dt };
    (* 7 *) P.Store { dst = 4; value = 6 };
  |]

let make_nested_loops () =
  let dt = Dtype.float32 in
  let ptr = global_ptr dt in
  [|
    (* 0 *) P.Param { idx = 0; dtype = ptr };
    (* 1 *) P.Const { value = Int 10; dtype = Dtype.int32 };
    (* 2 *) P.Const { value = Int 5; dtype = Dtype.int32 };
    (* 3 *)
    P.Range { size = 1; dtype = Dtype.int32; axis = 0; kind = Ir.Loop };
    (* 4 *)
    P.Range { size = 2; dtype = Dtype.int32; axis = 1; kind = Ir.Loop };
    (* 5 *) P.Add { lhs = 3; rhs = 4; dtype = Dtype.int32 };
    (* 6 *) P.Index { ptr = 0; idxs = [ 5 ]; gate = None; dtype = ptr };
    (* 7 *) P.Load { src = 6; alt = None; dtype = dt };
    (* 8 *) P.Index { ptr = 0; idxs = [ 5 ]; gate = None; dtype = ptr };
    (* 9 *) P.Store { dst = 8; value = 7 };
    (* 10 *) P.End_range { range = 4 };
    (* 11 *) P.End_range { range = 3 };
  |]

let make_multi_param () =
  let dt = Dtype.float32 in
  let ptr = global_ptr dt in
  [|
    (* 0 *) P.Param { idx = 0; dtype = ptr };
    (* 1 *) P.Param { idx = 1; dtype = ptr };
    (* 2 *) P.Param { idx = 2; dtype = ptr };
    (* 3 *) P.Param { idx = 3; dtype = ptr };
    (* 4 *) P.Const { value = Int 0; dtype = Dtype.int32 };
    (* 5 *) P.Index { ptr = 0; idxs = [ 4 ]; gate = None; dtype = ptr };
    (* 6 *) P.Index { ptr = 1; idxs = [ 4 ]; gate = None; dtype = ptr };
    (* 7 *) P.Load { src = 5; alt = None; dtype = dt };
    (* 8 *) P.Load { src = 6; alt = None; dtype = dt };
    (* 9 *) P.Add { lhs = 7; rhs = 8; dtype = dt };
    (* 10 *) P.Index { ptr = 3; idxs = [ 4 ]; gate = None; dtype = ptr };
    (* 11 *) P.Store { dst = 10; value = 9 };
  |]

let make_unary_sqrt_f32 () =
  let dt = Dtype.float32 in
  let ptr = global_ptr dt in
  [|
    (* 0 *) P.Param { idx = 0; dtype = ptr };
    (* 1 *) P.Param { idx = 1; dtype = ptr };
    (* 2 *) P.Const { value = Int 0; dtype = Dtype.int32 };
    (* 3 *) P.Index { ptr = 0; idxs = [ 2 ]; gate = None; dtype = ptr };
    (* 4 *) P.Load { src = 3; alt = None; dtype = dt };
    (* 5 *) P.Sqrt { src = 4; dtype = dt };
    (* 6 *) P.Index { ptr = 1; idxs = [ 2 ]; gate = None; dtype = ptr };
    (* 7 *) P.Store { dst = 6; value = 5 };
  |]

let make_unary_sqrt_f16 () =
  let dt = Dtype.float16 in
  let ptr = global_ptr dt in
  [|
    (* 0 *) P.Param { idx = 0; dtype = ptr };
    (* 1 *) P.Param { idx = 1; dtype = ptr };
    (* 2 *) P.Const { value = Int 0; dtype = Dtype.int32 };
    (* 3 *) P.Index { ptr = 0; idxs = [ 2 ]; gate = None; dtype = ptr };
    (* 4 *) P.Load { src = 3; alt = None; dtype = dt };
    (* 5 *) P.Sqrt { src = 4; dtype = dt };
    (* 6 *) P.Index { ptr = 1; idxs = [ 2 ]; gate = None; dtype = ptr };
    (* 7 *) P.Store { dst = 6; value = 5 };
  |]

let make_special_dims () =
  let dt = Dtype.float32 in
  let ptr = global_ptr dt in
  [|
    (* 0 *) P.Param { idx = 0; dtype = ptr };
    (* 1 *) P.Const { value = Int 32; dtype = Dtype.int32 };
    (* 2 *) P.Special { dim = Group_id 0; size = 1; dtype = Dtype.int32 };
    (* 3 *) P.Special { dim = Local_id 0; size = 1; dtype = Dtype.int32 };
    (* 4 *) P.Add { lhs = 2; rhs = 3; dtype = Dtype.int32 };
    (* 5 *) P.Index { ptr = 0; idxs = [ 4 ]; gate = None; dtype = ptr };
    (* 6 *) P.Load { src = 5; alt = None; dtype = dt };
    (* 7 *) P.Index { ptr = 0; idxs = [ 4 ]; gate = None; dtype = ptr };
    (* 8 *) P.Store { dst = 7; value = 6 };
  |]

let make_bitcast_f32_to_i32 () =
  let from_dt = Dtype.float32 in
  let to_dt = Dtype.int32 in
  let from_ptr = global_ptr from_dt in
  let to_ptr = global_ptr to_dt in
  [|
    (* 0 *) P.Param { idx = 0; dtype = from_ptr };
    (* 1 *) P.Param { idx = 1; dtype = to_ptr };
    (* 2 *) P.Const { value = Int 0; dtype = Dtype.int32 };
    (* 3 *)
    P.Index { ptr = 0; idxs = [ 2 ]; gate = None; dtype = from_ptr };
    (* 4 *) P.Load { src = 3; alt = None; dtype = from_dt };
    (* 5 *) P.Bitcast { src = 4; dtype = to_dt };
    (* 6 *)
    P.Index { ptr = 1; idxs = [ 2 ]; gate = None; dtype = to_ptr };
    (* 7 *) P.Store { dst = 6; value = 5 };
  |]

let make_conditional () =
  let dt = Dtype.float32 in
  let ptr = global_ptr dt in
  [|
    (* 0 *) P.Param { idx = 0; dtype = ptr };
    (* 1 *) P.Const { value = Int 0; dtype = Dtype.int32 };
    (* 2 *) P.Const { value = Bool true; dtype = Dtype.bool };
    (* 3 *) P.If { cond = 2; idx_for_dedup = 1 };
    (* 4 *) P.Index { ptr = 0; idxs = [ 1 ]; gate = None; dtype = ptr };
    (* 5 *) P.Const { value = Float 1.0; dtype = dt };
    (* 6 *) P.Store { dst = 4; value = 5 };
    (* 7 *) P.Endif { if_ = 3 };
  |]

let make_const_inf_nan () =
  let dt = Dtype.float32 in
  let ptr = global_ptr dt in
  [|
    (* 0 *) P.Param { idx = 0; dtype = ptr };
    (* 1 *) P.Const { value = Int 0; dtype = Dtype.int32 };
    (* 2 *) P.Const { value = Int 1; dtype = Dtype.int32 };
    (* 3 *) P.Const { value = Float infinity; dtype = dt };
    (* 4 *) P.Const { value = Float nan; dtype = dt };
    (* 5 *) P.Index { ptr = 0; idxs = [ 1 ]; gate = None; dtype = ptr };
    (* 6 *) P.Store { dst = 5; value = 3 };
    (* 7 *) P.Index { ptr = 0; idxs = [ 2 ]; gate = None; dtype = ptr };
    (* 8 *) P.Store { dst = 7; value = 4 };
  |]

let make_vectorize_gep () =
  let dt = Dtype.float32 in
  let vdt = Dtype.vec dt 4 in
  let ptr = global_ptr dt in
  [|
    (* 0 *) P.Param { idx = 0; dtype = ptr };
    (* 1 *) P.Param { idx = 1; dtype = ptr };
    (* 2 *) P.Const { value = Int 0; dtype = Dtype.int32 };
    (* 3 *) P.Const { value = Int 1; dtype = Dtype.int32 };
    (* 4 *) P.Const { value = Int 2; dtype = Dtype.int32 };
    (* 5 *) P.Const { value = Int 3; dtype = Dtype.int32 };
    (* 6 *) P.Index { ptr = 0; idxs = [ 2 ]; gate = None; dtype = ptr };
    (* 7 *) P.Index { ptr = 0; idxs = [ 3 ]; gate = None; dtype = ptr };
    (* 8 *) P.Index { ptr = 0; idxs = [ 4 ]; gate = None; dtype = ptr };
    (* 9 *) P.Index { ptr = 0; idxs = [ 5 ]; gate = None; dtype = ptr };
    (* 10 *) P.Load { src = 6; alt = None; dtype = dt };
    (* 11 *) P.Load { src = 7; alt = None; dtype = dt };
    (* 12 *) P.Load { src = 8; alt = None; dtype = dt };
    (* 13 *) P.Load { src = 9; alt = None; dtype = dt };
    (* 14 *) P.Vectorize { srcs = [ 10; 11; 12; 13 ]; dtype = vdt };
    (* 15 *) P.Gep { src = 14; idx = 2; dtype = dt };
    (* 16 *) P.Index { ptr = 1; idxs = [ 2 ]; gate = None; dtype = ptr };
    (* 17 *) P.Store { dst = 16; value = 15 };
  |]

(* Test cases: (name, builder, backends). backends = None means all backends,
   Some [...] limits to those. *)

type test_case = {
  name : string;
  prog : Ir.Program.t;
  backends : (string * Renderer.t) list;
}

let all_renderers =
  [
    ("clang", Cstyle.clang_no_abi);
    ("metal", Cstyle.metal);
    ("opencl", Cstyle.opencl);
  ]

let gpu_renderers = List.filter (fun (name, _) -> name <> "clang") all_renderers

let test_cases =
  [
    {
      name = "simple_add_f32";
      prog = make_simple_add_f32 ();
      backends = all_renderers;
    };
    {
      name = "simple_mul_i32";
      prog = make_simple_mul_i32 ();
      backends = all_renderers;
    };
    { name = "loop"; prog = make_loop (); backends = all_renderers };
    { name = "gated_load"; prog = make_gated_load (); backends = all_renderers };
    {
      name = "shared_memory";
      prog = make_shared_memory ();
      backends = gpu_renderers;
    };
    {
      name = "where_select";
      prog = make_where_select ();
      backends = all_renderers;
    };
    {
      name = "cast_f16_to_f32";
      prog = make_cast_f16_to_f32 ();
      backends = all_renderers;
    };
    {
      name = "nested_loops";
      prog = make_nested_loops ();
      backends = all_renderers;
    };
    {
      name = "multi_param";
      prog = make_multi_param ();
      backends = all_renderers;
    };
    {
      name = "unary_sqrt_f32";
      prog = make_unary_sqrt_f32 ();
      backends = all_renderers;
    };
    {
      name = "unary_sqrt_f16";
      prog = make_unary_sqrt_f16 ();
      backends = all_renderers;
    };
    {
      name = "special_dims";
      prog = make_special_dims ();
      backends = gpu_renderers;
    };
    {
      name = "bitcast_f32_to_i32";
      prog = make_bitcast_f32_to_i32 ();
      backends = all_renderers;
    };
    {
      name = "conditional";
      prog = make_conditional ();
      backends = all_renderers;
    };
    {
      name = "const_inf_nan";
      prog = make_const_inf_nan ();
      backends = all_renderers;
    };
    {
      name = "vectorize_gep";
      prog = make_vectorize_gep ();
      backends = all_renderers;
    };
  ]

let () =
  let dir = Sys.argv.(1) in
  List.iter
    (fun { name; prog; backends } ->
      List.iter
        (fun (backend_name, renderer) ->
          let out = String.trim (Renderer.render renderer ~name:"test" prog) in
          let filename =
            Filename.concat dir
              (Printf.sprintf "%s_%s.actual" backend_name name)
          in
          let oc = open_out filename in
          output_string oc out;
          output_char oc '\n';
          close_out oc)
        backends)
    test_cases
