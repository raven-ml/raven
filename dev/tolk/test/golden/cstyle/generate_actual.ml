(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Generates .actual files for expect tests. Each file contains tolk's rendered
   output for a specific backend + test case, matching the programs in
   generate_expected.py. Dune diff rules compare .actual against .expected
   (generated from the reference renderer). *)

open Tolk
open Tolk_ir
module P = Program

let global_ptr dt = Dtype.Ptr.create dt ~addrspace:Global ()
let local_ptr dt = Dtype.Ptr.create dt ~addrspace:Local ()

(* IR program builders — must match generate_expected.py exactly. *)

let make_simple_add_f32 () =
  let dt = Dtype.float32 in
  let ptr = global_ptr dt in
  let b = P.create () in
  let p0 = P.emit b (Param { idx = 0; dtype = ptr }) in
  let p1 = P.emit b (Param { idx = 1; dtype = ptr }) in
  let p2 = P.emit b (Param { idx = 2; dtype = ptr }) in
  let c0 = P.emit b (Const { value = Const.int Dtype.int32 0; dtype = Dtype.int32 }) in
  let idx0 = P.emit b (Index { ptr = p0; idxs = [ c0 ]; gate = None; dtype = ptr }) in
  let idx1 = P.emit b (Index { ptr = p1; idxs = [ c0 ]; gate = None; dtype = ptr }) in
  let ld0 = P.emit b (Load { src = idx0; alt = None; dtype = dt }) in
  let ld1 = P.emit b (Load { src = idx1; alt = None; dtype = dt }) in
  let sum = P.emit b (Binary { op = `Add; lhs = ld0; rhs = ld1; dtype = dt }) in
  let idx2 = P.emit b (Index { ptr = p2; idxs = [ c0 ]; gate = None; dtype = ptr }) in
  let _ = P.emit b (Store { dst = idx2; value = sum }) in
  P.finish b

let make_simple_mul_i32 () =
  let dt = Dtype.int32 in
  let ptr = global_ptr dt in
  let b = P.create () in
  let p0 = P.emit b (Param { idx = 0; dtype = ptr }) in
  let p1 = P.emit b (Param { idx = 1; dtype = ptr }) in
  let p2 = P.emit b (Param { idx = 2; dtype = ptr }) in
  let c0 = P.emit b (Const { value = Const.int Dtype.int32 0; dtype = Dtype.int32 }) in
  let idx0 = P.emit b (Index { ptr = p0; idxs = [ c0 ]; gate = None; dtype = ptr }) in
  let idx1 = P.emit b (Index { ptr = p1; idxs = [ c0 ]; gate = None; dtype = ptr }) in
  let ld0 = P.emit b (Load { src = idx0; alt = None; dtype = dt }) in
  let ld1 = P.emit b (Load { src = idx1; alt = None; dtype = dt }) in
  let prod = P.emit b (Binary { op = `Mul; lhs = ld0; rhs = ld1; dtype = dt }) in
  let idx2 = P.emit b (Index { ptr = p2; idxs = [ c0 ]; gate = None; dtype = ptr }) in
  let _ = P.emit b (Store { dst = idx2; value = prod }) in
  P.finish b

let make_loop () =
  let dt = Dtype.float32 in
  let ptr = global_ptr dt in
  let b = P.create () in
  let p0 = P.emit b (Param { idx = 0; dtype = ptr }) in
  let c10 = P.emit b (Const { value = Const.int Dtype.int32 10; dtype = Dtype.int32 }) in
  let r = P.emit b (Range { size = c10; dtype = Dtype.int32; axis = 0; kind = Axis_kind.Loop }) in
  let idx0 = P.emit b (Index { ptr = p0; idxs = [ r ]; gate = None; dtype = ptr }) in
  let ld = P.emit b (Load { src = idx0; alt = None; dtype = dt }) in
  let idx1 = P.emit b (Index { ptr = p0; idxs = [ r ]; gate = None; dtype = ptr }) in
  let _ = P.emit b (Store { dst = idx1; value = ld }) in
  let _ = P.emit b (End_range { range = r }) in
  P.finish b

let make_gated_load () =
  let dt = Dtype.float32 in
  let ptr = global_ptr dt in
  let b = P.create () in
  let p0 = P.emit b (Param { idx = 0; dtype = ptr }) in
  let p1 = P.emit b (Param { idx = 1; dtype = ptr }) in
  let c0 = P.emit b (Const { value = Const.int Dtype.int32 0; dtype = Dtype.int32 }) in
  let gate = P.emit b (Const { value = Const.bool true; dtype = Dtype.bool }) in
  let idx0 = P.emit b (Index { ptr = p0; idxs = [ c0 ]; gate = Some gate; dtype = ptr }) in
  let alt = P.emit b (Const { value = Const.float dt 0.0; dtype = dt }) in
  let ld = P.emit b (Load { src = idx0; alt = Some alt; dtype = dt }) in
  let idx1 = P.emit b (Index { ptr = p1; idxs = [ c0 ]; gate = None; dtype = ptr }) in
  let _ = P.emit b (Store { dst = idx1; value = ld }) in
  P.finish b

let make_shared_memory () =
  let dt = Dtype.float32 in
  let gptr = global_ptr dt in
  let lptr = local_ptr dt in
  let b = P.create () in
  let p0 = P.emit b (Param { idx = 0; dtype = gptr }) in
  let dl = P.emit b (Define_local { size = 256; dtype = lptr }) in
  let c0 = P.emit b (Const { value = Const.int Dtype.int32 0; dtype = Dtype.int32 }) in
  let lidx = P.emit b (Index { ptr = dl; idxs = [ c0 ]; gate = None; dtype = lptr }) in
  let fzero = P.emit b (Const { value = Const.float dt 0.0; dtype = dt }) in
  let _ = P.emit b (Store { dst = lidx; value = fzero }) in
  let _ = P.emit b Barrier in
  let ld = P.emit b (Load { src = lidx; alt = None; dtype = dt }) in
  let gidx = P.emit b (Index { ptr = p0; idxs = [ c0 ]; gate = None; dtype = gptr }) in
  let _ = P.emit b (Store { dst = gidx; value = ld }) in
  P.finish b

let make_where_select () =
  let dt = Dtype.float32 in
  let ptr = global_ptr dt in
  let b = P.create () in
  let p0 = P.emit b (Param { idx = 0; dtype = ptr }) in
  let p1 = P.emit b (Param { idx = 1; dtype = ptr }) in
  let p2 = P.emit b (Param { idx = 2; dtype = ptr }) in
  let c0 = P.emit b (Const { value = Const.int Dtype.int32 0; dtype = Dtype.int32 }) in
  let idx0 = P.emit b (Index { ptr = p0; idxs = [ c0 ]; gate = None; dtype = ptr }) in
  let idx1 = P.emit b (Index { ptr = p1; idxs = [ c0 ]; gate = None; dtype = ptr }) in
  let ld0 = P.emit b (Load { src = idx0; alt = None; dtype = dt }) in
  let ld1 = P.emit b (Load { src = idx1; alt = None; dtype = dt }) in
  let cond = P.emit b (Const { value = Const.bool true; dtype = Dtype.bool }) in
  let w = P.emit b (Ternary { op = `Where; a = cond; b = ld0; c = ld1; dtype = dt }) in
  let idx2 = P.emit b (Index { ptr = p2; idxs = [ c0 ]; gate = None; dtype = ptr }) in
  let _ = P.emit b (Store { dst = idx2; value = w }) in
  P.finish b

let make_cast_f16_to_f32 () =
  let from_dt = Dtype.float16 in
  let to_dt = Dtype.float32 in
  let from_ptr = global_ptr from_dt in
  let to_ptr = global_ptr to_dt in
  let b = P.create () in
  let p0 = P.emit b (Param { idx = 0; dtype = from_ptr }) in
  let p1 = P.emit b (Param { idx = 1; dtype = to_ptr }) in
  let c0 = P.emit b (Const { value = Const.int Dtype.int32 0; dtype = Dtype.int32 }) in
  let idx0 = P.emit b (Index { ptr = p0; idxs = [ c0 ]; gate = None; dtype = from_ptr }) in
  let idx1 = P.emit b (Index { ptr = p1; idxs = [ c0 ]; gate = None; dtype = to_ptr }) in
  let ld = P.emit b (Load { src = idx0; alt = None; dtype = from_dt }) in
  let cast = P.emit b (Cast { src = ld; dtype = to_dt }) in
  let _ = P.emit b (Store { dst = idx1; value = cast }) in
  P.finish b

let make_nested_loops () =
  let dt = Dtype.float32 in
  let ptr = global_ptr dt in
  let b = P.create () in
  let p0 = P.emit b (Param { idx = 0; dtype = ptr }) in
  let c10 = P.emit b (Const { value = Const.int Dtype.int32 10; dtype = Dtype.int32 }) in
  let c5 = P.emit b (Const { value = Const.int Dtype.int32 5; dtype = Dtype.int32 }) in
  let r0 = P.emit b (Range { size = c10; dtype = Dtype.int32; axis = 0; kind = Axis_kind.Loop }) in
  let r1 = P.emit b (Range { size = c5; dtype = Dtype.int32; axis = 1; kind = Axis_kind.Loop }) in
  let sum = P.emit b (Binary { op = `Add; lhs = r0; rhs = r1; dtype = Dtype.int32 }) in
  let idx0 = P.emit b (Index { ptr = p0; idxs = [ sum ]; gate = None; dtype = ptr }) in
  let ld = P.emit b (Load { src = idx0; alt = None; dtype = dt }) in
  let idx1 = P.emit b (Index { ptr = p0; idxs = [ sum ]; gate = None; dtype = ptr }) in
  let _ = P.emit b (Store { dst = idx1; value = ld }) in
  let _ = P.emit b (End_range { range = r1 }) in
  let _ = P.emit b (End_range { range = r0 }) in
  P.finish b

let make_multi_param () =
  let dt = Dtype.float32 in
  let ptr = global_ptr dt in
  let b = P.create () in
  let p0 = P.emit b (Param { idx = 0; dtype = ptr }) in
  let p1 = P.emit b (Param { idx = 1; dtype = ptr }) in
  let _ = P.emit b (Param { idx = 2; dtype = ptr }) in
  let p3 = P.emit b (Param { idx = 3; dtype = ptr }) in
  let c0 = P.emit b (Const { value = Const.int Dtype.int32 0; dtype = Dtype.int32 }) in
  let idx0 = P.emit b (Index { ptr = p0; idxs = [ c0 ]; gate = None; dtype = ptr }) in
  let idx1 = P.emit b (Index { ptr = p1; idxs = [ c0 ]; gate = None; dtype = ptr }) in
  let ld0 = P.emit b (Load { src = idx0; alt = None; dtype = dt }) in
  let ld1 = P.emit b (Load { src = idx1; alt = None; dtype = dt }) in
  let sum = P.emit b (Binary { op = `Add; lhs = ld0; rhs = ld1; dtype = dt }) in
  let idx3 = P.emit b (Index { ptr = p3; idxs = [ c0 ]; gate = None; dtype = ptr }) in
  let _ = P.emit b (Store { dst = idx3; value = sum }) in
  P.finish b

let make_unary_sqrt_f32 () =
  let dt = Dtype.float32 in
  let ptr = global_ptr dt in
  let b = P.create () in
  let p0 = P.emit b (Param { idx = 0; dtype = ptr }) in
  let p1 = P.emit b (Param { idx = 1; dtype = ptr }) in
  let c0 = P.emit b (Const { value = Const.int Dtype.int32 0; dtype = Dtype.int32 }) in
  let idx0 = P.emit b (Index { ptr = p0; idxs = [ c0 ]; gate = None; dtype = ptr }) in
  let ld = P.emit b (Load { src = idx0; alt = None; dtype = dt }) in
  let sq = P.emit b (Unary { op = `Sqrt; src = ld; dtype = dt }) in
  let idx1 = P.emit b (Index { ptr = p1; idxs = [ c0 ]; gate = None; dtype = ptr }) in
  let _ = P.emit b (Store { dst = idx1; value = sq }) in
  P.finish b

let make_unary_sqrt_f16 () =
  let dt = Dtype.float16 in
  let ptr = global_ptr dt in
  let b = P.create () in
  let p0 = P.emit b (Param { idx = 0; dtype = ptr }) in
  let p1 = P.emit b (Param { idx = 1; dtype = ptr }) in
  let c0 = P.emit b (Const { value = Const.int Dtype.int32 0; dtype = Dtype.int32 }) in
  let idx0 = P.emit b (Index { ptr = p0; idxs = [ c0 ]; gate = None; dtype = ptr }) in
  let ld = P.emit b (Load { src = idx0; alt = None; dtype = dt }) in
  let sq = P.emit b (Unary { op = `Sqrt; src = ld; dtype = dt }) in
  let idx1 = P.emit b (Index { ptr = p1; idxs = [ c0 ]; gate = None; dtype = ptr }) in
  let _ = P.emit b (Store { dst = idx1; value = sq }) in
  P.finish b

let make_special_dims () =
  let dt = Dtype.float32 in
  let ptr = global_ptr dt in
  let b = P.create () in
  let p0 = P.emit b (Param { idx = 0; dtype = ptr }) in
  let c32 = P.emit b (Const { value = Const.int Dtype.int32 32; dtype = Dtype.int32 }) in
  let gid = P.emit b (Special { dim = Special_dim.Group_id 0; size = c32; dtype = Dtype.int32 }) in
  let lid = P.emit b (Special { dim = Special_dim.Local_id 0; size = c32; dtype = Dtype.int32 }) in
  let sum = P.emit b (Binary { op = `Add; lhs = gid; rhs = lid; dtype = Dtype.int32 }) in
  let idx0 = P.emit b (Index { ptr = p0; idxs = [ sum ]; gate = None; dtype = ptr }) in
  let ld = P.emit b (Load { src = idx0; alt = None; dtype = dt }) in
  let idx1 = P.emit b (Index { ptr = p0; idxs = [ sum ]; gate = None; dtype = ptr }) in
  let _ = P.emit b (Store { dst = idx1; value = ld }) in
  P.finish b

let make_bitcast_f32_to_i32 () =
  let from_dt = Dtype.float32 in
  let to_dt = Dtype.int32 in
  let from_ptr = global_ptr from_dt in
  let to_ptr = global_ptr to_dt in
  let b = P.create () in
  let p0 = P.emit b (Param { idx = 0; dtype = from_ptr }) in
  let p1 = P.emit b (Param { idx = 1; dtype = to_ptr }) in
  let c0 = P.emit b (Const { value = Const.int Dtype.int32 0; dtype = Dtype.int32 }) in
  let idx0 = P.emit b (Index { ptr = p0; idxs = [ c0 ]; gate = None; dtype = from_ptr }) in
  let ld = P.emit b (Load { src = idx0; alt = None; dtype = from_dt }) in
  let bc = P.emit b (Bitcast { src = ld; dtype = to_dt }) in
  let idx1 = P.emit b (Index { ptr = p1; idxs = [ c0 ]; gate = None; dtype = to_ptr }) in
  let _ = P.emit b (Store { dst = idx1; value = bc }) in
  P.finish b

let make_conditional () =
  let dt = Dtype.float32 in
  let ptr = global_ptr dt in
  let b = P.create () in
  let p0 = P.emit b (Param { idx = 0; dtype = ptr }) in
  let c0 = P.emit b (Const { value = Const.int Dtype.int32 0; dtype = Dtype.int32 }) in
  let cond = P.emit b (Const { value = Const.bool true; dtype = Dtype.bool }) in
  let if_ = P.emit b (If { cond; idx_for_dedup = c0 }) in
  let idx0 = P.emit b (Index { ptr = p0; idxs = [ c0 ]; gate = None; dtype = ptr }) in
  let fone = P.emit b (Const { value = Const.float dt 1.0; dtype = dt }) in
  let _ = P.emit b (Store { dst = idx0; value = fone }) in
  let _ = P.emit b (Endif { if_ }) in
  P.finish b

let make_const_inf_nan () =
  let dt = Dtype.float32 in
  let ptr = global_ptr dt in
  let b = P.create () in
  let p0 = P.emit b (Param { idx = 0; dtype = ptr }) in
  let c0 = P.emit b (Const { value = Const.int Dtype.int32 0; dtype = Dtype.int32 }) in
  let c1 = P.emit b (Const { value = Const.int Dtype.int32 1; dtype = Dtype.int32 }) in
  let finf = P.emit b (Const { value = Const.float dt infinity; dtype = dt }) in
  let fnan = P.emit b (Const { value = Const.float dt nan; dtype = dt }) in
  let idx0 = P.emit b (Index { ptr = p0; idxs = [ c0 ]; gate = None; dtype = ptr }) in
  let _ = P.emit b (Store { dst = idx0; value = finf }) in
  let idx1 = P.emit b (Index { ptr = p0; idxs = [ c1 ]; gate = None; dtype = ptr }) in
  let _ = P.emit b (Store { dst = idx1; value = fnan }) in
  P.finish b

let make_vectorize_gep () =
  let dt = Dtype.float32 in
  let vdt = Dtype.vec dt 4 in
  let ptr = global_ptr dt in
  let b = P.create () in
  let p0 = P.emit b (Param { idx = 0; dtype = ptr }) in
  let p1 = P.emit b (Param { idx = 1; dtype = ptr }) in
  let c0 = P.emit b (Const { value = Const.int Dtype.int32 0; dtype = Dtype.int32 }) in
  let c1 = P.emit b (Const { value = Const.int Dtype.int32 1; dtype = Dtype.int32 }) in
  let c2 = P.emit b (Const { value = Const.int Dtype.int32 2; dtype = Dtype.int32 }) in
  let c3 = P.emit b (Const { value = Const.int Dtype.int32 3; dtype = Dtype.int32 }) in
  let idx0 = P.emit b (Index { ptr = p0; idxs = [ c0 ]; gate = None; dtype = ptr }) in
  let idx1 = P.emit b (Index { ptr = p0; idxs = [ c1 ]; gate = None; dtype = ptr }) in
  let idx2 = P.emit b (Index { ptr = p0; idxs = [ c2 ]; gate = None; dtype = ptr }) in
  let idx3 = P.emit b (Index { ptr = p0; idxs = [ c3 ]; gate = None; dtype = ptr }) in
  let ld0 = P.emit b (Load { src = idx0; alt = None; dtype = dt }) in
  let ld1 = P.emit b (Load { src = idx1; alt = None; dtype = dt }) in
  let ld2 = P.emit b (Load { src = idx2; alt = None; dtype = dt }) in
  let ld3 = P.emit b (Load { src = idx3; alt = None; dtype = dt }) in
  let vec = P.emit b (Vectorize { srcs = [ ld0; ld1; ld2; ld3 ]; dtype = vdt }) in
  let gep = P.emit b (Gep { src = vec; idx = 2; dtype = dt }) in
  let oidx = P.emit b (Index { ptr = p1; idxs = [ c0 ]; gate = None; dtype = ptr }) in
  let _ = P.emit b (Store { dst = oidx; value = gep }) in
  P.finish b

(* Test cases: (name, builder, backends). backends = None means all backends,
   Some [...] limits to those. *)

type test_case = {
  name : string;
  prog : Program.t;
  backends : (string * Renderer.t) list;
}

let all_renderers =
  [
    ("clang", Cstyle.clang_no_abi);
    ("cuda", Cstyle.cuda Gpu_target.SM80);
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
