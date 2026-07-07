(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Generates .actual files for expect tests. Each file contains tolk's rendered
   output for a specific backend + test case, matching the programs in
   generate_expected.py. Dune diff rules compare .actual against .expected
   (generated from the reference renderer). *)

open Tolk
open Tolk_uop
module B = Program_spec_builder

let global_ptr dt = Dtype.Ptr.create dt ~addrspace:Global ~size:(-1)

(* IR program builders — must match generate_expected.py exactly. *)

let make_simple_add_f32 () =
  let dt = Dtype.Val.float32 in
  let ptr = global_ptr dt in
  let b = B.create () in
  let p0 = B.emit b (Param { slot = 0; dtype = ptr }) in
  let p1 = B.emit b (Param { slot = 1; dtype = ptr }) in
  let p2 = B.emit b (Param { slot = 2; dtype = ptr }) in
  let c0 = B.emit b (Const { value = Const.int Dtype.Val.int32 0; dtype = Dtype.Val.int32 }) in
  let idx0 = B.emit b (Index { ptr = p0; idxs = [ c0 ]; dtype = ptr }) in
  let idx1 = B.emit b (Index { ptr = p1; idxs = [ c0 ]; dtype = ptr }) in
  let ld0 = B.emit b (Load { src = idx0; alt = None; gate = None; dtype = dt }) in
  let ld1 = B.emit b (Load { src = idx1; alt = None; gate = None; dtype = dt }) in
  let sum = B.emit b (Binary { op = `Add; lhs = ld0; rhs = ld1; dtype = dt }) in
  let idx2 = B.emit b (Index { ptr = p2; idxs = [ c0 ]; dtype = ptr }) in
  let _ = B.emit b (Store { dst = idx2; value = sum; gate = None }) in
  B.finish b

let make_simple_mul_i32 () =
  let dt = Dtype.Val.int32 in
  let ptr = global_ptr dt in
  let b = B.create () in
  let p0 = B.emit b (Param { slot = 0; dtype = ptr }) in
  let p1 = B.emit b (Param { slot = 1; dtype = ptr }) in
  let p2 = B.emit b (Param { slot = 2; dtype = ptr }) in
  let c0 = B.emit b (Const { value = Const.int Dtype.Val.int32 0; dtype = Dtype.Val.int32 }) in
  let idx0 = B.emit b (Index { ptr = p0; idxs = [ c0 ]; dtype = ptr }) in
  let idx1 = B.emit b (Index { ptr = p1; idxs = [ c0 ]; dtype = ptr }) in
  let ld0 = B.emit b (Load { src = idx0; alt = None; gate = None; dtype = dt }) in
  let ld1 = B.emit b (Load { src = idx1; alt = None; gate = None; dtype = dt }) in
  let prod = B.emit b (Binary { op = `Mul; lhs = ld0; rhs = ld1; dtype = dt }) in
  let idx2 = B.emit b (Index { ptr = p2; idxs = [ c0 ]; dtype = ptr }) in
  let _ = B.emit b (Store { dst = idx2; value = prod; gate = None }) in
  B.finish b

let make_loop () =
  let dt = Dtype.Val.float32 in
  let ptr = global_ptr dt in
  let b = B.create () in
  let p0 = B.emit b (Param { slot = 0; dtype = ptr }) in
  let c10 = B.emit b (Const { value = Const.int Dtype.Val.int32 10; dtype = Dtype.Val.int32 }) in
  let r = B.emit b (Range { size = c10; dtype = Dtype.Val.int32; axis = 0; sub = []; kind = Axis_type.Loop }) in
  let idx0 = B.emit b (Index { ptr = p0; idxs = [ r ]; dtype = ptr }) in
  let ld = B.emit b (Load { src = idx0; alt = None; gate = None; dtype = dt }) in
  let idx1 = B.emit b (Index { ptr = p0; idxs = [ r ]; dtype = ptr }) in
  let _ = B.emit b (Store { dst = idx1; value = ld; gate = None }) in
  let _ = B.emit b (End_range { dep = ld; range = r }) in
  B.finish b

let make_gated_load () =
  let dt = Dtype.Val.float32 in
  let ptr = global_ptr dt in
  let b = B.create () in
  let p0 = B.emit b (Param { slot = 0; dtype = ptr }) in
  let p1 = B.emit b (Param { slot = 1; dtype = ptr }) in
  let c0 = B.emit b (Const { value = Const.int Dtype.Val.int32 0; dtype = Dtype.Val.int32 }) in
  let gate = B.emit b (Const { value = Const.bool true; dtype = Dtype.Val.bool }) in
  let idx0 = B.emit b (Index { ptr = p0; idxs = [ c0 ]; dtype = ptr }) in
  let alt = B.emit b (Const { value = Const.float dt 0.0; dtype = dt }) in
  let ld =
    B.emit b (Load { src = idx0; alt = Some alt; gate = Some gate; dtype = dt })
  in
  let idx1 = B.emit b (Index { ptr = p1; idxs = [ c0 ]; dtype = ptr }) in
  let _ = B.emit b (Store { dst = idx1; value = ld; gate = None }) in
  B.finish b

let make_shared_memory () =
  let dt = Dtype.Val.float32 in
  let gptr = global_ptr dt in
  let lptr = Dtype.Ptr.create dt ~addrspace:Local ~size:256 in
  let b = B.create () in
  let p0 = B.emit b (Param { slot = 0; dtype = gptr }) in
  let dl = B.emit b (Buffer { slot = Some 0; size = 256; dtype = lptr }) in
  let c0 = B.emit b (Const { value = Const.int Dtype.Val.int32 0; dtype = Dtype.Val.int32 }) in
  let lidx = B.emit b (Index { ptr = dl; idxs = [ c0 ]; dtype = lptr }) in
  let fzero = B.emit b (Const { value = Const.float dt 0.0; dtype = dt }) in
  let _ = B.emit b (Store { dst = lidx; value = fzero; gate = None }) in
  let _ = B.emit b Barrier in
  let ld = B.emit b (Load { src = lidx; alt = None; gate = None; dtype = dt }) in
  let gidx = B.emit b (Index { ptr = p0; idxs = [ c0 ]; dtype = gptr }) in
  let _ = B.emit b (Store { dst = gidx; value = ld; gate = None }) in
  B.finish b

let make_where_select () =
  let dt = Dtype.Val.float32 in
  let ptr = global_ptr dt in
  let b = B.create () in
  let p0 = B.emit b (Param { slot = 0; dtype = ptr }) in
  let p1 = B.emit b (Param { slot = 1; dtype = ptr }) in
  let p2 = B.emit b (Param { slot = 2; dtype = ptr }) in
  let c0 = B.emit b (Const { value = Const.int Dtype.Val.int32 0; dtype = Dtype.Val.int32 }) in
  let idx0 = B.emit b (Index { ptr = p0; idxs = [ c0 ]; dtype = ptr }) in
  let idx1 = B.emit b (Index { ptr = p1; idxs = [ c0 ]; dtype = ptr }) in
  let ld0 = B.emit b (Load { src = idx0; alt = None; gate = None; dtype = dt }) in
  let ld1 = B.emit b (Load { src = idx1; alt = None; gate = None; dtype = dt }) in
  let cond = B.emit b (Const { value = Const.bool true; dtype = Dtype.Val.bool }) in
  let w = B.emit b (Ternary { op = `Where; a = cond; b = ld0; c = ld1; dtype = dt }) in
  let idx2 = B.emit b (Index { ptr = p2; idxs = [ c0 ]; dtype = ptr }) in
  let _ = B.emit b (Store { dst = idx2; value = w; gate = None }) in
  B.finish b

let make_cast_f16_to_f32 () =
  let from_dt = Dtype.Val.float16 in
  let to_dt = Dtype.Val.float32 in
  let from_ptr = global_ptr from_dt in
  let to_ptr = global_ptr to_dt in
  let b = B.create () in
  let p0 = B.emit b (Param { slot = 0; dtype = from_ptr }) in
  let p1 = B.emit b (Param { slot = 1; dtype = to_ptr }) in
  let c0 = B.emit b (Const { value = Const.int Dtype.Val.int32 0; dtype = Dtype.Val.int32 }) in
  let idx0 = B.emit b (Index { ptr = p0; idxs = [ c0 ]; dtype = from_ptr }) in
  let idx1 = B.emit b (Index { ptr = p1; idxs = [ c0 ]; dtype = to_ptr }) in
  let ld = B.emit b (Load { src = idx0; alt = None; gate = None; dtype = from_dt }) in
  let cast = B.emit b (Cast { src = ld; dtype = to_dt }) in
  let _ = B.emit b (Store { dst = idx1; value = cast; gate = None }) in
  B.finish b

let make_nested_loops () =
  let dt = Dtype.Val.float32 in
  let ptr = global_ptr dt in
  let b = B.create () in
  let p0 = B.emit b (Param { slot = 0; dtype = ptr }) in
  let c10 = B.emit b (Const { value = Const.int Dtype.Val.int32 10; dtype = Dtype.Val.int32 }) in
  let c5 = B.emit b (Const { value = Const.int Dtype.Val.int32 5; dtype = Dtype.Val.int32 }) in
  let r0 = B.emit b (Range { size = c10; dtype = Dtype.Val.int32; axis = 0; sub = []; kind = Axis_type.Loop }) in
  let r1 = B.emit b (Range { size = c5; dtype = Dtype.Val.int32; axis = 1; sub = []; kind = Axis_type.Loop }) in
  let sum = B.emit b (Binary { op = `Add; lhs = r0; rhs = r1; dtype = Dtype.Val.int32 }) in
  let idx0 = B.emit b (Index { ptr = p0; idxs = [ sum ]; dtype = ptr }) in
  let ld = B.emit b (Load { src = idx0; alt = None; gate = None; dtype = dt }) in
  let idx1 = B.emit b (Index { ptr = p0; idxs = [ sum ]; dtype = ptr }) in
  let _ = B.emit b (Store { dst = idx1; value = ld; gate = None }) in
  let _ = B.emit b (End_range { dep = ld; range = r1 }) in
  let _ = B.emit b (End_range { dep = r0; range = r0 }) in
  B.finish b

let make_multi_param () =
  let dt = Dtype.Val.float32 in
  let ptr = global_ptr dt in
  let b = B.create () in
  let p0 = B.emit b (Param { slot = 0; dtype = ptr }) in
  let p1 = B.emit b (Param { slot = 1; dtype = ptr }) in
  let _ = B.emit b (Param { slot = 2; dtype = ptr }) in
  let p3 = B.emit b (Param { slot = 3; dtype = ptr }) in
  let c0 = B.emit b (Const { value = Const.int Dtype.Val.int32 0; dtype = Dtype.Val.int32 }) in
  let idx0 = B.emit b (Index { ptr = p0; idxs = [ c0 ]; dtype = ptr }) in
  let idx1 = B.emit b (Index { ptr = p1; idxs = [ c0 ]; dtype = ptr }) in
  let ld0 = B.emit b (Load { src = idx0; alt = None; gate = None; dtype = dt }) in
  let ld1 = B.emit b (Load { src = idx1; alt = None; gate = None; dtype = dt }) in
  let sum = B.emit b (Binary { op = `Add; lhs = ld0; rhs = ld1; dtype = dt }) in
  let idx3 = B.emit b (Index { ptr = p3; idxs = [ c0 ]; dtype = ptr }) in
  let _ = B.emit b (Store { dst = idx3; value = sum; gate = None }) in
  B.finish b

let make_unary_sqrt_f32 () =
  let dt = Dtype.Val.float32 in
  let ptr = global_ptr dt in
  let b = B.create () in
  let p0 = B.emit b (Param { slot = 0; dtype = ptr }) in
  let p1 = B.emit b (Param { slot = 1; dtype = ptr }) in
  let c0 = B.emit b (Const { value = Const.int Dtype.Val.int32 0; dtype = Dtype.Val.int32 }) in
  let idx0 = B.emit b (Index { ptr = p0; idxs = [ c0 ]; dtype = ptr }) in
  let ld = B.emit b (Load { src = idx0; alt = None; gate = None; dtype = dt }) in
  let sq = B.emit b (Unary { op = `Sqrt; src = ld; dtype = dt }) in
  let idx1 = B.emit b (Index { ptr = p1; idxs = [ c0 ]; dtype = ptr }) in
  let _ = B.emit b (Store { dst = idx1; value = sq; gate = None }) in
  B.finish b

let make_unary_sqrt_f16 () =
  let dt = Dtype.Val.float16 in
  let ptr = global_ptr dt in
  let b = B.create () in
  let p0 = B.emit b (Param { slot = 0; dtype = ptr }) in
  let p1 = B.emit b (Param { slot = 1; dtype = ptr }) in
  let c0 = B.emit b (Const { value = Const.int Dtype.Val.int32 0; dtype = Dtype.Val.int32 }) in
  let idx0 = B.emit b (Index { ptr = p0; idxs = [ c0 ]; dtype = ptr }) in
  let ld = B.emit b (Load { src = idx0; alt = None; gate = None; dtype = dt }) in
  let sq = B.emit b (Unary { op = `Sqrt; src = ld; dtype = dt }) in
  let idx1 = B.emit b (Index { ptr = p1; idxs = [ c0 ]; dtype = ptr }) in
  let _ = B.emit b (Store { dst = idx1; value = sq; gate = None }) in
  B.finish b

let make_special_dims () =
  let dt = Dtype.Val.float32 in
  let ptr = global_ptr dt in
  let b = B.create () in
  let p0 = B.emit b (Param { slot = 0; dtype = ptr }) in
  let c32 = B.emit b (Const { value = Const.int Dtype.Val.int32 32; dtype = Dtype.Val.int32 }) in
  let gid = B.emit b (Special { dim = Gpu_dim.Group_id 0; size = c32; dtype = Dtype.Val.int32 }) in
  let lid = B.emit b (Special { dim = Gpu_dim.Local_id 0; size = c32; dtype = Dtype.Val.int32 }) in
  let sum = B.emit b (Binary { op = `Add; lhs = gid; rhs = lid; dtype = Dtype.Val.int32 }) in
  let idx0 = B.emit b (Index { ptr = p0; idxs = [ sum ]; dtype = ptr }) in
  let ld = B.emit b (Load { src = idx0; alt = None; gate = None; dtype = dt }) in
  let idx1 = B.emit b (Index { ptr = p0; idxs = [ sum ]; dtype = ptr }) in
  let _ = B.emit b (Store { dst = idx1; value = ld; gate = None }) in
  B.finish b

let make_bitcast_f32_to_i32 () =
  let from_dt = Dtype.Val.float32 in
  let to_dt = Dtype.Val.int32 in
  let from_ptr = global_ptr from_dt in
  let to_ptr = global_ptr to_dt in
  let b = B.create () in
  let p0 = B.emit b (Param { slot = 0; dtype = from_ptr }) in
  let p1 = B.emit b (Param { slot = 1; dtype = to_ptr }) in
  let c0 = B.emit b (Const { value = Const.int Dtype.Val.int32 0; dtype = Dtype.Val.int32 }) in
  let idx0 = B.emit b (Index { ptr = p0; idxs = [ c0 ]; dtype = from_ptr }) in
  let ld = B.emit b (Load { src = idx0; alt = None; gate = None; dtype = from_dt }) in
  let bc = B.emit b (Bitcast { src = ld; dtype = to_dt }) in
  let idx1 = B.emit b (Index { ptr = p1; idxs = [ c0 ]; dtype = to_ptr }) in
  let _ = B.emit b (Store { dst = idx1; value = bc; gate = None }) in
  B.finish b

let make_conditional () =
  let dt = Dtype.Val.float32 in
  let ptr = global_ptr dt in
  let b = B.create () in
  let p0 = B.emit b (Param { slot = 0; dtype = ptr }) in
  let c0 = B.emit b (Const { value = Const.int Dtype.Val.int32 0; dtype = Dtype.Val.int32 }) in
  let cond = B.emit b (Const { value = Const.bool true; dtype = Dtype.Val.bool }) in
  let if_ = B.emit b (If { cond; idx_for_dedup = c0 }) in
  let idx0 = B.emit b (Index { ptr = p0; idxs = [ c0 ]; dtype = ptr }) in
  let fone = B.emit b (Const { value = Const.float dt 1.0; dtype = dt }) in
  let _ = B.emit b (Store { dst = idx0; value = fone; gate = None }) in
  let _ = B.emit b (Endif { if_ }) in
  B.finish b

let make_const_inf_nan () =
  let dt = Dtype.Val.float32 in
  let ptr = global_ptr dt in
  let b = B.create () in
  let p0 = B.emit b (Param { slot = 0; dtype = ptr }) in
  let c0 = B.emit b (Const { value = Const.int Dtype.Val.int32 0; dtype = Dtype.Val.int32 }) in
  let c1 = B.emit b (Const { value = Const.int Dtype.Val.int32 1; dtype = Dtype.Val.int32 }) in
  let finf = B.emit b (Const { value = Const.float dt infinity; dtype = dt }) in
  let fnan = B.emit b (Const { value = Const.float dt nan; dtype = dt }) in
  let idx0 = B.emit b (Index { ptr = p0; idxs = [ c0 ]; dtype = ptr }) in
  let _ = B.emit b (Store { dst = idx0; value = finf; gate = None }) in
  let idx1 = B.emit b (Index { ptr = p0; idxs = [ c1 ]; dtype = ptr }) in
  let _ = B.emit b (Store { dst = idx1; value = fnan; gate = None }) in
  B.finish b

let make_vectorize_index () =
  let dt = Dtype.Val.float32 in
  let vdt = Dtype.Val.vec 4 dt in
  let ptr = global_ptr dt in
  let b = B.create () in
  let p0 = B.emit b (Param { slot = 0; dtype = ptr }) in
  let p1 = B.emit b (Param { slot = 1; dtype = ptr }) in
  let c0 = B.emit b (Const { value = Const.int Dtype.Val.int32 0; dtype = Dtype.Val.int32 }) in
  let c1 = B.emit b (Const { value = Const.int Dtype.Val.int32 1; dtype = Dtype.Val.int32 }) in
  let c2 = B.emit b (Const { value = Const.int Dtype.Val.int32 2; dtype = Dtype.Val.int32 }) in
  let c3 = B.emit b (Const { value = Const.int Dtype.Val.int32 3; dtype = Dtype.Val.int32 }) in
  let idx0 = B.emit b (Index { ptr = p0; idxs = [ c0 ]; dtype = ptr }) in
  let idx1 = B.emit b (Index { ptr = p0; idxs = [ c1 ]; dtype = ptr }) in
  let idx2 = B.emit b (Index { ptr = p0; idxs = [ c2 ]; dtype = ptr }) in
  let idx3 = B.emit b (Index { ptr = p0; idxs = [ c3 ]; dtype = ptr }) in
  let ld0 = B.emit b (Load { src = idx0; alt = None; gate = None; dtype = dt }) in
  let ld1 = B.emit b (Load { src = idx1; alt = None; gate = None; dtype = dt }) in
  let ld2 = B.emit b (Load { src = idx2; alt = None; gate = None; dtype = dt }) in
  let ld3 = B.emit b (Load { src = idx3; alt = None; gate = None; dtype = dt }) in
  let vec = B.emit b (Stack { srcs = [ ld0; ld1; ld2; ld3 ]; dtype = vdt }) in
  let lane = B.emit b (Value_index { src = vec; idxs = [ c2 ]; dtype = dt }) in
  let oidx = B.emit b (Index { ptr = p1; idxs = [ c0 ]; dtype = ptr }) in
  let _ = B.emit b (Store { dst = oidx; value = lane; gate = None }) in
  B.finish b

let make_vectorize_index_scalarized () =
  let dt = Dtype.Val.float32 in
  let ptr = global_ptr dt in
  let b = B.create () in
  let p0 = B.emit b (Param { slot = 0; dtype = ptr }) in
  let p1 = B.emit b (Param { slot = 1; dtype = ptr }) in
  let c0 = B.emit b (Const { value = Const.int Dtype.Val.int32 0; dtype = Dtype.Val.int32 }) in
  let c1 = B.emit b (Const { value = Const.int Dtype.Val.int32 1; dtype = Dtype.Val.int32 }) in
  let c2 = B.emit b (Const { value = Const.int Dtype.Val.int32 2; dtype = Dtype.Val.int32 }) in
  let c3 = B.emit b (Const { value = Const.int Dtype.Val.int32 3; dtype = Dtype.Val.int32 }) in
  let idx0 = B.emit b (Index { ptr = p0; idxs = [ c0 ]; dtype = ptr }) in
  let idx1 = B.emit b (Index { ptr = p0; idxs = [ c1 ]; dtype = ptr }) in
  let idx2 = B.emit b (Index { ptr = p0; idxs = [ c2 ]; dtype = ptr }) in
  let idx3 = B.emit b (Index { ptr = p0; idxs = [ c3 ]; dtype = ptr }) in
  let _ = B.emit b (Load { src = idx0; alt = None; gate = None; dtype = dt }) in
  let _ = B.emit b (Load { src = idx1; alt = None; gate = None; dtype = dt }) in
  let ld2 = B.emit b (Load { src = idx2; alt = None; gate = None; dtype = dt }) in
  let _ = B.emit b (Load { src = idx3; alt = None; gate = None; dtype = dt }) in
  let oidx = B.emit b (Index { ptr = p1; idxs = [ c0 ]; dtype = ptr }) in
  let _ = B.emit b (Store { dst = oidx; value = ld2; gate = None }) in
  B.finish b

(* Test cases: (name, builder, backends). backends = None means all backends,
   Some [...] limits to those. *)

type test_case = {
  name : string;
  prog : Program_spec.program;
  backends : (string * Renderer.t) list;
}

let all_renderers =
  [
    ("clang", Cstyle.clang_no_abi Gpu_target.X86_64);
    ("cuda", Cstyle.cuda Gpu_target.SM80);
    ("metal", Cstyle.metal (Gpu_target.Apple 7));
    ("opencl", Cstyle.opencl "");
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
      name = "vectorize_index";
      prog = make_vectorize_index ();
      backends = [ ("metal", Cstyle.metal (Gpu_target.Apple 7)); ("opencl", Cstyle.opencl "") ];
    };
    {
      name = "vectorize_index";
      prog = make_vectorize_index_scalarized ();
      backends =
        [ ("clang", Cstyle.clang_no_abi Gpu_target.X86_64); ("cuda", Cstyle.cuda Gpu_target.SM80) ];
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
