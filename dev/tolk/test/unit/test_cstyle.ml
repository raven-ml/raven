(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tolk
open Tolk_ir
module P = Program

let all_renderers =
  [
    ("clang", Cstyle.clang);
    ("cuda", Cstyle.cuda Gpu_target.SM80);
    ("metal", Cstyle.metal);
    ("opencl", Cstyle.opencl);
  ]

let gpu_renderers = List.filter (fun (name, _) -> name <> "clang") all_renderers

(* Helpers *)

let dt = Dtype.float32
let global_ptr dt = Dtype.Ptr.create dt ~addrspace:Global ()
let local_ptr dt = Dtype.Ptr.create dt ~addrspace:Local ()
let render r prog = Renderer.render r prog
let render_with_images r prog = Renderer.render r (Images.rewrite r prog)
let int32_c n = Const.int Dtype.int32 n
let float_c dt v = Const.float dt v

let contains haystack needle =
  let hl = String.length haystack and nl = String.length needle in
  if nl = 0 then true
  else if nl > hl then false
  else
    let rec loop i =
      if i > hl - nl then false
      else if String.sub haystack i nl = needle then true
      else loop (i + 1)
    in
    loop 0

let count_char s c =
  let n = ref 0 in
  String.iter (fun ch -> if ch = c then incr n) s;
  !n

let count_substring s sub =
  let sl = String.length s and nl = String.length sub in
  let rec loop i acc =
    if i > sl - nl then acc
    else if String.sub s i nl = sub then loop (i + 1) (acc + 1)
    else loop (i + 1) acc
  in
  loop 0 0

let assert_contains msg haystack needle =
  if not (contains haystack needle) then
    failwith
      (Printf.sprintf "%s: expected output to contain %S, got:\n%s" msg needle
         haystack)

let assert_not_contains msg haystack needle =
  if contains haystack needle then
    failwith
      (Printf.sprintf "%s: expected output NOT to contain %S, got:\n%s" msg
         needle haystack)

let for_each_renderer renderers f =
  List.iter (fun (name, renderer) -> f name renderer) renderers

let assert_equal_string msg expected actual =
  if not (String.equal expected actual) then
    failwith
      (Printf.sprintf "%s: expected:\n%s\n\ngot:\n%s" msg expected actual)

(* IR Program Builders *)

let make_store_const dt const_value =
  let ptr = global_ptr dt in
  let b = P.create () in
  let p0 = P.emit b (Param { idx = 0; dtype = ptr }) in
  let cv = P.emit b (Const { value = const_value; dtype = dt }) in
  let c0 = P.emit b (Const { value = int32_c 0; dtype = Dtype.int32 }) in
  let idx = P.emit b (Index { ptr = p0; idxs = [ c0 ]; gate = None; dtype = ptr }) in
  let _ = P.emit b (Store { dst = idx; value = cv }) in
  P.finish b

let make_binop dt mk_op =
  let ptr = global_ptr dt in
  let b = P.create () in
  let p0 = P.emit b (Param { idx = 0; dtype = ptr }) in
  let p1 = P.emit b (Param { idx = 1; dtype = ptr }) in
  let p2 = P.emit b (Param { idx = 2; dtype = ptr }) in
  let c0 = P.emit b (Const { value = int32_c 0; dtype = Dtype.int32 }) in
  let idx0 = P.emit b (Index { ptr = p0; idxs = [ c0 ]; gate = None; dtype = ptr }) in
  let idx1 = P.emit b (Index { ptr = p1; idxs = [ c0 ]; gate = None; dtype = ptr }) in
  let idx2 = P.emit b (Index { ptr = p2; idxs = [ c0 ]; gate = None; dtype = ptr }) in
  let ld0 = P.emit b (Load { src = idx0; alt = None; dtype = dt }) in
  let ld1 = P.emit b (Load { src = idx1; alt = None; dtype = dt }) in
  let op_result = P.emit b (mk_op ld0 ld1 dt) in
  let _ = P.emit b (Store { dst = idx2; value = op_result }) in
  P.finish b

let make_simple_add_f32 () =
  make_binop dt (fun lhs rhs dtype ->
      P.Binary { op = `Add; lhs; rhs; dtype })

let make_unop dt mk_op =
  let ptr = global_ptr dt in
  let b = P.create () in
  let p0 = P.emit b (Param { idx = 0; dtype = ptr }) in
  let p1 = P.emit b (Param { idx = 1; dtype = ptr }) in
  let c0 = P.emit b (Const { value = int32_c 0; dtype = Dtype.int32 }) in
  let idx0 = P.emit b (Index { ptr = p0; idxs = [ c0 ]; gate = None; dtype = ptr }) in
  let idx1 = P.emit b (Index { ptr = p1; idxs = [ c0 ]; gate = None; dtype = ptr }) in
  let ld = P.emit b (Load { src = idx0; alt = None; dtype = dt }) in
  let op_result = P.emit b (mk_op ld dt) in
  let _ = P.emit b (Store { dst = idx1; value = op_result }) in
  P.finish b

let make_ternary_where dt =
  let ptr = global_ptr dt in
  let b = P.create () in
  let p0 = P.emit b (Param { idx = 0; dtype = ptr }) in
  let p1 = P.emit b (Param { idx = 1; dtype = ptr }) in
  let p2 = P.emit b (Param { idx = 2; dtype = ptr }) in
  let c0 = P.emit b (Const { value = int32_c 0; dtype = Dtype.int32 }) in
  let idx0 = P.emit b (Index { ptr = p0; idxs = [ c0 ]; gate = None; dtype = ptr }) in
  let idx1 = P.emit b (Index { ptr = p1; idxs = [ c0 ]; gate = None; dtype = ptr }) in
  let idx2 = P.emit b (Index { ptr = p2; idxs = [ c0 ]; gate = None; dtype = ptr }) in
  let ld0 = P.emit b (Load { src = idx0; alt = None; dtype = dt }) in
  let ld1 = P.emit b (Load { src = idx1; alt = None; dtype = dt }) in
  let cond = P.emit b (Const { value = Const.bool true; dtype = Dtype.bool }) in
  let w = P.emit b (Ternary { op = `Where; a = cond; b = ld0; c = ld1; dtype = dt }) in
  let _ = P.emit b (Store { dst = idx2; value = w }) in
  P.finish b

let make_mulacc dt =
  let ptr = global_ptr dt in
  let b = P.create () in
  let p0 = P.emit b (Param { idx = 0; dtype = ptr }) in
  let p1 = P.emit b (Param { idx = 1; dtype = ptr }) in
  let p2 = P.emit b (Param { idx = 2; dtype = ptr }) in
  let p3 = P.emit b (Param { idx = 3; dtype = ptr }) in
  let c0 = P.emit b (Const { value = int32_c 0; dtype = Dtype.int32 }) in
  let idx0 = P.emit b (Index { ptr = p0; idxs = [ c0 ]; gate = None; dtype = ptr }) in
  let idx1 = P.emit b (Index { ptr = p1; idxs = [ c0 ]; gate = None; dtype = ptr }) in
  let idx2 = P.emit b (Index { ptr = p2; idxs = [ c0 ]; gate = None; dtype = ptr }) in
  let idx3 = P.emit b (Index { ptr = p3; idxs = [ c0 ]; gate = None; dtype = ptr }) in
  let ld0 = P.emit b (Load { src = idx0; alt = None; dtype = dt }) in
  let ld1 = P.emit b (Load { src = idx1; alt = None; dtype = dt }) in
  let ld2 = P.emit b (Load { src = idx2; alt = None; dtype = dt }) in
  let mac = P.emit b (Ternary { op = `Mulacc; a = ld0; b = ld1; c = ld2; dtype = dt }) in
  let _ = P.emit b (Store { dst = idx3; value = mac }) in
  P.finish b

let make_loop () =
  let ptr = global_ptr dt in
  let b = P.create () in
  let p0 = P.emit b (Param { idx = 0; dtype = ptr }) in
  let c10 = P.emit b (Const { value = int32_c 10; dtype = Dtype.int32 }) in
  let r = P.emit b (Range { size = c10; dtype = Dtype.int32; axis = 0; kind = Axis_kind.Loop }) in
  let idx0 = P.emit b (Index { ptr = p0; idxs = [ r ]; gate = None; dtype = ptr }) in
  let ld = P.emit b (Load { src = idx0; alt = None; dtype = dt }) in
  let idx1 = P.emit b (Index { ptr = p0; idxs = [ r ]; gate = None; dtype = ptr }) in
  let _ = P.emit b (Store { dst = idx1; value = ld }) in
  let _ = P.emit b (End_range { range = r }) in
  P.finish b

let make_nested_loops () =
  let ptr = global_ptr dt in
  let b = P.create () in
  let p0 = P.emit b (Param { idx = 0; dtype = ptr }) in
  let c10 = P.emit b (Const { value = int32_c 10; dtype = Dtype.int32 }) in
  let c5 = P.emit b (Const { value = int32_c 5; dtype = Dtype.int32 }) in
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

let make_special dim =
  let ptr = global_ptr Dtype.int32 in
  let b = P.create () in
  let p0 = P.emit b (Param { idx = 0; dtype = ptr }) in
  let c64 = P.emit b (Const { value = int32_c 64; dtype = Dtype.int32 }) in
  let sp = P.emit b (Special { dim; size = c64; dtype = Dtype.int32 }) in
  let idx = P.emit b (Index { ptr = p0; idxs = [ sp ]; gate = None; dtype = ptr }) in
  let _ = P.emit b (Store { dst = idx; value = sp }) in
  P.finish b

let make_shared_memory () =
  let gptr = global_ptr dt in
  let lptr = local_ptr dt in
  let b = P.create () in
  let p0 = P.emit b (Param { idx = 0; dtype = gptr }) in
  let dl = P.emit b (Define_local { size = 256; dtype = lptr }) in
  let c0 = P.emit b (Const { value = int32_c 0; dtype = Dtype.int32 }) in
  let lidx = P.emit b (Index { ptr = dl; idxs = [ c0 ]; gate = None; dtype = lptr }) in
  let fzero = P.emit b (Const { value = float_c dt 0.0; dtype = dt }) in
  let _ = P.emit b (Store { dst = lidx; value = fzero }) in
  let _ = P.emit b Barrier in
  let ld = P.emit b (Load { src = lidx; alt = None; dtype = dt }) in
  let gidx = P.emit b (Index { ptr = p0; idxs = [ c0 ]; gate = None; dtype = gptr }) in
  let _ = P.emit b (Store { dst = gidx; value = ld }) in
  P.finish b

let make_gated_load () =
  let ptr = global_ptr dt in
  let b = P.create () in
  let p0 = P.emit b (Param { idx = 0; dtype = ptr }) in
  let p1 = P.emit b (Param { idx = 1; dtype = ptr }) in
  let c0 = P.emit b (Const { value = int32_c 0; dtype = Dtype.int32 }) in
  let gate = P.emit b (Const { value = Const.bool true; dtype = Dtype.bool }) in
  let idx0 = P.emit b (Index { ptr = p0; idxs = [ c0 ]; gate = Some gate; dtype = ptr }) in
  let alt = P.emit b (Const { value = float_c dt 0.0; dtype = dt }) in
  let ld = P.emit b (Load { src = idx0; alt = Some alt; dtype = dt }) in
  let idx1 = P.emit b (Index { ptr = p1; idxs = [ c0 ]; gate = None; dtype = ptr }) in
  let _ = P.emit b (Store { dst = idx1; value = ld }) in
  P.finish b

let make_image_load () =
  let image_ptr = global_ptr dt in
  let vec_dt = Dtype.vec dt 4 in
  let vec_ptr = global_ptr vec_dt in
  let b = P.create () in
  let p0 = P.emit b (Param_image { idx = 0; dtype = image_ptr; width = 4; height = 4 }) in
  let p1 = P.emit b (Param { idx = 1; dtype = vec_ptr }) in
  let c0 = P.emit b (Const { value = int32_c 0; dtype = Dtype.int32 }) in
  let c1 = P.emit b (Const { value = int32_c 1; dtype = Dtype.int32 }) in
  let idx0 = P.emit b (Index { ptr = p0; idxs = [ c0; c1 ]; gate = None; dtype = image_ptr }) in
  let ld = P.emit b (Load { src = idx0; alt = None; dtype = vec_dt }) in
  let idx1 = P.emit b (Index { ptr = p1; idxs = [ c0 ]; gate = None; dtype = vec_ptr }) in
  let _ = P.emit b (Store { dst = idx1; value = ld }) in
  P.finish b

let make_image_store () =
  let image_ptr = global_ptr dt in
  let vec_dt = Dtype.vec dt 4 in
  let vec_ptr = global_ptr vec_dt in
  let b = P.create () in
  let p0 = P.emit b (Param_image { idx = 0; dtype = image_ptr; width = 4; height = 4 }) in
  let p1 = P.emit b (Param { idx = 1; dtype = vec_ptr }) in
  let c0 = P.emit b (Const { value = int32_c 0; dtype = Dtype.int32 }) in
  let c1 = P.emit b (Const { value = int32_c 1; dtype = Dtype.int32 }) in
  let idx0 = P.emit b (Index { ptr = p1; idxs = [ c0 ]; gate = None; dtype = vec_ptr }) in
  let ld = P.emit b (Load { src = idx0; alt = None; dtype = vec_dt }) in
  let idx1 = P.emit b (Index { ptr = p0; idxs = [ c0; c1 ]; gate = None; dtype = image_ptr }) in
  let _ = P.emit b (Store { dst = idx1; value = ld }) in
  P.finish b

let make_type_convert ~from_dt ~to_dt mk_convert =
  let from_ptr = global_ptr from_dt in
  let to_ptr = global_ptr to_dt in
  let b = P.create () in
  let p0 = P.emit b (Param { idx = 0; dtype = from_ptr }) in
  let p1 = P.emit b (Param { idx = 1; dtype = to_ptr }) in
  let c0 = P.emit b (Const { value = int32_c 0; dtype = Dtype.int32 }) in
  let idx0 = P.emit b (Index { ptr = p0; idxs = [ c0 ]; gate = None; dtype = from_ptr }) in
  let idx1 = P.emit b (Index { ptr = p1; idxs = [ c0 ]; gate = None; dtype = to_ptr }) in
  let ld = P.emit b (Load { src = idx0; alt = None; dtype = from_dt }) in
  let converted = P.emit b (mk_convert ld) in
  let _ = P.emit b (Store { dst = idx1; value = converted }) in
  P.finish b

let make_cast ~from_dt ~to_dt =
  make_type_convert ~from_dt ~to_dt (fun src -> P.Cast { src; dtype = to_dt })

let make_bitcast ~from_dt ~to_dt =
  make_type_convert ~from_dt ~to_dt (fun src -> P.Bitcast { src; dtype = to_dt })

let make_vectorize_gep () =
  let vdt = Dtype.vec dt 4 in
  let ptr = global_ptr dt in
  let b = P.create () in
  let p0 = P.emit b (Param { idx = 0; dtype = ptr }) in
  let p1 = P.emit b (Param { idx = 1; dtype = ptr }) in
  let c0 = P.emit b (Const { value = int32_c 0; dtype = Dtype.int32 }) in
  let c1 = P.emit b (Const { value = int32_c 1; dtype = Dtype.int32 }) in
  let c2 = P.emit b (Const { value = int32_c 2; dtype = Dtype.int32 }) in
  let c3 = P.emit b (Const { value = int32_c 3; dtype = Dtype.int32 }) in
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

let make_custom () =
  let ptr = global_ptr dt in
  let b = P.create () in
  let p0 = P.emit b (Param { idx = 0; dtype = ptr }) in
  let c0 = P.emit b (Const { value = int32_c 0; dtype = Dtype.int32 }) in
  let idx = P.emit b (Index { ptr = p0; idxs = [ c0 ]; gate = None; dtype = ptr }) in
  let ld = P.emit b (Load { src = idx; alt = None; dtype = dt }) in
  let ci = P.emit b (Custom_inline { fmt = "custom_func({0}, {0})"; args = [ ld ]; dtype = dt }) in
  let _ = P.emit b (Store { dst = idx; value = ci }) in
  P.finish b

let make_define_var () =
  let ptr = global_ptr dt in
  let b = P.create () in
  let p0 = P.emit b (Param { idx = 0; dtype = ptr }) in
  let dv = P.emit b (Define_var { name = "n"; lo = 0; hi = 1024; dtype = Dtype.int32 }) in
  let idx = P.emit b (Index { ptr = p0; idxs = [ dv ]; gate = None; dtype = ptr }) in
  let ld = P.emit b (Load { src = idx; alt = None; dtype = dt }) in
  let _ = P.emit b (Store { dst = idx; value = ld }) in
  P.finish b

let make_chained_binop dt mk_op n =
  let ptr = global_ptr dt in
  let b = P.create () in
  let p0 = P.emit b (Param { idx = 0; dtype = ptr }) in
  let p1 = P.emit b (Param { idx = 1; dtype = ptr }) in
  let c0 = P.emit b (Const { value = int32_c 0; dtype = Dtype.int32 }) in
  let idx_in = P.emit b (Index { ptr = p1; idxs = [ c0 ]; gate = None; dtype = ptr }) in
  let ld = P.emit b (Load { src = idx_in; alt = None; dtype = dt }) in
  let result = ref ld in
  for _ = 0 to n - 1 do
    result := P.emit b (mk_op !result ld dt)
  done;
  let idx_out = P.emit b (Index { ptr = p0; idxs = [ c0 ]; gate = None; dtype = ptr }) in
  let _ = P.emit b (Store { dst = idx_out; value = !result }) in
  P.finish b

let make_conditional () =
  let ptr = global_ptr dt in
  let b = P.create () in
  let p0 = P.emit b (Param { idx = 0; dtype = ptr }) in
  let c0 = P.emit b (Const { value = int32_c 0; dtype = Dtype.int32 }) in
  let idx = P.emit b (Index { ptr = p0; idxs = [ c0 ]; gate = None; dtype = ptr }) in
  let cond = P.emit b (Const { value = Const.bool true; dtype = Dtype.bool }) in
  let if_ = P.emit b (If { cond; idx_for_dedup = idx }) in
  let fval = P.emit b (Const { value = float_c dt 42.0; dtype = dt }) in
  let _ = P.emit b (Store { dst = idx; value = fval }) in
  let _ = P.emit b (Endif { if_ }) in
  P.finish b

let make_launch_bounds () =
  let ptr = global_ptr Dtype.int32 in
  let b = P.create () in
  let p0 = P.emit b (Param { idx = 0; dtype = ptr }) in
  let c64 = P.emit b (Const { value = int32_c 64; dtype = Dtype.int32 }) in
  let lid0 = P.emit b (Special { dim = Special_dim.Local_id 0; size = c64; dtype = Dtype.int32 }) in
  let c4 = P.emit b (Const { value = int32_c 4; dtype = Dtype.int32 }) in
  let lid1 = P.emit b (Special { dim = Special_dim.Local_id 1; size = c4; dtype = Dtype.int32 }) in
  let sum = P.emit b (Binary { op = `Add; lhs = lid0; rhs = lid1; dtype = Dtype.int32 }) in
  let idx = P.emit b (Index { ptr = p0; idxs = [ sum ]; gate = None; dtype = ptr }) in
  let _ = P.emit b (Store { dst = idx; value = sum }) in
  P.finish b

(* Frequently-used programs *)

let f32_1 = make_store_const dt (float_c dt 1.0)

(* Comparison program builder: loads from two float32 inputs, applies cmp, stores bool *)
let make_comparison mk_op =
  let in_ptr = global_ptr dt in
  let out_ptr = global_ptr Dtype.bool in
  let b = P.create () in
  let p0 = P.emit b (Param { idx = 0; dtype = in_ptr }) in
  let p1 = P.emit b (Param { idx = 1; dtype = in_ptr }) in
  let p2 = P.emit b (Param { idx = 2; dtype = out_ptr }) in
  let c0 = P.emit b (Const { value = int32_c 0; dtype = Dtype.int32 }) in
  let idx0 = P.emit b (Index { ptr = p0; idxs = [ c0 ]; gate = None; dtype = in_ptr }) in
  let idx1 = P.emit b (Index { ptr = p1; idxs = [ c0 ]; gate = None; dtype = in_ptr }) in
  let idx2 = P.emit b (Index { ptr = p2; idxs = [ c0 ]; gate = None; dtype = out_ptr }) in
  let ld0 = P.emit b (Load { src = idx0; alt = None; dtype = dt }) in
  let ld1 = P.emit b (Load { src = idx1; alt = None; dtype = dt }) in
  let cmp = P.emit b (mk_op ld0 ld1 Dtype.bool) in
  let _ = P.emit b (Store { dst = idx2; value = cmp }) in
  P.finish b

(* Property test support *)

let renderer_testable =
  let gen = Gen.oneofl all_renderers in
  let pp fmt (name, _) = Format.pp_print_string fmt name in
  testable ~pp ~equal:(fun (a, _) (b, _) -> String.equal a b) ~gen ()

let safe_dtypes = [ Dtype.int32; Dtype.float32; Dtype.float64; Dtype.uint32 ]

let safe_dtype =
  let gen = Gen.oneofl safe_dtypes in
  testable ~pp:Dtype.pp ~equal:Dtype.equal ~gen ()

(* Runner *)

let () =
  run "Renderer"
    [
      group "Constants"
        [
          test "int constant" (fun () ->
            let prog = make_store_const Dtype.int32 (int32_c 42) in
            for_each_renderer all_renderers (fun name r ->
                assert_contains (name ^ " int 42") (render r prog) "42"));
          test "float32 constant" (fun () ->
            let prog = make_store_const dt (float_c dt 3.14) in
            for_each_renderer all_renderers (fun name r ->
                let out = render r prog in
                assert_contains (name ^ " float32 3.14") out "3.14";
                assert_contains (name ^ " float32 f suffix") out "f"));
          test "float64 constant" (fun () ->
            let prog = make_store_const Dtype.float64 (float_c Dtype.float64 3.14) in
            for_each_renderer all_renderers (fun name r ->
                assert_contains (name ^ " float64 3.14") (render r prog) "3.14"));
          test "bool constants" (fun () ->
            for_each_renderer all_renderers (fun name r ->
                assert_contains (name ^ " bool true")
                  (render r (make_store_const Dtype.bool (Const.bool true))) "1";
                assert_contains (name ^ " bool false")
                  (render r (make_store_const Dtype.bool (Const.bool false))) "0"));
          test "nan/inf constants" (fun () ->
            let nan_prog = make_store_const dt (float_c dt Float.nan) in
            let inf_prog = make_store_const dt (float_c dt Float.infinity) in
            let neg_inf_prog = make_store_const dt (float_c dt Float.neg_infinity) in
            List.iter
              (fun (name, r) ->
                assert_contains (name ^ " NAN") (render r nan_prog) "NAN";
                assert_contains (name ^ " INFINITY") (render r inf_prog) "INFINITY";
                assert_contains (name ^ " -INFINITY") (render r neg_inf_prog) "INFINITY")
              [
                ("cuda", Cstyle.cuda Gpu_target.SM80);
                ("metal", Cstyle.metal);
                ("opencl", Cstyle.opencl);
              ];
            let nan_out = render Cstyle.clang nan_prog in
            assert_contains "clang NAN" nan_out "__builtin_nanf";
            assert_contains "clang INF" (render Cstyle.clang inf_prog) "__builtin_inff");
          test "int64 suffix" (fun () ->
            let prog = make_store_const Dtype.int64 (Const.int Dtype.int64 12345) in
            for_each_renderer all_renderers (fun name r ->
                assert_contains (name ^ " int64 ll suffix") (render r prog) "12345ll"));
          test "uint32 suffix" (fun () ->
            let prog = make_store_const Dtype.uint32 (Const.int Dtype.uint32 42) in
            for_each_renderer all_renderers (fun name r ->
                assert_contains (name ^ " uint32 u suffix") (render r prog) "42u"));
          test "uint64 suffix" (fun () ->
            let prog = make_store_const Dtype.uint64 (Const.int Dtype.uint64 42) in
            for_each_renderer all_renderers (fun name r ->
                assert_contains (name ^ " uint64 ull suffix") (render r prog) "42ull"));
        ];
      group "ALU Operations"
        [
          group "Binary"
            [
              test "arithmetic operators" (fun () ->
                let ops =
                  [
                    ("Add +", (fun l r dt -> P.Binary { op = `Add; lhs = l; rhs = r; dtype = dt }), "+");
                    ("Sub -", (fun l r dt -> P.Binary { op = `Sub; lhs = l; rhs = r; dtype = dt }), "-");
                    ("Mul *", (fun l r dt -> P.Binary { op = `Mul; lhs = l; rhs = r; dtype = dt }), "*");
                    ("Fdiv /", (fun l r dt -> P.Binary { op = `Fdiv; lhs = l; rhs = r; dtype = dt }), "/");
                    ("Mod %", (fun l r dt -> P.Binary { op = `Mod; lhs = l; rhs = r; dtype = dt }), "%");
                    ("Shl <<", (fun l r dt -> P.Binary { op = `Shl; lhs = l; rhs = r; dtype = dt }), "<<");
                    ("Shr >>", (fun l r dt -> P.Binary { op = `Shr; lhs = l; rhs = r; dtype = dt }), ">>");
                    ("And &", (fun l r dt -> P.Binary { op = `And; lhs = l; rhs = r; dtype = dt }), "&");
                    ("Or |", (fun l r dt -> P.Binary { op = `Or; lhs = l; rhs = r; dtype = dt }), "|");
                    ("Xor ^", (fun l r dt -> P.Binary { op = `Xor; lhs = l; rhs = r; dtype = dt }), "^");
                  ]
                in
                List.iter
                  (fun (label, mk_op, expected) ->
                    let op_dt =
                      if String.length expected = 1 && expected.[0] = '/' then dt
                      else Dtype.int32
                    in
                    let prog = make_binop op_dt mk_op in
                    for_each_renderer all_renderers (fun name r ->
                        assert_contains (name ^ " " ^ label) (render r prog) expected))
                  ops);
              test "integer division" (fun () ->
                let prog =
                  make_binop Dtype.int32 (fun l r dt ->
                      P.Binary { op = `Idiv; lhs = l; rhs = r; dtype = dt })
                in
                for_each_renderer all_renderers (fun name r ->
                    assert_contains (name ^ " Idiv /") (render r prog) "/"));
              test "comparison operators" (fun () ->
                let ops =
                  [
                    ("Cmplt <", (fun l r dt -> P.Binary { op = `Cmplt; lhs = l; rhs = r; dtype = dt }), "<");
                    ("Cmpeq ==", (fun l r dt -> P.Binary { op = `Cmpeq; lhs = l; rhs = r; dtype = dt }), "==");
                    ("Cmpne !=", (fun l r dt -> P.Binary { op = `Cmpne; lhs = l; rhs = r; dtype = dt }), "!=");
                  ]
                in
                List.iter
                  (fun (label, mk_op, expected) ->
                    let prog = make_comparison mk_op in
                    for_each_renderer all_renderers (fun name r ->
                        assert_contains (name ^ " " ^ label) (render r prog) expected))
                  ops);
              test "max" (fun () ->
                let prog =
                  make_binop dt (fun l r dt ->
                      P.Binary { op = `Max; lhs = l; rhs = r; dtype = dt })
                in
                for_each_renderer all_renderers (fun name r ->
                    raises_match
                      (function
                        | Invalid_argument msg -> contains msg "not handled"
                        | _ -> false)
                      (fun () ->
                        ignore (render r prog);
                        failwith (name ^ " should reject raw Max in renderer"))));
            ];
          group "Unary"
            [
              test "operators" (fun () ->
                let ops =
                  [
                    ("Neg", (fun s dt -> P.Unary { op = `Neg; src = s; dtype = dt }), "-");
                    ("Exp2", (fun s dt -> P.Unary { op = `Exp2; src = s; dtype = dt }), "exp2");
                    ("Log2", (fun s dt -> P.Unary { op = `Log2; src = s; dtype = dt }), "log2");
                    ("Sin", (fun s dt -> P.Unary { op = `Sin; src = s; dtype = dt }), "sin");
                    ("Sqrt", (fun s dt -> P.Unary { op = `Sqrt; src = s; dtype = dt }), "sqrt");
                    ("Trunc", (fun s dt -> P.Unary { op = `Trunc; src = s; dtype = dt }), "trunc");
                  ]
                in
                List.iter
                  (fun (label, mk_op, expected) ->
                    let prog = make_unop dt mk_op in
                    for_each_renderer all_renderers (fun name r ->
                        assert_contains (name ^ " " ^ label) (render r prog) expected))
                  ops);
              test "reciprocal" (fun () ->
                let prog =
                  make_unop dt (fun s dt -> P.Unary { op = `Recip; src = s; dtype = dt })
                in
                for_each_renderer all_renderers (fun name r ->
                    assert_contains (name ^ " Recip") (render r prog) "1/"));
            ];
          group "Ternary"
            [
              test "where" (fun () ->
                let prog = make_ternary_where dt in
                for_each_renderer all_renderers (fun name r ->
                    let out = render r prog in
                    assert_contains (name ^ " Where ?") out "?";
                    assert_contains (name ^ " Where :") out ":"));
              test "mulacc" (fun () ->
                let prog = make_mulacc dt in
                for_each_renderer all_renderers (fun name r ->
                    raises_match
                      (function
                        | Invalid_argument msg -> contains msg "not handled"
                        | _ -> false)
                      (fun () ->
                        ignore (render r prog);
                        failwith (name ^ " should reject raw Mulacc in renderer"))));
            ];
          group "Backend-specific"
            [
              test "CUDA half intrinsics" (fun () ->
                let cuda = Cstyle.cuda Gpu_target.SM80 in
                List.iter
                  (fun (expected, mk_op) ->
                    let out = render cuda (make_unop Dtype.float16 mk_op) in
                    assert_contains ("CUDA " ^ expected) out expected)
                  [
                    ("hexp2", fun s dt -> P.Unary { op = `Exp2; src = s; dtype = dt });
                    ("hlog2", fun s dt -> P.Unary { op = `Log2; src = s; dtype = dt });
                    ("hsin", fun s dt -> P.Unary { op = `Sin; src = s; dtype = dt });
                    ("hsqrt", fun s dt -> P.Unary { op = `Sqrt; src = s; dtype = dt });
                    ("hrcp", fun s dt -> P.Unary { op = `Recip; src = s; dtype = dt });
                    ("htrunc", fun s dt -> P.Unary { op = `Trunc; src = s; dtype = dt });
                  ]);
              test "Metal precise sin" (fun () ->
                let prog =
                  make_unop dt (fun s dt -> P.Unary { op = `Sin; src = s; dtype = dt })
                in
                assert_contains "Metal precise::sin" (render Cstyle.metal prog) "precise::sin");
              test "Clang builtins" (fun () ->
                let clang = Cstyle.clang in
                let sqrt_out =
                  render clang
                    (make_unop dt (fun s dt -> P.Unary { op = `Sqrt; src = s; dtype = dt }))
                in
                assert_contains "clang __builtin_sqrtf" sqrt_out "__builtin_sqrtf";
                let trunc_out =
                  render clang
                    (make_unop dt (fun s dt -> P.Unary { op = `Trunc; src = s; dtype = dt }))
                in
                assert_contains "clang __builtin_truncf" trunc_out "__builtin_truncf");
            ];
          test "paren stripping" (fun () ->
            let mk_add l r dt = P.Binary { op = `Add; lhs = l; rhs = r; dtype = dt } in
            let mk_sub l r dt = P.Binary { op = `Sub; lhs = l; rhs = r; dtype = dt } in
            let mk_mul l r dt = P.Binary { op = `Mul; lhs = l; rhs = r; dtype = dt } in
            let mk_xor l r dt = P.Binary { op = `Xor; lhs = l; rhs = r; dtype = dt } in
            let mk_or l r dt = P.Binary { op = `Or; lhs = l; rhs = r; dtype = dt } in
            let mk_and l r dt = P.Binary { op = `And; lhs = l; rhs = r; dtype = dt } in
            let prog_add = make_chained_binop dt mk_add 5 in
            let prog_sub = make_chained_binop dt mk_sub 5 in
            let prog_mul = make_chained_binop dt mk_mul 5 in
            let prog_xor = make_chained_binop Dtype.int32 mk_xor 5 in
            let prog_or = make_chained_binop Dtype.int32 mk_or 5 in
            let prog_and = make_chained_binop Dtype.int32 mk_and 5 in
            for_each_renderer all_renderers (fun name r ->
                assert_not_contains (name ^ " Add no deep parens") (render r prog_add) "(((((";
                assert_not_contains (name ^ " Mul no deep parens") (render r prog_mul) "(((((";
                assert_not_contains (name ^ " Xor no deep parens") (render r prog_xor) "(((((";
                assert_not_contains (name ^ " Or no deep parens") (render r prog_or) "(((((";
                assert_not_contains (name ^ " And no deep parens") (render r prog_and) "(((((";
                assert_contains (name ^ " Sub deep parens") (render r prog_sub) "((((("));
        ];
      group "Control Flow"
        [
          test "for loop" (fun () ->
            let prog = make_loop () in
            for_each_renderer all_renderers (fun name r ->
                assert_contains (name ^ " for loop") (render r prog) "for ("));
          test "nested loops" (fun () ->
            let prog = make_nested_loops () in
            for_each_renderer all_renderers (fun name r ->
                let out = render r prog in
                let count = count_substring out "for " in
                if count < 2 then
                  failwith
                    (Printf.sprintf "%s: expected 2 'for ' occurrences, got %d" name count)));
          test "conditional" (fun () ->
            let prog = make_conditional () in
            for_each_renderer all_renderers (fun name r ->
                assert_contains (name ^ " if") (render r prog) "if ("));
        ];
      group "Memory"
        [
          test "simple load/store" (fun () ->
            let prog =
              make_binop dt (fun l r dt ->
                  P.Binary { op = `Add; lhs = l; rhs = r; dtype = dt })
            in
            for_each_renderer all_renderers (fun name r ->
                assert_contains (name ^ " dereference") (render r prog) "*"));
          test "gated load" (fun () ->
            let prog = make_gated_load () in
            for_each_renderer all_renderers (fun name r ->
                assert_contains (name ^ " gated load ternary") (render r prog) "?"));
          test "opencl image load/store" (fun () ->
            let load_out = render_with_images Cstyle.opencl (make_image_load ()) in
            assert_contains "opencl image param" load_out "read_only image2d_t";
            assert_contains "opencl sampler preamble" load_out "const sampler_t smp";
            assert_contains "opencl read_imagef" load_out "read_imagef(";
            let store_out = render_with_images Cstyle.opencl (make_image_store ()) in
            assert_contains "opencl mutable image param" store_out "write_only image2d_t";
            assert_contains "opencl write_imagef" store_out "write_imagef(");
          test "non-opencl image rejected" (fun () ->
            raises_match
              (function
                | Failure msg -> contains msg "does not support OpenCL image parameters"
                | _ -> false)
              (fun () -> ignore (Images.rewrite Cstyle.metal (make_image_load ()))));
        ];
      group "Cast and Bitcast"
        [
          test "cast per backend" (fun () ->
            let prog = make_cast ~from_dt:Dtype.int32 ~to_dt:dt in
            let metal_out = render Cstyle.metal prog in
            assert_contains "metal cast" metal_out "(float)";
            let cuda_out = render (Cstyle.cuda Gpu_target.SM80) prog in
            assert_contains "cuda cast" cuda_out "(float)";
            let opencl_out = render Cstyle.opencl prog in
            assert_contains "opencl cast" opencl_out "(float)";
            assert_contains "clang cast" (render Cstyle.clang prog) "(float)");
          test "bitcast per backend" (fun () ->
            let prog = make_bitcast ~from_dt:dt ~to_dt:Dtype.int32 in
            assert_contains "clang __builtin_bit_cast"
              (render Cstyle.clang prog) "__builtin_bit_cast";
            assert_contains "cuda tg_bitcast"
              (render (Cstyle.cuda Gpu_target.SM80) prog) "tg_bitcast";
            assert_contains "metal as_type"
              (render Cstyle.metal prog) "as_type<";
            assert_contains "opencl as_"
              (render Cstyle.opencl prog) "as_");
        ];
      group "Special Dimensions"
        [
          test "Group_id" (fun () ->
            let prog = make_special (Special_dim.Group_id 0) in
            assert_contains "cuda blockIdx.x"
              (render (Cstyle.cuda Gpu_target.SM80) prog) "blockIdx.x";
            assert_contains "metal gid.x"
              (render Cstyle.metal prog) "gid.x";
            assert_contains "opencl get_group_id(0)"
              (render Cstyle.opencl prog) "get_group_id(0)");
          test "Local_id" (fun () ->
            let prog = make_special (Special_dim.Local_id 1) in
            assert_contains "cuda threadIdx.y"
              (render (Cstyle.cuda Gpu_target.SM80) prog) "threadIdx.y";
            assert_contains "metal lid.y"
              (render Cstyle.metal prog) "lid.y";
            assert_contains "opencl get_local_id(1)"
              (render Cstyle.opencl prog) "get_local_id(1)");
          test "Global_idx" (fun () ->
            let prog = make_special (Special_dim.Global_idx 2) in
            assert_contains "cuda blockIdx.z"
              (render (Cstyle.cuda Gpu_target.SM80) prog) "blockIdx.z";
            assert_contains "metal gid.z"
              (render Cstyle.metal prog) "gid.z";
            assert_contains "opencl get_global_id(2)"
              (render Cstyle.opencl prog) "get_global_id(2)");
          test "Clang fails" (fun () ->
            raises_match
              (function Failure _ -> true | _ -> false)
              (fun () -> ignore (render Cstyle.clang (make_special (Special_dim.Group_id 0)))));
        ];
      group "Shared Memory and Barrier"
        [
          test "shared memory qualifiers" (fun () ->
            let prog = make_shared_memory () in
            assert_contains "cuda __shared__"
              (render (Cstyle.cuda Gpu_target.SM80) prog) "__shared__";
            assert_contains "metal threadgroup"
              (render Cstyle.metal prog) "threadgroup";
            assert_contains "opencl __local"
              (render Cstyle.opencl prog) "__local");
          test "barrier syntax" (fun () ->
            let prog = make_shared_memory () in
            assert_contains "cuda __syncthreads"
              (render (Cstyle.cuda Gpu_target.SM80) prog) "__syncthreads()";
            assert_contains "metal threadgroup_barrier"
              (render Cstyle.metal prog) "threadgroup_barrier";
            assert_contains "opencl barrier"
              (render Cstyle.opencl prog) "barrier(CLK_LOCAL_MEM_FENCE)");
        ];
      group "Vectorize and Gep"
        [
          test "vectorize" (fun () ->
            let prog = make_vectorize_gep () in
            for_each_renderer all_renderers (fun name r ->
                assert_contains (name ^ " vectorize val elements")
                  (render r prog) "val0,val1,val2,val3"));
          test "gep" (fun () ->
            let prog = make_vectorize_gep () in
            for_each_renderer all_renderers (fun name r ->
                let out = render r prog in
                if not (contains out "[2]" || contains out ".z") then
                  failwith
                    (Printf.sprintf
                       "%s: expected GEP element 2 access ([2] or .z), got:\n%s"
                       name out)));
        ];
      group "Kernel Signature"
        [
          test "function prefix" (fun () ->
            let cuda_out = render (Cstyle.cuda Gpu_target.SM80) f32_1 in
            assert_contains "cuda extern C" cuda_out {|extern "C"|};
            assert_contains "cuda __global__" cuda_out "__global__";
            assert_contains "metal kernel void"
              (render Cstyle.metal f32_1) "kernel void";
            assert_contains "opencl __kernel"
              (render Cstyle.opencl f32_1) "__kernel";
            assert_contains "clang void" (render Cstyle.clang f32_1) "void");
          test "kernel name" (fun () ->
            for_each_renderer all_renderers (fun name r ->
                assert_contains (name ^ " kernel name")
                  (Renderer.render r ~name:"my_test_kernel" f32_1) "my_test_kernel"));
          test "parameter qualifiers" (fun () ->
            assert_contains "opencl __global"
              (render Cstyle.opencl f32_1) "__global";
            assert_contains "metal device"
              (render Cstyle.metal f32_1) "device");
          test "scalar parameter" (fun () ->
            let prog = make_define_var () in
            for_each_renderer all_renderers (fun name r ->
                assert_contains (name ^ " scalar param n") (render r prog) "n"));
        ];
      group "Preamble"
        [
          test "CUDA bitcast template" (fun () ->
            let prog = make_bitcast ~from_dt:dt ~to_dt:Dtype.int32 in
            assert_contains "cuda tg_bitcast template"
              (render (Cstyle.cuda Gpu_target.SM80) prog) "tg_bitcast");
          test "CUDA fp16 include" (fun () ->
            let prog = make_store_const Dtype.float16 (float_c Dtype.float16 1.0) in
            assert_contains "cuda fp16 include"
              (render (Cstyle.cuda Gpu_target.SM80) prog) "cuda_fp16");
          test "Metal stdlib" (fun () ->
            assert_contains "metal stdlib" (render Cstyle.metal f32_1) "metal_stdlib");
          test "OpenCL fp16 pragma" (fun () ->
            let prog = make_store_const Dtype.float16 (float_c Dtype.float16 1.0) in
            assert_contains "opencl fp16 pragma"
              (render Cstyle.opencl prog) "cl_khr_fp16");
        ];
      group "Non-native Rewrites"
        [
          test "bf16 promoted to f32" (fun () ->
            let prog =
              make_binop Dtype.bfloat16 (fun l r dt ->
                  P.Binary { op = `Add; lhs = l; rhs = r; dtype = dt })
            in
            assert_contains "clang bf16 promoted to float"
              (render Cstyle.clang prog) "float");
        ];
      group "Clang ABI"
        [
          test "fixed ABI wrapper" (fun () ->
            let out = Renderer.render Cstyle.clang ~name:"kern" f32_1 in
            assert_contains "clang fixed ABI" out
              "void kern(const unsigned long long *bufs");
          test "fixed ABI wraps inner kernel" (fun () ->
            let out = Renderer.render Cstyle.clang ~name:"kern" f32_1 in
            assert_contains "clang fixed ABI static inner" out "static void kern_(";
            assert_contains "clang fixed ABI wrapper signature" out
              "void kern(const unsigned long long *bufs, const long long *vals)";
            assert_contains "clang fixed ABI wrapper call" out "kern_((float*)bufs[0]);");
        ];
      group "CUDA Launch Bounds"
        [
          test "launch bounds" (fun () ->
            assert_contains "cuda __launch_bounds__"
              (render (Cstyle.cuda Gpu_target.SM80) (make_launch_bounds ()))
              "__launch_bounds__");
        ];
      group "Variable Naming"
        [
          test "range variable prefix" (fun () ->
            let prog = make_loop () in
            for_each_renderer all_renderers (fun name r ->
                assert_contains (name ^ " Loop prefix Lidx") (render r prog) "Lidx0"));
          test "special variable names" (fun () ->
            let prog = make_special (Special_dim.Group_id 0) in
            for_each_renderer gpu_renderers (fun name r ->
                assert_contains (name ^ " gidx0") (render r prog) "gidx0");
            let prog_lid = make_special (Special_dim.Local_id 1) in
            for_each_renderer gpu_renderers (fun name r ->
                assert_contains (name ^ " lidx1") (render r prog_lid) "lidx1"));
        ];
      group "Custom"
        [
          test "custom_inline" (fun () ->
            let prog = make_custom () in
            for_each_renderer all_renderers (fun name r ->
                assert_contains (name ^ " custom_func") (render r prog) "custom_func"));
        ];
      group "AMD/HIP"
        [
          test "special dims" (fun () ->
            let rdna3 = Cstyle.amd Gpu_target.RDNA3 in
            assert_contains "amd group_id"
              (render rdna3 (make_special (Special_dim.Group_id 0)))
              "__ockl_get_group_id(0)";
            assert_contains "amd local_id"
              (render rdna3 (make_special (Special_dim.Local_id 0)))
              "__ockl_get_local_id(0)");
          test "transcendentals" (fun () ->
            let rdna3 = Cstyle.amd Gpu_target.RDNA3 in
            assert_contains "amd __ocml_sqrt_f32"
              (render rdna3
                 (make_unop dt (fun s dt -> P.Unary { op = `Sqrt; src = s; dtype = dt })))
              "__ocml_sqrt_f32";
            assert_contains "amd __ocml_sin_f32"
              (render rdna3
                 (make_unop dt (fun s dt -> P.Unary { op = `Sin; src = s; dtype = dt })))
              "__ocml_sin_f32");
          test "barrier" (fun () ->
            let out = render (Cstyle.amd Gpu_target.RDNA3) (make_shared_memory ()) in
            assert_contains "amd fence" out "__builtin_amdgcn_fence";
            assert_contains "amd s_barrier" out "__builtin_amdgcn_s_barrier");
          test "kernel attribute" (fun () ->
            assert_contains "amd amdgpu_flat_work_group_size"
              (render (Cstyle.amd Gpu_target.RDNA3) f32_1)
              "amdgpu_flat_work_group_size");
          test "bf16 target paths" (fun () ->
            let prog =
              make_binop Dtype.bfloat16 (fun l r dt ->
                  P.Binary { op = `Add; lhs = l; rhs = r; dtype = dt })
            in
            let rdna3_out = render (Cstyle.amd Gpu_target.RDNA3) prog in
            assert_contains "amd rdna3 uses hip_bfloat16" rdna3_out "hip_bfloat16";
            assert_contains "amd rdna3 typedefs software bf16" rdna3_out
              "typedef unsigned short hip_bfloat16;";
            let cdna4_out = render (Cstyle.amd Gpu_target.CDNA4) prog in
            assert_contains "amd cdna4 typedefs __bf16 hip_bfloat16" cdna4_out
              "typedef __bf16 hip_bfloat16;";
            assert_not_contains "amd cdna4 does not typedef ushort hip_bfloat16"
              cdna4_out "typedef unsigned short hip_bfloat16;");
        ];
      group "Intel"
        [
          test "kernel attribute" (fun () ->
            assert_contains "intel sub_group_size"
              (render Cstyle.intel f32_1) "intel_reqd_sub_group_size(8)");
        ];
      group "Properties"
        [
          prop "non-empty output"
            (pair safe_dtype renderer_testable)
            (fun (dt, (_name, renderer)) ->
              let const_value =
                match dt.Dtype.scalar with
                | Dtype.Float32 | Dtype.Float64 -> Const.float dt 1.0
                | _ -> Const.int dt 1
              in
              String.length (render renderer (make_store_const dt const_value)) > 0);
          prop "contains kernel name" renderer_testable (fun (_name, renderer) ->
            contains
              (Renderer.render renderer ~name:"test_prop_kernel" f32_1)
              "test_prop_kernel");
          prop "balanced braces" renderer_testable (fun (_name, renderer) ->
            let output = render renderer (make_loop ()) in
            count_char output '{' = count_char output '}');
          prop "deterministic" renderer_testable (fun (_name, renderer) ->
            String.equal (render renderer f32_1) (render renderer f32_1));
        ];
    ]
