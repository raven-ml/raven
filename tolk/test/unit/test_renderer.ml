(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tolk
module P = Ir.Program

(* ───── Renderers Under Test ───── *)

(* Core 4 backends. *)
let all_renderers =
  [
    ("clang", Cstyle.clang);
    ("cuda", Cstyle.cuda ());
    ("metal", Cstyle.metal);
    ("opencl", Cstyle.opencl);
  ]

let gpu_renderers = List.filter (fun (name, _) -> name <> "clang") all_renderers

(* ───── Helpers ───── *)

let global_ptr dt = Dtype.Ptr.create dt ~addrspace:Global ()
let local_ptr dt = Dtype.Ptr.create dt ~addrspace:Local ()
let render r prog = Renderer.render r prog

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

(* ───── IR Program Builders ───── *)

(* Simplest: one param, one const value, index at 0, store the const. *)
let make_store_const dt value =
  let ptr = global_ptr dt in
  [|
    P.Param { idx = 0; dtype = ptr };
    P.Const { value; dtype = dt };
    P.Const { value = Int 0; dtype = Dtype.int32 };
    P.Index { ptr = 0; idxs = [ 2 ]; gate = None; dtype = ptr };
    P.Store { dst = 3; value = 1 };
  |]

(* Two params in + one param out, load both, apply binary op, store. *)
let make_binop dt mk_op =
  let ptr = global_ptr dt in
  [|
    (* 0 *) P.Param { idx = 0; dtype = ptr };
    (* 1 *) P.Param { idx = 1; dtype = ptr };
    (* 2 *) P.Param { idx = 2; dtype = ptr };
    (* 3 *) P.Const { value = Int 0; dtype = Dtype.int32 };
    (* 4 *) P.Index { ptr = 0; idxs = [ 3 ]; gate = None; dtype = ptr };
    (* 5 *) P.Index { ptr = 1; idxs = [ 3 ]; gate = None; dtype = ptr };
    (* 6 *) P.Index { ptr = 2; idxs = [ 3 ]; gate = None; dtype = ptr };
    (* 7 *) P.Load { src = 4; alt = None; dtype = dt };
    (* 8 *) P.Load { src = 5; alt = None; dtype = dt };
    (* 9 *) mk_op 7 8 dt;
    (* 10 *) P.Store { dst = 6; value = 9 };
  |]

(* One param in + one param out, load, apply unary op, store. *)
let make_unop dt mk_op =
  let ptr = global_ptr dt in
  [|
    (* 0 *) P.Param { idx = 0; dtype = ptr };
    (* 1 *) P.Param { idx = 1; dtype = ptr };
    (* 2 *) P.Const { value = Int 0; dtype = Dtype.int32 };
    (* 3 *) P.Index { ptr = 0; idxs = [ 2 ]; gate = None; dtype = ptr };
    (* 4 *) P.Index { ptr = 1; idxs = [ 2 ]; gate = None; dtype = ptr };
    (* 5 *) P.Load { src = 3; alt = None; dtype = dt };
    (* 6 *) mk_op 5 dt;
    (* 7 *) P.Store { dst = 4; value = 6 };
  |]

(* Where ternary: two params, load both, Where with true cond, store. *)
let make_ternary_where dt =
  let ptr = global_ptr dt in
  [|
    (* 0 *) P.Param { idx = 0; dtype = ptr };
    (* 1 *) P.Param { idx = 1; dtype = ptr };
    (* 2 *) P.Param { idx = 2; dtype = ptr };
    (* 3 *) P.Const { value = Int 0; dtype = Dtype.int32 };
    (* 4 *) P.Index { ptr = 0; idxs = [ 3 ]; gate = None; dtype = ptr };
    (* 5 *) P.Index { ptr = 1; idxs = [ 3 ]; gate = None; dtype = ptr };
    (* 6 *) P.Index { ptr = 2; idxs = [ 3 ]; gate = None; dtype = ptr };
    (* 7 *) P.Load { src = 4; alt = None; dtype = dt };
    (* 8 *) P.Load { src = 5; alt = None; dtype = dt };
    (* 9 *) P.Const { value = Bool true; dtype = Dtype.bool };
    (* 10 *) P.Where { cond = 9; a = 7; b = 8; dtype = dt };
    (* 11 *) P.Store { dst = 6; value = 10 };
  |]

(* Mulacc: three params, load all, fused multiply-accumulate, store. *)
let make_mulacc dt =
  let ptr = global_ptr dt in
  [|
    (* 0 *) P.Param { idx = 0; dtype = ptr };
    (* 1 *) P.Param { idx = 1; dtype = ptr };
    (* 2 *) P.Param { idx = 2; dtype = ptr };
    (* 3 *) P.Param { idx = 3; dtype = ptr };
    (* 4 *) P.Const { value = Int 0; dtype = Dtype.int32 };
    (* 5 *) P.Index { ptr = 0; idxs = [ 4 ]; gate = None; dtype = ptr };
    (* 6 *) P.Index { ptr = 1; idxs = [ 4 ]; gate = None; dtype = ptr };
    (* 7 *) P.Index { ptr = 2; idxs = [ 4 ]; gate = None; dtype = ptr };
    (* 8 *) P.Index { ptr = 3; idxs = [ 4 ]; gate = None; dtype = ptr };
    (* 9 *) P.Load { src = 5; alt = None; dtype = dt };
    (* 10 *) P.Load { src = 6; alt = None; dtype = dt };
    (* 11 *) P.Load { src = 7; alt = None; dtype = dt };
    (* 12 *) P.Mulacc { a = 9; b = 10; c = 11; dtype = dt };
    (* 13 *) P.Store { dst = 8; value = 12 };
  |]

(* Loop: one param, range 0..9, load at loop index, end. *)
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
    (* 5 *) P.End_range { range = 2 };
  |]

(* Nested loops: two nested Range/End_range pairs. *)
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
    (* 8 *) P.End_range { range = 4 };
    (* 9 *) P.End_range { range = 3 };
  |]

(* Special dimension: one param, special dim, store workitem id. *)
let make_special dim =
  let dt = Dtype.int32 in
  let ptr = global_ptr dt in
  [|
    (* 0 *) P.Param { idx = 0; dtype = ptr };
    (* 1 *) P.Const { value = Int 64; dtype = Dtype.int32 };
    (* 2 *) P.Special { dim; size = 1; dtype = dt };
    (* 3 *) P.Index { ptr = 0; idxs = [ 2 ]; gate = None; dtype = ptr };
    (* 4 *) P.Store { dst = 3; value = 2 };
  |]

(* Shared memory: define_local + barrier + load/store. *)
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

(* Gated load: index with gate + load with alt. *)
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

(* Cast from one dtype to another. *)
let make_cast ~from_dt ~to_dt =
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

(* Bitcast between same-size types. *)
let make_bitcast ~from_dt ~to_dt =
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
    (* 6 *) P.Bitcast { src = 5; dtype = to_dt };
    (* 7 *) P.Store { dst = 4; value = 6 };
  |]

(* Vectorize 4 scalars then extract element with Gep. *)
let make_vectorize_gep () =
  let dt = Dtype.float32 in
  let vdt = Dtype.vec dt 4 in
  let ptr = global_ptr dt in
  [|
    (* 0 *) P.Param { idx = 0; dtype = ptr };
    (* 1 *) P.Const { value = Float 1.0; dtype = dt };
    (* 2 *) P.Const { value = Float 2.0; dtype = dt };
    (* 3 *) P.Const { value = Float 3.0; dtype = dt };
    (* 4 *) P.Const { value = Float 4.0; dtype = dt };
    (* 5 *) P.Vectorize { srcs = [ 1; 2; 3; 4 ]; dtype = vdt };
    (* 6 *) P.Gep { src = 5; idx = 2; dtype = dt };
    (* 7 *) P.Const { value = Int 0; dtype = Dtype.int32 };
    (* 8 *) P.Index { ptr = 0; idxs = [ 7 ]; gate = None; dtype = ptr };
    (* 9 *) P.Store { dst = 8; value = 6 };
  |]

(* Custom code injection. *)
let make_custom () =
  let dt = Dtype.float32 in
  let ptr = global_ptr dt in
  [|
    (* 0 *) P.Param { idx = 0; dtype = ptr };
    (* 1 *) P.Const { value = Int 0; dtype = Dtype.int32 };
    (* 2 *) P.Index { ptr = 0; idxs = [ 1 ]; gate = None; dtype = ptr };
    (* 3 *) P.Load { src = 2; alt = None; dtype = dt };
    (* 4 *)
    P.Custom_inline { fmt = "custom_func({0}, {0})"; args = [ 3 ]; dtype = dt };
    (* 5 *) P.Store { dst = 2; value = 4 };
  |]

(* Define_var scalar parameter. *)
let make_define_var () =
  let dt = Dtype.float32 in
  let ptr = global_ptr dt in
  [|
    (* 0 *) P.Param { idx = 0; dtype = ptr };
    (* 1 *) P.Define_var { name = "n"; lo = 0; hi = 1024; dtype = Dtype.int32 };
    (* 2 *) P.Index { ptr = 0; idxs = [ 1 ]; gate = None; dtype = ptr };
    (* 3 *) P.Load { src = 2; alt = None; dtype = dt };
    (* 4 *) P.Store { dst = 2; value = 3 };
  |]

(* Chain of n binary ops for paren stripping tests. *)
let make_chained_binop dt mk_op n =
  let ptr = global_ptr dt in
  let instrs =
    Array.make (5 + n) (P.Const { value = Int 0; dtype = Dtype.int32 })
  in
  (* 0: output param *)
  instrs.(0) <- P.Param { idx = 0; dtype = ptr };
  (* 1: input param *)
  instrs.(1) <- P.Param { idx = 1; dtype = ptr };
  (* 2: const index 0 *)
  instrs.(2) <- P.Const { value = Int 0; dtype = Dtype.int32 };
  (* 3: index into input *)
  instrs.(3) <- P.Index { ptr = 1; idxs = [ 2 ]; gate = None; dtype = ptr };
  (* 4: first load *)
  instrs.(4) <- P.Load { src = 3; alt = None; dtype = dt };
  (* 5..4+n: chain of ops, each uses previous result and the load *)
  for i = 0 to n - 1 do
    instrs.(5 + i) <- mk_op (if i = 0 then 4 else 4 + i) 4 dt
  done;
  (* Append: index output, store *)
  let result_idx = 4 + n in
  let idx_out = result_idx + 1 in
  let store = result_idx + 2 in
  let full =
    Array.make (store + 1) (P.Const { value = Int 0; dtype = Dtype.int32 })
  in
  Array.blit instrs 0 full 0 (result_idx + 1);
  full.(idx_out) <- P.Index { ptr = 0; idxs = [ 2 ]; gate = None; dtype = ptr };
  full.(store) <- P.Store { dst = idx_out; value = result_idx };
  full

(* Program with If/Endif. Needs an Index for the idx_for_dedup field. *)
let make_conditional () =
  let dt = Dtype.float32 in
  let ptr = global_ptr dt in
  [|
    (* 0 *) P.Param { idx = 0; dtype = ptr };
    (* 1 *) P.Const { value = Int 0; dtype = Dtype.int32 };
    (* 2 *) P.Index { ptr = 0; idxs = [ 1 ]; gate = None; dtype = ptr };
    (* 3 *) P.Const { value = Bool true; dtype = Dtype.bool };
    (* 4 *) P.If { cond = 3; idx_for_dedup = 2 };
    (* 5 *) P.Const { value = Float 42.0; dtype = dt };
    (* 6 *) P.Store { dst = 2; value = 5 };
    (* 7 *) P.Endif { if_ = 4 };
  |]

(* Launch bounds: two local dims. *)
let make_launch_bounds () =
  let dt = Dtype.int32 in
  let ptr = global_ptr dt in
  [|
    (* 0 *) P.Param { idx = 0; dtype = ptr };
    (* 1 *) P.Const { value = Int 64; dtype = Dtype.int32 };
    (* 2 *) P.Special { dim = Local_id 0; size = 1; dtype = dt };
    (* 3 *) P.Const { value = Int 4; dtype = Dtype.int32 };
    (* 4 *) P.Special { dim = Local_id 1; size = 3; dtype = dt };
    (* 5 *) P.Add { lhs = 2; rhs = 4; dtype = dt };
    (* 6 *) P.Index { ptr = 0; idxs = [ 5 ]; gate = None; dtype = ptr };
    (* 7 *) P.Store { dst = 6; value = 5 };
  |]

(* ───── Constants ───── *)

let test_int_constant () =
  let prog = make_store_const Dtype.int32 (Int 42) in
  for_each_renderer all_renderers (fun name r ->
      let out = render r prog in
      assert_contains (name ^ " int 42") out "42")

let test_float32_constant () =
  let prog = make_store_const Dtype.float32 (Float 3.14) in
  for_each_renderer all_renderers (fun name r ->
      let out = render r prog in
      assert_contains (name ^ " float32 3.14") out "3.14";
      assert_contains (name ^ " float32 f suffix") out "f")

let test_float64_constant () =
  let prog = make_store_const Dtype.float64 (Float 3.14) in
  for_each_renderer all_renderers (fun name r ->
      let out = render r prog in
      assert_contains (name ^ " float64 3.14") out "3.14")

let test_bool_constants () =
  for_each_renderer all_renderers (fun name r ->
      let out_t = render r (make_store_const Dtype.bool (Bool true)) in
      assert_contains (name ^ " bool true") out_t "1";
      let out_f = render r (make_store_const Dtype.bool (Bool false)) in
      assert_contains (name ^ " bool false") out_f "0")

let test_nan_inf_constants () =
  let nan_prog = make_store_const Dtype.float32 (Float Float.nan) in
  let inf_prog = make_store_const Dtype.float32 (Float Float.infinity) in
  let neg_inf_prog =
    make_store_const Dtype.float32 (Float Float.neg_infinity)
  in
  (* CUDA, Metal, OpenCL use NAN/INFINITY *)
  List.iter
    (fun (name, r) ->
      let nan_out = render r nan_prog in
      assert_contains (name ^ " NAN") nan_out "NAN";
      let inf_out = render r inf_prog in
      assert_contains (name ^ " INFINITY") inf_out "INFINITY";
      let neg_inf_out = render r neg_inf_prog in
      assert_contains (name ^ " -INFINITY") neg_inf_out "INFINITY")
    [
      ("cuda", Cstyle.cuda ());
      ("metal", Cstyle.metal);
      ("opencl", Cstyle.opencl);
    ];
  (* Clang uses __builtin_nanf / __builtin_inff *)
  let clang = Cstyle.clang in
  let nan_out = render clang nan_prog in
  assert_contains "clang NAN" nan_out "__builtin_nanf";
  let inf_out = render clang inf_prog in
  assert_contains "clang INF" inf_out "__builtin_inff"

let test_int64_suffix () =
  let prog = make_store_const Dtype.int64 (Int 12345) in
  for_each_renderer all_renderers (fun name r ->
      let out = render r prog in
      assert_contains (name ^ " int64 ll suffix") out "12345ll")

let test_uint32_suffix () =
  let prog = make_store_const Dtype.uint32 (Int 42) in
  for_each_renderer all_renderers (fun name r ->
      let out = render r prog in
      assert_contains (name ^ " uint32 u suffix") out "42u")

let test_uint64_suffix () =
  let prog = make_store_const Dtype.uint64 (Int 42) in
  for_each_renderer all_renderers (fun name r ->
      let out = render r prog in
      assert_contains (name ^ " uint64 ull suffix") out "42ull")

(* ───── ALU Operations ───── *)

let test_binary_operators () =
  let ops =
    [
      ("Add +", (fun l r dt -> P.Add { lhs = l; rhs = r; dtype = dt }), "+");
      ("Sub -", (fun l r dt -> P.Sub { lhs = l; rhs = r; dtype = dt }), "-");
      ("Mul *", (fun l r dt -> P.Mul { lhs = l; rhs = r; dtype = dt }), "*");
      ("Fdiv /", (fun l r dt -> P.Fdiv { lhs = l; rhs = r; dtype = dt }), "/");
      ("Mod %", (fun l r dt -> P.Mod { lhs = l; rhs = r; dtype = dt }), "%");
      ("Shl <<", (fun l r dt -> P.Shl { lhs = l; rhs = r; dtype = dt }), "<<");
      ("Shr >>", (fun l r dt -> P.Shr { lhs = l; rhs = r; dtype = dt }), ">>");
      ("And &", (fun l r dt -> P.And { lhs = l; rhs = r; dtype = dt }), "&");
      ("Or |", (fun l r dt -> P.Or { lhs = l; rhs = r; dtype = dt }), "|");
      ("Xor ^", (fun l r dt -> P.Xor { lhs = l; rhs = r; dtype = dt }), "^");
    ]
  in
  List.iter
    (fun (label, mk_op, expected) ->
      let dt =
        if String.length expected = 1 && expected.[0] = '/' then Dtype.float32
        else Dtype.int32
      in
      let prog = make_binop dt mk_op in
      for_each_renderer all_renderers (fun name r ->
          let out = render r prog in
          assert_contains (name ^ " " ^ label) out expected))
    ops

let test_int_division () =
  let prog =
    make_binop Dtype.int32 (fun l r dt ->
        P.Idiv { lhs = l; rhs = r; dtype = dt })
  in
  for_each_renderer all_renderers (fun name r ->
      let out = render r prog in
      assert_contains (name ^ " Idiv /") out "/")

let test_comparison_operators () =
  let ops =
    [
      ("Cmplt <", (fun l r dt -> P.Cmplt { lhs = l; rhs = r; dtype = dt }), "<");
      ( "Cmpeq ==",
        (fun l r dt -> P.Cmpeq { lhs = l; rhs = r; dtype = dt }),
        "==" );
      ( "Cmpne !=",
        (fun l r dt -> P.Cmpne { lhs = l; rhs = r; dtype = dt }),
        "!=" );
    ]
  in
  (* Comparisons take a non-bool operand type but produce bool. We construct
     programs that load float32 and compare. *)
  List.iter
    (fun (label, mk_op, expected) ->
      let in_dt = Dtype.float32 in
      let out_dt = Dtype.bool in
      let in_ptr = global_ptr in_dt in
      let out_ptr = global_ptr out_dt in
      let prog =
        [|
          (* 0 *) P.Param { idx = 0; dtype = in_ptr };
          (* 1 *) P.Param { idx = 1; dtype = in_ptr };
          (* 2 *) P.Param { idx = 2; dtype = out_ptr };
          (* 3 *) P.Const { value = Int 0; dtype = Dtype.int32 };
          (* 4 *)
          P.Index { ptr = 0; idxs = [ 3 ]; gate = None; dtype = in_ptr };
          (* 5 *)
          P.Index { ptr = 1; idxs = [ 3 ]; gate = None; dtype = in_ptr };
          (* 6 *)
          P.Index { ptr = 2; idxs = [ 3 ]; gate = None; dtype = out_ptr };
          (* 7 *) P.Load { src = 4; alt = None; dtype = in_dt };
          (* 8 *) P.Load { src = 5; alt = None; dtype = in_dt };
          (* 9 *) mk_op 7 8 out_dt;
          (* 10 *) P.Store { dst = 6; value = 9 };
        |]
      in
      for_each_renderer all_renderers (fun name r ->
          let out = render r prog in
          assert_contains (name ^ " " ^ label) out expected))
    ops

let test_max_operator () =
  let prog =
    make_binop Dtype.float32 (fun l r dt ->
        P.Max { lhs = l; rhs = r; dtype = dt })
  in
  for_each_renderer all_renderers (fun name r ->
      let out = render r prog in
      assert_contains (name ^ " Max") out "max")

let test_unary_operators () =
  let ops =
    [
      ("Neg", (fun s dt -> P.Neg { src = s; dtype = dt }), "-");
      ("Exp2", (fun s dt -> P.Exp2 { src = s; dtype = dt }), "exp2");
      ("Log2", (fun s dt -> P.Log2 { src = s; dtype = dt }), "log2");
      ("Sin", (fun s dt -> P.Sin { src = s; dtype = dt }), "sin");
      ("Sqrt", (fun s dt -> P.Sqrt { src = s; dtype = dt }), "sqrt");
      ("Trunc", (fun s dt -> P.Trunc { src = s; dtype = dt }), "trunc");
    ]
  in
  List.iter
    (fun (label, mk_op, expected) ->
      let prog = make_unop Dtype.float32 mk_op in
      for_each_renderer all_renderers (fun name r ->
          let out = render r prog in
          assert_contains (name ^ " " ^ label) out expected))
    ops

let test_recip () =
  let prog =
    make_unop Dtype.float32 (fun s dt -> P.Recip { src = s; dtype = dt })
  in
  for_each_renderer all_renderers (fun name r ->
      let out = render r prog in
      assert_contains (name ^ " Recip") out "1/")

let test_where () =
  let prog = make_ternary_where Dtype.float32 in
  for_each_renderer all_renderers (fun name r ->
      let out = render r prog in
      assert_contains (name ^ " Where ?") out "?";
      assert_contains (name ^ " Where :") out ":")

let test_mulacc () =
  let prog = make_mulacc Dtype.float32 in
  for_each_renderer all_renderers (fun name r ->
      let out = render r prog in
      assert_contains (name ^ " Mulacc *") out "*";
      assert_contains (name ^ " Mulacc +") out "+")

let test_cuda_half_intrinsics () =
  let cuda = Cstyle.cuda () in
  let ops =
    [
      ("hexp2", fun s dt -> P.Exp2 { src = s; dtype = dt });
      ("hlog2", fun s dt -> P.Log2 { src = s; dtype = dt });
      ("hsin", fun s dt -> P.Sin { src = s; dtype = dt });
      ("hsqrt", fun s dt -> P.Sqrt { src = s; dtype = dt });
      ("hrcp", fun s dt -> P.Recip { src = s; dtype = dt });
      ("htrunc", fun s dt -> P.Trunc { src = s; dtype = dt });
    ]
  in
  List.iter
    (fun (expected, mk_op) ->
      let prog = make_unop Dtype.float16 mk_op in
      let out = render cuda prog in
      assert_contains ("CUDA " ^ expected) out expected)
    ops

let test_metal_precise_sin () =
  let prog =
    make_unop Dtype.float32 (fun s dt -> P.Sin { src = s; dtype = dt })
  in
  let out = render Cstyle.metal prog in
  assert_contains "Metal precise::sin" out "precise::sin"

let test_clang_builtins () =
  let clang = Cstyle.clang in
  let sqrt_prog =
    make_unop Dtype.float32 (fun s dt -> P.Sqrt { src = s; dtype = dt })
  in
  let trunc_prog =
    make_unop Dtype.float32 (fun s dt -> P.Trunc { src = s; dtype = dt })
  in
  let sqrt_out = render clang sqrt_prog in
  assert_contains "clang __builtin_sqrtf" sqrt_out "__builtin_sqrtf";
  let trunc_out = render clang trunc_prog in
  assert_contains "clang __builtin_truncf" trunc_out "__builtin_truncf"

let test_paren_stripping () =
  let mk_add l r dt = P.Add { lhs = l; rhs = r; dtype = dt } in
  let mk_sub l r dt = P.Sub { lhs = l; rhs = r; dtype = dt } in
  let prog_add = make_chained_binop Dtype.float32 mk_add 5 in
  let prog_sub = make_chained_binop Dtype.float32 mk_sub 5 in
  for_each_renderer all_renderers (fun name r ->
      let add_out = render r prog_add in
      assert_not_contains (name ^ " Add no deep parens") add_out "(((((";
      let sub_out = render r prog_sub in
      assert_contains (name ^ " Sub deep parens") sub_out "(((((")

(* ───── Control Flow ───── *)

let test_for_loop () =
  let prog = make_loop () in
  for_each_renderer all_renderers (fun name r ->
      let out = render r prog in
      assert_contains (name ^ " for loop") out "for (")

let test_nested_loops () =
  let prog = make_nested_loops () in
  for_each_renderer all_renderers (fun name r ->
      let out = render r prog in
      (* Count "for (" occurrences *)
      let count =
        let rec loop i acc =
          if i > String.length out - 4 then acc
          else if String.sub out i 4 = "for " then loop (i + 1) (acc + 1)
          else loop (i + 1) acc
        in
        loop 0 0
      in
      if count < 2 then
        failwith
          (Printf.sprintf "%s: expected 2 'for ' occurrences, got %d" name count))

let test_conditional () =
  let prog = make_conditional () in
  for_each_renderer all_renderers (fun name r ->
      let out = render r prog in
      assert_contains (name ^ " if") out "if (")

(* ───── Memory ───── *)

let test_simple_load_store () =
  let prog =
    make_binop Dtype.float32 (fun l r dt ->
        P.Add { lhs = l; rhs = r; dtype = dt })
  in
  for_each_renderer all_renderers (fun name r ->
      let out = render r prog in
      (* Load uses *ptr and store uses *ptr = ... *)
      assert_contains (name ^ " dereference") out "*")

let test_gated_load () =
  let prog = make_gated_load () in
  for_each_renderer all_renderers (fun name r ->
      let out = render r prog in
      assert_contains (name ^ " gated load ternary") out "?")

(* ───── Cast and Bitcast ───── *)

let test_cast_per_backend () =
  let prog = make_cast ~from_dt:Dtype.int32 ~to_dt:Dtype.float32 in
  let metal_out = render Cstyle.metal prog in
  assert_contains "metal cast" metal_out "(float)";
  let cuda_out = render (Cstyle.cuda ()) prog in
  assert_contains "cuda cast" cuda_out "(float)";
  let opencl_out = render Cstyle.opencl prog in
  assert_contains "opencl cast" opencl_out "(float)";
  let clang_out = render Cstyle.clang prog in
  assert_contains "clang cast" clang_out "(float)"

let test_bitcast_per_backend () =
  let prog = make_bitcast ~from_dt:Dtype.float32 ~to_dt:Dtype.int32 in
  let clang_out = render Cstyle.clang prog in
  assert_contains "clang __builtin_bit_cast" clang_out "__builtin_bit_cast";
  let cuda_out = render (Cstyle.cuda ()) prog in
  assert_contains "cuda tg_bitcast" cuda_out "tg_bitcast";
  let metal_out = render Cstyle.metal prog in
  assert_contains "metal as_type" metal_out "as_type<";
  let opencl_out = render Cstyle.opencl prog in
  assert_contains "opencl as_" opencl_out "as_"

(* ───── Special Dimensions ───── *)

let test_special_group_id () =
  let prog = make_special (Ir.Group_id 0) in
  let cuda_out = render (Cstyle.cuda ()) prog in
  assert_contains "cuda blockIdx.x" cuda_out "blockIdx.x";
  let metal_out = render Cstyle.metal prog in
  assert_contains "metal gid.x" metal_out "gid.x";
  let opencl_out = render Cstyle.opencl prog in
  assert_contains "opencl get_group_id(0)" opencl_out "get_group_id(0)"

let test_special_local_id () =
  let prog = make_special (Ir.Local_id 1) in
  let cuda_out = render (Cstyle.cuda ()) prog in
  assert_contains "cuda threadIdx.y" cuda_out "threadIdx.y";
  let metal_out = render Cstyle.metal prog in
  assert_contains "metal lid.y" metal_out "lid.y";
  let opencl_out = render Cstyle.opencl prog in
  assert_contains "opencl get_local_id(1)" opencl_out "get_local_id(1)"

let test_special_global_idx () =
  let prog = make_special (Ir.Global_idx 2) in
  let cuda_out = render (Cstyle.cuda ()) prog in
  assert_contains "cuda blockIdx.z" cuda_out "blockIdx.z";
  let metal_out = render Cstyle.metal prog in
  assert_contains "metal tid.z" metal_out "tid.z";
  let opencl_out = render Cstyle.opencl prog in
  assert_contains "opencl get_global_id(2)" opencl_out "get_global_id(2)"

let test_special_clang_fails () =
  let prog = make_special (Ir.Group_id 0) in
  raises_match
    (function
      | Failure msg -> contains msg "no GPU thread support" | _ -> false)
    (fun () -> ignore (render Cstyle.clang prog))

(* ───── Shared Memory and Barrier ───── *)

let test_shared_memory_qualifiers () =
  let prog = make_shared_memory () in
  let cuda_out = render (Cstyle.cuda ()) prog in
  assert_contains "cuda __shared__" cuda_out "__shared__";
  let metal_out = render Cstyle.metal prog in
  assert_contains "metal threadgroup" metal_out "threadgroup";
  let opencl_out = render Cstyle.opencl prog in
  assert_contains "opencl __local" opencl_out "__local"

let test_barrier_syntax () =
  let prog = make_shared_memory () in
  let cuda_out = render (Cstyle.cuda ()) prog in
  assert_contains "cuda __syncthreads" cuda_out "__syncthreads()";
  let metal_out = render Cstyle.metal prog in
  assert_contains "metal threadgroup_barrier" metal_out "threadgroup_barrier";
  let opencl_out = render Cstyle.opencl prog in
  assert_contains "opencl barrier" opencl_out "barrier(CLK_LOCAL_MEM_FENCE)"

(* ───── Vectorize and Gep ───── *)

let test_vectorize () =
  let prog = make_vectorize_gep () in
  for_each_renderer all_renderers (fun name r ->
      let out = render r prog in
      (* All backends should reference the 4 float values *)
      assert_contains (name ^ " has 1.0") out "1.0";
      assert_contains (name ^ " has 4.0") out "4.0")

let test_gep () =
  let prog = make_vectorize_gep () in
  (* Clang has gep_arr_threshold = 0, so uses array notation *)
  let clang_out = render Cstyle.clang prog in
  assert_contains "clang gep [2]" clang_out "[2]";
  (* CUDA has gep_arr_threshold = 8, so vec4 uses swizzle *)
  let cuda_out = render (Cstyle.cuda ()) prog in
  assert_contains "cuda gep .z" cuda_out ".z"

(* ───── Kernel Signature ───── *)

let test_function_prefix () =
  let prog = make_store_const Dtype.float32 (Float 1.0) in
  let cuda_out = render (Cstyle.cuda ()) prog in
  assert_contains "cuda extern C" cuda_out {|extern "C"|};
  assert_contains "cuda __global__" cuda_out "__global__";
  let metal_out = render Cstyle.metal prog in
  assert_contains "metal kernel void" metal_out "kernel void";
  let opencl_out = render Cstyle.opencl prog in
  assert_contains "opencl __kernel" opencl_out "__kernel";
  let clang_out = render Cstyle.clang prog in
  assert_contains "clang static" clang_out "static"

let test_kernel_name () =
  let prog = make_store_const Dtype.float32 (Float 1.0) in
  for_each_renderer all_renderers (fun name r ->
      let out = Renderer.render r ~name:"my_test_kernel" prog in
      assert_contains (name ^ " kernel name") out "my_test_kernel")

let test_param_qualifiers () =
  let prog = make_store_const Dtype.float32 (Float 1.0) in
  let opencl_out = render Cstyle.opencl prog in
  assert_contains "opencl __global" opencl_out "__global";
  let metal_out = render Cstyle.metal prog in
  assert_contains "metal device" metal_out "device"

let test_scalar_parameter () =
  let prog = make_define_var () in
  for_each_renderer all_renderers (fun name r ->
      let out = render r prog in
      assert_contains (name ^ " scalar param n") out "n")

(* ───── Preamble ───── *)

let test_cuda_bitcast_template () =
  (* Any CUDA program with a bitcast needs tg_bitcast *)
  let prog = make_bitcast ~from_dt:Dtype.float32 ~to_dt:Dtype.int32 in
  let out = render (Cstyle.cuda ()) prog in
  assert_contains "cuda tg_bitcast template" out "tg_bitcast"

let test_cuda_fp16_include () =
  let prog = make_store_const Dtype.float16 (Float 1.0) in
  let out = render (Cstyle.cuda ()) prog in
  assert_contains "cuda fp16 include" out "cuda_fp16"

let test_metal_stdlib () =
  let prog = make_store_const Dtype.float32 (Float 1.0) in
  let out = render Cstyle.metal prog in
  assert_contains "metal stdlib" out "metal_stdlib"

let test_opencl_fp16_pragma () =
  let prog = make_store_const Dtype.float16 (Float 1.0) in
  let out = render Cstyle.opencl prog in
  assert_contains "opencl fp16 pragma" out "cl_khr_fp16"

(* ───── Clang ABI ───── *)

let test_clang_fixed_abi () =
  let prog = make_store_const Dtype.float32 (Float 1.0) in
  let out = Renderer.render Cstyle.clang ~name:"kern" prog in
  assert_contains "clang fixed ABI" out
    "void kern(const unsigned long long *bufs"

(* ───── CUDA Launch Bounds ───── *)

let test_cuda_launch_bounds () =
  let prog = make_launch_bounds () in
  let out = render (Cstyle.cuda ()) prog in
  assert_contains "cuda __launch_bounds__" out "__launch_bounds__"

(* ───── Variable Naming ───── *)

let test_range_variable_prefix () =
  let prog = make_loop () in
  for_each_renderer all_renderers (fun name r ->
      let out = render r prog in
      assert_contains (name ^ " Loop prefix Lidx") out "Lidx0")

let test_special_variable_names () =
  let prog = make_special (Ir.Group_id 0) in
  for_each_renderer gpu_renderers (fun name r ->
      let out = render r prog in
      assert_contains (name ^ " gidx0") out "gidx0");
  let prog_lid = make_special (Ir.Local_id 1) in
  for_each_renderer gpu_renderers (fun name r ->
      let out = render r prog_lid in
      assert_contains (name ^ " lidx1") out "lidx1")

(* ───── Custom Code Injection ───── *)

let test_custom_inline () =
  let prog = make_custom () in
  for_each_renderer all_renderers (fun name r ->
      let out = render r prog in
      assert_contains (name ^ " custom_func") out "custom_func")

(* ───── Non-native Float Rewrites ───── *)

let test_bf16_promoted_to_f32 () =
  let prog =
    make_binop Dtype.bfloat16 (fun l r dt ->
        P.Add { lhs = l; rhs = r; dtype = dt })
  in
  let clang_out = render Cstyle.clang prog in
  assert_contains "clang bf16 promoted to float" clang_out "float"

(* ───── Property Tests ───── *)

let renderer_testable =
  let gen = Gen.oneofl all_renderers in
  let pp fmt (name, _) = Format.pp_print_string fmt name in
  testable ~pp ~equal:(fun (a, _) (b, _) -> String.equal a b) ~gen ()

let safe_dtypes = [ Dtype.int32; Dtype.float32; Dtype.float64; Dtype.uint32 ]

let safe_dtype =
  let gen = Gen.oneofl safe_dtypes in
  testable ~pp:Dtype.pp ~equal:Dtype.equal ~gen ()

let prop_non_empty (dt, (_name, renderer)) =
  let prog =
    make_store_const dt
      (match dt.Dtype.scalar with
      | Dtype.Float32 | Dtype.Float64 -> Float 1.0
      | _ -> Int 1)
  in
  let output = render renderer prog in
  String.length output > 0

let prop_contains_kernel_name (_name, renderer) =
  let prog = make_store_const Dtype.float32 (Float 1.0) in
  let output = Renderer.render renderer ~name:"test_prop_kernel" prog in
  contains output "test_prop_kernel"

let prop_balanced_braces (_name, renderer) =
  let prog = make_loop () in
  let output = render renderer prog in
  let opens = count_char output '{' in
  let closes = count_char output '}' in
  opens = closes

let prop_deterministic (_name, renderer) =
  let prog = make_store_const Dtype.float32 (Float 1.0) in
  let a = render renderer prog in
  let b = render renderer prog in
  String.equal a b

(* ───── Runner ───── *)

let () =
  run "Renderer"
    [
      group "Constants"
        [
          test "int constant" test_int_constant;
          test "float32 constant" test_float32_constant;
          test "float64 constant" test_float64_constant;
          test "bool constants" test_bool_constants;
          test "nan/inf constants" test_nan_inf_constants;
          test "int64 suffix" test_int64_suffix;
          test "uint32 suffix" test_uint32_suffix;
          test "uint64 suffix" test_uint64_suffix;
        ];
      group "ALU Operations"
        [
          group "Binary"
            [
              test "arithmetic operators" test_binary_operators;
              test "integer division" test_int_division;
              test "comparison operators" test_comparison_operators;
              test "max" test_max_operator;
            ];
          group "Unary"
            [
              test "operators" test_unary_operators;
              test "reciprocal" test_recip;
            ];
          group "Ternary" [ test "where" test_where; test "mulacc" test_mulacc ];
          group "Backend-specific"
            [
              test "CUDA half intrinsics" test_cuda_half_intrinsics;
              test "Metal precise sin" test_metal_precise_sin;
              test "Clang builtins" test_clang_builtins;
            ];
          test "paren stripping" test_paren_stripping;
        ];
      group "Control Flow"
        [
          test "for loop" test_for_loop;
          test "nested loops" test_nested_loops;
          test "conditional" test_conditional;
        ];
      group "Memory"
        [
          test "simple load/store" test_simple_load_store;
          test "gated load" test_gated_load;
        ];
      group "Cast and Bitcast"
        [
          test "cast per backend" test_cast_per_backend;
          test "bitcast per backend" test_bitcast_per_backend;
        ];
      group "Special Dimensions"
        [
          test "Group_id" test_special_group_id;
          test "Local_id" test_special_local_id;
          test "Global_idx" test_special_global_idx;
          test "Clang fails" test_special_clang_fails;
        ];
      group "Shared Memory and Barrier"
        [
          test "shared memory qualifiers" test_shared_memory_qualifiers;
          test "barrier syntax" test_barrier_syntax;
        ];
      group "Vectorize and Gep"
        [ test "vectorize" test_vectorize; test "gep" test_gep ];
      group "Kernel Signature"
        [
          test "function prefix" test_function_prefix;
          test "kernel name" test_kernel_name;
          test "parameter qualifiers" test_param_qualifiers;
          test "scalar parameter" test_scalar_parameter;
        ];
      group "Preamble"
        [
          test "CUDA bitcast template" test_cuda_bitcast_template;
          test "CUDA fp16 include" test_cuda_fp16_include;
          test "Metal stdlib" test_metal_stdlib;
          test "OpenCL fp16 pragma" test_opencl_fp16_pragma;
        ];
      group "Non-native Rewrites"
        [ test "bf16 promoted to f32" test_bf16_promoted_to_f32 ];
      group "Clang ABI" [ test "fixed ABI wrapper" test_clang_fixed_abi ];
      group "CUDA Launch Bounds"
        [ test "launch bounds" test_cuda_launch_bounds ];
      group "Variable Naming"
        [
          test "range variable prefix" test_range_variable_prefix;
          test "special variable names" test_special_variable_names;
        ];
      group "Custom" [ test "custom_inline" test_custom_inline ];
      group "Properties"
        [
          prop "non-empty output"
            (pair safe_dtype renderer_testable)
            prop_non_empty;
          prop "contains kernel name" renderer_testable
            prop_contains_kernel_name;
          prop "balanced braces" renderer_testable prop_balanced_braces;
          prop "deterministic" renderer_testable prop_deterministic;
        ];
    ]
