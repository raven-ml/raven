(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tolk
open Tolk_uop
module U = Uop

let clang_renderer = Cstyle.clang Gpu_target.X86_64
let metal_renderer = Cstyle.metal (Gpu_target.Apple 7)
let opencl_renderer = Cstyle.opencl ""
let intel_renderer = Cstyle.intel ""

let all_renderers =
  [
    ("clang", clang_renderer);
    ("cuda", Cstyle.cuda Gpu_target.SM80);
    ("metal", metal_renderer);
    ("opencl", opencl_renderer);
  ]

let gpu_renderers = List.filter (fun (name, _) -> name <> "clang") all_renderers

(* Helpers *)

let dt = Dtype.Val.float32
let global_ptr dt = Dtype.Ptr.create dt ~addrspace:Global ~size:(-1)
let local_ptr dt = Dtype.Ptr.create dt ~addrspace:Local ~size:(-1)
let render r prog = Renderer.render r prog
let render_kernel r kernel = Renderer.render r (Linearizer.linearize kernel)
let int32_c n = Const.int Dtype.Val.int32 n
let float_c dt v = Const.float dt v

let with_env name value f =
  let old = Sys.getenv_opt name in
  Unix.putenv name value;
  Fun.protect
    ~finally:(fun () ->
      match old with
      | Some value -> Unix.putenv name value
      | None -> Unix.putenv name "")
    f

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

let apply_extra_matcher renderer node =
  match Renderer.extra_matcher renderer with
  | Some f -> f node
  | None -> None

(* IR Program Builders *)

let param idx ptr = U.param ~slot:idx ~dtype:(Dtype.Ptr ptr) ()
let const value = U.const value
let c0_i32 () = const (int32_c 0)
let ptr_index ptr idx () = U.index ~ptr ~idxs:[idx] ~as_ptr:true ()
let load ?alt ?gate src = U.load ~src ?alt ?gate ()
let store dst value = U.store ~dst ~value ()
let unary op src _dtype = U.alu_unary ~op ~src
let binary op lhs rhs _dtype = U.alu_binary ~op ~lhs ~rhs
let ternary op a b c _dtype = U.alu_ternary ~op ~a ~b ~c
let cast_to dtype src = U.cast ~src ~dtype:(Dtype.Val dtype)
let bitcast_to dtype src = U.bitcast ~src ~dtype:(Dtype.Val dtype)

let make_store_const dt const_value =
  let ptr = global_ptr dt in
  let p0 = param 0 ptr in
  let cv = const const_value in
  let c0 = c0_i32 () in
  let idx = ptr_index p0 c0 () in
  [ p0; cv; c0; idx; store idx cv ]

let make_binop dt mk_op =
  let ptr = global_ptr dt in
  let p0 = param 0 ptr in
  let p1 = param 1 ptr in
  let p2 = param 2 ptr in
  let c0 = c0_i32 () in
  let idx0 = ptr_index p0 c0 () in
  let idx1 = ptr_index p1 c0 () in
  let idx2 = ptr_index p2 c0 () in
  let ld0 = load idx0 in
  let ld1 = load idx1 in
  let op_result = mk_op ld0 ld1 dt in
  [ p0; p1; p2; c0; idx0; idx1; idx2; ld0; ld1; op_result; store idx2 op_result ]

let make_simple_add_f32 () =
  make_binop dt (binary Ops.Add)

let make_unop dt mk_op =
  let ptr = global_ptr dt in
  let p0 = param 0 ptr in
  let p1 = param 1 ptr in
  let c0 = c0_i32 () in
  let idx0 = ptr_index p0 c0 () in
  let idx1 = ptr_index p1 c0 () in
  let ld = load idx0 in
  let op_result = mk_op ld dt in
  [ p0; p1; c0; idx0; idx1; ld; op_result; store idx1 op_result ]

let make_ternary_where dt =
  let ptr = global_ptr dt in
  let p0 = param 0 ptr and p1 = param 1 ptr and p2 = param 2 ptr in
  let c0 = c0_i32 () in
  let idx0 = ptr_index p0 c0 () in
  let idx1 = ptr_index p1 c0 () in
  let idx2 = ptr_index p2 c0 () in
  let ld0 = load idx0 in
  let ld1 = load idx1 in
  let cond = const (Const.bool true) in
  let w = U.alu_ternary ~op:Ops.Where ~a:cond ~b:ld0 ~c:ld1 in
  [ p0; p1; p2; c0; idx0; idx1; idx2; ld0; ld1; cond; w; store idx2 w ]

let make_mulacc dt =
  let ptr = global_ptr dt in
  let p0 = param 0 ptr and p1 = param 1 ptr in
  let p2 = param 2 ptr and p3 = param 3 ptr in
  let c0 = c0_i32 () in
  let idx0 = ptr_index p0 c0 () and idx1 = ptr_index p1 c0 () in
  let idx2 = ptr_index p2 c0 () and idx3 = ptr_index p3 c0 () in
  let ld0 = load idx0 and ld1 = load idx1 and ld2 = load idx2 in
  let mac = U.alu_ternary ~op:Ops.Mulacc ~a:ld0 ~b:ld1 ~c:ld2 in
  [ p0; p1; p2; p3; c0; idx0; idx1; idx2; idx3; ld0; ld1; ld2; mac; store idx3 mac ]

let make_loop () =
  let ptr = global_ptr dt in
  let p0 = param 0 ptr in
  let c10 = const (int32_c 10) in
  let r = U.range ~size:c10 ~axis:0 ~sub:[] ~kind:Axis_type.Loop
      ~dtype:Dtype.Val.int32 () in
  let idx0 = ptr_index p0 r () in
  let ld = load idx0 in
  let idx1 = ptr_index p0 r () in
  let st = store idx1 ld in
  [ p0; c10; r; idx0; ld; idx1; st; U.end_ ~value:ld ~ranges:[ r ] ]

let make_nested_loops () =
  let ptr = global_ptr dt in
  let p0 = param 0 ptr in
  let c10 = const (int32_c 10) and c5 = const (int32_c 5) in
  let r0 = U.range ~size:c10 ~axis:0 ~sub:[] ~kind:Axis_type.Loop
      ~dtype:Dtype.Val.int32 () in
  let r1 = U.range ~size:c5 ~axis:1 ~sub:[] ~kind:Axis_type.Loop
      ~dtype:Dtype.Val.int32 ~parents:[ r0 ] () in
  let sum = U.alu_binary ~op:Ops.Add ~lhs:r0 ~rhs:r1 in
  let idx0 = ptr_index p0 sum () in
  let ld = load idx0 in
  let idx1 = ptr_index p0 sum () in
  let st = store idx1 ld in
  [ p0; c10; c5; r0; r1; sum; idx0; ld; idx1; st;
    U.end_ ~value:ld ~ranges:[ r1 ]; U.end_ ~value:r0 ~ranges:[ r0 ] ]

let make_special dim =
  let ptr = global_ptr Dtype.Val.int32 in
  let p0 = param 0 ptr in
  let c64 = const (int32_c 64) in
  let sp =
    U.special ~name:(Gpu_dim.to_special_name dim) ~size:c64
      ~dtype:Dtype.Val.int32 ()
  in
  let idx = ptr_index p0 sp () in
  [ p0; c64; sp; idx; store idx sp ]

let make_shared_memory () =
  let gptr = global_ptr dt in
  let lptr = local_ptr dt in
  let p0 = param 0 gptr in
  let dl =
    U.buffer ~slot:0 ~dtype:(Dtype.Ptr (Dtype.Ptr.with_size 256 lptr)) ()
  in
  let c0 = c0_i32 () in
  let lidx = ptr_index dl c0 () in
  let fzero = const (float_c dt 0.0) in
  let st0 = store lidx fzero in
  let barrier = U.barrier () in
  let ld = load lidx in
  let gidx = ptr_index p0 c0 () in
  [ p0; dl; c0; lidx; fzero; st0; barrier; ld; gidx; store gidx ld ]

let make_gated_load () =
  let ptr = global_ptr dt in
  let p0 = param 0 ptr and p1 = param 1 ptr in
  let c0 = c0_i32 () in
  let gate = const (Const.bool true) in
  let idx0 = ptr_index p0 c0 () in
  let alt = const (float_c dt 0.0) in
  let ld = load ~alt ~gate idx0 in
  let idx1 = ptr_index p1 c0 () in
  [ p0; p1; c0; gate; idx0; alt; ld; idx1; store idx1 ld ]

let make_gated_store () =
  let ptr = global_ptr dt in
  let p0 = param 0 ptr in
  let c0 = c0_i32 () in
  let gate = const (Const.bool true) in
  let value = const (float_c dt 2.0) in
  let idx = ptr_index p0 c0 () in
  [ p0; c0; gate; value; idx; U.store ~dst:idx ~value ~gate () ]

let make_vector_access () =
  let vdt = Dtype.Val.vec 4 dt in
  let ptr = global_ptr vdt in
  let p0 = param 0 ptr and p1 = param 1 ptr in
  let c0 = c0_i32 () in
  let idx0 = ptr_index p0 c0 () in
  let idx1 = ptr_index p1 c0 () in
  let ld = load idx0 in
  [ p0; p1; c0; idx0; idx1; ld; store idx1 ld ]

let make_dtype_changing_load () =
  let in_ptr = global_ptr Dtype.Val.int32 in
  let out_ptr = global_ptr dt in
  let p0 = param 0 in_ptr and p1 = param 1 out_ptr in
  let c0 = c0_i32 () in
  let idx0 = ptr_index p0 c0 () in
  let idx1 = ptr_index p1 c0 () in
  let ld = U.replace (load idx0) ~dtype:(Dtype.Val dt) () in
  [ p0; p1; c0; idx0; idx1; ld; store idx1 ld ]

let make_pointer_bitcast_load () =
  let src_ptr = global_ptr dt in
  let cast_ptr = global_ptr Dtype.Val.int32 in
  let out_ptr = global_ptr Dtype.Val.int32 in
  let p0 = param 0 src_ptr and p1 = param 1 out_ptr in
  let c0 = c0_i32 () in
  let cast = U.bitcast ~src:p0 ~dtype:(Dtype.Ptr cast_ptr) in
  let src = ptr_index cast c0 () in
  let dst = ptr_index p1 c0 () in
  let ld = load src in
  [ p0; p1; c0; cast; src; dst; ld; store dst ld ]

let make_shrink_load () =
  let ptr = global_ptr dt in
  let p0 = param 0 ptr and p1 = param 1 ptr in
  let c0 = c0_i32 () in
  let shrunk = U.shrink ~src:p0 ~offset:c0 ~size:c0 in
  let dst = ptr_index p1 c0 () in
  let ld = load shrunk in
  [ p0; p1; c0; shrunk; dst; ld; store dst ld ]

let make_image_load () =
  let img_ptr = Dtype.Image.imagef [ 4; 4; 4 ] in
  let buf_ptr = global_ptr dt in
  let img = U.param ~slot:0 ~dtype:(Dtype.Ptr img_ptr) () in
  let buf = param 1 buf_ptr in
  let c0 = const (int32_c 0) and c1 = const (int32_c 1) in
  let coord = U.stack ~dtype:Dtype.Val.int32 [ c0; c1 ] in
  let src = ptr_index img coord () in
  let dst = ptr_index buf c0 () in
  U.sink [ store dst (load src) ]

let make_image_store () =
  let img_ptr = Dtype.Image.imagef [ 4; 4; 4 ] in
  let buf_ptr = global_ptr dt in
  let img = U.param ~slot:0 ~dtype:(Dtype.Ptr img_ptr) () in
  let buf = param 1 buf_ptr in
  let c0 = const (int32_c 0) and c1 = const (int32_c 1) in
  let coord = U.stack ~dtype:Dtype.Val.int32 [ c0; c1 ] in
  let src = ptr_index buf c0 () in
  let dst = ptr_index img coord () in
  U.sink [ store dst (load src) ]

let make_gated_image_store () =
  let img_ptr = Dtype.Image.imagef [ 4; 4; 4 ] in
  let buf_ptr = global_ptr dt in
  let img = U.param ~slot:0 ~dtype:(Dtype.Ptr img_ptr) () in
  let buf = param 1 buf_ptr in
  let c0 = const (int32_c 0) and c1 = const (int32_c 1) in
  let gate = const (Const.bool true) in
  let coord = U.stack ~dtype:Dtype.Val.int32 [ c0; c1 ] in
  let src = ptr_index buf c0 () in
  let dst = ptr_index img coord () in
  U.sink [ U.store ~dst ~value:(load src) ~gate () ]

let make_type_convert ~from_dt ~to_dt mk_convert =
  let from_ptr = global_ptr from_dt in
  let to_ptr = global_ptr to_dt in
  let p0 = param 0 from_ptr and p1 = param 1 to_ptr in
  let c0 = c0_i32 () in
  let idx0 = ptr_index p0 c0 () and idx1 = ptr_index p1 c0 () in
  let ld = load idx0 in
  let converted = mk_convert ld in
  [ p0; p1; c0; idx0; idx1; ld; converted; store idx1 converted ]

let make_cast ~from_dt ~to_dt = make_type_convert ~from_dt ~to_dt (cast_to to_dt)

let make_bitcast ~from_dt ~to_dt =
  make_type_convert ~from_dt ~to_dt (bitcast_to to_dt)

let float_vec scalar count value =
  let dtype = Dtype.Val.of_scalar scalar in
  U.stack
    ~dtype:(Dtype.Val.vec count dtype)
    (List.init count (fun _ -> const (Const.float dtype value)))

let make_wmma ?(device = "AMD") ?(threads = 64)
    ?(upcast_axes = ([], [], [])) ~name ~dims ~dtype_in ~dtype_out ~a_count
    ~b_count ~c_count () =
  let a = float_vec dtype_in a_count 1.0 in
  let b = float_vec dtype_in b_count 1.0 in
  let c = float_vec dtype_out c_count 0.0 in
  let info : U.wmma_info =
    {
      name;
      dims;
      dtype_in;
      dtype_out;
      device;
      threads;
      upcast_axes;
      reduce_axes = [];
    }
  in
  let dtype = Dtype.Val.vec c_count (Dtype.Val.of_scalar dtype_out) in
  U.toposort (U.wmma ~a ~b ~c ~info ~dtype)

let make_vectorize_index () =
  let vdt = Dtype.Val.vec 4 dt in
  let ptr = global_ptr dt in
  let p0 = param 0 ptr and p1 = param 1 ptr in
  let c0 = const (int32_c 0) and c1 = const (int32_c 1) in
  let c2 = const (int32_c 2) and c3 = const (int32_c 3) in
  let idx0 = ptr_index p0 c0 () and idx1 = ptr_index p0 c1 () in
  let idx2 = ptr_index p0 c2 () and idx3 = ptr_index p0 c3 () in
  let ld0 = load idx0 and ld1 = load idx1 in
  let ld2 = load idx2 and ld3 = load idx3 in
  let vec = U.stack ~dtype:vdt [ ld0; ld1; ld2; ld3 ] in
  let lane = ptr_index vec c2 () in
  let oidx = ptr_index p1 c0 () in
  [
    p0;
    p1;
    c0;
    c1;
    c2;
    c3;
    idx0;
    idx1;
    idx2;
    idx3;
    ld0;
    ld1;
    ld2;
    ld3;
    vec;
    lane;
    oidx;
    store oidx lane;
  ]

let make_custom () =
  let ptr = global_ptr dt in
  let p0 = param 0 ptr in
  let c0 = c0_i32 () in
  let idx = ptr_index p0 c0 () in
  let ld = load idx in
  let ci = U.custom_inline ~fmt:"custom_func({0}, {0})" ~args:[ ld ] ~dtype:dt in
  [ p0; c0; idx; ld; ci; store idx ci ]

let make_define_var () =
  let ptr = global_ptr dt in
  let p0 = param 0 ptr in
  let dv = U.variable ~name:"n" ~min_val:0 ~max_val:1024 ~dtype:Dtype.Val.int32 () in
  let idx = ptr_index p0 dv () in
  let ld = load idx in
  [ p0; dv; idx; ld; store idx ld ]

let make_chained_binop dt mk_op n =
  let ptr = global_ptr dt in
  let p0 = param 0 ptr and p1 = param 1 ptr in
  let c0 = c0_i32 () in
  let idx_in = ptr_index p1 c0 () in
  let ld = load idx_in in
  let chain = ref [] in
  let result = ref ld in
  for _ = 0 to n - 1 do
    let next = mk_op !result ld dt in
    chain := next :: !chain;
    result := next
  done;
  let idx_out = ptr_index p0 c0 () in
  [ p0; p1; c0; idx_in; ld ] @ List.rev !chain @
  [ idx_out; store idx_out !result ]

let make_conditional () =
  let ptr = global_ptr dt in
  let p0 = param 0 ptr in
  let c0 = c0_i32 () in
  let idx = ptr_index p0 c0 () in
  let cond = const (Const.bool true) in
  let if_ = U.if_ ~cond ~idx_for_dedup:idx in
  let fval = const (float_c dt 42.0) in
  [ p0; c0; idx; cond; if_; fval; store idx fval; U.endif ~if_ ]

let make_launch_bounds () =
  let ptr = global_ptr Dtype.Val.int32 in
  let p0 = param 0 ptr in
  let c64 = const (int32_c 64) in
  let lid0 = U.special ~name:(Gpu_dim.to_special_name (Gpu_dim.Local_id 0)) ~size:c64
      ~dtype:Dtype.Val.int32 () in
  let c4 = const (int32_c 4) in
  let lid1 = U.special ~name:(Gpu_dim.to_special_name (Gpu_dim.Local_id 1)) ~size:c4
      ~dtype:Dtype.Val.int32 () in
  let sum = U.alu_binary ~op:Ops.Add ~lhs:lid0 ~rhs:lid1 in
  let idx = ptr_index p0 sum () in
  [ p0; c64; lid0; c4; lid1; sum; idx; store idx sum ]

(* Frequently-used programs *)

let f32_1 = make_store_const dt (float_c dt 1.0)

(* Comparison program builder: loads from two float32 inputs, applies cmp, stores bool *)
let make_comparison mk_op =
  let in_ptr = global_ptr dt in
  let out_ptr = global_ptr Dtype.Val.bool in
  let p0 = param 0 in_ptr and p1 = param 1 in_ptr and p2 = param 2 out_ptr in
  let c0 = c0_i32 () in
  let idx0 = ptr_index p0 c0 () and idx1 = ptr_index p1 c0 () in
  let idx2 = ptr_index p2 c0 () in
  let ld0 = load idx0 and ld1 = load idx1 in
  let cmp = mk_op ld0 ld1 Dtype.Val.bool in
  [ p0; p1; p2; c0; idx0; idx1; idx2; ld0; ld1; cmp; store idx2 cmp ]

(* Property test support *)

let renderer_testable =
  let gen = Gen.oneofl all_renderers in
  let pp fmt (name, _) = Format.pp_print_string fmt name in
  testable ~pp ~equal:(fun (a, _) (b, _) -> String.equal a b) ~gen ()

let safe_dtypes = [ Dtype.Val.int32; Dtype.Val.float32; Dtype.Val.float64; Dtype.Val.uint32 ]

let safe_dtype =
  let gen = Gen.oneofl safe_dtypes in
  testable ~pp:Dtype.Val.pp ~equal:Dtype.Val.equal ~gen ()

(* Runner *)

let () =
  run "Renderer"
    [
      group "Constants"
        [
          test "int constant" (fun () ->
            let prog = make_store_const Dtype.Val.int32 (int32_c 42) in
            for_each_renderer all_renderers (fun name r ->
                assert_contains (name ^ " int 42") (render r prog) "42"));
          test "float32 constant" (fun () ->
            let prog = make_store_const dt (float_c dt 3.14) in
            for_each_renderer all_renderers (fun name r ->
                let out = render r prog in
                assert_contains (name ^ " float32 3.14") out "3.14";
                assert_contains (name ^ " float32 f suffix") out "f"));
          test "float64 constant" (fun () ->
            let prog = make_store_const Dtype.Val.float64 (float_c Dtype.Val.float64 3.14) in
            for_each_renderer all_renderers (fun name r ->
                assert_contains (name ^ " float64 3.14") (render r prog) "3.14"));
          test "bool constants" (fun () ->
            for_each_renderer all_renderers (fun name r ->
                assert_contains (name ^ " bool true")
                  (render r (make_store_const Dtype.Val.bool (Const.bool true))) "1";
                assert_contains (name ^ " bool false")
                  (render r (make_store_const Dtype.Val.bool (Const.bool false))) "0"));
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
                ("metal", metal_renderer);
                ("opencl", opencl_renderer);
              ];
            let nan_out = render clang_renderer nan_prog in
            assert_contains "clang NAN" nan_out "__builtin_nanf";
            assert_contains "clang INF" (render clang_renderer inf_prog) "__builtin_inff");
          test "int64 suffix" (fun () ->
            let prog = make_store_const Dtype.Val.int64 (Const.int Dtype.Val.int64 12345) in
            for_each_renderer all_renderers (fun name r ->
                assert_contains (name ^ " int64 l suffix") (render r prog) "12345l"));
          test "uint32 suffix" (fun () ->
            let prog = make_store_const Dtype.Val.uint32 (Const.int Dtype.Val.uint32 42) in
            for_each_renderer all_renderers (fun name r ->
                let out = render r prog in
                assert_contains (name ^ " uint32 u suffix") out "42u";
                assert_not_contains (name ^ " uint32 has no uu suffix") out "42uu"));
          test "uint64 suffix" (fun () ->
            let prog = make_store_const Dtype.Val.uint64 (Const.int Dtype.Val.uint64 42) in
            for_each_renderer all_renderers (fun name r ->
                assert_contains (name ^ " uint64 ul suffix") (render r prog) "42ul"));
          test "uint64 constants are rendered unsigned" (fun () ->
            let prog =
              make_store_const Dtype.Val.uint64
                (Const.int64 Dtype.Val.uint64 Int64.minus_one)
            in
            for_each_renderer all_renderers (fun name r ->
                assert_contains (name ^ " uint64 truncation") (render r prog)
                  "18446744073709551615ul"));
        ];
      group "ALU Operations"
        [
          group "Binary"
            [
              test "arithmetic operators" (fun () ->
                let ops =
                  [
                    ("Add +", (fun l r dt -> binary Ops.Add l r dt), "+");
                    ("Sub -", (fun l r dt -> binary Ops.Sub l r dt), "-");
                    ("Mul *", (fun l r dt -> binary Ops.Mul l r dt), "*");
                    ("Mod %", (fun l r dt -> binary Ops.Cmod l r dt), "%");
                    ("Shl <<", (fun l r dt -> binary Ops.Shl l r dt), "<<");
                    ("Shr >>", (fun l r dt -> binary Ops.Shr l r dt), ">>");
                    ("And &", (fun l r dt -> binary Ops.And l r dt), "&");
                    ("Or |", (fun l r dt -> binary Ops.Or l r dt), "|");
                    ("Xor ^", (fun l r dt -> binary Ops.Xor l r dt), "^");
                  ]
                in
                List.iter
                  (fun (label, mk_op, expected) ->
                    let op_dt =
                      if String.length expected = 1 && expected.[0] = '/' then dt
                      else Dtype.Val.int32
                    in
                    let prog = make_binop op_dt mk_op in
                    for_each_renderer all_renderers (fun name r ->
                        assert_contains (name ^ " " ^ label) (render r prog) expected))
                  ops);
              test "raw Fdiv is Clang-only" (fun () ->
                let prog = make_binop dt (fun l r dt -> binary Ops.Fdiv l r dt) in
                assert_contains "clang Fdiv /" (render clang_renderer prog) "/";
                for_each_renderer gpu_renderers (fun name r ->
                    raises_match
                      (function
                        | Invalid_argument msg -> contains msg "unhandled op FDIV"
                        | _ -> false)
                      (fun () ->
                        ignore (render r prog);
                        failwith (name ^ " should reject raw Fdiv"))));
              test "integer division" (fun () ->
                let prog =
                  make_binop Dtype.Val.int32 (fun l r dt ->
                      binary Ops.Cdiv l r dt)
                in
                for_each_renderer all_renderers (fun name r ->
                    assert_contains (name ^ " Idiv /") (render r prog) "/"));
              test "comparison operators" (fun () ->
                let ops =
                  [
                    ("Cmplt <", (fun l r dt -> binary Ops.Cmplt l r dt), "<");
                    ("Cmpeq ==", (fun l r dt -> binary Ops.Cmpeq l r dt), "==");
                    ("Cmpne !=", (fun l r dt -> binary Ops.Cmpne l r dt), "!=");
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
                      binary Ops.Max l r dt)
                in
                for_each_renderer all_renderers (fun name r ->
                    raises_match
                      (function
                        | Invalid_argument msg -> contains msg "unhandled"
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
                    ("Neg", (fun s dt -> unary Ops.Neg s dt), "-");
                    ("Exp2", (fun s dt -> unary Ops.Exp2 s dt), "exp2");
                    ("Log2", (fun s dt -> unary Ops.Log2 s dt), "log2");
                    ("Sin", (fun s dt -> unary Ops.Sin s dt), "sin");
                    ("Sqrt", (fun s dt -> unary Ops.Sqrt s dt), "sqrt");
                    ("Trunc", (fun s dt -> unary Ops.Trunc s dt), "trunc");
                  ]
                in
                List.iter
                  (fun (label, mk_op, expected) ->
                    let prog = make_unop dt mk_op in
                    let renderers =
                      match label with
                      | "Neg" | "Sqrt" | "Trunc" -> all_renderers
                      | _ -> gpu_renderers
                    in
                    for_each_renderer renderers (fun name r ->
                        assert_contains (name ^ " " ^ label) (render r prog) expected))
                  ops);
              test "reciprocal" (fun () ->
                let prog =
                  make_unop dt (fun s dt -> unary Ops.Reciprocal s dt)
                in
                for_each_renderer gpu_renderers (fun name r ->
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
                        | Invalid_argument msg -> contains msg "unhandled"
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
                    let out = render cuda (make_unop Dtype.Val.float16 mk_op) in
                    assert_contains ("CUDA " ^ expected) out expected)
                  [
                    ("hexp2", fun s dt -> unary Ops.Exp2 s dt);
                    ("hlog2", fun s dt -> unary Ops.Log2 s dt);
                    ("hsin", fun s dt -> unary Ops.Sin s dt);
                    ("hsqrt", fun s dt -> unary Ops.Sqrt s dt);
                    ("hrcp", fun s dt -> unary Ops.Reciprocal s dt);
                    ("htrunc", fun s dt -> unary Ops.Trunc s dt);
                  ]);
              test "Metal precise sin" (fun () ->
                let prog =
                  make_unop dt (fun s dt -> unary Ops.Sin s dt)
                in
                assert_contains "Metal precise::sin" (render metal_renderer prog) "precise::sin");
              test "Clang builtins" (fun () ->
                let clang = clang_renderer in
                let sqrt_out =
                  render clang
                    (make_unop dt (fun s dt -> unary Ops.Sqrt s dt))
                in
                assert_contains "clang __builtin_sqrtf" sqrt_out "__builtin_sqrtf";
                let trunc_out =
                  render clang
                    (make_unop dt (fun s dt -> unary Ops.Trunc s dt))
                in
                assert_contains "clang __builtin_truncf" trunc_out "__builtin_truncf");
            ];
          test "paren stripping" (fun () ->
            let mk_add l r dt = binary Ops.Add l r dt in
            let mk_sub l r dt = binary Ops.Sub l r dt in
            let mk_mul l r dt = binary Ops.Mul l r dt in
            let mk_xor l r dt = binary Ops.Xor l r dt in
            let mk_or l r dt = binary Ops.Or l r dt in
            let mk_and l r dt = binary Ops.And l r dt in
            let prog_add = make_chained_binop dt mk_add 5 in
            let prog_sub = make_chained_binop dt mk_sub 5 in
            let prog_mul = make_chained_binop dt mk_mul 5 in
            let prog_xor = make_chained_binop Dtype.Val.int32 mk_xor 5 in
            let prog_or = make_chained_binop Dtype.Val.int32 mk_or 5 in
            let prog_and = make_chained_binop Dtype.Val.int32 mk_and 5 in
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
                  binary Ops.Add l r dt)
            in
            for_each_renderer all_renderers (fun name r ->
                assert_contains (name ^ " dereference") (render r prog) "*"));
          test "gated load" (fun () ->
            let prog = make_gated_load () in
            for_each_renderer all_renderers (fun name r ->
                assert_contains (name ^ " gated load ternary") (render r prog) "?"));
          test "gated store" (fun () ->
            let out = render clang_renderer (make_gated_store ()) in
            assert_contains "gated store if" out "if (1)";
            assert_contains "gated store assignment" out " = 2.0f;");
          test "vector load/store casts access pointer" (fun () ->
            for_each_renderer all_renderers (fun name r ->
                let out = render r (make_vector_access ()) in
                assert_contains (name ^ " vector access cast") out "*((";
                assert_contains (name ^ " vector access dtype") out "float4*"));
          test "dtype-changing load casts access pointer" (fun () ->
            let out = render clang_renderer (make_dtype_changing_load ()) in
            assert_contains "dtype-changing load casts" out "*((float*)";
            assert_not_contains "dtype-changing load does not cast to source"
              out "*((int*)");
          test "shrink renders like index" (fun () ->
            let out = render clang_renderer (make_shrink_load ()) in
            assert_contains "shrink renders pointer offset" out "+0";
            assert_not_contains "shrink not named temp" out "alu0 =");
          test "opencl image load/store" (fun () ->
            let load_out = render_kernel opencl_renderer (make_image_load ()) in
            assert_contains "opencl image param" load_out "read_only image2d_t";
            assert_contains "opencl sampler preamble" load_out "const sampler_t smp";
            assert_contains "opencl read_imagef" load_out "read_imagef(";
            let store_out = render_kernel opencl_renderer (make_image_store ()) in
            assert_contains "opencl mutable image param" store_out "write_only image2d_t";
            assert_contains "opencl write_imagef" store_out "write_imagef(";
            let gated_store_out =
              render_kernel opencl_renderer (make_gated_image_store ())
            in
            assert_contains "opencl gated mutable image param" gated_store_out
              "write_only image2d_t";
            assert_contains "opencl gated write_imagef" gated_store_out
              "if (1)";
            assert_contains "opencl gated write_imagef body" gated_store_out
              "write_imagef(");
          test "non-opencl image rejected" (fun () ->
            raises_match
              (function
                | Failure msg -> contains msg "does not support images"
                | _ -> false)
              (fun () ->
                ignore
                  (render_kernel metal_renderer (make_image_load ()))))
        ];
      group "Cast and Bitcast"
        [
          test "cast per backend" (fun () ->
            let prog = make_cast ~from_dt:Dtype.Val.int32 ~to_dt:dt in
            let metal_out = render metal_renderer prog in
            assert_contains "metal cast" metal_out "(float)";
            let cuda_out = render (Cstyle.cuda Gpu_target.SM80) prog in
            assert_contains "cuda cast" cuda_out "(float)";
            let opencl_out = render opencl_renderer prog in
            assert_contains "opencl cast" opencl_out "(float)";
            assert_contains "clang cast" (render clang_renderer prog) "(float)");
          test "bitcast per backend" (fun () ->
            let prog = make_bitcast ~from_dt:dt ~to_dt:Dtype.Val.int32 in
            assert_contains "clang __builtin_bit_cast"
              (render clang_renderer prog) "__builtin_bit_cast";
            assert_contains "cuda tg_bitcast"
              (render (Cstyle.cuda Gpu_target.SM80) prog) "tg_bitcast";
            assert_contains "metal as_type"
              (render metal_renderer prog) "as_type<";
            assert_contains "opencl as_"
              (render opencl_renderer prog) "as_");
          test "pointer bitcast in global memory is passthrough" (fun () ->
            let out = render clang_renderer (make_pointer_bitcast_load ()) in
            assert_not_contains "pointer bitcast passthrough" out
              "__builtin_bit_cast");
        ];
      group "Special Dimensions"
        [
          test "Group_id" (fun () ->
            let prog = make_special (Gpu_dim.Group_id 0) in
            assert_contains "cuda blockIdx.x"
              (render (Cstyle.cuda Gpu_target.SM80) prog) "blockIdx.x";
            assert_contains "metal gid.x"
              (render metal_renderer prog) "gid.x";
            assert_contains "opencl get_group_id(0)"
              (render opencl_renderer prog) "get_group_id(0)");
          test "Local_id" (fun () ->
            let prog = make_special (Gpu_dim.Local_id 1) in
            assert_contains "cuda threadIdx.y"
              (render (Cstyle.cuda Gpu_target.SM80) prog) "threadIdx.y";
            assert_contains "metal lid.y"
              (render metal_renderer prog) "lid.y";
            assert_contains "opencl get_local_id(1)"
              (render opencl_renderer prog) "get_local_id(1)");
          test "Global_idx" (fun () ->
            let prog = make_special (Gpu_dim.Global_idx 2) in
            assert_contains "cuda global idx formula"
              (render (Cstyle.cuda Gpu_target.SM80) prog)
              "(blockIdx.z*blockDim.z+threadIdx.z)";
            raises_match
              (function Failure _ | Invalid_argument _ -> true | _ -> false)
              (fun () -> ignore (render metal_renderer prog));
            assert_contains "opencl get_global_id(2)"
              (render opencl_renderer prog) "get_global_id(2)");
          test "Clang fails" (fun () ->
            raises_match
              (function Failure _ | Invalid_argument _ -> true | _ -> false)
              (fun () -> ignore (render clang_renderer (make_special (Gpu_dim.Group_id 0)))));
        ];
      group "Shared Memory and Barrier"
        [
          test "shared memory qualifiers" (fun () ->
            let prog = make_shared_memory () in
            assert_contains "cuda __shared__"
              (render (Cstyle.cuda Gpu_target.SM80) prog) "__shared__";
            assert_contains "metal threadgroup"
              (render metal_renderer prog) "threadgroup";
            assert_contains "opencl __local"
              (render opencl_renderer prog) "__local");
          test "barrier syntax" (fun () ->
            let prog = make_shared_memory () in
            assert_contains "cuda __syncthreads"
              (render (Cstyle.cuda Gpu_target.SM80) prog) "__syncthreads()";
            assert_contains "metal threadgroup_barrier"
              (render metal_renderer prog) "threadgroup_barrier";
            assert_contains "opencl barrier"
              (render opencl_renderer prog) "barrier(CLK_LOCAL_MEM_FENCE)");
        ];
      group "Vectorize and Index"
        [
          test "vectorize" (fun () ->
            let prog = make_vectorize_index () in
            for_each_renderer all_renderers (fun name r ->
                let out = render r prog in
                assert_contains (name ^ " vectorize val0") out "val0";
                assert_contains (name ^ " vectorize val1") out "val1";
                assert_contains (name ^ " vectorize val2") out "val2";
                assert_contains (name ^ " vectorize val3") out "val3"));
          test "index" (fun () ->
            let prog = make_vectorize_index () in
            for_each_renderer all_renderers (fun name r ->
                let out = render r prog in
                if not (contains out "[2]" || contains out ".z" || contains out "+2") then
                  failwith
                    (Printf.sprintf
                       "%s: expected index element 2 access ([2], .z, or +2), got:\n%s"
                       name out)));
        ];
      group "Kernel Signature"
        [
          test "function prefix" (fun () ->
            let cuda_out = render (Cstyle.cuda Gpu_target.SM80) f32_1 in
            assert_contains "cuda extern C" cuda_out {|extern "C"|};
            assert_contains "cuda __global__" cuda_out "__global__";
            assert_contains "metal kernel void"
              (render metal_renderer f32_1) "kernel void";
            assert_contains "opencl __kernel"
              (render opencl_renderer f32_1) "__kernel";
            assert_contains "clang void" (render clang_renderer f32_1) "void");
          test "kernel name" (fun () ->
            for_each_renderer all_renderers (fun name r ->
                assert_contains (name ^ " kernel name")
                  (Renderer.render r ~name:"my_test_kernel" f32_1) "my_test_kernel"));
          test "parameter qualifiers" (fun () ->
            assert_contains "opencl __global"
              (render opencl_renderer f32_1) "__global";
            assert_contains "metal device"
              (render metal_renderer f32_1) "device");
          test "scalar parameter" (fun () ->
            let prog = make_define_var () in
            assert_contains "clang scalar param in inner signature"
              (render clang_renderer prog) "static void test_(float* restrict data0_-1, const int n)";
            assert_contains "clang scalar param forwarded through vals"
              (render clang_renderer prog) "test_((float*)bufs[0], (int)vals[0]);";
            assert_contains "opencl scalar param in signature"
              (render opencl_renderer prog) "__global float* data0_-1, const int n";
            assert_contains "metal scalar param in signature"
              (render metal_renderer prog) "device float* data0_-1, constant int& n";
            assert_contains "cuda scalar param in signature"
              (render (Cstyle.cuda Gpu_target.SM80) prog) "float* data0_-1, const int n");
          test "buffer parameter base uses canonical name, body uses type_map" (fun () ->
            (* tinygrad quirk: PtrDType.scalar() returns the pointer, so a
               pointer param's base misses type_map and falls back to the raw
               scalar name, while value-typed uses in the body do apply
               type_map. A bf16 metal buffer is thus [device __bf16*] in the
               signature but [bfloat] in the body. *)
            let prog = make_store_const Dtype.Val.bfloat16 (float_c Dtype.Val.bfloat16 1.0) in
            let metal_out = render (Cstyle.metal (Gpu_target.Apple 7)) prog in
            assert_contains "metal bf16 buffer signature" metal_out "device __bf16*";
            assert_contains "metal bf16 body cast" metal_out "(bfloat)");
          test "renderer op capabilities match cstyle render surface" (fun () ->
            let check_ops name r =
              let ops = Renderer.code_for_op r in
              if List.mem Renderer.Max ops || List.mem Renderer.Mulacc ops
                 || List.mem Renderer.Threefry ops then
                failwith (name ^ " should not advertise unrendered composite ops")
            in
            for_each_renderer
              (all_renderers @ [ ("qcom", Cstyle.qcom); ("amd", Cstyle.amd Gpu_target.CDNA3) ])
              check_ops);
          test "renderer dtype capabilities are backend-specific" (fun () ->
            let fp8 = Dtype.of_scalar Dtype.Fp8e4m3 in
            if Renderer.supports_dtype Cstyle.qcom Dtype.float64 then
              failwith "qcom should not advertise float64";
            if Renderer.supports_dtype Cstyle.qcom Dtype.float16 then
              failwith "qcom should not advertise float16";
            with_env "IMAGE" "1" (fun () ->
                if Renderer.supports_dtype Cstyle.qcom Dtype.float16 then
                  failwith "qcom should require FLOAT16 for float16");
            with_env "FLOAT16" "1" (fun () ->
                if Renderer.supports_dtype Cstyle.qcom Dtype.float16 then
                  failwith "qcom should require IMAGE for float16");
            with_env "IMAGE" "1" (fun () ->
                with_env "FLOAT16" "1" (fun () ->
                    if not (Renderer.supports_dtype Cstyle.qcom Dtype.float16) then
                      failwith "qcom should advertise float16 with IMAGE and FLOAT16"));
            if Renderer.supports_dtype Cstyle.qcom fp8 then
              failwith "qcom should not advertise fp8";
            if not (Renderer.supports_dtype (Cstyle.clang Gpu_target.X86_64) Dtype.bfloat16) then
              failwith "x86_64 clang should advertise bfloat16";
            if not (Renderer.supports_dtype (Cstyle.clang Gpu_target.Arm64) Dtype.bfloat16) then
              failwith "arm64 clang should advertise bfloat16";
            if Renderer.supports_dtype (Cstyle.clang Gpu_target.Riscv64) Dtype.bfloat16 then
              failwith "riscv64 clang should not advertise bfloat16";
            if Renderer.emulated_float_dtypes (Cstyle.clang Gpu_target.X86_64) <> [] then
              failwith "x86_64 clang should not storage-emulate bfloat16";
            if
              Renderer.emulated_float_dtypes (Cstyle.clang Gpu_target.Riscv64)
              <> [ (Dtype.Bfloat16, Dtype.Float32) ]
            then
              failwith "riscv64 clang should storage-emulate bfloat16";
            if
              Renderer.emulated_float_dtypes (Cstyle.opencl "cl_khr_fp16")
              <> []
            then
              failwith "opencl should not decompose native float16";
            if Renderer.emulated_float_dtypes Cstyle.qcom <> [] then
              failwith "qcom should not install hidden float decompositions";
            if Renderer.supports_dtype (Cstyle.opencl "") Dtype.float16 then
              failwith "opencl without cl_khr_fp16 should not advertise float16";
            if not (Renderer.supports_dtype (Cstyle.opencl "cl_khr_fp16") Dtype.float16) then
              failwith "opencl with cl_khr_fp16 should advertise float16";
            if Renderer.supports_dtype (Cstyle.opencl "") Dtype.float64 then
              failwith "opencl without cl_khr_fp64 should not advertise float64";
            if not (Renderer.supports_dtype (Cstyle.opencl "cl_khr_fp64") Dtype.float64) then
              failwith "opencl with cl_khr_fp64 should advertise float64";
            if not (Renderer.supports_dtype (Cstyle.opencl "") Dtype.bfloat16) then
              failwith "opencl should advertise rewritten bfloat16";
            if Renderer.supports_dtype (Cstyle.opencl "cl_khr_fp16") fp8 then
              failwith "opencl should not advertise fp8";
            if
              Renderer.image_pitch_alignment
                (Cstyle.opencl "cl_khr_fp16,IMAGE_PITCH_ALIGNMENT=64")
              <> Some 64
            then
              failwith "opencl should parse IMAGE_PITCH_ALIGNMENT";
            if Renderer.supports_dtype (Cstyle.cuda Gpu_target.SM80) fp8 then
              failwith "sm80 should not advertise fp8";
            if not (Renderer.supports_dtype (Cstyle.cuda Gpu_target.SM89) fp8) then
              failwith "sm89 should advertise ocp fp8";
            if Renderer.supports_dtype (Cstyle.amd Gpu_target.CDNA3) fp8 then
              failwith "cdna3 should not advertise ocp fp8";
            if not (Renderer.supports_dtype (Cstyle.amd Gpu_target.CDNA4) fp8) then
              failwith "cdna4 should advertise ocp fp8";
            if Renderer.supports_dtype (Cstyle.metal (Gpu_target.Apple 5)) Dtype.bfloat16 then
              failwith "Apple5 should not advertise Metal bfloat16";
            if not (Renderer.supports_dtype (Cstyle.metal (Gpu_target.Apple 6)) Dtype.bfloat16) then
              failwith "Apple6 should advertise Metal bfloat16";
            if Renderer.supports_dtype (Cstyle.metal (Gpu_target.Mac 2)) Dtype.bfloat16 then
              failwith "Mac families should not advertise Metal bfloat16";
            if Renderer.supports_dtype (Cstyle.metal (Gpu_target.Apple 7)) Dtype.float64 then
              failwith "Metal should not advertise float64");
          test "metal tensor cores follow Apple GPU family" (fun () ->
            if Gpu_target.parse_metal_arch "Apple7" <> Some (Gpu_target.Apple 7) then
              failwith "Apple7 should parse";
            if Gpu_target.parse_metal_arch "Mac2" <> Some (Gpu_target.Mac 2) then
              failwith "Mac2 should parse";
            if Gpu_target.parse_metal_arch "Apple0" <> None then
              failwith "Apple0 should not parse";
            if Renderer.tensor_cores (Cstyle.metal (Gpu_target.Apple 6)) <> [] then
              failwith "Apple6 should not advertise Metal tensor cores";
            if Renderer.tensor_cores (Cstyle.metal (Gpu_target.Mac 2)) <> [] then
              failwith "Mac families should not advertise Metal tensor cores";
            if Renderer.tensor_cores (Cstyle.metal (Gpu_target.Apple 7)) = [] then
              failwith "Apple7 should advertise Metal tensor cores");
        ];
      group "Preamble"
        [
          test "CUDA bitcast template" (fun () ->
            let prog = make_bitcast ~from_dt:dt ~to_dt:Dtype.Val.int32 in
            assert_contains "cuda tg_bitcast template"
              (render (Cstyle.cuda Gpu_target.SM80) prog) "tg_bitcast");
          test "CUDA fp16 include" (fun () ->
            let prog = make_store_const Dtype.Val.float16 (float_c Dtype.Val.float16 1.0) in
            assert_contains "cuda fp16 include"
              (render (Cstyle.cuda Gpu_target.SM80) prog) "cuda_fp16");
          test "CUDA WMMA helper follows tinygrad asm preamble" (fun () ->
            let name = "WMMA_8_16_16_half_float" in
            let prog =
              make_wmma ~device:"CUDA" ~threads:32 ~name ~dims:(8, 16, 16)
                ~dtype_in:Dtype.Float16 ~dtype_out:Dtype.Float32
                ~upcast_axes:([ (0, 8) ], [ (0, 4) ], [ (0, 4) ])
                ~a_count:8 ~b_count:4 ~c_count:4 ()
            in
            let out = render (Cstyle.cuda Gpu_target.SM80) prog in
            assert_contains "cuda wmma helper signature" out
              "__device__ float4 __WMMA_8_16_16_half_float";
            assert_contains "cuda wmma helper asm" out
              "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32");
          test "CUDA WMMA helper is emitted once per signature" (fun () ->
            let name = "WMMA_8_16_16_half_float" in
            let one =
              make_wmma ~device:"CUDA" ~threads:32 ~name ~dims:(8, 16, 16)
                ~dtype_in:Dtype.Float16 ~dtype_out:Dtype.Float32
                ~upcast_axes:([ (0, 8) ], [ (0, 4) ], [ (0, 4) ])
                ~a_count:8 ~b_count:4 ~c_count:4 ()
            in
            let out = render (Cstyle.cuda Gpu_target.SM80) (one @ one) in
            equal int 1
              (count_substring out
                 "__device__ float4 __WMMA_8_16_16_half_float"));
          test "Metal stdlib" (fun () ->
            assert_contains "metal stdlib" (render metal_renderer f32_1) "metal_stdlib");
          test "Metal WMMA helper follows tinygrad simdgroup preamble" (fun () ->
            let name = "WMMA_8_8_8_float_float" in
            let prog =
              make_wmma ~device:"METAL" ~threads:32 ~name ~dims:(8, 8, 8)
                ~dtype_in:Dtype.Float32 ~dtype_out:Dtype.Float32
                ~upcast_axes:([ (0, 2) ], [ (0, 2) ], [ (0, 2) ])
                ~a_count:2 ~b_count:2 ~c_count:2 ()
            in
            let out = render metal_renderer prog in
            assert_contains "metal wmma helper signature" out
              "float2 __WMMA_8_8_8_float_float";
            assert_contains "metal wmma helper simdgroup" out
              "simdgroup_multiply_accumulate");
          test "OpenCL fp16 pragma" (fun () ->
            let prog = make_store_const Dtype.Val.float16 (float_c Dtype.Val.float16 1.0) in
            assert_contains "opencl fp16 pragma"
              (render opencl_renderer prog) "cl_khr_fp16");
          test "OpenCL aux groups params by slot" (fun () ->
            let prog = make_simple_add_f32 () in
            let expected_dtype =
              Dtype.repr (Dtype.Ptr (global_ptr Dtype.Val.float32))
            in
            equal (list string)
              [
                Printf.sprintf "((0,%s))" expected_dtype;
                Printf.sprintf "((1,%s))" expected_dtype;
                Printf.sprintf "((2,%s))" expected_dtype;
              ]
              (Renderer.aux opencl_renderer prog));
          test "OpenCL aux uses dense parameter ordinals" (fun () ->
            let c0 = c0_i32 () in
            let p0 = param 0 (global_ptr Dtype.Val.float32) in
            let p1 = param 1 (global_ptr Dtype.Val.float32) in
            let prog = [ c0; p0; p1 ] in
            let expected_dtype =
              Dtype.repr (Dtype.Ptr (global_ptr Dtype.Val.float32))
            in
            equal (list string)
              [
                Printf.sprintf "((0,%s))" expected_dtype;
                Printf.sprintf "((1,%s))" expected_dtype;
              ]
              (Renderer.aux opencl_renderer prog));
          test "GPU pointer bitcasts use backend bitcast syntax" (fun () ->
            let f32_ptr = global_ptr Dtype.Val.float32 in
            let i32_ptr = global_ptr Dtype.Val.int32 in
            let p0 = param 0 f32_ptr in
            let p1 = param 1 i32_ptr in
            let c0 = c0_i32 () in
            let src = ptr_index p0 c0 () in
            let dst = ptr_index p1 c0 () in
            let bc = U.bitcast ~src ~dtype:(Dtype.Ptr i32_ptr) in
            let ld = load bc in
            let prog = [ p0; p1; c0; src; dst; bc; ld; store dst ld ] in
            assert_contains "cuda pointer bitcast"
              (render (Cstyle.cuda Gpu_target.SM80) prog) "tg_bitcast<";
            assert_contains "opencl pointer bitcast"
              (render opencl_renderer prog) "as_";
            assert_contains "metal pointer bitcast"
              (render metal_renderer prog) "as_type<");
        ];
      group "Non-native Rewrites"
        [
          (* bf16 promotion is handled by extra_matcher at the Kernel level
             (during codegen), not at render time.  Verify the matcher is set. *)
          test "clang has bf16 extra_matcher" (fun () ->
            match Renderer.extra_matcher clang_renderer with
            | None -> failwith "clang should have extra_matcher for bf16 promotion"
            | Some _ -> ());
          test "amd promotes bf16 ALU through float32" (fun () ->
            let a = U.const (Const.float Dtype.Val.bfloat16 1.0) in
            let b = U.const (Const.float Dtype.Val.bfloat16 2.0) in
            let add = U.alu_binary ~op:Ops.Add ~lhs:a ~rhs:b in
            match apply_extra_matcher (Cstyle.amd Gpu_target.RDNA3) add with
            | Some r ->
                is_true ~msg:"amd bf16 ALU rewrites to casted f32 op"
                  (U.op r = Ops.Cast
                   && Dtype.equal (U.dtype r) Dtype.bfloat16)
            | None -> failwith "amd bf16 ALU extra matcher did not fire");
          test "amd casts bf16 constants through software bf16 path" (fun () ->
            let c = U.const (Const.float Dtype.Val.bfloat16 1.0) in
            match apply_extra_matcher (Cstyle.amd Gpu_target.CDNA4) c with
            | Some r ->
                is_true ~msg:"amd bf16 const keeps bf16 dtype"
                  (Dtype.equal (U.dtype r) Dtype.bfloat16)
            | None -> failwith "amd bf16 const extra matcher did not fire");
          test "amd manual bf16 casts are skipped on CDNA4" (fun () ->
            let src = U.const (Const.float Dtype.Val.bfloat16 1.0) in
            let cast = U.cast ~src ~dtype:Dtype.float32 in
            (match apply_extra_matcher (Cstyle.amd Gpu_target.RDNA3) cast with
             | Some _ -> ()
             | None -> failwith "rdna3 should use manual bf16 cast");
            match apply_extra_matcher (Cstyle.amd Gpu_target.CDNA4) cast with
            | None -> ()
            | Some _ -> failwith "cdna4 should not use manual bf16 cast");
          test "amd bitcasts fp8 WMMA inputs" (fun () ->
            let a = float_vec Dtype.Fp8e4m3 8 1.0 in
            let b = float_vec Dtype.Fp8e4m3 8 1.0 in
            let c = float_vec Dtype.Float32 4 0.0 in
            let info : U.wmma_info =
              {
                name = "WMMA_16_16_128_float8_e4m3_float";
                dims = (16, 16, 128);
                dtype_in = Dtype.Fp8e4m3;
                dtype_out = Dtype.Float32;
                device = "AMD";
                threads = 64;
                upcast_axes = ([], [], []);
                reduce_axes = [];
              }
            in
            let wmma =
              U.wmma ~a ~b ~c ~info
                ~dtype:(Dtype.Val.vec 4 Dtype.Val.float32)
            in
            match apply_extra_matcher (Cstyle.amd Gpu_target.CDNA4) wmma with
            | Some r -> (
                match U.as_wmma r with
                | Some { a; b; _ } ->
                    is_true ~msg:"fp8 WMMA operands are bitcast to uint64"
                      (U.op a = Ops.Bitcast && U.op b = Ops.Bitcast
                       && Dtype.equal (U.dtype a) Dtype.uint64
                       && Dtype.equal (U.dtype b) Dtype.uint64)
                | None -> failwith "expected rewritten WMMA")
            | None -> failwith "amd fp8 WMMA extra matcher did not fire");
        ];
      group "Clang ABI"
        [
          test "fixed ABI wrapper" (fun () ->
            let out = Renderer.render clang_renderer ~name:"kern" f32_1 in
            assert_contains "clang fixed ABI" out
              "void kern(const unsigned long long *bufs");
          test "fixed ABI wraps inner kernel" (fun () ->
            let out = Renderer.render clang_renderer ~name:"kern" f32_1 in
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
            let prog = make_special (Gpu_dim.Group_id 0) in
            for_each_renderer gpu_renderers (fun name r ->
                assert_contains (name ^ " gidx0") (render r prog) "gidx0");
            let prog_lid = make_special (Gpu_dim.Local_id 1) in
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
              (render rdna3 (make_special (Gpu_dim.Group_id 0)))
              "__ockl_get_group_id(0)";
            assert_contains "amd local_id"
              (render rdna3 (make_special (Gpu_dim.Local_id 0)))
              "__ockl_get_local_id(0)");
          test "transcendentals" (fun () ->
            let rdna3 = Cstyle.amd Gpu_target.RDNA3 in
            assert_contains "amd __ocml_sqrt_f32"
              (render rdna3
                 (make_unop dt (fun s dt -> unary Ops.Sqrt s dt)))
              "__ocml_sqrt_f32";
            assert_contains "amd __ocml_sin_f32"
              (render rdna3
                 (make_unop dt (fun s dt -> unary Ops.Sin s dt)))
              "__ocml_sin_f32";
            let f32 value = const (Const.float dt value) in
            let vec = U.stack [ f32 1.0; f32 4.0; f32 9.0; f32 16.0 ] in
            let sqrt = U.alu_unary ~op:Ops.Sqrt ~src:vec in
            let out = Renderer.render rdna3 (U.toposort sqrt) in
            assert_contains "amd vector sqrt declares scalar ocml" out
              "__ocml_sqrt_f32(float)";
            assert_not_contains "amd vector sqrt does not declare f128" out
              "__ocml_sqrt_f128");
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
              make_binop Dtype.Val.bfloat16 (fun l r dt ->
                  binary Ops.Add l r dt)
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
          test "cdna fp8 constants use f32_to_fp8 helper" (fun () ->
            let prog =
              make_store_const Dtype.Val.fp8e4m3
                (Const.float Dtype.Val.fp8e4m3 1.0)
            in
            let out = render (Cstyle.amd Gpu_target.CDNA4) prog in
            assert_contains "amd cdna fp8 helper" out "f32_to_fp8";
            assert_contains "amd cdna fp8 const" out "f32_to_fp8(1.0f, 0)");
          test "cdna fp8 casts use f32_to_fp8 helper" (fun () ->
            let prog =
              make_cast ~from_dt:Dtype.Val.float32 ~to_dt:Dtype.Val.fp8e5m2
            in
            let out = render (Cstyle.amd Gpu_target.CDNA4) prog in
            assert_contains "amd cdna bf8 helper" out "f32_to_fp8";
            assert_contains "amd cdna bf8 cast" out "f32_to_fp8(val0, 1)");
          test "cdna WMMA emits MFMA macro and extra call arguments" (fun () ->
            let name = "WMMA_16_16_128_float8_e4m3_float" in
            let prog =
              make_wmma ~name ~dims:(16, 16, 128)
                ~dtype_in:Dtype.Fp8e4m3 ~dtype_out:Dtype.Float32
                ~a_count:8 ~b_count:8 ~c_count:4 ()
            in
            let out = render (Cstyle.amd Gpu_target.CDNA4) prog in
            assert_contains "amd cdna mfma scale macro" out
              "#define __WMMA_16_16_128_float8_e4m3_float __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4";
            assert_contains "amd cdna wmma call extras" out
              ", 0, 0, 0, 0, 0, 0)");
          test "rdna4 WMMA emits gfx12 builtin macro" (fun () ->
            let name = "WMMA_16_16_16___bf16___bf16" in
            let prog =
              make_wmma ~name ~dims:(16, 16, 16)
                ~dtype_in:Dtype.Bfloat16 ~dtype_out:Dtype.Bfloat16
                ~a_count:8 ~b_count:8 ~c_count:8 ()
            in
            let out = render (Cstyle.amd Gpu_target.RDNA4) prog in
            assert_contains "amd rdna4 wmma macro" out
              "#define __WMMA_16_16_16___bf16___bf16 __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32_gfx12");
          test "rdna3 half output WMMA emits wrapper" (fun () ->
            let name = "WMMA_16_16_16_half_half" in
            let prog =
              make_wmma ~name ~dims:(16, 16, 16)
                ~dtype_in:Dtype.Float16 ~dtype_out:Dtype.Float16
                ~a_count:16 ~b_count:16 ~c_count:8 ()
            in
            let out = render (Cstyle.amd Gpu_target.RDNA3) prog in
            assert_contains "amd rdna3 half wrapper" out
              "static inline __attribute__((device)) half8 __WMMA_16_16_16_half_half");
        ];
      group "Intel"
        [
          test "kernel attribute" (fun () ->
            assert_contains "intel sub_group_size"
              (render intel_renderer f32_1) "intel_reqd_sub_group_size(8)");
        ];
      group "Properties"
        [
          prop "non-empty output"
            (pair safe_dtype renderer_testable)
            (fun (dt, (_name, renderer)) ->
              let const_value =
                match Dtype.Val.scalar dt with
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
