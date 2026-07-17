(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Generates .actual files for codegen pipeline golden tests. Each file
   contains tolk's rendered output for a specific backend + test case after
   running the full codegen pipeline (Pipeline.full_rewrite_to_sink ->
   Linearizer.linearize -> Renderer.render). Dune diff rules compare .actual
   against .expected. *)

open Tolk
open Tolk_uop
module U = Uop

let global_fptr = Dtype.float32
let idx n = U.const (Const.int Dtype.weakint n)

let kernel_info ?(axis_types = []) name =
  {
    U.name;
    axis_types;
    dont_use_locals = false;
    applied_opts = [];
    opts_to_apply = Some [];
    estimates = None;
  beam = 0;
  }

(* Extract the kernel name from a pipeline-processed Sink. *)
let name_of_sink sink =
  match U.as_kernel_info sink with Some ki -> ki.name | None -> "kernel"

(* Full pipeline chain: Kernel.t -> source string. *)
let pipeline_to_source ?(optimize = true) ren sink =
  let processed = Codegen.full_rewrite_to_sink ~optimize ren sink in
  let name = name_of_sink processed in
  let program = Linearizer.linearize processed in
  String.trim (Renderer.render ren ~name program)

(* ── Kernel AST builders ── *)

let make_elementwise_add () =
  let p0 = U.param ~slot:0 ~dtype:global_fptr ~shape:(U.const_int (-1)) () in
  let p1 = U.param ~slot:1 ~dtype:global_fptr ~shape:(U.const_int (-1)) () in
  let p2 = U.param ~slot:2 ~dtype:global_fptr ~shape:(U.const_int (-1)) () in
  let r0 = U.range ~size:(U.const_int 256) ~axis:0 ~kind:Axis_type.Global () in
  let ld_a = U.load ~src:(U.index ~ptr:p0 ~idxs:[r0] ()) () in
  let ld_b = U.load ~src:(U.index ~ptr:p1 ~idxs:[r0] ()) () in
  let add = U.alu_binary ~op:Ops.Add ~lhs:ld_a ~rhs:ld_b in
  let st = U.store ~dst:(U.index ~ptr:p2 ~idxs:[r0] ()) ~value:add () in
  let e = U.end_ ~value:st ~ranges:[ r0 ] in
  U.sink ~kernel_info:(kernel_info ~axis_types:[ Axis_type.Global ] "elementwise_add") [ e ]

let make_sum_reduce () =
  let p0 = U.param ~slot:0 ~dtype:global_fptr ~shape:(U.const_int (-1)) () in
  let p1 = U.param ~slot:1 ~dtype:global_fptr ~shape:(U.const_int (-1)) () in
  let r0 = U.range ~size:(U.const_int 256) ~axis:0 ~kind:Axis_type.Reduce () in
  let ld = U.load ~src:(U.index ~ptr:p0 ~idxs:[r0] ()) () in
  let red = U.reduce ~op:Ops.Add ~src:ld ~ranges:[ r0 ] ~dtype:Dtype.float32 in
  let st =
    U.store ~dst:(U.index ~ptr:p1 ~idxs:[(idx 0)] ()) ~value:red ()
  in
  U.sink ~kernel_info:(kernel_info ~axis_types:[ Axis_type.Reduce ] "sum_reduce") [ st ]

let make_max_reduce () =
  let p0 = U.param ~slot:0 ~dtype:global_fptr ~shape:(U.const_int (-1)) () in
  let p1 = U.param ~slot:1 ~dtype:global_fptr ~shape:(U.const_int (-1)) () in
  let r0 = U.range ~size:(U.const_int 64) ~axis:0 ~kind:Axis_type.Reduce () in
  let ld = U.load ~src:(U.index ~ptr:p0 ~idxs:[ r0 ] ()) () in
  let red =
    U.reduce ~op:Ops.Max ~src:ld ~ranges:[ r0 ] ~dtype:Dtype.float32
  in
  let st =
    U.store ~dst:(U.index ~ptr:p1 ~idxs:[ idx 0 ] ()) ~value:red ()
  in
  U.sink ~kernel_info:(kernel_info ~axis_types:[ Axis_type.Reduce ] "max_reduce") [ st ]

let make_dot_product () =
  let p0 = U.param ~slot:0 ~dtype:global_fptr ~shape:(U.const_int (-1)) () in
  let p1 = U.param ~slot:1 ~dtype:global_fptr ~shape:(U.const_int (-1)) () in
  let p2 = U.param ~slot:2 ~dtype:global_fptr ~shape:(U.const_int (-1)) () in
  let r0 = U.range ~size:(U.const_int 128) ~axis:0 ~kind:Axis_type.Reduce () in
  let ld_a = U.load ~src:(U.index ~ptr:p0 ~idxs:[r0] ()) () in
  let ld_b = U.load ~src:(U.index ~ptr:p1 ~idxs:[r0] ()) () in
  let mul = U.alu_binary ~op:Ops.Mul ~lhs:ld_a ~rhs:ld_b in
  let red = U.reduce ~op:Ops.Add ~src:mul ~ranges:[ r0 ] ~dtype:Dtype.float32 in
  let st =
    U.store ~dst:(U.index ~ptr:p2 ~idxs:[(idx 0)] ()) ~value:red ()
  in
  U.sink ~kernel_info:(kernel_info ~axis_types:[ Axis_type.Reduce ] "dot_product") [ st ]

let make_matmul_small () =
  let m, n, k = (4, 4, 4) in
  let pA = U.param ~slot:0 ~dtype:global_fptr ~shape:(U.const_int (-1)) () in
  let pB = U.param ~slot:1 ~dtype:global_fptr ~shape:(U.const_int (-1)) () in
  let pC = U.param ~slot:2 ~dtype:global_fptr ~shape:(U.const_int (-1)) () in
  let ri = U.range ~size:(U.const_int m) ~axis:0 ~kind:Axis_type.Global () in
  let rj = U.range ~size:(U.const_int n) ~axis:1 ~kind:Axis_type.Global () in
  let rk = U.range ~size:(U.const_int k) ~axis:2 ~kind:Axis_type.Reduce () in
  let open U.O in
  let a_idx = ri * int_ k + rk in
  let b_idx = rk * int_ n + rj in
  let c_idx = ri * int_ n + rj in
  let ld_a = U.load ~src:(U.index ~ptr:pA ~idxs:[a_idx] ()) () in
  let ld_b = U.load ~src:(U.index ~ptr:pB ~idxs:[b_idx] ()) () in
  let mul = U.alu_binary ~op:Ops.Mul ~lhs:ld_a ~rhs:ld_b in
  let red = U.reduce ~op:Ops.Add ~src:mul ~ranges:[ rk ] ~dtype:Dtype.float32 in
  let st =
    U.store ~dst:(U.index ~ptr:pC ~idxs:[c_idx] ()) ~value:red ()
  in
  let e = U.end_ ~value:st ~ranges:[ ri; rj ] in
  U.sink
    ~kernel_info:
      (kernel_info
         ~axis_types:[ Axis_type.Global; Axis_type.Global; Axis_type.Reduce ]
         "matmul_small")
    [ e ]

let make_elementwise_2d () =
  let rows, cols = (8, 16) in
  let p0 = U.param ~slot:0 ~dtype:global_fptr ~shape:(U.const_int (-1)) () in
  let p1 = U.param ~slot:1 ~dtype:global_fptr ~shape:(U.const_int (-1)) () in
  let p2 = U.param ~slot:2 ~dtype:global_fptr ~shape:(U.const_int (-1)) () in
  let ri = U.range ~size:(U.const_int rows) ~axis:0 ~kind:Axis_type.Global () in
  let rj = U.range ~size:(U.const_int cols) ~axis:1 ~kind:Axis_type.Global () in
  let open U.O in
  let flat = ri * int_ cols + rj in
  let ld_a = U.load ~src:(U.index ~ptr:p0 ~idxs:[flat] ()) () in
  let ld_b = U.load ~src:(U.index ~ptr:p1 ~idxs:[flat] ()) () in
  let add = U.alu_binary ~op:Ops.Add ~lhs:ld_a ~rhs:ld_b in
  let st =
    U.store ~dst:(U.index ~ptr:p2 ~idxs:[flat] ()) ~value:add ()
  in
  let e = U.end_ ~value:st ~ranges:[ ri; rj ] in
  U.sink
    ~kernel_info:
      (kernel_info
         ~axis_types:[ Axis_type.Global; Axis_type.Global ]
         "elementwise_2d")
    [ e ]

let make_reduce_rows () =
  let rows, cols = (8, 32) in
  let p0 = U.param ~slot:0 ~dtype:global_fptr ~shape:(U.const_int (-1)) () in
  let p1 = U.param ~slot:1 ~dtype:global_fptr ~shape:(U.const_int (-1)) () in
  let ri = U.range ~size:(U.const_int rows) ~axis:0 ~kind:Axis_type.Global () in
  let rj = U.range ~size:(U.const_int cols) ~axis:1 ~kind:Axis_type.Reduce () in
  let open U.O in
  let flat = ri * int_ cols + rj in
  let ld = U.load ~src:(U.index ~ptr:p0 ~idxs:[flat] ()) () in
  let red = U.reduce ~op:Ops.Add ~src:ld ~ranges:[ rj ] ~dtype:Dtype.float32 in
  let st =
    U.store ~dst:(U.index ~ptr:p1 ~idxs:[ri] ()) ~value:red ()
  in
  let e = U.end_ ~value:st ~ranges:[ ri ] in
  U.sink
    ~kernel_info:
      (kernel_info
         ~axis_types:[ Axis_type.Global; Axis_type.Reduce ]
         "reduce_rows")
    [ e ]

let make_no_optimize () =
  let p0 = U.param ~slot:0 ~dtype:global_fptr ~shape:(U.const_int (-1)) () in
  let p1 = U.param ~slot:1 ~dtype:global_fptr ~shape:(U.const_int (-1)) () in
  let p2 = U.param ~slot:2 ~dtype:global_fptr ~shape:(U.const_int (-1)) () in
  let r0 = U.range ~size:(U.const_int 256) ~axis:0 ~kind:Axis_type.Global () in
  let ld_a = U.load ~src:(U.index ~ptr:p0 ~idxs:[r0] ()) () in
  let ld_b = U.load ~src:(U.index ~ptr:p1 ~idxs:[r0] ()) () in
  let add = U.alu_binary ~op:Ops.Add ~lhs:ld_a ~rhs:ld_b in
  let st = U.store ~dst:(U.index ~ptr:p2 ~idxs:[r0] ()) ~value:add () in
  let e = U.end_ ~value:st ~ranges:[ r0 ] in
  U.sink ~kernel_info:(kernel_info ~axis_types:[ Axis_type.Global ] "no_optimize") [ e ]

let make_multi_output () =
  let p0 = U.param ~slot:0 ~dtype:global_fptr ~shape:(U.const_int (-1)) () in
  let p1 = U.param ~slot:1 ~dtype:global_fptr ~shape:(U.const_int (-1)) () in
  let p2 = U.param ~slot:2 ~dtype:global_fptr ~shape:(U.const_int (-1)) () in
  let r0 = U.range ~size:(U.const_int 256) ~axis:0 ~kind:Axis_type.Global () in
  let ld_a = U.load ~src:(U.index ~ptr:p0 ~idxs:[r0] ()) () in
  let one = U.const (Const.float Dtype.float32 1.0) in
  let two = U.const (Const.float Dtype.float32 2.0) in
  let st1 =
    U.store
      ~dst:(U.index ~ptr:p1 ~idxs:[r0] ())
      ~value:(U.alu_binary ~op:Ops.Add ~lhs:ld_a ~rhs:one) ()
  in
  let e1 = U.end_ ~value:st1 ~ranges:[ r0 ] in
  let st2 =
    U.store
      ~dst:(U.index ~ptr:p2 ~idxs:[r0] ())
      ~value:(U.alu_binary ~op:Ops.Mul ~lhs:ld_a ~rhs:two) ()
  in
  let e2 = U.end_ ~value:st2 ~ranges:[ r0 ] in
  U.sink ~kernel_info:(kernel_info ~axis_types:[ Axis_type.Global ] "multi_output") [ e1; e2 ]

let make_gated_store () =
  let p0 = U.param ~slot:0 ~dtype:global_fptr ~shape:(U.const_int (-1)) () in
  let p1 = U.param ~slot:1 ~dtype:global_fptr ~shape:(U.const_int (-1)) () in
  let p2 = U.param ~slot:2 ~dtype:global_fptr ~shape:(U.const_int (-1)) () in
  let r0 = U.range ~size:(U.const_int 256) ~axis:0 ~kind:Axis_type.Global () in
  let ld_a = U.load ~src:(U.index ~ptr:p0 ~idxs:[r0] ()) () in
  let ld_b = U.load ~src:(U.index ~ptr:p1 ~idxs:[r0] ()) () in
  let add = U.alu_binary ~op:Ops.Add ~lhs:ld_a ~rhs:ld_b in
  let gate = U.alu_binary ~op:Ops.Cmplt ~lhs:r0 ~rhs:(idx 200) in
  let value = U.O.where gate add (U.invalid ~dtype:Dtype.float32 ()) in
  let st =
    U.store
      ~dst:(U.index ~ptr:p2 ~idxs:[r0] ())
      ~value ()
  in
  let e = U.end_ ~value:st ~ranges:[ r0 ] in
  U.sink ~kernel_info:(kernel_info ~axis_types:[ Axis_type.Global ] "gated_store") [ e ]

let make_elementwise_where () =
  let p0 = U.param ~slot:0 ~dtype:global_fptr ~shape:(U.const_int (-1)) () in
  let p1 = U.param ~slot:1 ~dtype:global_fptr ~shape:(U.const_int (-1)) () in
  let r0 = U.range ~size:(U.const_int 256) ~axis:0 ~kind:Axis_type.Global () in
  let ld = U.load ~src:(U.index ~ptr:p0 ~idxs:[r0] ()) () in
  let zero = U.const (Const.float Dtype.float32 0.0) in
  let cond = U.alu_binary ~op:Ops.Cmplt ~lhs:zero ~rhs:ld in
  let w = U.alu_ternary ~op:Ops.Where ~a:cond ~b:ld ~c:zero in
  let st = U.store ~dst:(U.index ~ptr:p1 ~idxs:[r0] ()) ~value:w () in
  let e = U.end_ ~value:st ~ranges:[ r0 ] in
  U.sink ~kernel_info:(kernel_info ~axis_types:[ Axis_type.Global ] "elementwise_where") [ e ]

let make_elementwise_cast_f16 () =
  (* c[i] = (float32)a_f16[i] + b[i]. Param order: 0=f16, 1=f32, 2=out_f32.
     Build the Add as cast(ld_f16) + ld_f32 to match the reference load ordering. *)
  let p0 = U.param ~slot:0 ~dtype:Dtype.float16 ~shape:(U.const_int (-1)) () in
  let p1 = U.param ~slot:1 ~dtype:global_fptr ~shape:(U.const_int (-1)) () in
  let p2 = U.param ~slot:2 ~dtype:global_fptr ~shape:(U.const_int (-1)) () in
  let r0 = U.range ~size:(U.const_int 256) ~axis:0 ~kind:Axis_type.Global () in
  let ld_a = U.load ~src:(U.index ~ptr:p0 ~idxs:[r0] ()) () in
  let cast_a = U.cast ~src:ld_a ~dtype:Dtype.float32 in
  let ld_b = U.load ~src:(U.index ~ptr:p1 ~idxs:[r0] ()) () in
  let add = U.alu_binary ~op:Ops.Add ~lhs:cast_a ~rhs:ld_b in
  let st = U.store ~dst:(U.index ~ptr:p2 ~idxs:[r0] ()) ~value:add () in
  let e = U.end_ ~value:st ~ranges:[ r0 ] in
  U.sink ~kernel_info:(kernel_info ~axis_types:[ Axis_type.Global ] "elementwise_cast_f16") [ e ]

let make_elementwise_sqrt () =
  let p0 = U.param ~slot:0 ~dtype:global_fptr ~shape:(U.const_int (-1)) () in
  let p1 = U.param ~slot:1 ~dtype:global_fptr ~shape:(U.const_int (-1)) () in
  let r0 = U.range ~size:(U.const_int 256) ~axis:0 ~kind:Axis_type.Global () in
  let ld = U.load ~src:(U.index ~ptr:p0 ~idxs:[r0] ()) () in
  let sq = U.alu_unary ~op:Ops.Sqrt ~src:ld in
  let st = U.store ~dst:(U.index ~ptr:p1 ~idxs:[r0] ()) ~value:sq () in
  let e = U.end_ ~value:st ~ranges:[ r0 ] in
  U.sink ~kernel_info:(kernel_info ~axis_types:[ Axis_type.Global ] "elementwise_sqrt") [ e ]

let make_parallel_reduce () =
  let p0 = U.param ~slot:0 ~dtype:global_fptr ~shape:(U.const_int (-1)) () in
  let p1 = U.param ~slot:1 ~dtype:global_fptr ~shape:(U.const_int (-1)) () in
  let p2 = U.param ~slot:2 ~dtype:global_fptr ~shape:(U.const_int (-1)) () in
  let r0 = U.range ~size:(U.const_int 128) ~axis:0 ~kind:Axis_type.Reduce () in
  let ld = U.load ~src:(U.index ~ptr:p0 ~idxs:[r0] ()) () in
  let red1 = U.reduce ~op:Ops.Add ~src:ld ~ranges:[ r0 ] ~dtype:Dtype.float32 in
  let sq = U.alu_binary ~op:Ops.Mul ~lhs:ld ~rhs:ld in
  let red2 = U.reduce ~op:Ops.Add ~src:sq ~ranges:[ r0 ] ~dtype:Dtype.float32 in
  let c0 = idx 0 in
  let st1 =
    U.store ~dst:(U.index ~ptr:p1 ~idxs:[c0] ()) ~value:red1 ()
  in
  let st2 =
    U.store ~dst:(U.index ~ptr:p2 ~idxs:[c0] ()) ~value:red2 ()
  in
  U.sink ~kernel_info:(kernel_info ~axis_types:[ Axis_type.Reduce ] "parallel_reduce") [ st1; st2 ]

let make_elementwise_int32 () =
  let p0 = U.param ~slot:0 ~dtype:Dtype.int32 ~shape:(U.const_int (-1)) () in
  let p1 = U.param ~slot:1 ~dtype:Dtype.int32 ~shape:(U.const_int (-1)) () in
  let p2 = U.param ~slot:2 ~dtype:Dtype.int32 ~shape:(U.const_int (-1)) () in
  let r0 = U.range ~size:(U.const_int 256) ~axis:0 ~kind:Axis_type.Global () in
  let ld_a = U.load ~src:(U.index ~ptr:p0 ~idxs:[r0] ()) () in
  let ld_b = U.load ~src:(U.index ~ptr:p1 ~idxs:[r0] ()) () in
  let add = U.alu_binary ~op:Ops.Add ~lhs:ld_a ~rhs:ld_b in
  let st = U.store ~dst:(U.index ~ptr:p2 ~idxs:[r0] ()) ~value:add () in
  let e = U.end_ ~value:st ~ranges:[ r0 ] in
  U.sink ~kernel_info:(kernel_info ~axis_types:[ Axis_type.Global ] "elementwise_int32") [ e ]

let make_llama_rmsnorm backend =
  let p0 = U.param ~slot:0 ~dtype:global_fptr ~shape:(U.const_int 2) () in
  let p1 = U.param ~slot:1 ~dtype:global_fptr ~shape:(U.const_int 16) () in
  let ri = U.range ~size:(U.const_int 2) ~axis:1 ~kind:Axis_type.Loop () in
  let rr = U.range ~size:(U.const_int 8) ~axis:0 ~kind:Axis_type.Reduce () in
  let open U.O in
  let in_idx = (ri * int_ 8) + rr in
  let ld = U.load ~src:(U.index ~ptr:p1 ~idxs:[ in_idx ] ()) () in
  let sq = U.alu_binary ~op:Ops.Mul ~lhs:ld ~rhs:ld in
  let sum = U.reduce ~op:Ops.Add ~src:sq ~ranges:[ rr ] ~dtype:Dtype.float32 in
  let mean =
    U.alu_binary ~op:Ops.Mul ~lhs:sum
      ~rhs:(U.const (Const.float Dtype.float32 0.125))
  in
  let eps =
    U.alu_binary ~op:Ops.Add ~lhs:mean
      ~rhs:(U.const (Const.float Dtype.float32 0.00001))
  in
  let sqrt = U.alu_unary ~op:Ops.Sqrt ~src:eps in
  let rsqrt = U.alu_unary ~op:Ops.Reciprocal ~src:sqrt in
  let st =
    U.store ~dst:(U.index ~ptr:p0 ~idxs:[ ri ] ()) ~value:rsqrt ()
  in
  let e = U.end_ ~value:st ~ranges:[ ri ] in
  let name, opts_to_apply =
    match backend with
    | "clang" -> ("r_2_8n1", [ U.Opt.Unroll { axis = 0; amount = 0 } ])
    | "cuda" ->
        ( "r_2_8n2",
          [
            U.Opt.Unroll { axis = 0; amount = 0 };
            U.Opt.Local { axis = 0; amount = 2 };
          ] )
    | "metal" ->
        ( "r_2_8n3",
          [
            U.Opt.Unroll { axis = 0; amount = 0 };
            U.Opt.Local { axis = 0; amount = 2 };
          ] )
    | "opencl" ->
        ( "r_2_8n4",
          [
            U.Opt.Unroll { axis = 0; amount = 0 };
            U.Opt.Local { axis = 0; amount = 2 };
          ] )
    | backend -> invalid_arg (Printf.sprintf "unknown backend %S" backend)
  in
  U.sink
    ~kernel_info:
      {
        U.name = name;
        axis_types = [];
        dont_use_locals = false;
        applied_opts = [];
        opts_to_apply = Some opts_to_apply;
        estimates = None;
        beam = 0;
      }
    [ e ]

let model_kernel_info name opts_to_apply =
  {
    U.name;
    axis_types = [];
    dont_use_locals = false;
    applied_opts = [];
    opts_to_apply = Some opts_to_apply;
    estimates = None;
    beam = 0;
  }

let make_llama_embedding backend =
  let p0 = U.param ~slot:0 ~dtype:global_fptr ~shape:(U.const_int 16) () in
  let p1 = U.param ~slot:1 ~dtype:Dtype.int32 ~shape:(U.const_int 2) () in
  let p2 = U.param ~slot:2 ~dtype:global_fptr ~shape:(U.const_int 256) () in
  let ri = U.range ~size:(U.const_int 2) ~axis:1 ~kind:Axis_type.Loop () in
  let rj = U.range ~size:(U.const_int 8) ~axis:2 ~kind:Axis_type.Loop () in
  let rv = U.range ~size:(U.const_int 32) ~axis:0 ~kind:Axis_type.Reduce () in
  let open U.O in
  let out_idx = (ri * int_ 8) + rj in
  let token = U.load ~src:(U.index ~ptr:p1 ~idxs:[ ri ] ()) () in
  let vocab = U.cast ~src:rv ~dtype:Dtype.int32 in
  let gate = U.alu_binary ~op:Ops.Cmpne ~lhs:vocab ~rhs:token in
  let emb_idx = (rv * int_ 8) + rj in
  let emb = U.load ~src:(U.index ~ptr:p2 ~idxs:[ emb_idx ] ()) () in
  let zero = U.const (Const.float Dtype.float32 0.0) in
  let selected = U.alu_ternary ~op:Ops.Where ~a:gate ~b:zero ~c:emb in
  let value =
    U.reduce ~op:Ops.Add ~src:selected ~ranges:[ rv ] ~dtype:Dtype.float32
  in
  let st =
    U.store ~dst:(U.index ~ptr:p0 ~idxs:[ out_idx ] ()) ~value ()
  in
  let e = U.end_ ~value:st ~ranges:[ ri; rj ] in
  let name, opts_to_apply =
    match backend with
    | "clang" -> ("E_8_2n1", [ U.Opt.Upcast { axis = 0; amount = 0 } ])
    | "cuda" ->
        ( "E_8_2n2",
          [
            U.Opt.Upcast { axis = 0; amount = 0 };
            U.Opt.Local { axis = 0; amount = 8 };
          ] )
    | "metal" ->
        ( "E_8_2n3",
          [
            U.Opt.Upcast { axis = 0; amount = 0 };
            U.Opt.Local { axis = 0; amount = 8 };
          ] )
    | "opencl" ->
        ( "E_8_2n4",
          [
            U.Opt.Upcast { axis = 0; amount = 0 };
            U.Opt.Local { axis = 0; amount = 8 };
          ] )
    | backend -> invalid_arg (Printf.sprintf "unknown backend %S" backend)
  in
  U.sink ~kernel_info:(model_kernel_info name opts_to_apply) [ e ]

let make_llama_ffn_gate backend =
  let p0 = U.param ~slot:0 ~dtype:global_fptr ~shape:(U.const_int 16) () in
  let p1 = U.param ~slot:1 ~dtype:global_fptr ~shape:(U.const_int 16) () in
  let p2 = U.param ~slot:2 ~dtype:global_fptr ~shape:(U.const_int 2) () in
  let p3 = U.param ~slot:3 ~dtype:global_fptr ~shape:(U.const_int 8) () in
  let p4 = U.param ~slot:4 ~dtype:global_fptr ~shape:(U.const_int 64) () in
  let r3 = U.range ~size:(U.const_int 2) ~axis:3 ~kind:Axis_type.Loop () in
  let r4 = U.range ~size:(U.const_int 2) ~axis:4 ~kind:Axis_type.Loop () in
  let r2 = U.range ~size:(U.const_int 2) ~axis:2 ~kind:Axis_type.Loop () in
  let r1 = U.range ~size:(U.const_int 2) ~axis:1 ~kind:Axis_type.Loop () in
  let rr = U.range ~size:(U.const_int 8) ~axis:0 ~kind:Axis_type.Reduce () in
  let open U.O in
  let lane = (r3 * int_ 2) + r4 + (r2 * int_ 4) in
  let out_idx = lane + (r1 * int_ 8) in
  let input = U.load ~src:(U.index ~ptr:p1 ~idxs:[ (r1 * int_ 8) + rr ] ()) () in
  let norm = U.load ~src:(U.index ~ptr:p2 ~idxs:[ r1 ] ()) () in
  let weight = U.load ~src:(U.index ~ptr:p3 ~idxs:[ rr ] ()) () in
  let matrix =
    U.load ~src:(U.index ~ptr:p4 ~idxs:[ (lane * int_ 8) + rr ] ()) ()
  in
  let value =
    U.alu_binary ~op:Ops.Mul
      ~lhs:
        (U.alu_binary ~op:Ops.Mul
           ~lhs:(U.alu_binary ~op:Ops.Mul ~lhs:input ~rhs:norm)
           ~rhs:weight)
      ~rhs:matrix
  in
  let red = U.reduce ~op:Ops.Add ~src:value ~ranges:[ rr ] ~dtype:Dtype.float32 in
  let st =
    U.store ~dst:(U.index ~ptr:p0 ~idxs:[ out_idx ] ()) ~value:red ()
  in
  let e = U.end_ ~value:st ~ranges:[ r1; r2; r3; r4 ] in
  let name, opts_to_apply =
    match backend with
    | "clang" -> ("r_2_8_8n1", [ U.Opt.Unroll { axis = 0; amount = 0 } ])
    | "cuda" ->
        ( "r_2_8_8n2",
          [
            U.Opt.Unroll { axis = 0; amount = 0 };
            U.Opt.Local { axis = 0; amount = 2 };
            U.Opt.Local { axis = 0; amount = 8 };
          ] )
    | "metal" ->
        ( "r_2_8_8n3",
          [
            U.Opt.Unroll { axis = 0; amount = 0 };
            U.Opt.Local { axis = 0; amount = 2 };
            U.Opt.Local { axis = 0; amount = 8 };
          ] )
    | "opencl" ->
        ( "r_2_8_8n4",
          [
            U.Opt.Unroll { axis = 0; amount = 0 };
            U.Opt.Local { axis = 0; amount = 2 };
            U.Opt.Local { axis = 0; amount = 8 };
          ] )
    | backend -> invalid_arg (Printf.sprintf "unknown backend %S" backend)
  in
  U.sink ~kernel_info:(model_kernel_info name opts_to_apply) [ e ]

let make_llama_vector_scale backend =
  let p0 = U.param ~slot:0 ~dtype:global_fptr ~shape:(U.const_int 16) () in
  let p1 = U.param ~slot:1 ~dtype:global_fptr ~shape:(U.const_int 16) () in
  let p2 = U.param ~slot:2 ~dtype:global_fptr ~shape:(U.const_int 2) () in
  let p3 = U.param ~slot:3 ~dtype:global_fptr ~shape:(U.const_int 8) () in
  let ri = U.range ~size:(U.const_int 2) ~axis:0 ~kind:Axis_type.Loop () in
  let rj = U.range ~size:(U.const_int 8) ~axis:1 ~kind:Axis_type.Loop () in
  let open U.O in
  let flat = (ri * int_ 8) + rj in
  let lhs = U.load ~src:(U.index ~ptr:p1 ~idxs:[ flat ] ()) () in
  let scale = U.load ~src:(U.index ~ptr:p2 ~idxs:[ ri ] ()) () in
  let weight = U.load ~src:(U.index ~ptr:p3 ~idxs:[ rj ] ()) () in
  let value =
    U.alu_binary ~op:Ops.Mul
      ~lhs:(U.alu_binary ~op:Ops.Mul ~lhs ~rhs:scale)
      ~rhs:weight
  in
  let st =
    U.store ~dst:(U.index ~ptr:p0 ~idxs:[ flat ] ()) ~value ()
  in
  let e = U.end_ ~value:st ~ranges:[ ri; rj ] in
  let name, opts_to_apply =
    match backend with
    | "clang" -> ("E_2_2_4n1", [ U.Opt.Upcast { axis = 1; amount = 4 } ])
    | "cuda" ->
        ( "E_2_2_4n2",
          [
            U.Opt.Upcast { axis = 1; amount = 4 };
            U.Opt.Local { axis = 0; amount = 2 };
            U.Opt.Local { axis = 0; amount = 2 };
          ] )
    | "metal" ->
        ( "E_2_2_4n3",
          [
            U.Opt.Upcast { axis = 1; amount = 4 };
            U.Opt.Local { axis = 0; amount = 2 };
            U.Opt.Local { axis = 0; amount = 2 };
          ] )
    | "opencl" ->
        ( "E_2_2_4n4",
          [
            U.Opt.Upcast { axis = 1; amount = 4 };
            U.Opt.Local { axis = 0; amount = 2 };
            U.Opt.Local { axis = 0; amount = 2 };
          ] )
    | backend -> invalid_arg (Printf.sprintf "unknown backend %S" backend)
  in
  U.sink ~kernel_info:(model_kernel_info name opts_to_apply) [ e ]

let make_llama_output_projection backend =
  let p0 = U.param ~slot:0 ~dtype:global_fptr ~shape:(U.const_int 64) () in
  let p1 = U.param ~slot:1 ~dtype:global_fptr ~shape:(U.const_int 16) () in
  let p2 = U.param ~slot:2 ~dtype:global_fptr ~shape:(U.const_int 256) () in
  let ri = U.range ~size:(U.const_int 2) ~axis:1 ~kind:Axis_type.Loop () in
  let rj = U.range ~size:(U.const_int 32) ~axis:2 ~kind:Axis_type.Loop () in
  let rr = U.range ~size:(U.const_int 8) ~axis:0 ~kind:Axis_type.Reduce () in
  let open U.O in
  let input = U.load ~src:(U.index ~ptr:p1 ~idxs:[ (ri * int_ 8) + rr ] ()) () in
  let weight = U.load ~src:(U.index ~ptr:p2 ~idxs:[ (rj * int_ 8) + rr ] ()) () in
  let mul = U.alu_binary ~op:Ops.Mul ~lhs:input ~rhs:weight in
  let red = U.reduce ~op:Ops.Add ~src:mul ~ranges:[ rr ] ~dtype:Dtype.float32 in
  let st =
    U.store
      ~dst:(U.index ~ptr:p0 ~idxs:[ (ri * int_ 32) + rj ] ())
      ~value:red ()
  in
  let e = U.end_ ~value:st ~ranges:[ ri; rj ] in
  let name, opts_to_apply =
    match backend with
    | "clang" -> ("r_2_32_8n1", [ U.Opt.Unroll { axis = 0; amount = 0 } ])
    | "cuda" ->
        ( "r_2_2_16_8",
          [
            U.Opt.Unroll { axis = 0; amount = 0 };
            U.Opt.Local { axis = 0; amount = 2 };
            U.Opt.Local { axis = 0; amount = 16 };
          ] )
    | "metal" ->
        ( "r_2_2_16_8n1",
          [
            U.Opt.Unroll { axis = 0; amount = 0 };
            U.Opt.Local { axis = 0; amount = 2 };
            U.Opt.Local { axis = 0; amount = 16 };
          ] )
    | "opencl" ->
        ( "r_2_2_16_8n2",
          [
            U.Opt.Unroll { axis = 0; amount = 0 };
            U.Opt.Local { axis = 0; amount = 2 };
            U.Opt.Local { axis = 0; amount = 16 };
          ] )
    | backend -> invalid_arg (Printf.sprintf "unknown backend %S" backend)
  in
  U.sink ~kernel_info:(model_kernel_info name opts_to_apply) [ e ]

(* ── Test cases ── *)

type test_case = {
  name : string;
  kernel : string -> U.t;
  backends : (string * Renderer.t) list;
  optimize : bool;
}

let fixed kernel _backend = kernel

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
    { name = "elementwise_add"; kernel = fixed (make_elementwise_add ());
      backends = all_renderers; optimize = true };
    { name = "sum_reduce"; kernel = fixed (make_sum_reduce ());
      backends = all_renderers; optimize = true };
    { name = "max_reduce"; kernel = fixed (make_max_reduce ());
      backends = all_renderers; optimize = true };
    { name = "dot_product"; kernel = fixed (make_dot_product ());
      backends = all_renderers; optimize = true };
    { name = "matmul_small"; kernel = fixed (make_matmul_small ());
      backends = gpu_renderers; optimize = true };
    { name = "elementwise_2d"; kernel = fixed (make_elementwise_2d ());
      backends = gpu_renderers; optimize = true };
    { name = "reduce_rows"; kernel = fixed (make_reduce_rows ());
      backends = all_renderers; optimize = true };
    { name = "no_optimize"; kernel = fixed (make_no_optimize ());
      backends = all_renderers; optimize = false };
    { name = "multi_output"; kernel = fixed (make_multi_output ());
      backends = all_renderers; optimize = true };
    { name = "gated_store"; kernel = fixed (make_gated_store ());
      backends = all_renderers; optimize = true };
    { name = "elementwise_where"; kernel = fixed (make_elementwise_where ());
      backends = all_renderers; optimize = true };
    { name = "elementwise_cast_f16"; kernel = fixed (make_elementwise_cast_f16 ());
      backends = all_renderers; optimize = true };
    { name = "elementwise_sqrt"; kernel = fixed (make_elementwise_sqrt ());
      backends = all_renderers; optimize = true };
    { name = "parallel_reduce"; kernel = fixed (make_parallel_reduce ());
      backends = all_renderers; optimize = true };
    { name = "elementwise_int32"; kernel = fixed (make_elementwise_int32 ());
      backends = all_renderers; optimize = true };
    { name = "llama_embedding"; kernel = make_llama_embedding;
      backends = all_renderers; optimize = true };
    { name = "llama_rmsnorm"; kernel = make_llama_rmsnorm;
      backends = all_renderers; optimize = true };
    { name = "llama_ffn_gate"; kernel = make_llama_ffn_gate;
      backends = all_renderers; optimize = true };
    { name = "llama_vector_scale"; kernel = make_llama_vector_scale;
      backends = all_renderers; optimize = true };
    { name = "llama_output_projection"; kernel = make_llama_output_projection;
      backends = all_renderers; optimize = true };
  ]

let () =
  let dir = Sys.argv.(1) in
  let test_cases =
    match Sys.getenv_opt "ONLY" with
    | None -> test_cases
    | Some only -> List.filter (fun { name; _ } -> String.equal name only) test_cases
  in
  List.iter
    (fun { name; kernel; backends; optimize } ->
      List.iter
        (fun (backend_name, renderer) ->
          let kernel = kernel backend_name in
          let snap = Printf.sprintf "%s_%s" backend_name name in
          match pipeline_to_source ~optimize renderer kernel with
          | out ->
              let filename = Filename.concat dir (snap ^ ".actual") in
              let oc = open_out filename in
              output_string oc out;
              output_char oc '\n';
              close_out oc
          | exception exn ->
              Printf.eprintf "FAIL %s: %s\n%!" snap (Printexc.to_string exn);
              raise exn)
        backends)
    test_cases
