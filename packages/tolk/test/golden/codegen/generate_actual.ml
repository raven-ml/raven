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
open Tolk_ir
module K = Kernel

let global_fptr = Dtype.ptr_of Dtype.float32 ~addrspace:Global ~size:(-1)
let idx n = K.const (Const.int Dtype.index n)

let kernel_info ?(axis_kinds = []) name =
  {
    K.name;
    axis_kinds;
    dont_use_locals = false;
    applied_opts = [];
    opts_to_apply = Some [];
    estimates = None;
  }

(* Extract the kernel name from a pipeline-processed Sink. *)
let name_of_sink sink =
  match K.view sink with
  | K.Sink { kernel_info = Some ki; _ } -> ki.name
  | _ -> "kernel"

(* Full pipeline chain: Kernel.t -> source string. *)
let pipeline_to_source ?(optimize = true) ren sink =
  let processed = Pipeline.full_rewrite_to_sink ~optimize ren sink in
  let name = name_of_sink processed in
  let program = Linearizer.linearize processed in
  String.trim (Renderer.render ren ~name program)

(* ── Kernel AST builders ── *)

let make_elementwise_add () =
  let p0 = K.param ~idx:0 ~dtype:global_fptr in
  let p1 = K.param ~idx:1 ~dtype:global_fptr in
  let p2 = K.param ~idx:2 ~dtype:global_fptr in
  let r0 = K.range ~size:(idx 256) ~axis:0 ~kind:Axis_kind.Global () in
  let ld_a = K.load ~src:(K.index ~ptr:p0 ~idxs:[ r0 ] ()) () in
  let ld_b = K.load ~src:(K.index ~ptr:p1 ~idxs:[ r0 ] ()) () in
  let add = K.binary ~op:`Add ~lhs:ld_a ~rhs:ld_b in
  let st = K.store ~dst:(K.index ~ptr:p2 ~idxs:[ r0 ] ()) ~value:add ~ranges:[] in
  let e = K.end_ ~value:st ~ranges:[ r0 ] () in
  K.sink ~kernel_info:(kernel_info ~axis_kinds:[ Axis_kind.Global ] "elementwise_add") [ e ]

let make_sum_reduce () =
  let p0 = K.param ~idx:0 ~dtype:global_fptr in
  let p1 = K.param ~idx:1 ~dtype:global_fptr in
  let r0 = K.range ~size:(idx 256) ~axis:0 ~kind:Axis_kind.Reduce () in
  let ld = K.load ~src:(K.index ~ptr:p0 ~idxs:[ r0 ] ()) () in
  let red = K.reduce ~op:`Add ~src:ld ~ranges:[ r0 ] ~dtype:Dtype.float32 in
  let st =
    K.store ~dst:(K.index ~ptr:p1 ~idxs:[ idx 0 ] ()) ~value:red ~ranges:[]
  in
  K.sink ~kernel_info:(kernel_info ~axis_kinds:[ Axis_kind.Reduce ] "sum_reduce") [ st ]

(* max_reduce is excluded: requires Max->Where decomposition (pipeline Steps
   18-21) which is not yet ported. The expected file is generated for reference
   but has no matching .actual until decompositions land. *)

let make_dot_product () =
  let p0 = K.param ~idx:0 ~dtype:global_fptr in
  let p1 = K.param ~idx:1 ~dtype:global_fptr in
  let p2 = K.param ~idx:2 ~dtype:global_fptr in
  let r0 = K.range ~size:(idx 128) ~axis:0 ~kind:Axis_kind.Reduce () in
  let ld_a = K.load ~src:(K.index ~ptr:p0 ~idxs:[ r0 ] ()) () in
  let ld_b = K.load ~src:(K.index ~ptr:p1 ~idxs:[ r0 ] ()) () in
  let mul = K.binary ~op:`Mul ~lhs:ld_a ~rhs:ld_b in
  let red = K.reduce ~op:`Add ~src:mul ~ranges:[ r0 ] ~dtype:Dtype.float32 in
  let st =
    K.store ~dst:(K.index ~ptr:p2 ~idxs:[ idx 0 ] ()) ~value:red ~ranges:[]
  in
  K.sink ~kernel_info:(kernel_info ~axis_kinds:[ Axis_kind.Reduce ] "dot_product") [ st ]

let make_matmul_small () =
  let m, n, k = (4, 4, 4) in
  let pA = K.param ~idx:0 ~dtype:global_fptr in
  let pB = K.param ~idx:1 ~dtype:global_fptr in
  let pC = K.param ~idx:2 ~dtype:global_fptr in
  let ri = K.range ~size:(idx m) ~axis:0 ~kind:Axis_kind.Global () in
  let rj = K.range ~size:(idx n) ~axis:1 ~kind:Axis_kind.Global () in
  let rk = K.range ~size:(idx k) ~axis:2 ~kind:Axis_kind.Reduce () in
  let open K.O in
  let a_idx = ri * int_ k + rk in
  let b_idx = rk * int_ n + rj in
  let c_idx = ri * int_ n + rj in
  let ld_a = K.load ~src:(K.index ~ptr:pA ~idxs:[ a_idx ] ()) () in
  let ld_b = K.load ~src:(K.index ~ptr:pB ~idxs:[ b_idx ] ()) () in
  let mul = K.binary ~op:`Mul ~lhs:ld_a ~rhs:ld_b in
  let red = K.reduce ~op:`Add ~src:mul ~ranges:[ rk ] ~dtype:Dtype.float32 in
  let st =
    K.store ~dst:(K.index ~ptr:pC ~idxs:[ c_idx ] ()) ~value:red ~ranges:[]
  in
  let e = K.end_ ~value:st ~ranges:[ ri; rj ] () in
  K.sink
    ~kernel_info:
      (kernel_info
         ~axis_kinds:[ Axis_kind.Global; Axis_kind.Global; Axis_kind.Reduce ]
         "matmul_small")
    [ e ]

let make_elementwise_2d () =
  let rows, cols = (8, 16) in
  let p0 = K.param ~idx:0 ~dtype:global_fptr in
  let p1 = K.param ~idx:1 ~dtype:global_fptr in
  let p2 = K.param ~idx:2 ~dtype:global_fptr in
  let ri = K.range ~size:(idx rows) ~axis:0 ~kind:Axis_kind.Global () in
  let rj = K.range ~size:(idx cols) ~axis:1 ~kind:Axis_kind.Global () in
  let open K.O in
  let flat = ri * int_ cols + rj in
  let ld_a = K.load ~src:(K.index ~ptr:p0 ~idxs:[ flat ] ()) () in
  let ld_b = K.load ~src:(K.index ~ptr:p1 ~idxs:[ flat ] ()) () in
  let add = K.binary ~op:`Add ~lhs:ld_a ~rhs:ld_b in
  let st =
    K.store ~dst:(K.index ~ptr:p2 ~idxs:[ flat ] ()) ~value:add ~ranges:[]
  in
  let e = K.end_ ~value:st ~ranges:[ ri; rj ] () in
  K.sink
    ~kernel_info:
      (kernel_info
         ~axis_kinds:[ Axis_kind.Global; Axis_kind.Global ]
         "elementwise_2d")
    [ e ]

let make_reduce_rows () =
  let rows, cols = (8, 32) in
  let p0 = K.param ~idx:0 ~dtype:global_fptr in
  let p1 = K.param ~idx:1 ~dtype:global_fptr in
  let ri = K.range ~size:(idx rows) ~axis:0 ~kind:Axis_kind.Global () in
  let rj = K.range ~size:(idx cols) ~axis:1 ~kind:Axis_kind.Reduce () in
  let open K.O in
  let flat = ri * int_ cols + rj in
  let ld = K.load ~src:(K.index ~ptr:p0 ~idxs:[ flat ] ()) () in
  let red = K.reduce ~op:`Add ~src:ld ~ranges:[ rj ] ~dtype:Dtype.float32 in
  let st =
    K.store ~dst:(K.index ~ptr:p1 ~idxs:[ ri ] ()) ~value:red ~ranges:[]
  in
  let e = K.end_ ~value:st ~ranges:[ ri ] () in
  K.sink
    ~kernel_info:
      (kernel_info
         ~axis_kinds:[ Axis_kind.Global; Axis_kind.Reduce ]
         "reduce_rows")
    [ e ]

let make_no_optimize () =
  let p0 = K.param ~idx:0 ~dtype:global_fptr in
  let p1 = K.param ~idx:1 ~dtype:global_fptr in
  let p2 = K.param ~idx:2 ~dtype:global_fptr in
  let r0 = K.range ~size:(idx 256) ~axis:0 ~kind:Axis_kind.Global () in
  let ld_a = K.load ~src:(K.index ~ptr:p0 ~idxs:[ r0 ] ()) () in
  let ld_b = K.load ~src:(K.index ~ptr:p1 ~idxs:[ r0 ] ()) () in
  let add = K.binary ~op:`Add ~lhs:ld_a ~rhs:ld_b in
  let st = K.store ~dst:(K.index ~ptr:p2 ~idxs:[ r0 ] ()) ~value:add ~ranges:[] in
  let e = K.end_ ~value:st ~ranges:[ r0 ] () in
  K.sink ~kernel_info:(kernel_info ~axis_kinds:[ Axis_kind.Global ] "no_optimize") [ e ]

let make_multi_output () =
  let p0 = K.param ~idx:0 ~dtype:global_fptr in
  let p1 = K.param ~idx:1 ~dtype:global_fptr in
  let p2 = K.param ~idx:2 ~dtype:global_fptr in
  let r0 = K.range ~size:(idx 256) ~axis:0 ~kind:Axis_kind.Global () in
  let ld_a = K.load ~src:(K.index ~ptr:p0 ~idxs:[ r0 ] ()) () in
  let one = K.const (Const.float Dtype.float32 1.0) in
  let two = K.const (Const.float Dtype.float32 2.0) in
  let st1 =
    K.store
      ~dst:(K.index ~ptr:p1 ~idxs:[ r0 ] ())
      ~value:(K.binary ~op:`Add ~lhs:ld_a ~rhs:one)
      ~ranges:[]
  in
  let e1 = K.end_ ~value:st1 ~ranges:[ r0 ] () in
  let st2 =
    K.store
      ~dst:(K.index ~ptr:p2 ~idxs:[ r0 ] ())
      ~value:(K.binary ~op:`Mul ~lhs:ld_a ~rhs:two)
      ~ranges:[]
  in
  let e2 = K.end_ ~value:st2 ~ranges:[ r0 ] () in
  K.sink ~kernel_info:(kernel_info ~axis_kinds:[ Axis_kind.Global ] "multi_output") [ e1; e2 ]

let make_gated_store () =
  let p0 = K.param ~idx:0 ~dtype:global_fptr in
  let p1 = K.param ~idx:1 ~dtype:global_fptr in
  let p2 = K.param ~idx:2 ~dtype:global_fptr in
  let r0 = K.range ~size:(idx 256) ~axis:0 ~kind:Axis_kind.Global () in
  let ld_a = K.load ~src:(K.index ~ptr:p0 ~idxs:[ r0 ] ()) () in
  let ld_b = K.load ~src:(K.index ~ptr:p1 ~idxs:[ r0 ] ()) () in
  let add = K.binary ~op:`Add ~lhs:ld_a ~rhs:ld_b in
  let gate = K.binary ~op:`Cmplt ~lhs:r0 ~rhs:(idx 200) in
  let st =
    K.store
      ~dst:(K.index ~ptr:p2 ~idxs:[ r0 ] ~gate ())
      ~value:add ~ranges:[]
  in
  let e = K.end_ ~value:st ~ranges:[ r0 ] () in
  K.sink ~kernel_info:(kernel_info ~axis_kinds:[ Axis_kind.Global ] "gated_store") [ e ]

let make_elementwise_where () =
  let p0 = K.param ~idx:0 ~dtype:global_fptr in
  let p1 = K.param ~idx:1 ~dtype:global_fptr in
  let r0 = K.range ~size:(idx 256) ~axis:0 ~kind:Axis_kind.Global () in
  let ld = K.load ~src:(K.index ~ptr:p0 ~idxs:[ r0 ] ()) () in
  let zero = K.const (Const.float Dtype.float32 0.0) in
  let cond = K.binary ~op:`Cmplt ~lhs:zero ~rhs:ld in
  let w = K.ternary ~op:`Where ~a:cond ~b:ld ~c:zero in
  let st = K.store ~dst:(K.index ~ptr:p1 ~idxs:[ r0 ] ()) ~value:w ~ranges:[] in
  let e = K.end_ ~value:st ~ranges:[ r0 ] () in
  K.sink ~kernel_info:(kernel_info ~axis_kinds:[ Axis_kind.Global ] "elementwise_where") [ e ]

let make_elementwise_cast_f16 () =
  (* c[i] = (float32)a_f16[i] + b[i]. Param order: 0=f16, 1=f32, 2=out_f32.
     Build the Add as cast(ld_f16) + ld_f32 to match the reference load ordering. *)
  let f16_ptr = Dtype.ptr_of Dtype.float16 ~addrspace:Global ~size:(-1) in
  let p0 = K.param ~idx:0 ~dtype:f16_ptr in
  let p1 = K.param ~idx:1 ~dtype:global_fptr in
  let p2 = K.param ~idx:2 ~dtype:global_fptr in
  let r0 = K.range ~size:(idx 256) ~axis:0 ~kind:Axis_kind.Global () in
  let ld_a = K.load ~src:(K.index ~ptr:p0 ~idxs:[ r0 ] ()) () in
  let cast_a = K.cast ~src:ld_a ~dtype:(Dtype.to_any Dtype.float32) in
  let ld_b = K.load ~src:(K.index ~ptr:p1 ~idxs:[ r0 ] ()) () in
  let add = K.binary ~op:`Add ~lhs:cast_a ~rhs:ld_b in
  let st = K.store ~dst:(K.index ~ptr:p2 ~idxs:[ r0 ] ()) ~value:add ~ranges:[] in
  let e = K.end_ ~value:st ~ranges:[ r0 ] () in
  K.sink ~kernel_info:(kernel_info ~axis_kinds:[ Axis_kind.Global ] "elementwise_cast_f16") [ e ]

let make_elementwise_sqrt () =
  let p0 = K.param ~idx:0 ~dtype:global_fptr in
  let p1 = K.param ~idx:1 ~dtype:global_fptr in
  let r0 = K.range ~size:(idx 256) ~axis:0 ~kind:Axis_kind.Global () in
  let ld = K.load ~src:(K.index ~ptr:p0 ~idxs:[ r0 ] ()) () in
  let sq = K.unary ~op:`Sqrt ~src:ld in
  let st = K.store ~dst:(K.index ~ptr:p1 ~idxs:[ r0 ] ()) ~value:sq ~ranges:[] in
  let e = K.end_ ~value:st ~ranges:[ r0 ] () in
  K.sink ~kernel_info:(kernel_info ~axis_kinds:[ Axis_kind.Global ] "elementwise_sqrt") [ e ]

let make_parallel_reduce () =
  let p0 = K.param ~idx:0 ~dtype:global_fptr in
  let p1 = K.param ~idx:1 ~dtype:global_fptr in
  let p2 = K.param ~idx:2 ~dtype:global_fptr in
  let r0 = K.range ~size:(idx 128) ~axis:0 ~kind:Axis_kind.Reduce () in
  let ld = K.load ~src:(K.index ~ptr:p0 ~idxs:[ r0 ] ()) () in
  let red1 = K.reduce ~op:`Add ~src:ld ~ranges:[ r0 ] ~dtype:Dtype.float32 in
  let sq = K.binary ~op:`Mul ~lhs:ld ~rhs:ld in
  let red2 = K.reduce ~op:`Add ~src:sq ~ranges:[ r0 ] ~dtype:Dtype.float32 in
  let c0 = idx 0 in
  let st1 =
    K.store ~dst:(K.index ~ptr:p1 ~idxs:[ c0 ] ()) ~value:red1 ~ranges:[]
  in
  let st2 =
    K.store ~dst:(K.index ~ptr:p2 ~idxs:[ c0 ] ()) ~value:red2 ~ranges:[]
  in
  K.sink ~kernel_info:(kernel_info ~axis_kinds:[ Axis_kind.Reduce ] "parallel_reduce") [ st1; st2 ]

let make_elementwise_int32 () =
  let i32_ptr = Dtype.ptr_of Dtype.int32 ~addrspace:Global ~size:(-1) in
  let p0 = K.param ~idx:0 ~dtype:i32_ptr in
  let p1 = K.param ~idx:1 ~dtype:i32_ptr in
  let p2 = K.param ~idx:2 ~dtype:i32_ptr in
  let r0 = K.range ~size:(idx 256) ~axis:0 ~kind:Axis_kind.Global () in
  let ld_a = K.load ~src:(K.index ~ptr:p0 ~idxs:[ r0 ] ()) () in
  let ld_b = K.load ~src:(K.index ~ptr:p1 ~idxs:[ r0 ] ()) () in
  let add = K.binary ~op:`Add ~lhs:ld_a ~rhs:ld_b in
  let st = K.store ~dst:(K.index ~ptr:p2 ~idxs:[ r0 ] ()) ~value:add ~ranges:[] in
  let e = K.end_ ~value:st ~ranges:[ r0 ] () in
  K.sink ~kernel_info:(kernel_info ~axis_kinds:[ Axis_kind.Global ] "elementwise_int32") [ e ]

(* ── Test cases ── *)

type test_case = {
  name : string;
  kernel : Kernel.t;
  backends : (string * Renderer.t) list;
  optimize : bool;
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
    { name = "elementwise_add"; kernel = make_elementwise_add ();
      backends = all_renderers; optimize = true };
    { name = "sum_reduce"; kernel = make_sum_reduce ();
      backends = all_renderers; optimize = true };
    (* max_reduce excluded: requires Max→Where decomposition (Steps 18-21). *)
    { name = "dot_product"; kernel = make_dot_product ();
      backends = all_renderers; optimize = true };
    { name = "matmul_small"; kernel = make_matmul_small ();
      backends = gpu_renderers; optimize = true };
    { name = "elementwise_2d"; kernel = make_elementwise_2d ();
      backends = gpu_renderers; optimize = true };
    { name = "reduce_rows"; kernel = make_reduce_rows ();
      backends = all_renderers; optimize = true };
    { name = "no_optimize"; kernel = make_no_optimize ();
      backends = all_renderers; optimize = false };
    { name = "multi_output"; kernel = make_multi_output ();
      backends = all_renderers; optimize = true };
    { name = "gated_store"; kernel = make_gated_store ();
      backends = all_renderers; optimize = true };
    { name = "elementwise_where"; kernel = make_elementwise_where ();
      backends = all_renderers; optimize = true };
    { name = "elementwise_cast_f16"; kernel = make_elementwise_cast_f16 ();
      backends = all_renderers; optimize = true };
    { name = "elementwise_sqrt"; kernel = make_elementwise_sqrt ();
      backends = all_renderers; optimize = true };
    { name = "parallel_reduce"; kernel = make_parallel_reduce ();
      backends = all_renderers; optimize = true };
    { name = "elementwise_int32"; kernel = make_elementwise_int32 ();
      backends = all_renderers; optimize = true };
  ]

let () =
  let dir = Sys.argv.(1) in
  List.iter
    (fun { name; kernel; backends; optimize } ->
      List.iter
        (fun (backend_name, renderer) ->
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
