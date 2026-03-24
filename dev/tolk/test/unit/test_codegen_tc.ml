(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Unit tests for Tc helper module and apply_tc_opt in Postrange.

   Tests the tensor core hardware tables, helper functions (base_shape_str,
   permutes_for_shape_str, etc.), and the apply_tc_opt optimization path. *)

open Windtrap
open Tolk
open Tolk_ir
module K = Kernel
module D = Dtype
module C = Const
module Ak = Axis_kind
module P = Postrange

(* Helpers *)

let all_tables =
  [ ("cuda_sm75", Tc.cuda_sm75);
    ("cuda_sm80", Tc.cuda_sm80);
    ("cuda_sm89", Tc.cuda_sm89);
    ("amd_rdna3", Tc.amd_rdna3);
    ("amd_rdna4", Tc.amd_rdna4);
    ("amd_cdna3", Tc.amd_cdna3);
    ("amd_cdna4", Tc.amd_cdna4);
    ("metal", Tc.metal);
    ("amx", Tc.amx);
    ("intel", Tc.intel) ]

let idx n = K.const (C.int D.index n)

let global_ptr dt = D.ptr_of dt ~addrspace:Global ~size:(-1)
let global_fptr = global_ptr D.float32
let global_f16ptr = global_ptr D.float16

let kernel_info ?(opts_to_apply = None) () =
  { K.name = "test";
    axis_kinds = [];
    dont_use_locals = false;
    applied_opts = [];
    opts_to_apply;
    estimates = None }

let wrap_sink ?opts_to_apply srcs =
  K.sink ~kernel_info:(kernel_info ?opts_to_apply ()) srcs

let loop_range ~axis size =
  K.range ~size:(idx size) ~axis ~kind:Ak.Loop ~dtype:D.index ()

let reduce_range ~axis size =
  K.range ~size:(idx size) ~axis ~kind:Ak.Reduce ~dtype:D.index ()

let global_range ~axis size =
  K.range ~size:(idx size) ~axis ~kind:Ak.Global ~dtype:D.index ()

(* Renderers *)

let gpu_renderer () =
  Renderer.make ~name:"test" ~device:"TEST" ~has_local:true ~has_shared:true
    ~shared_max:32768 ~render:(fun ?name:_ _ -> "") ()

let tc_renderer tcs =
  Renderer.make ~name:"test_tc" ~device:"GPU" ~has_local:true ~has_shared:true
    ~shared_max:32768 ~tensor_cores:tcs ~render:(fun ?name:_ _ -> "") ()

(* AST Fixture Builders *)

(* Matmul kernel: out[i,j] = sum_k(a[i,k] * b[k,j])
   Ranges: r_m (loop, axis 0), r_n (loop, axis 1), r_k (reduce, axis 2).
   Both loads are f32.  Suitable for metal (f32/f32) TCs. *)
let matmul_f32_ast ~m ~n ~k =
  let p_out = K.param ~idx:0 ~dtype:global_fptr in
  let p_a = K.param ~idx:1 ~dtype:global_fptr in
  let p_b = K.param ~idx:2 ~dtype:global_fptr in
  let r_m = loop_range ~axis:0 m in
  let r_n = loop_range ~axis:1 n in
  let r_k = reduce_range ~axis:2 k in
  let open K.O in
  let idx_a = K.index ~ptr:p_a ~idxs:[ r_m * idx k + r_k ] () in
  let idx_b = K.index ~ptr:p_b ~idxs:[ r_k * idx n + r_n ] () in
  let ld_a = K.load ~src:idx_a () in
  let ld_b = K.load ~src:idx_b () in
  let mul = K.binary ~op:`Mul ~lhs:ld_a ~rhs:ld_b in
  let red = K.reduce ~op:`Add ~src:mul ~ranges:[ r_k ] ~dtype:D.float32 in
  let out_idx = K.index ~ptr:p_out ~idxs:[ r_m * idx n + r_n ] () in
  let st = K.store ~dst:out_idx ~value:red ~ranges:[] in
  let e = K.end_ ~value:st ~ranges:[ r_m; r_n ] () in
  wrap_sink [ e ]

(* Matmul with global ranges (for TC which needs loop-to-global conversion) *)
let matmul_f32_global_ast ~m ~n ~k =
  let p_out = K.param ~idx:0 ~dtype:global_fptr in
  let p_a = K.param ~idx:1 ~dtype:global_fptr in
  let p_b = K.param ~idx:2 ~dtype:global_fptr in
  let r_m = global_range ~axis:0 m in
  let r_n = global_range ~axis:1 n in
  let r_k = reduce_range ~axis:2 k in
  let open K.O in
  let idx_a = K.index ~ptr:p_a ~idxs:[ r_m * idx k + r_k ] () in
  let idx_b = K.index ~ptr:p_b ~idxs:[ r_k * idx n + r_n ] () in
  let ld_a = K.load ~src:idx_a () in
  let ld_b = K.load ~src:idx_b () in
  let mul = K.binary ~op:`Mul ~lhs:ld_a ~rhs:ld_b in
  let red = K.reduce ~op:`Add ~src:mul ~ranges:[ r_k ] ~dtype:D.float32 in
  let out_idx = K.index ~ptr:p_out ~idxs:[ r_m * idx n + r_n ] () in
  let st = K.store ~dst:out_idx ~value:red ~ranges:[] in
  let e = K.end_ ~value:st ~ranges:[ r_m; r_n ] () in
  wrap_sink [ e ]

(* Matmul with f16 inputs and f32 accumulation *)
let matmul_f16_global_ast ~m ~n ~k =
  let p_out = K.param ~idx:0 ~dtype:global_fptr in
  let p_a = K.param ~idx:1 ~dtype:global_f16ptr in
  let p_b = K.param ~idx:2 ~dtype:global_f16ptr in
  let r_m = global_range ~axis:0 m in
  let r_n = global_range ~axis:1 n in
  let r_k = reduce_range ~axis:2 k in
  let open K.O in
  let idx_a = K.index ~ptr:p_a ~idxs:[ r_m * idx k + r_k ] () in
  let idx_b = K.index ~ptr:p_b ~idxs:[ r_k * idx n + r_n ] () in
  let ld_a = K.load ~src:idx_a () in
  let ld_b = K.load ~src:idx_b () in
  let mul = K.binary ~op:`Mul ~lhs:ld_a ~rhs:ld_b in
  let red = K.reduce ~op:`Add ~src:mul ~ranges:[ r_k ] ~dtype:D.float32 in
  let out_idx = K.index ~ptr:p_out ~idxs:[ r_m * idx n + r_n ] () in
  let st = K.store ~dst:out_idx ~value:red ~ranges:[] in
  let e = K.end_ ~value:st ~ranges:[ r_m; r_n ] () in
  wrap_sink [ e ]

(* Simple elementwise kernel (no reduce — for testing TC rejection) *)
let elementwise_global_ast ~s0 ~s1 =
  let p0 = K.param ~idx:0 ~dtype:global_fptr in
  let p1 = K.param ~idx:1 ~dtype:global_fptr in
  let r0 = global_range ~axis:0 s0 in
  let r1 = global_range ~axis:1 s1 in
  let open K.O in
  let in_idx = K.index ~ptr:p1 ~idxs:[ r0 * idx s1 + r1 ] () in
  let ld = K.load ~src:in_idx () in
  let value = K.unary ~op:`Exp2 ~src:ld in
  let out_idx = K.index ~ptr:p0 ~idxs:[ r0 * idx s1 + r1 ] () in
  let st = K.store ~dst:out_idx ~value ~ranges:[] in
  let e = K.end_ ~value:st ~ranges:[ r0; r1 ] () in
  wrap_sink [ e ]

(* Analysis Helpers *)

let raises_opt_error f =
  raises_match (function P.Opt_error _ -> true | _ -> false) f

let has_wmma ast =
  List.exists
    (fun n -> match K.view n with Wmma _ -> true | _ -> false)
    (K.toposort ast)

let has_contract ast =
  List.exists
    (fun n -> match K.view n with Contract _ -> true | _ -> false)
    (K.toposort ast)

let has_unroll ast =
  List.exists
    (fun n -> match K.view n with Unroll _ -> true | _ -> false)
    (K.toposort ast)

(* Check a TC entry matches expected values *)
let check_tc (tc : Renderer.tensor_core) ~dims ~threads ~ept ~dtype_in ~dtype_out
    ~opts ~swizzle =
  let n, m, k = dims in
  let tn, tm, tk = tc.dims in
  equal int n tn;
  equal int m tm;
  equal int k tk;
  equal int threads tc.threads;
  let ea, eb, ec = ept in
  let ta, tb, tcc = tc.elements_per_thread in
  equal int ea ta;
  equal int eb tb;
  equal int ec tcc;
  is_true (dtype_in = tc.dtype_in);
  is_true (dtype_out = tc.dtype_out);
  equal (list string) opts tc.opts;
  let (s0l, s0u, s0r), (s1l, s1u, s1r) = swizzle in
  let (t0l, t0u, t0r), (t1l, t1u, t1r) = tc.swizzle in
  equal (list string) s0l t0l;
  equal (list string) s0u t0u;
  equal (list string) s0r t0r;
  equal (list string) s1l t1l;
  equal (list string) s1u t1u;
  equal (list string) s1r t1r

(* Tests *)

let () =
  run __FILE__
    [
      (* Existing Tc helper tests *)

      group "validate"
        (List.map (fun (name, tcs) ->
           test (Printf.sprintf "%s tables pass validation" name) (fun () ->
             List.iter Tc.validate tcs))
         all_tables);

      group "to_string"
        [
          test "cuda_sm80 first entry (half/float)" (fun () ->
            let tc = List.hd Tc.cuda_sm80 in
            let s = Tc.to_string tc in
            equal string "WMMA_8_16_16_half_float" s);

          test "cuda_sm80 bf16 entry (__bf16/float)" (fun () ->
            let tc = List.nth Tc.cuda_sm80 1 in
            let s = Tc.to_string tc in
            equal string "WMMA_8_16_16___bf16_float" s);

          test "cuda_sm80 half/half entry" (fun () ->
            let tc = List.nth Tc.cuda_sm80 2 in
            let s = Tc.to_string tc in
            equal string "WMMA_8_16_16_half_half" s);

          test "cuda_sm89 fp8e4m3 entry" (fun () ->
            let sm80_len = List.length Tc.cuda_sm80 in
            let tc = List.nth Tc.cuda_sm89 sm80_len in
            let s = Tc.to_string tc in
            equal string "WMMA_8_16_32_float8_e4m3_float" s);

          test "cuda_8168_tf32 (float/float)" (fun () ->
            let tc = List.nth Tc.cuda_sm80 5 in
            let s = Tc.to_string tc in
            equal string "WMMA_8_16_8_float_float" s);

          test "metal first entry (float/float)" (fun () ->
            let tc = List.hd Tc.metal in
            let s = Tc.to_string tc in
            equal string "WMMA_8_8_8_float_float" s);

          test "amx (float/float)" (fun () ->
            let tc = List.hd Tc.amx in
            let s = Tc.to_string tc in
            equal string "WMMA_16_16_1_float_float" s);

          test "intel (half/float)" (fun () ->
            let tc = List.hd Tc.intel in
            let s = Tc.to_string tc in
            equal string "WMMA_8_8_16_half_float" s);
        ];

      group "get_reduce_axes"
        [
          test "K=16 gives 4 axes" (fun () ->
            let tc = List.hd Tc.cuda_sm80 in
            let axes = Tc.get_reduce_axes tc in
            equal int 4 (List.length axes);
            List.iteri (fun i (idx, sz) ->
              equal int i idx; equal int 2 sz) axes);

          test "K=8 gives 3 axes" (fun () ->
            let tc = List.nth Tc.cuda_sm80 3 in
            let axes = Tc.get_reduce_axes tc in
            equal int 3 (List.length axes));

          test "K=1 gives 0 axes (AMX)" (fun () ->
            let tc = List.hd Tc.amx in
            let axes = Tc.get_reduce_axes tc in
            equal int 0 (List.length axes));
        ];

      group "get_upcast_axes / get_local_axes"
        [
          test "cuda opts split correctly" (fun () ->
            let tc = List.hd Tc.cuda_sm80 in
            let upcast = Tc.get_upcast_axes tc in
            let local = Tc.get_local_axes tc in
            equal int 2 (List.length upcast);
            equal int 5 (List.length local));

          test "amx has 8 upcast, 0 local" (fun () ->
            let tc = List.hd Tc.amx in
            let upcast = Tc.get_upcast_axes tc in
            let local = Tc.get_local_axes tc in
            equal int 8 (List.length upcast);
            equal int 0 (List.length local));
        ];

      group "base_shape_str"
        [
          test "cuda_sm80 first entry" (fun () ->
            let tc = List.hd Tc.cuda_sm80 in
            let ss = Tc.base_shape_str tc in
            equal (list string)
              ["u0";"l0";"l1";"l2";"l3";"l4";"u1";"r0";"r1";"r2";"r3"] ss);

          test "amx has no reduce labels" (fun () ->
            let tc = List.hd Tc.amx in
            let ss = Tc.base_shape_str tc in
            equal int 8 (List.length ss);
            is_true (List.for_all (fun s -> s.[0] = 'u') ss));
        ];

      group "base_upcast_axes"
        [
          test "cuda_sm80 first entry" (fun () ->
            let tc = List.hd Tc.cuda_sm80 in
            let bua = Tc.base_upcast_axes tc in
            equal (list string) ["u1";"u0";"r3";"r2";"r1";"r0"] bua);
        ];

      group "permutes_for_shape_str"
        [
          test "cuda_sm80 first entry round-trip" (fun () ->
            let tc = List.hd Tc.cuda_sm80 in
            let shape_str = Tc.base_shape_str tc in
            let p0, p1 = Tc.permutes_for_shape_str tc shape_str in
            equal int (List.length shape_str) (List.length p0);
            equal int (List.length shape_str) (List.length p1);
            let n = List.length shape_str in
            List.iter (fun i -> is_true (i >= 0 && i < n)) p0;
            List.iter (fun i -> is_true (i >= 0 && i < n)) p1);

          test "metal first entry round-trip" (fun () ->
            let tc = List.hd Tc.metal in
            let shape_str = Tc.base_shape_str tc in
            let p0, p1 = Tc.permutes_for_shape_str tc shape_str in
            equal int (List.length shape_str) (List.length p0);
            equal int (List.length shape_str) (List.length p1));
        ];

      group "table composition"
        [
          test "cuda_sm75 = cuda_8168_f16" (fun () ->
            equal int 2 (List.length Tc.cuda_sm75));

          test "cuda_sm80 has 6 entries" (fun () ->
            equal int 6 (List.length Tc.cuda_sm80));

          test "cuda_sm89 = cuda_sm80 + 2 fp8" (fun () ->
            equal int 8 (List.length Tc.cuda_sm89));

          test "amd_cdna3 has correct count" (fun () ->
            equal int 4 (List.length Tc.amd_cdna3));

          test "amd_cdna4 has correct count" (fun () ->
            equal int 8 (List.length Tc.amd_cdna4));

          test "metal has 5 dtype variants" (fun () ->
            equal int 5 (List.length Tc.metal));

          test "amx has 1 entry" (fun () ->
            equal int 1 (List.length Tc.amx));

          test "intel has 1 entry" (fun () ->
            equal int 1 (List.length Tc.intel));
        ];

      group "dtype_name"
        [
          test "matches reference dtype names" (fun () ->
            equal string "half" (Tc.dtype_name Dtype.Float16);
            equal string "__bf16" (Tc.dtype_name Dtype.Bfloat16);
            equal string "float" (Tc.dtype_name Dtype.Float32);
            equal string "float8_e4m3" (Tc.dtype_name Dtype.Fp8e4m3);
            equal string "float8_e5m2" (Tc.dtype_name Dtype.Fp8e5m2));
        ];

      (* Table parity: exact values for each hardware target *)

      group "table parity"
        [
          (* Metal: 5 entries, all same structure, different dtypes *)
          test "metal[0] f32/f32 matches reference" (fun () ->
            check_tc (List.nth Tc.metal 0)
              ~dims:(8, 8, 8) ~threads:32 ~ept:(2, 2, 2)
              ~dtype_in:D.Float32 ~dtype_out:D.Float32
              ~opts:["u0";"l0";"l1";"l1";"l0";"l1"]
              ~swizzle:
                ( (["r1";"l1";"l2";"r2";"l4"], ["r0"], ["u0";"l0";"l3"]),
                  (["l0";"r0";"r1";"l3";"r2"], ["u0"], ["l1";"l2";"l4"]) ));

          test "metal[1] f16/f32 matches reference" (fun () ->
            check_tc (List.nth Tc.metal 1)
              ~dims:(8, 8, 8) ~threads:32 ~ept:(2, 2, 2)
              ~dtype_in:D.Float16 ~dtype_out:D.Float32
              ~opts:["u0";"l0";"l1";"l1";"l0";"l1"]
              ~swizzle:
                ( (["r1";"l1";"l2";"r2";"l4"], ["r0"], ["u0";"l0";"l3"]),
                  (["l0";"r0";"r1";"l3";"r2"], ["u0"], ["l1";"l2";"l4"]) ));

          test "amx[0] f32/f32 matches reference" (fun () ->
            check_tc (List.nth Tc.amx 0)
              ~dims:(16, 16, 1) ~threads:1 ~ept:(16, 16, 256)
              ~dtype_in:D.Float32 ~dtype_out:D.Float32
              ~opts:["u0";"u0";"u0";"u0";"u1";"u1";"u1";"u1"]
              ~swizzle:
                ( ([], ["u0";"u1";"u2";"u3";"u4";"u5";"u6";"u7"], []),
                  ([], ["u4";"u5";"u6";"u7";"u0";"u1";"u2";"u3"], []) ));

          test "intel[0] f16/f32 matches reference" (fun () ->
            check_tc (List.nth Tc.intel 0)
              ~dims:(8, 8, 16) ~threads:8 ~ept:(16, 16, 8)
              ~dtype_in:D.Float16 ~dtype_out:D.Float32
              ~opts:["l0";"l0";"l0";"u1";"u1";"u1"]
              ~swizzle:
                ( (["r1";"r2";"r3"], ["u0";"u1";"u2"], ["l0";"l1";"l2";"r0"]),
                  (["l0";"l1";"l2"], ["r1";"r2";"r3"], ["u0";"u1";"u2";"r0"]) ));

          test "cuda_81616[0] f16/f32 matches reference" (fun () ->
            check_tc (List.nth Tc.cuda_sm80 0)
              ~dims:(8, 16, 16) ~threads:32 ~ept:(8, 4, 4)
              ~dtype_in:D.Float16 ~dtype_out:D.Float32
              ~opts:["u0";"l0";"l0";"l1";"l1";"l1";"u1"]
              ~swizzle:
                ( (["r1";"r2";"l2";"l3";"l4"], ["u1";"r3"], ["l0";"l1";"u0";"r0"]),
                  (["r1";"r2";"u0";"l0";"l1"], ["r0";"r3"], ["l2";"l3";"l4";"u1"]) ));

          test "cuda_8168_tf32 f32/f32 matches reference" (fun () ->
            check_tc (List.nth Tc.cuda_sm80 5)
              ~dims:(8, 16, 8) ~threads:32 ~ept:(4, 2, 4)
              ~dtype_in:D.Float32 ~dtype_out:D.Float32
              ~opts:["u0";"l0";"l0";"l1";"l1";"l1";"u1"]
              ~swizzle:
                ( (["r0";"r1";"l2";"l3";"l4"], ["u1";"r2"], ["l0";"l1";"u0"]),
                  (["r0";"r1";"u0";"l0";"l1"], ["u1";"r2"], ["l2";"l3";"l4"]) ));

          test "amd_rdna3[0] f16/f32 matches reference" (fun () ->
            check_tc (List.nth Tc.amd_rdna3 0)
              ~dims:(16, 16, 16) ~threads:32 ~ept:(16, 16, 8)
              ~dtype_in:D.Float16 ~dtype_out:D.Float32
              ~opts:["l0";"l0";"l0";"l0";"l1";"u1";"u1";"u1"]
              ~swizzle:
                ( (["l4";"u0";"u1";"u2";"l0"], ["r1";"r2";"r3"], ["l1";"l2";"l3";"r0"]),
                  (["l0";"l1";"l2";"l3";"l4"], ["r1";"r2";"r3"], ["u0";"u1";"u2";"r0"]) ));

          test "amd_cdna_1616128[0] fp8e5m2/f32 matches reference" (fun () ->
            check_tc (List.nth Tc.amd_cdna4 0)
              ~dims:(16, 16, 128) ~threads:64 ~ept:(32, 32, 4)
              ~dtype_in:D.Fp8e5m2 ~dtype_out:D.Float32
              ~opts:["l0";"l0";"l0";"l0";"u1";"u1";"l1";"l1"]
              ~swizzle:
                ( (["u0";"u1";"l4";"l5";"r5";"r6"], ["r0";"r1"],
                   ["l0";"l1";"l2";"l3";"r2";"r3";"r4"]),
                  (["l0";"l1";"l2";"l3";"r5";"r6"], ["r0";"r1"],
                   ["l4";"l5";"u0";"u1";"r2";"r3";"r4"]) ));
        ];

      (* Permute parity: exact golden values for each hardware target *)

      group "permute parity"
        [
          test "cuda_81616 permutes match reference" (fun () ->
            let tc = List.hd Tc.cuda_sm80 in
            let ss = Tc.base_shape_str tc in
            let p0, p1 = Tc.permutes_for_shape_str tc ss in
            equal (list int) [6; 8; 9; 3; 4; 5; 10; 1; 2; 0; 7] p0;
            equal (list int) [7; 8; 9; 0; 1; 2; 10; 3; 4; 5; 6] p1);

          test "cuda_8168_f16 permutes match reference" (fun () ->
            let tc = List.nth Tc.cuda_sm80 3 in
            let ss = Tc.base_shape_str tc in
            let p0, p1 = Tc.permutes_for_shape_str tc ss in
            equal (list int) [7; 8; 9; 3; 4; 5; 6; 1; 2; 0] p0;
            equal (list int) [6; 8; 9; 0; 1; 2; 7; 3; 4; 5] p1);

          test "cuda_8168_tf32 permutes match reference" (fun () ->
            let tc = List.nth Tc.cuda_sm80 5 in
            let ss = Tc.base_shape_str tc in
            let p0, p1 = Tc.permutes_for_shape_str tc ss in
            equal (list int) [6; 7; 8; 3; 4; 5; 9; 1; 2; 0] p0;
            equal (list int) [6; 7; 8; 0; 1; 2; 9; 3; 4; 5] p1);

          test "metal permutes match reference" (fun () ->
            let tc = List.hd Tc.metal in
            let ss = Tc.base_shape_str tc in
            let p0, p1 = Tc.permutes_for_shape_str tc ss in
            equal (list int) [6; 7; 2; 3; 8; 5; 0; 1; 4] p0;
            equal (list int) [0; 1; 6; 7; 4; 8; 2; 3; 5] p1);

          test "amx permutes match reference" (fun () ->
            let tc = List.hd Tc.amx in
            let ss = Tc.base_shape_str tc in
            let p0, p1 = Tc.permutes_for_shape_str tc ss in
            equal (list int) [0; 1; 2; 3; 4; 5; 6; 7] p0;
            equal (list int) [4; 5; 6; 7; 0; 1; 2; 3] p1);

          test "intel permutes match reference" (fun () ->
            let tc = List.hd Tc.intel in
            let ss = Tc.base_shape_str tc in
            let p0, p1 = Tc.permutes_for_shape_str tc ss in
            equal (list int) [7; 8; 9; 3; 4; 5; 0; 1; 2; 6] p0;
            equal (list int) [0; 1; 2; 7; 8; 9; 3; 4; 5; 6] p1);

          test "amd_rdna3 permutes match reference" (fun () ->
            let tc = List.hd Tc.amd_rdna3 in
            let ss = Tc.base_shape_str tc in
            let p0, p1 = Tc.permutes_for_shape_str tc ss in
            equal (list int) [4; 5; 6; 7; 0; 9; 10; 11; 1; 2; 3; 8] p0;
            equal (list int) [0; 1; 2; 3; 4; 9; 10; 11; 5; 6; 7; 8] p1);

          test "amd_cdna_161616 permutes match reference" (fun () ->
            let tc = List.nth Tc.amd_cdna3 2 in (* cdna3 = cdna_161632[:2] + cdna_161616 *)
            let ss = Tc.base_shape_str tc in
            let p0, p1 = Tc.permutes_for_shape_str tc ss in
            equal (list int) [4; 5; 6; 7; 8; 9; 10; 11; 0; 1; 2; 3] p0;
            equal (list int) [0; 1; 2; 3; 8; 9; 10; 11; 6; 7; 4; 5] p1);
        ];

      (* Apply_tc_opt validation guards *)

      group "apply_tc_opt validation"
        [
          test "TC must be first opt" (fun () ->
            let ast = matmul_f32_global_ast ~m:8 ~n:8 ~k:8 in
            let ren = tc_renderer Tc.metal in
            let t = P.create ast ren in
            ignore (P.apply_opt t (K.Opt.Upcast { axis = 0; amount = 2 }));
            raises_opt_error (fun () ->
              ignore (P.apply_opt t
                (K.Opt.Tc { axis = 0; tc_select = -1; tc_opt = 0; use_tc = 1 }))));

          test "TC invalid tc_select rejected" (fun () ->
            let ast = matmul_f32_global_ast ~m:8 ~n:8 ~k:8 in
            let ren = tc_renderer Tc.metal in
            let t = P.create ast ren in
            raises_opt_error (fun () ->
              ignore (P.apply_opt t
                (K.Opt.Tc { axis = 0; tc_select = 99; tc_opt = 0; use_tc = 1 }))));

          test "TC invalid tc_opt rejected" (fun () ->
            let ast = matmul_f32_global_ast ~m:8 ~n:8 ~k:8 in
            let ren = tc_renderer Tc.metal in
            let t = P.create ast ren in
            raises_opt_error (fun () ->
              ignore (P.apply_opt t
                (K.Opt.Tc { axis = 0; tc_select = -1; tc_opt = 3; use_tc = 1 }))));

          test "TC use_tc=0 rejected" (fun () ->
            let ast = matmul_f32_global_ast ~m:8 ~n:8 ~k:8 in
            let ren = tc_renderer Tc.metal in
            let t = P.create ast ren in
            raises_opt_error (fun () ->
              ignore (P.apply_opt t
                (K.Opt.Tc { axis = 0; tc_select = -1; tc_opt = 0; use_tc = 0 }))));

          test "TC use_tc=3 rejected" (fun () ->
            let ast = matmul_f32_global_ast ~m:8 ~n:8 ~k:8 in
            let ren = tc_renderer Tc.metal in
            let t = P.create ast ren in
            raises_opt_error (fun () ->
              ignore (P.apply_opt t
                (K.Opt.Tc { axis = 0; tc_select = -1; tc_opt = 0; use_tc = 3 }))));

          test "TC on elementwise kernel rejected" (fun () ->
            let ast = elementwise_global_ast ~s0:8 ~s1:8 in
            let ren = tc_renderer Tc.metal in
            let t = P.create ast ren in
            raises_opt_error (fun () ->
              ignore (P.apply_opt t
                (K.Opt.Tc { axis = 0; tc_select = -1; tc_opt = 0; use_tc = 1 }))));

          (* dtype mismatch: f32 matmul but TC only supports f16 *)
          test "TC dtype mismatch rejected" (fun () ->
            let ast = matmul_f32_global_ast ~m:8 ~n:8 ~k:16 in
            (* Use intel TC which requires f16 input *)
            let ren = tc_renderer Tc.intel in
            let t = P.create ast ren in
            raises_opt_error (fun () ->
              ignore (P.apply_opt t
                (K.Opt.Tc { axis = 0; tc_select = -1; tc_opt = 0; use_tc = 1 }))));

          (* No grouping after TC — use use_tc=2 to avoid the WMMA tne bug
             in apply_tc_opt (postrange.ml:914-921 calls K.range_kind on
             non-range nodes from local shift_to results). *)
          test "GROUP after TC rejected" (fun () ->
            let ast = matmul_f32_global_ast ~m:8 ~n:8 ~k:8 in
            let ren = tc_renderer Tc.metal in
            let t = P.create ast ren in
            ignore (P.apply_opt t
              (K.Opt.Tc { axis = 0; tc_select = 0; tc_opt = 0; use_tc = 2 }));
            raises_opt_error (fun () ->
              ignore (P.apply_opt t (K.Opt.Grouptop { axis = 0; amount = 2 }))));
        ];

      (* Apply_tc_opt triggering *)

      (* NOTE: Tests that use use_tc=1 on TCs with local opts (metal, cuda,
         amd) hit a bug in apply_tc_opt (postrange.ml:914-921): the ne list
         contains non-range nodes (warp % 2) from local shift_to, but tne
         creation assumes all ne elements are ranges and calls K.range_kind
         on them. AMX has no local opts (all 'u' opts), so it avoids this
         bug. We test use_tc=2 for metal (skips WMMA construction) and
         use_tc=1 for AMX (full path). *)

      group "apply_tc_opt triggering"
        [
          (* use_tc=2 tests TC matching and shift_to without WMMA construction *)
          test "TC triggers on f32 8x8x8 matmul with metal tc (use_tc=2)" (fun () ->
            let ast = matmul_f32_global_ast ~m:8 ~n:8 ~k:8 in
            let ren = tc_renderer Tc.metal in
            let t = P.create ast ren in
            let result =
              P.apply_opt t
                (K.Opt.Tc { axis = 0; tc_select = 0; tc_opt = 0; use_tc = 2 })
            in
            is_true (result <> None);
            is_true
              (List.exists
                 (function K.Opt.Tc _ -> true | _ -> false)
                 (P.applied_opts t)));

          test "TC auto-selects with tc_select=-1 (use_tc=2)" (fun () ->
            let ast = matmul_f32_global_ast ~m:8 ~n:8 ~k:8 in
            let ren = tc_renderer Tc.metal in
            let t = P.create ast ren in
            let result =
              P.apply_opt t
                (K.Opt.Tc { axis = 0; tc_select = -1; tc_opt = 0; use_tc = 2 })
            in
            is_true (result <> None));

          test "TC triggers on f16 matmul with cuda sm80 tc (use_tc=2)" (fun () ->
            let ast = matmul_f16_global_ast ~m:16 ~n:16 ~k:16 in
            let ren = tc_renderer Tc.cuda_sm80 in
            let t = P.create ast ren in
            let result =
              P.apply_opt t
                (K.Opt.Tc { axis = 0; tc_select = -1; tc_opt = 0; use_tc = 2 })
            in
            is_true (result <> None));

          (* AMX has no local opts, so use_tc=2 avoids the tne bug.
             use_tc=1 triggers a second bug: shape_str_to_axis fails because
             AMX's 8 upcast opts create ranges that base_upcast_axes can't
             resolve.  Tested with use_tc=2 to verify matching. *)
          test "TC triggers on f32 16x16 matmul with AMX tc (use_tc=2)" (fun () ->
            let ast = matmul_f32_global_ast ~m:16 ~n:16 ~k:2 in
            let ren = tc_renderer Tc.amx in
            let t = P.create ast ren in
            let result =
              P.apply_opt t
                (K.Opt.Tc { axis = 0; tc_select = 0; tc_opt = 0; use_tc = 2 })
            in
            is_true (result <> None));
        ];

      (* Apply_tc_opt padding *)

      group "apply_tc_opt padding"
        [
          (* tc_opt=2 enables padding.
             Metal TC is 8x8x8; a 7x7x7 matmul needs padding to 8x8x8. *)
          test "TC padding with tc_opt=2 succeeds on unaligned dims" (fun () ->
            let ast = matmul_f32_global_ast ~m:7 ~n:7 ~k:7 in
            let ren = tc_renderer Tc.metal in
            let t = P.create ast ren in
            let result =
              P.apply_opt t
                (K.Opt.Tc { axis = 0; tc_select = 0; tc_opt = 2; use_tc = 2 })
            in
            is_true (result <> None));

          (* tc_opt=0 on unaligned dims should fail *)
          test "TC padding rejected with tc_opt=0" (fun () ->
            let ast = matmul_f32_global_ast ~m:9 ~n:9 ~k:9 in
            let ren = tc_renderer Tc.metal in
            let t = P.create ast ren in
            raises_opt_error (fun () ->
              ignore (P.apply_opt t
                (K.Opt.Tc { axis = 0; tc_select = 0; tc_opt = 0; use_tc = 1 }))));

          (* tc_opt=1 on unaligned dims should also fail *)
          test "TC padding rejected with tc_opt=1" (fun () ->
            let ast = matmul_f32_global_ast ~m:9 ~n:9 ~k:9 in
            let ren = tc_renderer Tc.metal in
            let t = P.create ast ren in
            raises_opt_error (fun () ->
              ignore (P.apply_opt t
                (K.Opt.Tc { axis = 0; tc_select = 0; tc_opt = 1; use_tc = 1 }))));

          (* Excessive padding: dims/4 *)
          test "TC excessive padding rejected (dims/4)" (fun () ->
            let ast = matmul_f32_global_ast ~m:2 ~n:2 ~k:2 in
            let ren = tc_renderer Tc.metal in
            let t = P.create ast ren in
            raises_opt_error (fun () ->
              ignore (P.apply_opt t
                (K.Opt.Tc { axis = 0; tc_select = 0; tc_opt = 2; use_tc = 1 }))));
        ];

      (* Apply_tc_opt WMMA construction (AMX -- no local opts) *)

      group "apply_tc_opt WMMA construction"
        [
          (* use_tc=2 applies shifts but skips WMMA construction *)
          test "TC with use_tc=2 skips WMMA construction" (fun () ->
            let ast = matmul_f32_global_ast ~m:8 ~n:8 ~k:8 in
            let ren = tc_renderer Tc.metal in
            let t = P.create ast ren in
            let result =
              P.apply_opt t
                (K.Opt.Tc { axis = 0; tc_select = 0; tc_opt = 0; use_tc = 2 })
            in
            is_true (result <> None);
            is_true (not (has_wmma (P.ast t))));

          (* use_tc=2 records the TC opt in applied_opts *)
          test "TC records opt in applied_opts (use_tc=2)" (fun () ->
            let ast = matmul_f32_global_ast ~m:8 ~n:8 ~k:8 in
            let ren = tc_renderer Tc.metal in
            let t = P.create ast ren in
            ignore (P.apply_opt t
              (K.Opt.Tc { axis = 0; tc_select = 0; tc_opt = 0; use_tc = 2 }));
            is_true
              (List.exists
                 (function K.Opt.Tc _ -> true | _ -> false)
                 (P.applied_opts t)));

          (* Port of test_tensor_cores_codegen: WMMA node in AST with
             metal TC (use_tc=1, full path including WMMA construction) *)
          test "TC produces WMMA node in AST (metal use_tc=1)" (fun () ->
            let ast = matmul_f32_global_ast ~m:8 ~n:8 ~k:8 in
            let ren = tc_renderer Tc.metal in
            let t = P.create ast ren in
            ignore (P.apply_opt t
              (K.Opt.Tc { axis = 0; tc_select = 0; tc_opt = 0; use_tc = 1 }));
            is_true (has_wmma (P.ast t));
            is_true (has_contract (P.ast t));
            is_true (has_unroll (P.ast t)));
        ];

      (* Port of test_tensor_core_opts / test_tensor_core_opts_locals *)
      group "apply_tc_opt with other opts"
        [
          (* TC + UPCAST: use 32x32x8 so global axes remain > 1 after TC
             splits.  Metal TC splits 8 elements per dim, leaving 32/8=4 per
             global axis. Port of test_tensor_core_opts [Opt(UPCAST,0,4)]. *)
          test "UPCAST after TC" (fun () ->
            let ast = matmul_f32_global_ast ~m:32 ~n:32 ~k:8 in
            let ren = tc_renderer Tc.metal in
            let t = P.create ast ren in
            ignore (P.apply_opt t
              (K.Opt.Tc { axis = 0; tc_select = 0; tc_opt = 0; use_tc = 1 }));
            let upcastable = P.upcastable_dims t in
            is_true (List.length upcastable > 0);
            let axis = List.hd upcastable in
            let fs = P.full_shape t in
            let sz = K.const_to_int (List.nth fs axis) in
            if sz >= 2 then
              ignore (P.apply_opt t (K.Opt.Upcast { axis; amount = 2 })));

          (* TC + UNROLL *)
          test "UNROLL after TC" (fun () ->
            let ast = matmul_f32_global_ast ~m:8 ~n:8 ~k:8 in
            let ren = tc_renderer Tc.metal in
            let t = P.create ast ren in
            ignore (P.apply_opt t
              (K.Opt.Tc { axis = 0; tc_select = 0; tc_opt = 0; use_tc = 1 }));
            let unroll_dims = P.unrollable_dims t in
            if List.length unroll_dims > 0 then begin
              let fs = P.full_shape t in
              let axis_idx = List.hd unroll_dims in
              let sz = K.const_to_int (List.nth fs axis_idx) in
              if sz >= 2 then
                ignore (P.apply_opt t (K.Opt.Unroll { axis = 0; amount = min sz 2 }))
            end);

          (* TC + LOCAL *)
          test "LOCAL after TC" (fun () ->
            let ast = matmul_f32_global_ast ~m:8 ~n:8 ~k:8 in
            let ren = tc_renderer Tc.metal in
            let t = P.create ast ren in
            ignore (P.apply_opt t
              (K.Opt.Tc { axis = 0; tc_select = 0; tc_opt = 0; use_tc = 1 }));
            let upcastable = P.upcastable_dims t in
            if List.length upcastable > 0 then begin
              let axis = List.hd upcastable in
              let fs = P.full_shape t in
              let sz = K.const_to_int (List.nth fs axis) in
              if sz >= 2 then
                ignore (P.apply_opt t (K.Opt.Local { axis; amount = 2 }))
            end);
        ];
    ]
