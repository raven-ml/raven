(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Unit tests for Tc helper module and apply_tc_opt in Postrange.

   Tests hardware fragment tables, coordinate relabeling, and the
   apply_tc_opt optimization path. *)

open Windtrap
open Tolk
open Tolk_uop
module U = Uop
module D = Dtype
module Ak = Axis_type
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
    ("metal", Tc.metal) ]

let idx n = U.const_int n

let global_fptr = D.float32
let global_f16ptr = D.float16

let kernel_info ?(opts_to_apply = None) () =
  { U.name = "test";
    applied_opts = [];
    opts_to_apply;
    estimates = None;
    beam = 0 }

let wrap_sink ?opts_to_apply srcs =
  U.sink ~kernel_info:(kernel_info ?opts_to_apply ()) srcs

let loop_range ~axis size =
  U.range ~size:(idx size) ~axis ~kind:Ak.Weak ~dtype:D.weakint ()

let reduce_range ~axis size =
  U.range ~size:(idx size) ~axis ~kind:Ak.Reduce ~dtype:D.weakint ()

let global_range ~axis size =
  U.range ~size:(idx size) ~axis ~kind:Ak.Global ~dtype:D.weakint ()

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
  let p_out = U.param ~slot:0 ~dtype:(global_fptr) () in
  let p_a = U.param ~slot:1 ~dtype:(global_fptr) () in
  let p_b = U.param ~slot:2 ~dtype:(global_fptr) () in
  let r_m = loop_range ~axis:0 m in
  let r_n = loop_range ~axis:1 n in
  let r_k = reduce_range ~axis:2 k in
  let open U.O in
  let idx_a = U.index ~ptr:p_a ~idxs:[((r_m * idx k) + r_k)] () in
  let idx_b = U.index ~ptr:p_b ~idxs:[((r_k * idx n) + r_n)] () in
  let ld_a = U.load ~src:idx_a () in
  let ld_b = U.load ~src:idx_b () in
  let mul = U.alu_binary ~op:Ops.Mul ~lhs:ld_a ~rhs:ld_b in
  let red = U.reduce ~op:Ops.Add ~src:mul ~ranges:[ r_k ] in
  let out_idx =
    U.index ~ptr:p_out ~idxs:[((r_m * idx n) + r_n)] ()
  in
  let st = U.store ~dst:out_idx ~value:red () in
  let e = U.end_ ~value:st ~ranges:[ r_m; r_n ] in
  wrap_sink [ e ]

(* Matmul with global ranges (for TC which needs loop-to-global conversion) *)
let matmul_f32_global_ast ~m ~n ~k =
  let p_out = U.param ~slot:0 ~dtype:(global_fptr) () in
  let p_a = U.param ~slot:1 ~dtype:(global_fptr) () in
  let p_b = U.param ~slot:2 ~dtype:(global_fptr) () in
  let r_m = global_range ~axis:0 m in
  let r_n = global_range ~axis:1 n in
  let r_k = reduce_range ~axis:2 k in
  let open U.O in
  let idx_a = U.index ~ptr:p_a ~idxs:[((r_m * idx k) + r_k)] () in
  let idx_b = U.index ~ptr:p_b ~idxs:[((r_k * idx n) + r_n)] () in
  let ld_a = U.load ~src:idx_a () in
  let ld_b = U.load ~src:idx_b () in
  let mul = U.alu_binary ~op:Ops.Mul ~lhs:ld_a ~rhs:ld_b in
  let red = U.reduce ~op:Ops.Add ~src:mul ~ranges:[ r_k ] in
  let out_idx =
    U.index ~ptr:p_out ~idxs:[((r_m * idx n) + r_n)] ()
  in
  let st = U.store ~dst:out_idx ~value:red () in
  let e = U.end_ ~value:st ~ranges:[ r_m; r_n ] in
  wrap_sink [ e ]

(* out[j] = sum_i sum_k (a[i,k] * b[k,j]): a matmul whose M axis is reduced
   away rather than stored. The M range is a REDUCE, so it can be picked as a
   tensor-core X or Y axis — which a WMMA cannot express, since those axes
   index the accumulator tile. *)
let matmul_reduced_output_ast ~m ~n ~k =
  let p_out = U.param ~slot:0 ~dtype:global_fptr () in
  let p_a = U.param ~slot:1 ~dtype:global_fptr () in
  let p_b = U.param ~slot:2 ~dtype:global_fptr () in
  let r_m = reduce_range ~axis:0 m in
  let r_n = global_range ~axis:1 n in
  let r_k = reduce_range ~axis:2 k in
  let open U.O in
  let idx_a = U.index ~ptr:p_a ~idxs:[ (r_m * idx k) + r_k ] () in
  let idx_b = U.index ~ptr:p_b ~idxs:[ (r_k * idx n) + r_n ] () in
  let mul =
    U.alu_binary ~op:Ops.Mul
      ~lhs:(U.load ~src:idx_a ())
      ~rhs:(U.load ~src:idx_b ())
  in
  let red =
    U.reduce ~op:Ops.Add ~src:mul ~ranges:[ r_m; r_k ]
  in
  let st = U.store ~dst:(U.index ~ptr:p_out ~idxs:[ r_n ] ()) ~value:red () in
  wrap_sink [ U.end_ ~value:st ~ranges:[ r_n ] ]

(* Matmul with f16 inputs and f32 accumulation *)
let matmul_f16_global_ast ~m ~n ~k =
  let p_out = U.param ~slot:0 ~dtype:(global_fptr) () in
  let p_a = U.param ~slot:1 ~dtype:(global_f16ptr) () in
  let p_b = U.param ~slot:2 ~dtype:(global_f16ptr) () in
  let r_m = global_range ~axis:0 m in
  let r_n = global_range ~axis:1 n in
  let r_k = reduce_range ~axis:2 k in
  let open U.O in
  let idx_a = U.index ~ptr:p_a ~idxs:[((r_m * idx k) + r_k)] () in
  let idx_b = U.index ~ptr:p_b ~idxs:[((r_k * idx n) + r_n)] () in
  let ld_a = U.load ~src:idx_a () in
  let ld_b = U.load ~src:idx_b () in
  let mul = U.alu_binary ~op:Ops.Mul ~lhs:ld_a ~rhs:ld_b in
  let red = U.reduce ~op:Ops.Add ~src:mul ~ranges:[ r_k ] in
  let out_idx =
    U.index ~ptr:p_out ~idxs:[((r_m * idx n) + r_n)] ()
  in
  let st = U.store ~dst:out_idx ~value:red () in
  let e = U.end_ ~value:st ~ranges:[ r_m; r_n ] in
  wrap_sink [ e ]

(* Matmul over [dtype] loads widened to f32 before the multiply, the exact
   product of narrow operands, summed at [acc]. *)
let matmul_widened_global_ast ?(acc = D.float32) ~dtype ~m ~n ~k () =
  let p_out = U.param ~slot:0 ~dtype:acc () in
  let p_a = U.param ~slot:1 ~dtype () in
  let p_b = U.param ~slot:2 ~dtype () in
  let r_m = global_range ~axis:0 m in
  let r_n = global_range ~axis:1 n in
  let r_k = reduce_range ~axis:2 k in
  let open U.O in
  let widened p i =
    U.cast ~src:(U.load ~src:(U.index ~ptr:p ~idxs:[ i ] ()) ()) ~dtype:D.float32
  in
  let mul =
    U.alu_binary ~op:Ops.Mul
      ~lhs:(widened p_a ((r_m * idx k) + r_k))
      ~rhs:(widened p_b ((r_k * idx n) + r_n))
  in
  let src = if D.equal acc D.float32 then mul else U.cast ~src:mul ~dtype:acc in
  let red = U.reduce ~op:Ops.Add ~src ~ranges:[ r_k ] in
  let st =
    U.store ~dst:(U.index ~ptr:p_out ~idxs:[ (r_m * idx n) + r_n ] ()) ~value:red ()
  in
  wrap_sink [ U.end_ ~value:st ~ranges:[ r_m; r_n ] ]

let matmul_f16_symbolic_k_ast () =
  let n = U.variable ~name:"tc_k" ~min_val:1 ~max_val:2 () in
  let m = global_range ~axis:0 32 and col = global_range ~axis:1 32 in
  let k = U.range ~size:U.O.(n * idx 8) ~axis:2 ~kind:Ak.Reduce () in
  let a = U.param ~slot:0 ~dtype:D.float16 ~shape:(idx 512) () in
  let b = U.param ~slot:1 ~dtype:D.float16 ~shape:(idx 512) () in
  let out = U.param ~slot:2 ~dtype:D.float32 ~shape:(idx 1024) () in
  let a = U.index ~ptr:a ~idxs:[ U.O.(m * idx 16 + k) ] () in
  let b = U.index ~ptr:b ~idxs:[ U.O.(k * idx 32 + col) ] () in
  let product = U.cast ~src:(U.alu_binary ~op:Ops.Mul ~lhs:a ~rhs:b)
      ~dtype:D.float32 in
  let value = U.reduce ~op:Ops.Add ~src:product ~ranges:[ k ] in
  let dst = U.index ~ptr:out ~idxs:[ U.O.(m * idx 32 + col) ] () in
  wrap_sink [ U.end_ ~value:(U.store ~dst ~value ()) ~ranges:[ m; col ] ]

(* Simple elementwise kernel (no reduce — for testing TC rejection) *)
let elementwise_global_ast ~s0 ~s1 =
  let p0 = U.param ~slot:0 ~dtype:(global_fptr) () in
  let p1 = U.param ~slot:1 ~dtype:(global_fptr) () in
  let r0 = global_range ~axis:0 s0 in
  let r1 = global_range ~axis:1 s1 in
  let open U.O in
  let in_idx = U.index ~ptr:p1 ~idxs:[((r0 * idx s1) + r1)] () in
  let ld = U.load ~src:in_idx () in
  let value = U.alu_unary ~op:Ops.Exp2 ~src:ld in
  let out_idx =
    U.index ~ptr:p0 ~idxs:[((r0 * idx s1) + r1)] ()
  in
  let st = U.store ~dst:out_idx ~value () in
  let e = U.end_ ~value:st ~ranges:[ r0; r1 ] in
  wrap_sink [ e ]

(* Analysis Helpers *)

let raises_opt_error f =
  raises_match (function P.Opt_error _ -> true | _ -> false) f

let has_wmma ast =
  List.exists (fun n -> Option.is_some (U.as_wmma n)) (U.toposort ast)

let const_to_int u =
  match U.const_int_value u with
  | Some n -> n
  | None -> failwith "expected integer constant"

let create_like ?frag_a ?frag_b ?frag_c (tc : Tc.t) =
  Tc.create ~dtype_in:tc.dtype_in ~dtype_out:tc.dtype_out
    ~frag_a:(Option.value frag_a ~default:tc.frag_a)
    ~frag_b:(Option.value frag_b ~default:tc.frag_b)
    ~frag_c:(Option.value frag_c ~default:tc.frag_c)

let rejects_layout f =
  raises_match (function Invalid_argument _ -> true | _ -> false) (fun () -> ignore (f ()))

(* Tests *)

let () =
  exit (run __FILE__
    [
      (* Existing Tc helper tests *)

      group "create validation"
        (List.map (fun (name, tcs) ->
           test (Printf.sprintf "%s tables are constructed" name) (fun () ->
             is_true (List.length tcs > 0)))
         all_tables
         @ [
           test "rejects malformed coordinates" (fun () ->
             rejects_layout (fun () -> create_like (List.hd Tc.metal) ~frag_a:([ "bad" ], [])));
           test "rejects unequal lane counts" (fun () ->
             let tc = List.hd Tc.metal in
             rejects_layout (fun () -> create_like tc ~frag_a:(List.tl (fst tc.frag_a), snd tc.frag_a)));
           test "rejects missing own coordinates" (fun () ->
             let tc = List.hd Tc.metal in
             rejects_layout (fun () -> create_like tc ~frag_a:(fst tc.frag_a, [])));
           test "rejects duplicate coordinates" (fun () ->
             let tc = List.hd Tc.metal in
             rejects_layout (fun () -> create_like tc
               ~frag_a:([ "k1"; "m0"; "m0"; "k2"; "m2" ], snd tc.frag_a)));
           test "rejects foreign element bits" (fun () ->
             let tc = List.hd Tc.metal in
             rejects_layout (fun () -> create_like tc ~frag_a:(fst tc.frag_a, [ "n0" ])));
           test "rejects different input contraction permutations" (fun () ->
             let tc = List.hd Tc.metal in
             rejects_layout (fun () -> create_like tc
               ~frag_a:([ "k2"; "m0"; "m1"; "k1"; "m2" ], snd tc.frag_a)));
         ]);

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
        ];

      group "fragment coordinates"
        [
          test "CUDA tile bits and element slots match the reference" (fun () ->
            let tc = List.hd Tc.cuda_sm80 in
            equal (list string)
              [ "n0"; "n1"; "n2"; "m0"; "m1"; "m2"; "m3"; "k0"; "k1"; "k2"; "k3" ]
              (Tc.axis_coords tc);
            equal (list string) [ "m3"; "n0"; "k3"; "k2"; "k1"; "k0" ] (Tc.base_upcast_axes tc);
            equal (list (list (pair string string)))
              [ [ "k1", "n1"; "k2", "n2"; "m0", "m0"; "m1", "m1"; "m2", "m2";
                  "k0", "k3"; "m3", "n0"; "k3", "m3" ];
                [ "k1", "n1"; "k2", "n2"; "n0", "m0"; "n1", "m1"; "n2", "m2";
                  "k0", "n0"; "k3", "m3" ] ] (Tc.relabel tc));
          test "Metal tile bits map to SIMD fragment slots" (fun () ->
            let tc = List.hd Tc.metal in
            equal (list string) [ "n0"; "k2"; "k1"; "k0" ] (Tc.base_upcast_axes tc);
            equal (list (list (pair string string)))
              [ [ "k1", "n1"; "m0", "m0"; "m1", "m1"; "k2", "n2"; "m2", "m2"; "k0", "n0" ];
                [ "n1", "n1"; "k0", "m0"; "k1", "m1"; "n2", "n2"; "k2", "m2"; "n0", "n0" ] ]
              (Tc.relabel tc));
          test "CDNA K128 places the high contraction bit in an element slot" (fun () ->
            let tc = List.hd Tc.amd_cdna4 in
            equal (pair (list string) (list string))
              ([ "m0"; "m1"; "m2"; "m3"; "k4"; "k5" ], [ "k0"; "k1"; "k2"; "k3"; "k6" ])
              tc.frag_a;
            equal (list string) [ "m1"; "m0"; "k6"; "k5"; "k4"; "k3"; "k2"; "k1"; "k0" ]
              (Tc.base_upcast_axes tc));
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
        ];

      group "tinygrad table names"
        [
          test "cuda_sm75 names match tinygrad" (fun () ->
            equal (list string)
              [
                "WMMA_8_16_8_half_float";
                "WMMA_8_16_8_half_half";
              ]
              (List.map Tc.to_string Tc.cuda_sm75));

          test "cuda_sm80 names match tinygrad" (fun () ->
            equal (list string)
              [
                "WMMA_8_16_16_half_float";
                "WMMA_8_16_16___bf16_float";
                "WMMA_8_16_16_half_half";
                "WMMA_8_16_8_half_float";
                "WMMA_8_16_8_half_half";
                "WMMA_8_16_8_float_float";
              ]
              (List.map Tc.to_string Tc.cuda_sm80));

          test "cuda_sm89 names match tinygrad" (fun () ->
            equal (list string)
              [
                "WMMA_8_16_16_half_float";
                "WMMA_8_16_16___bf16_float";
                "WMMA_8_16_16_half_half";
                "WMMA_8_16_8_half_float";
                "WMMA_8_16_8_half_half";
                "WMMA_8_16_8_float_float";
                "WMMA_8_16_32_float8_e4m3_float";
                "WMMA_8_16_32_float8_e5m2_float";
              ]
              (List.map Tc.to_string Tc.cuda_sm89));

          test "amd_rdna names match tinygrad" (fun () ->
            equal (list string)
              [
                "WMMA_16_16_16_half_float";
                "WMMA_16_16_16_half_half";
                "WMMA_16_16_16___bf16_float";
                "WMMA_16_16_16_signed_char_int";
              ]
              (List.map Tc.to_string Tc.amd_rdna3);
            equal (list string)
              [
                "WMMA_16_16_16_half_float";
                "WMMA_16_16_16_half_half";
                "WMMA_16_16_16___bf16_float";
                "WMMA_16_16_16___bf16___bf16";
              ]
              (List.map Tc.to_string Tc.amd_rdna4));

          test "amd_cdna names match tinygrad" (fun () ->
            equal (list string)
              [
                "WMMA_16_16_32_float8_e5m2fnuz_float";
                "WMMA_16_16_32_float8_e4m3fnuz_float";
                "WMMA_16_16_16_half_float";
                "WMMA_16_16_16___bf16_float";
              ]
              (List.map Tc.to_string Tc.amd_cdna3);
            equal (list string)
              [
                "WMMA_16_16_128_float8_e5m2_float";
                "WMMA_16_16_128_float8_e4m3_float";
                "WMMA_16_16_32_float8_e5m2_float";
                "WMMA_16_16_32_float8_e4m3_float";
                "WMMA_16_16_32_half_float";
                "WMMA_16_16_32___bf16_float";
                "WMMA_16_16_16_half_float";
                "WMMA_16_16_16___bf16_float";
              ]
              (List.map Tc.to_string Tc.amd_cdna4));

          test "metal names match tinygrad" (fun () ->
            equal (list string)
              [
                "WMMA_8_8_8_float_float";
                "WMMA_8_8_8_half_float";
                "WMMA_8_8_8_half_half";
                "WMMA_8_8_8___bf16_float";
                "WMMA_8_8_8___bf16___bf16";
              ]
              (List.map Tc.to_string Tc.metal));
        ];

      group "apply_tc_opt validation"
        [
          test "TC must be first opt" (fun () ->
            let ast = matmul_f32_global_ast ~m:8 ~n:8 ~k:8 in
            let ren = tc_renderer Tc.metal in
            let t = P.create ast ren in
            ignore (P.apply_opt t (U.Opt.Split { kind = Axis_type.Upcast; top = false; axis = 0; amount = 2 }));
            raises_opt_error (fun () ->
              ignore (P.apply_opt t
                (U.Opt.Tc { axis = 0; tc_select = -1; tc_opt = 0; use_tc = 1 }))));

          test "TC invalid tc_select rejected" (fun () ->
            let ast = matmul_f32_global_ast ~m:8 ~n:8 ~k:8 in
            let ren = tc_renderer Tc.metal in
            let t = P.create ast ren in
            raises_opt_error (fun () ->
              ignore (P.apply_opt t
                (U.Opt.Tc { axis = 0; tc_select = 99; tc_opt = 0; use_tc = 1 }))));

          test "TC invalid tc_opt rejected" (fun () ->
            let ast = matmul_f32_global_ast ~m:8 ~n:8 ~k:8 in
            let ren = tc_renderer Tc.metal in
            let t = P.create ast ren in
            raises_opt_error (fun () ->
              ignore (P.apply_opt t
                (U.Opt.Tc { axis = 0; tc_select = -1; tc_opt = 3; use_tc = 1 }))));

          test "TC use_tc=0 rejected" (fun () ->
            let ast = matmul_f32_global_ast ~m:8 ~n:8 ~k:8 in
            let ren = tc_renderer Tc.metal in
            let t = P.create ast ren in
            raises_opt_error (fun () ->
              ignore (P.apply_opt t
                (U.Opt.Tc { axis = 0; tc_select = -1; tc_opt = 0; use_tc = 0 }))));

          test "TC use_tc=3 rejected" (fun () ->
            let ast = matmul_f32_global_ast ~m:8 ~n:8 ~k:8 in
            let ren = tc_renderer Tc.metal in
            let t = P.create ast ren in
            raises_opt_error (fun () ->
              ignore (P.apply_opt t
                (U.Opt.Tc { axis = 0; tc_select = -1; tc_opt = 0; use_tc = 3 }))));

          test "TC on elementwise kernel rejected" (fun () ->
            let ast = elementwise_global_ast ~s0:8 ~s1:8 in
            let ren = tc_renderer Tc.metal in
            let t = P.create ast ren in
            raises_opt_error (fun () ->
              ignore (P.apply_opt t
                (U.Opt.Tc { axis = 0; tc_select = -1; tc_opt = 0; use_tc = 1 }))));

          (* dtype mismatch: f32 matmul but TC only supports f16 *)
          test "TC dtype mismatch rejected" (fun () ->
            let ast = matmul_f32_global_ast ~m:16 ~n:8 ~k:8 in
            let ren = tc_renderer Tc.cuda_sm75 in
            let t = P.create ast ren in
            raises_opt_error (fun () ->
              ignore (P.apply_opt t
                (U.Opt.Tc { axis = 0; tc_select = -1; tc_opt = 0; use_tc = 1 }))));

          test "TC over a reduced output axis rejected" (fun () ->
            let tc = List.hd Tc.metal in
            let m, n, k = tc.Tc.dims in
            let ast = matmul_reduced_output_ast ~m:(m * 2) ~n ~k in
            let ren = tc_renderer Tc.metal in
            let t = P.create ast ren in
            raises_opt_error (fun () ->
              ignore (P.apply_opt t
                (U.Opt.Tc { axis = 0; tc_select = -1; tc_opt = 0; use_tc = 1 }))));

          test "TC rejects an output axis contracted through local threads" (fun () ->
            let ast = matmul_reduced_output_ast ~m:16 ~n:8 ~k:8 in
            let ast = U.graph_rewrite (fun node -> match U.as_range node with
                | Some { axis = 0; sub; _ } -> Some (U.replace node
                    ~arg:(U.Arg.Range_info { axis = 0; sub; kind = Ak.Local }) ())
                | _ -> None) ast in
            let t = P.create ast (tc_renderer Tc.metal) in
            raises_opt_error (fun () -> ignore (P.apply_opt t
              (U.Opt.Tc { axis = 0; tc_select = -1; tc_opt = 0; use_tc = 1 }))));

          test "reduction can be split into locals after TC" (fun () ->
            let ast = matmul_f32_global_ast ~m:16 ~n:16 ~k:32 in
            let t = P.create ast (tc_renderer Tc.metal) in
            ignore (P.apply_opt t
              (U.Opt.Tc { axis = 0; tc_select = 0; tc_opt = 0; use_tc = 2 }));
            let axis = List.hd (P.axes_of t [ Axis_type.Reduce ]) in
            ignore (P.apply_opt t (U.Opt.Split
              { kind = Axis_type.Local; top = true; axis; amount = 2 })));

        ];

      (* Apply_tc_opt triggering *)

      group "apply_tc_opt triggering"
        [
          test "TC retries smaller tiles after symbolic split rejection" (fun () ->
            let ast = matmul_f16_symbolic_k_ast () in
            let renderer = tc_renderer Tc.cuda_sm80 in
            let selected = P.create ast renderer and explicit = P.create ast renderer in
            let apply t tc_select = ignore (P.apply_opt t
                (U.Opt.Tc { axis = 0; tc_select; tc_opt = 0; use_tc = 2 })) in
            apply selected (-1);
            let tc = Option.get (P.tensor_core selected) in
            equal (triple int int int) (8, 16, 8) tc.dims;
            apply explicit 3;
            equal string (U.semantic_key (P.ast explicit))
              (U.semantic_key (P.ast selected)));
          test "TC split rejection restores state before another action" (fun () ->
            let ast = matmul_f16_symbolic_k_ast () in
            let renderer = tc_renderer Tc.cuda_sm80 in
            let retried = P.create ast renderer and fresh = P.create ast renderer in
            let apply t tc_select = ignore (P.apply_opt t
                (U.Opt.Tc { axis = 0; tc_select; tc_opt = 0; use_tc = 2 })) in
            raises_opt_error (fun () -> apply retried 0);
            is_true ~msg:"failed tile preserves the original AST" (U.equal ast (P.ast retried));
            equal (list string) [] (List.map U.Opt.to_string (P.applied_opts retried));
            is_true ~msg:"failed tile does not select a tensor core" (P.tensor_core retried = None);
            apply retried 3;
            apply fresh 3;
            equal string (U.semantic_key (P.ast fresh))
              (U.semantic_key (P.ast retried)));
          (* use_tc=2 tests TC matching and shift_to without WMMA construction *)
          test "TC triggers on f32 8x8x8 matmul with metal tc (use_tc=2)" (fun () ->
            let ast = matmul_f32_global_ast ~m:8 ~n:8 ~k:8 in
            let ren = tc_renderer Tc.metal in
            let t = P.create ast ren in
            let result =
              P.apply_opt t
                (U.Opt.Tc { axis = 0; tc_select = 0; tc_opt = 0; use_tc = 2 })
            in
            is_true (result <> None);
            is_true
              (List.exists
                 (function U.Opt.Tc _ -> true | _ -> false)
                 (P.applied_opts t)));

          test "TC auto-selects with tc_select=-1 (use_tc=2)" (fun () ->
            let ast = matmul_f32_global_ast ~m:8 ~n:8 ~k:8 in
            let ren = tc_renderer Tc.metal in
            let t = P.create ast ren in
            let result =
              P.apply_opt t
                (U.Opt.Tc { axis = 0; tc_select = -1; tc_opt = 0; use_tc = 2 })
            in
            is_true (result <> None));

          test "TC triggers on f16 matmul with cuda sm80 tc (use_tc=2)" (fun () ->
            let ast = matmul_f16_global_ast ~m:16 ~n:16 ~k:16 in
            let ren = tc_renderer Tc.cuda_sm80 in
            let t = P.create ast ren in
            let result =
              P.apply_opt t
                (U.Opt.Tc { axis = 0; tc_select = -1; tc_opt = 0; use_tc = 2 })
            in
            is_true (result <> None));
        ];

      group "apply_tc_opt widened operands"
        [
          test "a narrow-in, f32-out core multiplies the narrow loads" (fun () ->
            List.iter (fun dtype ->
                let ast = matmul_widened_global_ast ~dtype ~m:16 ~n:16 ~k:16 () in
                let t = P.create ast (tc_renderer Tc.cuda_sm80) in
                ignore (P.apply_opt t
                  (U.Opt.Tc { axis = 0; tc_select = -1; tc_opt = 0; use_tc = 1 }));
                match List.find_map U.as_wmma (U.toposort (P.ast t)) with
                | None -> is_true ~msg:"a WMMA node" false
                | Some w ->
                    let msg = D.to_string dtype in
                    is_true ~msg (D.equal w.info.dtype_in dtype);
                    is_true ~msg (D.equal (U.dtype w.a) dtype);
                    is_true ~msg (D.equal (U.dtype w.b) dtype))
              [ D.float16; D.bfloat16 ]);

          (* Summed at f16, the widened product matches the half/half core's
             output dtype, whose products round. *)
          test "a core with a narrow output takes no widened loads" (fun () ->
            let ast =
              matmul_widened_global_ast ~acc:D.float16 ~dtype:D.float16 ~m:8
                ~n:8 ~k:8 ()
            in
            let t = P.create ast (tc_renderer Tc.metal) in
            raises_opt_error (fun () ->
              ignore (P.apply_opt t
                (U.Opt.Tc { axis = 0; tc_select = 2; tc_opt = 0; use_tc = 1 }))));
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
                (U.Opt.Tc { axis = 0; tc_select = 0; tc_opt = 2; use_tc = 2 })
            in
            is_true (result <> None));

          (* tc_opt=0 on unaligned dims should fail *)
          test "TC padding rejected with tc_opt=0" (fun () ->
            let ast = matmul_f32_global_ast ~m:9 ~n:9 ~k:9 in
            let ren = tc_renderer Tc.metal in
            let t = P.create ast ren in
            raises_opt_error (fun () ->
              ignore (P.apply_opt t
                (U.Opt.Tc { axis = 0; tc_select = 0; tc_opt = 0; use_tc = 1 }))));

          test "failed TC leaves scheduler unchanged" (fun () ->
            let ast = matmul_f32_global_ast ~m:9 ~n:9 ~k:9 in
            let ren = tc_renderer Tc.metal in
            let t = P.create ast ren in
            raises_opt_error (fun () ->
              ignore (P.apply_opt t
                (U.Opt.Tc { axis = 0; tc_select = 0; tc_opt = 0; use_tc = 1 })));
            is_true (P.ast t == ast);
            equal (list string) [] (List.map U.Opt.to_string (P.applied_opts t));
            is_true (P.tensor_core t = None));

          (* tc_opt=1 on unaligned dims should also fail *)
          test "TC padding rejected with tc_opt=1" (fun () ->
            let ast = matmul_f32_global_ast ~m:9 ~n:9 ~k:9 in
            let ren = tc_renderer Tc.metal in
            let t = P.create ast ren in
            raises_opt_error (fun () ->
              ignore (P.apply_opt t
                (U.Opt.Tc { axis = 0; tc_select = 0; tc_opt = 1; use_tc = 1 }))));

          (* Excessive padding: dims/4 *)
          test "TC excessive padding rejected (dims/4)" (fun () ->
            let ast = matmul_f32_global_ast ~m:2 ~n:2 ~k:2 in
            let ren = tc_renderer Tc.metal in
            let t = P.create ast ren in
            raises_opt_error (fun () ->
              ignore (P.apply_opt t
                (U.Opt.Tc { axis = 0; tc_select = 0; tc_opt = 2; use_tc = 1 }))));
        ];

      (* Apply_tc_opt WMMA construction *)

      group "apply_tc_opt WMMA construction"
        [
          (* use_tc=2 applies shifts but skips WMMA construction *)
          test "TC with use_tc=2 skips WMMA construction" (fun () ->
            let ast = matmul_f32_global_ast ~m:8 ~n:8 ~k:8 in
            let ren = tc_renderer Tc.metal in
            let t = P.create ast ren in
            let result =
              P.apply_opt t
                (U.Opt.Tc { axis = 0; tc_select = 0; tc_opt = 0; use_tc = 2 })
            in
            is_true (result <> None);
            is_true (not (has_wmma (P.ast t))));

          (* use_tc=2 records the TC opt in applied_opts *)
          test "TC records opt in applied_opts (use_tc=2)" (fun () ->
            let ast = matmul_f32_global_ast ~m:8 ~n:8 ~k:8 in
            let ren = tc_renderer Tc.metal in
            let t = P.create ast ren in
            ignore (P.apply_opt t
              (U.Opt.Tc { axis = 0; tc_select = 0; tc_opt = 0; use_tc = 2 }));
            is_true
              (List.exists
                 (function U.Opt.Tc _ -> true | _ -> false)
                 (P.applied_opts t)));

          test "TC produces direct WMMA node in AST (metal use_tc=1)" (fun () ->
            let ast = matmul_f32_global_ast ~m:8 ~n:8 ~k:8 in
            let ren = tc_renderer Tc.metal in
            let t = P.create ast ren in
            let result =
              P.apply_opt t
                (U.Opt.Tc { axis = 0; tc_select = 0; tc_opt = 0; use_tc = 1 })
            in
            is_true (Option.is_some result);
            is_true (has_wmma (P.ast t)));
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
              (U.Opt.Tc { axis = 0; tc_select = 0; tc_opt = 0; use_tc = 1 }));
            let upcastable = P.upcastable_dims t in
            is_true (List.length upcastable > 0);
            let axis = List.hd upcastable in
            let fs = P.full_shape t in
            let sz = const_to_int (List.nth fs axis) in
            if sz >= 2 then
              ignore (P.apply_opt t (U.Opt.Split { kind = Axis_type.Upcast; top = false; axis; amount = 2 })));

          (* TC + UNROLL *)
          test "UNROLL after TC" (fun () ->
            let ast = matmul_f32_global_ast ~m:8 ~n:8 ~k:8 in
            let ren = tc_renderer Tc.metal in
            let t = P.create ast ren in
            ignore (P.apply_opt t
              (U.Opt.Tc { axis = 0; tc_select = 0; tc_opt = 0; use_tc = 1 }));
            let unroll_dims = P.unrollable_dims t in
            if List.length unroll_dims > 0 then begin
              let fs = P.full_shape t in
              let axis_idx = List.hd unroll_dims in
              let sz = const_to_int (List.nth fs axis_idx) in
              if sz >= 2 then
                ignore (P.apply_opt t (U.Opt.Split { kind = Axis_type.Unroll; top = false; axis = axis_idx; amount = min sz 2 }))
            end);

          (* TC + LOCAL *)
          test "LOCAL after TC" (fun () ->
            let ast = matmul_f32_global_ast ~m:8 ~n:8 ~k:8 in
            let ren = tc_renderer Tc.metal in
            let t = P.create ast ren in
            ignore (P.apply_opt t
              (U.Opt.Tc { axis = 0; tc_select = 0; tc_opt = 0; use_tc = 1 }));
            let upcastable = P.upcastable_dims t in
            if List.length upcastable > 0 then begin
              let axis = List.hd upcastable in
              let fs = P.full_shape t in
              let sz = const_to_int (List.nth fs axis) in
              if sz >= 2 then
                ignore (P.apply_opt t (U.Opt.Split { kind = Axis_type.Local; top = false; axis; amount = 2 }))
            end);
        ];
    ])
