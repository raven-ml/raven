(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Unit and integration tests for Heuristic.hand_coded_optimizations.

   Tests the 11-step heuristic optimization sequence. These tests verify
   structural decision logic by checking which opts are applied for specific
   kernel shapes and renderers. *)

open Windtrap
open Tolk
open Tolk_ir
module K = Kernel
module D = Dtype
module C = Const
module Ak = Axis_kind
module P = Postrange

(* Helpers *)

let idx n = K.const (C.int D.Val.index n)
let global_fptr = D.Ptr.create D.Val.float32 ~addrspace:Global ~size:(-1)

let kernel_info () =
  { K.name = "test";
    axis_kinds = [];
    dont_use_locals = false;
    applied_opts = [];
    opts_to_apply = None;
    estimates = None;
  }

let wrap_sink srcs = K.sink ~kernel_info:(kernel_info ()) srcs

let loop_range ~axis size =
  K.range ~size:(idx size) ~axis ~kind:Ak.Loop ~dtype:D.Val.index ()

let reduce_range ~axis size =
  K.range ~size:(idx size) ~axis ~kind:Ak.Reduce ~dtype:D.Val.index ()

let global_range ~axis size =
  K.range ~size:(idx size) ~axis ~kind:Ak.Global ~dtype:D.Val.index ()

(* Renderers *)

let gpu_renderer () =
  Renderer.make ~name:"test" ~device:"TEST" ~has_local:true ~has_shared:true
    ~shared_max:32768 ~render:(fun ?name:_ _ -> "") ()

let cpu_renderer () =
  Renderer.make ~name:"cpu" ~device:"CPU" ~has_local:false ~has_shared:false
    ~shared_max:0 ~render:(fun ?name:_ _ -> "") ()

let thread_renderer () =
  Renderer.make ~name:"thread" ~device:"CPU" ~has_local:false ~has_shared:false
    ~shared_max:0 ~has_threads:true ~global_max:[ 32; 32; 32 ]
    ~render:(fun ?name:_ _ -> "") ()

(* Opt Inspection Helpers *)

let run_heuristic ast ren =
  let t = P.create ast ren in
  let result = Heuristic.hand_coded_optimizations t in
  P.applied_opts result

let run_heuristic_scheduler ast ren =
  let t = P.create ast ren in
  Heuristic.hand_coded_optimizations t

let is_grouptop = function K.Opt.Grouptop _ -> true | _ -> false
let is_upcast = function K.Opt.Upcast _ -> true | _ -> false
let is_unroll = function K.Opt.Unroll _ -> true | _ -> false
let is_local = function K.Opt.Local _ -> true | _ -> false
let is_thread = function K.Opt.Thread _ -> true | _ -> false
let is_group = function K.Opt.Group _ -> true | _ -> false
let is_nolocals = function K.Opt.Nolocals -> true | _ -> false

let thread_axis = function K.Opt.Thread { axis; _ } -> Some axis | _ -> None
let local_axis = function K.Opt.Local { axis; _ } -> Some axis | _ -> None

let count pred opts = List.length (List.filter pred opts)
let has pred opts = List.exists pred opts

(* Env var helper: sets var, runs f, restores original. *)
let with_env name value f =
  let old = Sys.getenv_opt name in
  Unix.putenv name value;
  Fun.protect
    ~finally:(fun () ->
      match old with
      | Some v -> Unix.putenv name v
      | None ->
          (* Can't truly unsetenv in OCaml; empty string causes
             getenv_int to return the default via int_of_string failure. *)
          Unix.putenv name "")
    f

(* AST Fixtures *)

(* Elementwise: out[i,j] = exp2(in[i,j]) — Global ranges for GPU tests *)
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

(* Elementwise with Loop ranges — for thread renderer tests *)
let elementwise_loop_ast ~s0 ~s1 =
  let p0 = K.param ~idx:0 ~dtype:global_fptr in
  let p1 = K.param ~idx:1 ~dtype:global_fptr in
  let r0 = loop_range ~axis:0 s0 in
  let r1 = loop_range ~axis:1 s1 in
  let open K.O in
  let in_idx = K.index ~ptr:p1 ~idxs:[ r0 * idx s1 + r1 ] () in
  let ld = K.load ~src:in_idx () in
  let value = K.unary ~op:`Exp2 ~src:ld in
  let out_idx = K.index ~ptr:p0 ~idxs:[ r0 * idx s1 + r1 ] () in
  let st = K.store ~dst:out_idx ~value ~ranges:[] in
  let e = K.end_ ~value:st ~ranges:[ r0; r1 ] () in
  wrap_sink [ e ]

(* Reduce: out[i,j] = sum_k(in[i,k,j]) — Global output + Reduce *)
let reduce_global_ast ~s0 ~s1 ~sr =
  let p0 = K.param ~idx:0 ~dtype:global_fptr in
  let p1 = K.param ~idx:1 ~dtype:global_fptr in
  let r0 = global_range ~axis:0 s0 in
  let r1 = global_range ~axis:1 s1 in
  let rr = reduce_range ~axis:2 sr in
  let open K.O in
  let in_idx =
    K.index ~ptr:p1 ~idxs:[ r0 * idx sr * idx s1 + rr * idx s1 + r1 ] ()
  in
  let ld = K.load ~src:in_idx () in
  let red = K.reduce ~op:`Add ~src:ld ~ranges:[ rr ] ~dtype:D.Val.float32 in
  let out_idx = K.index ~ptr:p0 ~idxs:[ r0 * idx s1 + r1 ] () in
  let st = K.store ~dst:out_idx ~value:red ~ranges:[] in
  let e = K.end_ ~value:st ~ranges:[ r0; r1 ] () in
  wrap_sink [ e ]

(* Double reduce: out[i,j] = sum_{k1,k2}(in[i,j,k1,k2])
   Two reduce ranges for testing double unroll. *)
let double_reduce_global_ast ~s0 ~s1 ~sr1 ~sr2 =
  let p0 = K.param ~idx:0 ~dtype:global_fptr in
  let p1 = K.param ~idx:1 ~dtype:global_fptr in
  let r0 = global_range ~axis:0 s0 in
  let r1 = global_range ~axis:1 s1 in
  let rr1 = reduce_range ~axis:2 sr1 in
  let rr2 = reduce_range ~axis:3 sr2 in
  let open K.O in
  let in_idx =
    K.index ~ptr:p1
      ~idxs:
        [ r0 * idx (Stdlib.( * ) s1 (Stdlib.( * ) sr1 sr2))
          + r1 * idx (Stdlib.( * ) sr1 sr2)
          + rr1 * idx sr2
          + rr2
        ]
      ()
  in
  let ld = K.load ~src:in_idx () in
  let red =
    K.reduce ~op:`Add ~src:ld ~ranges:[ rr1; rr2 ] ~dtype:D.Val.float32
  in
  let out_idx = K.index ~ptr:p0 ~idxs:[ r0 * idx s1 + r1 ] () in
  let st = K.store ~dst:out_idx ~value:red ~ranges:[] in
  let e = K.end_ ~value:st ~ranges:[ r0; r1 ] () in
  wrap_sink [ e ]

(* Matmul: out[m,n] = sum_k(a[m,k] * b[k,n]) *)
let matmul_global_ast ~m ~n ~k =
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
  let red = K.reduce ~op:`Add ~src:mul ~ranges:[ r_k ] ~dtype:D.Val.float32 in
  let out_idx = K.index ~ptr:p_out ~idxs:[ r_m * idx n + r_n ] () in
  let st = K.store ~dst:out_idx ~value:red ~ranges:[] in
  let e = K.end_ ~value:st ~ranges:[ r_m; r_n ] () in
  wrap_sink [ e ]

(* Matvec: out[i] = sum_j(x[j] * A[i,j])

   NOTE: MUL operands are INDEX nodes directly (no LOAD). The heuristic's
   matvec detection checks for INDEX as MUL operands. In the standard Tolk IR,
   MUL operands are LOADs wrapping INDEXes. This fixture tests the matvec
   decision logic assuming the expected pattern. *)
let matvec_global_ast ~rows ~cols =
  let p_out = K.param ~idx:0 ~dtype:global_fptr in
  let p_x = K.param ~idx:1 ~dtype:global_fptr in
  let p_a = K.param ~idx:2 ~dtype:global_fptr in
  let r_i = global_range ~axis:0 rows in
  let r_j = reduce_range ~axis:1 cols in
  let open K.O in
  let idx_x = K.index ~ptr:p_x ~idxs:[ r_j ] () in
  let idx_a = K.index ~ptr:p_a ~idxs:[ r_i * idx cols + r_j ] () in
  let mul = K.binary ~op:`Mul ~lhs:idx_x ~rhs:idx_a in
  let red = K.reduce ~op:`Add ~src:mul ~ranges:[ r_j ] ~dtype:D.Val.float32 in
  let out_idx = K.index ~ptr:p_out ~idxs:[ r_i ] () in
  let st = K.store ~dst:out_idx ~value:red ~ranges:[] in
  let e = K.end_ ~value:st ~ranges:[ r_i ] () in
  wrap_sink [ e ]

(* Broadcast elementwise: out[i,j] = a[j] + b[i]
   Buffer a indexed by j only (broadcast on i), b indexed by i only
   (broadcast on j). Triggers heuristic upcast stride analysis. *)
let broadcast_ewise_global_ast ~s0 ~s1 =
  let p_out = K.param ~idx:0 ~dtype:global_fptr in
  let p_a = K.param ~idx:1 ~dtype:global_fptr in
  let p_b = K.param ~idx:2 ~dtype:global_fptr in
  let r0 = global_range ~axis:0 s0 in
  let r1 = global_range ~axis:1 s1 in
  let open K.O in
  let idx_a = K.index ~ptr:p_a ~idxs:[ r1 ] () in
  let ld_a = K.load ~src:idx_a () in
  let idx_b = K.index ~ptr:p_b ~idxs:[ r0 ] () in
  let ld_b = K.load ~src:idx_b () in
  let value = K.binary ~op:`Add ~lhs:ld_a ~rhs:ld_b in
  let out_idx = K.index ~ptr:p_out ~idxs:[ r0 * idx s1 + r1 ] () in
  let st = K.store ~dst:out_idx ~value ~ranges:[] in
  let e = K.end_ ~value:st ~ranges:[ r0; r1 ] () in
  wrap_sink [ e ]

(* Masked elementwise: out[i,j] = where(j < (s1-1), in[i,j], 0.0)
   Range j appears in WHERE condition → triggers masked upcast (step 6). *)
let masked_ewise_global_ast ~s0 ~s1 =
  let p0 = K.param ~idx:0 ~dtype:global_fptr in
  let p1 = K.param ~idx:1 ~dtype:global_fptr in
  let r0 = global_range ~axis:0 s0 in
  let r1 = global_range ~axis:1 s1 in
  let open K.O in
  let in_idx = K.index ~ptr:p1 ~idxs:[ r0 * idx s1 + r1 ] () in
  let ld = K.load ~src:in_idx () in
  let cond = K.binary ~op:`Cmplt ~lhs:r1 ~rhs:(idx (s1 - 1)) in
  let zero = K.const (C.float D.Val.float32 0.0) in
  let value = K.ternary ~op:`Where ~a:cond ~b:ld ~c:zero in
  let out_idx = K.index ~ptr:p0 ~idxs:[ r0 * idx s1 + r1 ] () in
  let st = K.store ~dst:out_idx ~value ~ranges:[] in
  let e = K.end_ ~value:st ~ranges:[ r0; r1 ] () in
  wrap_sink [ e ]

(* Partial broadcast: out[i,j] = a[i,j] + b[i]
   Buffer b indexed by r0 only → axis 1 is expand for b.
   Axis 0 is NOT expand (all bufs use r0). Tests local expand priority. *)
let partial_broadcast_global_ast ~s0 ~s1 =
  let p_out = K.param ~idx:0 ~dtype:global_fptr in
  let p_a = K.param ~idx:1 ~dtype:global_fptr in
  let p_b = K.param ~idx:2 ~dtype:global_fptr in
  let r0 = global_range ~axis:0 s0 in
  let r1 = global_range ~axis:1 s1 in
  let open K.O in
  let idx_a = K.index ~ptr:p_a ~idxs:[ r0 * idx s1 + r1 ] () in
  let ld_a = K.load ~src:idx_a () in
  let idx_b = K.index ~ptr:p_b ~idxs:[ r0 ] () in
  let ld_b = K.load ~src:idx_b () in
  let value = K.binary ~op:`Add ~lhs:ld_a ~rhs:ld_b in
  let out_idx = K.index ~ptr:p_out ~idxs:[ r0 * idx s1 + r1 ] () in
  let st = K.store ~dst:out_idx ~value ~ranges:[] in
  let e = K.end_ ~value:st ~ranges:[ r0; r1 ] () in
  wrap_sink [ e ]

(* Tests *)

(* Group 1: Grouping *)

let grouping_tests =
  group "grouping"
    [
      (* Small upcastable product triggers GROUPTOP.
         prod(upcastable) = 4*4 = 16 ≤ 2048 → GROUPTOP applied. *)
      test "applies GROUPTOP when upcastable prod small" (fun () ->
          let ast = reduce_global_ast ~s0:4 ~s1:4 ~sr:128 in
          let ren = gpu_renderer () in
          let opts = run_heuristic ast ren in
          is_true (has is_grouptop opts));
      (* After GROUPTOP, group_for_reduces > 0 → early return.
         No UPCAST, LOCAL, or UNROLL should appear. *)
      test "early return after grouping" (fun () ->
          let ast = reduce_global_ast ~s0:4 ~s1:4 ~sr:128 in
          let ren = gpu_renderer () in
          let opts = run_heuristic ast ren in
          is_true (has is_grouptop opts);
          is_true (not (has is_upcast opts));
          is_true (not (has is_local opts));
          is_true (not (has is_unroll opts)));
      (* Large upcastable product skips grouping.
         prod(upcastable) = 64*64 = 4096 > 2048 → no GROUPTOP. *)
      test "skips grouping when upcastable prod large" (fun () ->
          let ast = reduce_global_ast ~s0:64 ~s1:64 ~sr:128 in
          let ren = gpu_renderer () in
          let opts = run_heuristic ast ren in
          is_true (not (has is_grouptop opts)));
    ]

(* Group 2: Reduce unroll *)

let reduce_unroll_tests =
  group "reduce unroll"
    [
      (* Full unroll when reduce size ≤ 32.
         reduce_size=16, upcastable_prod=4096>2048 → skips grouping.
         Unroll amount=0 (full). *)
      test "full unrolls small reduce" (fun () ->
          let ast = reduce_global_ast ~s0:64 ~s1:64 ~sr:16 in
          let ren = gpu_renderer () in
          let opts = run_heuristic ast ren in
          is_true (has is_unroll opts));
      (* Split unroll by 4 when reduce > 32 and divisible by 4.
         reduce_size=64 > 32. 64%4=0 → UNROLL amount=4. *)
      test "split unrolls large reduce by 4" (fun () ->
          let ast = reduce_global_ast ~s0:64 ~s1:64 ~sr:64 in
          let ren = gpu_renderer () in
          let opts = run_heuristic ast ren in
          let unrolls = List.filter is_unroll opts in
          is_true (unrolls <> []);
          (match unrolls with
          | K.Opt.Unroll { amount; _ } :: _ -> equal int 4 amount
          | _ -> failwith "expected Unroll"));
      (* Double unroll when both reduce dims ≤ 3.
         Two reduce axes of size 3 each. Both get fully unrolled. *)
      test "double unrolls tiny reduces" (fun () ->
          let ast =
            double_reduce_global_ast ~s0:64 ~s1:64 ~sr1:3 ~sr2:3
          in
          let ren = gpu_renderer () in
          let opts = run_heuristic ast ren in
          equal int 2 (count is_unroll opts));
    ]

(* Group 3: Default upcast *)

let default_upcast_tests =
  group "default upcast"
    [
      (* When nothing is upcasted, applies 4x upcast on last upcastable dim.
         Elementwise on CPU: no locals, no threads, no reduce →
         only step 9 fires. *)
      test "applies 4x upcast when nothing upcasted" (fun () ->
          let ast = elementwise_loop_ast ~s0:128 ~s1:128 in
          let ren = cpu_renderer () in
          let opts = run_heuristic ast ren in
          is_true (has is_upcast opts);
          let upcasts = List.filter is_upcast opts in
          (match upcasts with
          | [ K.Opt.Upcast { amount; _ } ] -> equal int 4 amount
          | _ -> failwith "expected exactly one Upcast with amount=4"));
    ]

(* Group 4: Heuristic upcast (broadcast) *)

let heuristic_upcast_tests =
  group "heuristic upcast"
    [
      (* Broadcast triggers upcast; lower stride axis chosen first.
         out[i,j] = a[j] + b[i]: axis=1 has stride sum 2 vs axis=0 sum 65.
         On CPU to avoid local interference. *)
      test "broadcast upcast prefers lower stride axis" (fun () ->
          let ast = broadcast_ewise_global_ast ~s0:64 ~s1:64 in
          let ren = cpu_renderer () in
          let opts = run_heuristic ast ren in
          let upcasts = List.filter is_upcast opts in
          is_true (List.length upcasts >= 1);
          (match upcasts with
          | K.Opt.Upcast { axis = 1; _ } :: _ -> ()
          | K.Opt.Upcast { axis; _ } :: _ ->
              failwith (Printf.sprintf "expected first upcast axis=1, got %d" axis)
          | _ -> failwith "no upcast found"));
      (* Upcast size stays under 32. *)
      test "upcast size bounded by 32" (fun () ->
          let ast = broadcast_ewise_global_ast ~s0:64 ~s1:64 in
          let ren = cpu_renderer () in
          let result = run_heuristic_scheduler ast ren in
          is_true (P.upcast_size result <= 32));
    ]

(* Group 5: Matvec detection *)

let matvec_tests =
  group "matvec"
    [
      (* Matvec pattern detected: GROUP + LOCAL + UPCAST applied. *)
      test "detects matvec and applies GROUP LOCAL UPCAST" (fun () ->
          let ast = matvec_global_ast ~rows:128 ~cols:128 in
          let ren = gpu_renderer () in
          let opts = run_heuristic ast ren in
          is_true (has is_group opts);
          is_true (has is_local opts);
          is_true (has is_upcast opts));
      (* Matvec early return: no further opts after matvec. *)
      test "matvec early return prevents further opts" (fun () ->
          let ast = matvec_global_ast ~rows:128 ~cols:128 in
          let ren = gpu_renderer () in
          let opts = run_heuristic ast ren in
          is_true (not (has is_grouptop opts));
          is_true (not (has is_unroll opts));
          is_true (not (has is_thread opts)));
      (* Matvec skipped on CPU (no local/shared). *)
      test "matvec skipped on CPU" (fun () ->
          let ast = matvec_global_ast ~rows:128 ~cols:128 in
          let ren = cpu_renderer () in
          let opts = run_heuristic ast ren in
          is_true (not (has is_group opts)));
    ]

(* Group 6: Masked upcast *)

let masked_upcast_tests =
  group "masked upcast"
    [
      (* Small dim with WHERE guard triggers masked upcast (step 6).
         s1=5 ≤ 7, WHERE condition references r1, prod=1*5=5 ≤ 49.
         s0=1024 makes prod(upcastable)=5120 > 2048, skipping grouping. *)
      test "upcasts small WHERE-guarded dim" (fun () ->
          let ast = masked_ewise_global_ast ~s0:1024 ~s1:5 in
          let ren = gpu_renderer () in
          let opts = run_heuristic ast ren in
          is_true (has is_upcast opts));
    ]

(* Group 7: Local groups *)

let local_groups_tests =
  group "local groups"
    [
      (* GPU renderer applies LOCAL opts on elementwise. *)
      test "applies locals on GPU" (fun () ->
          let ast = elementwise_global_ast ~s0:128 ~s1:128 in
          let ren = gpu_renderer () in
          let opts = run_heuristic ast ren in
          is_true (has is_local opts));
      (* At most 3 LOCAL opts applied (take_n 3 in heuristic). *)
      test "at most 3 locals" (fun () ->
          let p0 = K.param ~idx:0 ~dtype:global_fptr in
          let p1 = K.param ~idx:1 ~dtype:global_fptr in
          let r0 = global_range ~axis:0 8 in
          let r1 = global_range ~axis:1 8 in
          let r2 = global_range ~axis:2 8 in
          let r3 = global_range ~axis:3 8 in
          let r4 = global_range ~axis:4 8 in
          let open K.O in
          let flat =
            r0 * idx 4096
            + r1 * idx 512
            + r2 * idx 64
            + r3 * idx 8
            + r4
          in
          let in_idx = K.index ~ptr:p1 ~idxs:[ flat ] () in
          let ld = K.load ~src:in_idx () in
          let value = K.unary ~op:`Exp2 ~src:ld in
          let out_idx = K.index ~ptr:p0 ~idxs:[ flat ] () in
          let st = K.store ~dst:out_idx ~value ~ranges:[] in
          let e = K.end_ ~value:st ~ranges:[ r0; r1; r2; r3; r4 ] () in
          let ast = wrap_sink [ e ] in
          let ren = gpu_renderer () in
          let opts = run_heuristic ast ren in
          is_true (count is_local opts <= 3));
      (* Local budget: product of local sizes ≤ 128. *)
      test "local budget respected" (fun () ->
          let ast = elementwise_global_ast ~s0:128 ~s1:128 in
          let ren = gpu_renderer () in
          let result = run_heuristic_scheduler ast ren in
          let local_ranges = P.ranges_of result [ Ak.Local ] in
          let local_prod =
            List.fold_left
              (fun acc r ->
                let sz = K.range_size r in
                if K.is_const sz then acc * K.const_to_int sz else acc)
              1 local_ranges
          in
          is_true (local_prod <= 128));
      (* NOLOCALS=1 applies Nolocals opt instead of LOCAL. *)
      test "NOLOCALS env applies Nolocals" (fun () ->
          let ast = elementwise_global_ast ~s0:128 ~s1:128 in
          let ren = gpu_renderer () in
          let opts =
            Helpers.Context_var.with_context
              [ B (Heuristic.nolocals_var, 1) ]
              (fun () -> run_heuristic ast ren)
          in
          is_true (has is_nolocals opts);
          is_true (not (has is_local opts)));
      (* NOLOCALS=1 changes grouping threshold from 2048 to 240.
         prod(upcastable)=256: ≤2048 (groups) but >240 (no group). *)
      test "NOLOCALS adjusts grouping threshold" (fun () ->
          let ren = gpu_renderer () in
          let ast1 = reduce_global_ast ~s0:8 ~s1:32 ~sr:128 in
          let opts_default = run_heuristic ast1 ren in
          is_true (has is_grouptop opts_default);
          let ast2 = reduce_global_ast ~s0:8 ~s1:32 ~sr:128 in
          let opts_nolocals =
            Helpers.Context_var.with_context
              [ B (Heuristic.nolocals_var, 1) ]
              (fun () -> run_heuristic ast2 ren)
          in
          is_true (not (has is_grouptop opts_nolocals)));
      (* Expand axes are prioritized for LOCAL.
         out[i,j] = a[i,j] + b[i]: axis 1 is expand for buffer b.
         Expand axis is ranked first, getting larger local_sz from the
         budget. Without priority, axis 0 (non-expand) would take 32
         first, leaving only 4 for axis 1. With priority, axis 1 gets
         16, axis 0 gets 8. Application order is by axis index. *)
      test "expand axis gets larger LOCAL from budget" (fun () ->
          let ast = partial_broadcast_global_ast ~s0:128 ~s1:128 in
          let ren = gpu_renderer () in
          let opts = run_heuristic ast ren in
          let locals =
            List.filter_map
              (function
                | K.Opt.Local { axis; amount } -> Some (axis, amount)
                | _ -> None)
              opts
          in
          (* Both axes get LOCAL'd *)
          is_true (List.length locals >= 2);
          (* Expand axis 1 gets local_sz=16 (ranked first, more budget).
             Without priority it would only get 4. *)
          let axis1_amount =
            List.find_map
              (fun (a, amt) -> if a = 1 then Some amt else None)
              locals
          in
          equal (option int) (Some 16) axis1_amount);
      (* deleted_shape tracking: when local_sz equals axis size, the axis
         is deleted and subsequent axis indices are adjusted.
         s0=16, s1=128: after upcast by 4, shape=[16,32,4].
         Axis 0 gets local_sz=16 (will_delete). Axis 1 shifts to 0. *)
      test "deleted_shape adjusts axis indices" (fun () ->
          let ast = elementwise_global_ast ~s0:16 ~s1:128 in
          let ren = gpu_renderer () in
          let result = run_heuristic_scheduler ast ren in
          let opts = P.applied_opts result in
          (* Two locals applied without crash *)
          is_true (count is_local opts >= 2);
          (* Local product respects 128 budget *)
          let local_ranges = P.ranges_of result [ Ak.Local ] in
          let local_prod =
            List.fold_left
              (fun acc r ->
                let sz = K.range_size r in
                if K.is_const sz then acc * K.const_to_int sz else acc)
              1 local_ranges
          in
          is_true (local_prod <= 128));
    ]

(* Group 7: Threading *)

let threading_tests =
  group "threading"
    [
      (* Large kernel: 4096×4096=16M. 16M/131072=128 ≥ 32 → THREAD. *)
      test "threading on large kernel" (fun () ->
          let ast = elementwise_loop_ast ~s0:4096 ~s1:4096 in
          let ren = thread_renderer () in
          let opts = run_heuristic ast ren in
          is_true (has is_thread opts));
      (* Small kernel: 4×4=16. 16/131072=0 < any thread count → no THREAD. *)
      test "threading skipped on small kernel" (fun () ->
          let ast = elementwise_loop_ast ~s0:4 ~s1:4 in
          let ren = thread_renderer () in
          let opts = run_heuristic ast ren in
          is_true (not (has is_thread opts)));
      (* First divisible loop axis is picked. Both axes div by 32; THREAD
         should target axis 0. *)
      test "threading picks first divisible axis" (fun () ->
          let ast = elementwise_loop_ast ~s0:4096 ~s1:4096 in
          let ren = thread_renderer () in
          let opts = run_heuristic ast ren in
          let axes = List.filter_map thread_axis opts in
          is_true (axes <> []);
          equal int 0 (List.hd axes));
      (* First axis not divisible by any thread count (size 7); second axis
         is divisible. THREAD should target the second loop axis. *)
      test "threading skips non-divisible axis" (fun () ->
          (* 7 × 4194304 = 29M, 29M/131072=224 ≥ 32. Axis 0 has size 7,
             not divisible by any of [32,16,12,8,6,5,4,3,2].
             After step 9 upcast by 4 on last dim: shape=[7, 1048576, 4].
             Axis 1 (1048576) is divisible by 32. *)
          let ast = elementwise_loop_ast ~s0:7 ~s1:4194304 in
          let ren = thread_renderer () in
          let opts = run_heuristic ast ren in
          let axes = List.filter_map thread_axis opts in
          is_true (axes <> []);
          (* Axis 1 after upcast — exact value depends on shape after
             upcast splitting, but it must NOT be axis 0 (size 7). *)
          is_true (List.hd axes <> 0));
    ]

(* Group 8: Integration *)

let integration_tests =
  group "integration"
    [
      (* Elementwise on GPU: default upcast + locals, no grouping. *)
      test "elementwise on GPU" (fun () ->
          let ast = elementwise_global_ast ~s0:128 ~s1:128 in
          let ren = gpu_renderer () in
          let opts = run_heuristic ast ren in
          is_true (has is_upcast opts);
          is_true (has is_local opts);
          is_true (not (has is_grouptop opts)));
      (* Reduce on GPU: small output → grouping → early return. *)
      test "reduce on GPU with grouping" (fun () ->
          let ast = reduce_global_ast ~s0:4 ~s1:4 ~sr:128 in
          let ren = gpu_renderer () in
          let opts = run_heuristic ast ren in
          is_true (has is_grouptop opts);
          is_true (not (has is_local opts)));
      (* Reduce on GPU: large output → no grouping → unroll + locals. *)
      test "reduce on GPU without grouping" (fun () ->
          let ast = reduce_global_ast ~s0:64 ~s1:64 ~sr:16 in
          let ren = gpu_renderer () in
          let opts = run_heuristic ast ren in
          is_true (not (has is_grouptop opts));
          is_true (has is_unroll opts);
          is_true (has is_local opts));
      (* Matmul on GPU: heuristic upcast + reduce unroll + locals. *)
      test "matmul on GPU" (fun () ->
          let ast = matmul_global_ast ~m:128 ~n:128 ~k:128 in
          let ren = gpu_renderer () in
          let opts = run_heuristic ast ren in
          is_true (has is_upcast opts);
          is_true (has is_unroll opts);
          is_true (has is_local opts);
          is_true (not (has is_grouptop opts)));
      (* Elementwise on CPU: default upcast only. *)
      test "elementwise on CPU" (fun () ->
          let ast = elementwise_loop_ast ~s0:128 ~s1:128 in
          let ren = cpu_renderer () in
          let opts = run_heuristic ast ren in
          is_true (has is_upcast opts);
          is_true (not (has is_local opts));
          is_true (not (has is_thread opts)));
      (* Large kernel on thread renderer: upcast + thread. *)
      test "large kernel on thread renderer" (fun () ->
          let ast = elementwise_loop_ast ~s0:4096 ~s1:4096 in
          let ren = thread_renderer () in
          let opts = run_heuristic ast ren in
          is_true (has is_upcast opts);
          is_true (has is_thread opts);
          is_true (not (has is_local opts)));
    ]

(* Entry *)

let () =
  run __FILE__
    [
      grouping_tests;
      reduce_unroll_tests;
      default_upcast_tests;
      heuristic_upcast_tests;
      matvec_tests;
      masked_upcast_tests;
      local_groups_tests;
      threading_tests;
      integration_tests;
    ]
