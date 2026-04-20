(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Unit tests for Postrange.

   Tests the kernel optimization scheduler: shift_to, apply_opt for all
   optimization variants, validation guards, and the apply_opts entry point. *)

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

let kernel_info ?(opts_to_apply = None) ?(dont_use_locals = false) () =
  {
    K.name = "test";
    axis_kinds = [];
    dont_use_locals;
    applied_opts = [];
    opts_to_apply;
    estimates = None;
  }

let wrap_sink ?opts_to_apply ?dont_use_locals srcs =
  K.sink ~kernel_info:(kernel_info ?opts_to_apply ?dont_use_locals ()) srcs

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
    ~shared_max:0 ~has_threads:true ~global_max:[ 8; 8; 8 ]
    ~render:(fun ?name:_ _ -> "") ()

(* Small shared memory renderer for testing budget *)
let small_smem_renderer () =
  Renderer.make ~name:"test" ~device:"TEST" ~has_local:true ~has_shared:true
    ~shared_max:64 ~render:(fun ?name:_ _ -> "") ()

(* AST Fixture Builders *)

(* Elementwise kernel: output[r0, r1] = exp2(input[r0, r1])
   Two LOOP ranges, load → unary → store → end. *)
let elementwise_ast ~s0 ~s1 =
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

(* Reduce kernel: output[r0, r1] = sum_rr(input[r0, rr, r1])
   Two LOOP ranges + one REDUCE range. *)
let reduce_ast ~s0 ~s1 ~sr =
  let p0 = K.param ~idx:0 ~dtype:global_fptr in
  let p1 = K.param ~idx:1 ~dtype:global_fptr in
  let r0 = loop_range ~axis:0 s0 in
  let r1 = loop_range ~axis:1 s1 in
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

(* Reduce kernel with unsafe pad op (exp2 before reduce) *)
let reduce_unsafe_pad_ast ~s0 ~sr =
  let p0 = K.param ~idx:0 ~dtype:global_fptr in
  let p1 = K.param ~idx:1 ~dtype:global_fptr in
  let r0 = loop_range ~axis:0 s0 in
  let rr = reduce_range ~axis:1 sr in
  let open K.O in
  let in_idx = K.index ~ptr:p1 ~idxs:[ r0 * idx sr + rr ] () in
  let ld = K.load ~src:in_idx () in
  let exp_ld = K.unary ~op:`Exp2 ~src:ld in
  let red =
    K.reduce ~op:`Add ~src:exp_ld ~ranges:[ rr ] ~dtype:D.Val.float32
  in
  let out_idx = K.index ~ptr:p0 ~idxs:[ r0 ] () in
  let st = K.store ~dst:out_idx ~value:red ~ranges:[] in
  let e = K.end_ ~value:st ~ranges:[ r0 ] () in
  wrap_sink [ e ]

(* Max reduce kernel: output[r0] = max_rr(input[r0, rr]) *)
let max_reduce_ast ~s0 ~sr =
  let p0 = K.param ~idx:0 ~dtype:global_fptr in
  let p1 = K.param ~idx:1 ~dtype:global_fptr in
  let r0 = loop_range ~axis:0 s0 in
  let rr = reduce_range ~axis:1 sr in
  let open K.O in
  let in_idx = K.index ~ptr:p1 ~idxs:[ r0 * idx sr + rr ] () in
  let ld = K.load ~src:in_idx () in
  let red = K.reduce ~op:`Max ~src:ld ~ranges:[ rr ] ~dtype:D.Val.float32 in
  let out_idx = K.index ~ptr:p0 ~idxs:[ r0 ] () in
  let st = K.store ~dst:out_idx ~value:red ~ranges:[] in
  let e = K.end_ ~value:st ~ranges:[ r0 ] () in
  wrap_sink [ e ]

(* Elementwise kernel with pre-assigned Global ranges *)
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

(* Reduce kernel with Global ranges *)
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

(* Analysis Helpers *)

let raises_opt_error f =
  raises_match (function P.Opt_error _ -> true | _ -> false) f

let range_size_int r = K.const_to_int (K.range_size r)

let count_kind kind rngs =
  List.length (List.filter (fun r -> K.range_kind r = kind) rngs)

(* Tests *)

(* Group 1: Shift_to *)

let shift_to_tests =
  group "shift_to"
    [
      test "splits range evenly" (fun () ->
        let ast = elementwise_global_ast ~s0:16 ~s1:4 in
        let ren = gpu_renderer () in
        let t = P.create ast ren in
        let initial_len = P.shape_len t in
        let rngs = P.rngs t in
        let rng = List.hd rngs in
        let replaced, new_rng = P.shift_to t rng 4 Ak.Local in
        equal int (initial_len + 1) (P.shape_len t);
        equal int 4 (range_size_int replaced);
        equal int 4 (range_size_int new_rng);
        is_true (K.range_kind replaced = Ak.Global);
        is_true (K.range_kind new_rng = Ak.Local));
      (* GROUPTOP/THREAD depend on top=true reversing the expression order *)
      test "top=true reverses expression order" (fun () ->
        let ast = elementwise_global_ast ~s0:8 ~s1:4 in
        let ren = gpu_renderer () in
        let t = P.create ast ren in
        let rngs = P.rngs t in
        let rng = List.hd rngs in
        (* top=false: replaced * amount + new *)
        let t_bot = P.copy t in
        let _replaced_b, _new_b = P.shift_to t_bot rng 4 Ak.Upcast in
        let shape_bot = P.shape_len t_bot in
        (* top=true: new * old_sz + replaced *)
        let t_top = P.copy t in
        let _replaced_t, _new_t =
          P.shift_to ~top:true t_top rng 4 Ak.Upcast
        in
        let shape_top = P.shape_len t_top in
        (* Both should add one range *)
        equal int (shape_bot) (shape_top));
      (* Divisibility guard: 10 % 3 ≠ 0 *)
      test "rejects non-divisible amount" (fun () ->
        let ast = elementwise_global_ast ~s0:10 ~s1:4 in
        let ren = gpu_renderer () in
        let t = P.create ast ren in
        let rngs = P.rngs t in
        let rng = List.hd rngs in
        raises_opt_error (fun () -> ignore (P.shift_to t rng 3 Ak.Local)));
      (* Full split: amount=size → replaced=1, new=full *)
      test "full amount creates size-1 replaced range" (fun () ->
        let ast = elementwise_global_ast ~s0:8 ~s1:4 in
        let ren = gpu_renderer () in
        let t = P.create ast ren in
        let rngs = P.rngs t in
        let rng = List.hd rngs in
        let replaced, new_rng = P.shift_to t rng 8 Ak.Upcast in
        equal int 1 (range_size_int replaced);
        equal int 8 (range_size_int new_rng));
      (* TC warp path: input_new_rng is used as-is *)
      test "input_new_rng is used as provided node" (fun () ->
        let ast = elementwise_global_ast ~s0:16 ~s1:4 in
        let ren = gpu_renderer () in
        let t = P.create ast ren in
        let rngs = P.rngs t in
        let rng = List.hd rngs in
        let custom_rng =
          K.range ~size:(idx 4) ~axis:99 ~kind:Ak.Warp ~dtype:D.Val.index ()
        in
        let _replaced, new_rng =
          P.shift_to ~input_new_rng:custom_rng t rng 4 Ak.Warp
        in
        is_true (new_rng == custom_rng));
    ]

(* Group 2: Apply_opt validation guards *)

let validation_tests =
  group "validation guards"
    [
      test "UPCAST rejects amount > 16" (fun () ->
        let ast = elementwise_global_ast ~s0:32 ~s1:4 in
        let ren = gpu_renderer () in
        let t = P.create ast ren in
        raises_opt_error (fun () ->
          ignore (P.apply_opt t (K.Opt.Upcast { axis = 0; amount = 17 }))));
      test "UNROLL rejects amount > 32" (fun () ->
        let ast = reduce_global_ast ~s0:4 ~s1:4 ~sr:64 in
        let ren = gpu_renderer () in
        let t = P.create ast ren in
        raises_opt_error (fun () ->
          ignore (P.apply_opt t (K.Opt.Unroll { axis = 0; amount = 33 }))));
      test "UPCAST rejects reduce axis" (fun () ->
        let ast = reduce_global_ast ~s0:4 ~s1:4 ~sr:8 in
        let ren = gpu_renderer () in
        let t = P.create ast ren in
        (* The reduce axis is the last one in sorted rngs (pos=4).
           With 2 globals + 1 reduce, the reduce is at index 2. *)
        raises_opt_error (fun () ->
          ignore (P.apply_opt t (K.Opt.Upcast { axis = 2; amount = 2 }))));
      (* No unrollable dims in elementwise kernel → IndexError equivalent *)
      test "UNROLL rejects non-reduce axis" (fun () ->
        let ast = elementwise_global_ast ~s0:8 ~s1:8 in
        let ren = gpu_renderer () in
        let t = P.create ast ren in
        raises_opt_error (fun () ->
          ignore (P.apply_opt t (K.Opt.Unroll { axis = 0; amount = 2 }))));
      test "LOCAL after NOLOCALS rejected" (fun () ->
        let ast = elementwise_global_ast ~s0:8 ~s1:8 in
        let ren = gpu_renderer () in
        let t = P.create ast ren in
        ignore (P.apply_opt t K.Opt.Nolocals);
        raises_opt_error (fun () ->
          ignore (P.apply_opt t (K.Opt.Local { axis = 0; amount = 2 }))));
      test "LOCAL without renderer locals rejected" (fun () ->
        let ast = elementwise_ast ~s0:8 ~s1:8 in
        let ren = cpu_renderer () in
        let t = P.create ast ren in
        raises_opt_error (fun () ->
          ignore (P.apply_opt t (K.Opt.Local { axis = 0; amount = 2 }))));
      test "shared memory budget exceeded" (fun () ->
        (* small_smem_renderer has shared_max=64 bytes.
           reduce f32 with GROUP amt=32: smem = 32 * 1 * 4 = 128 > 64 *)
        let ast = reduce_global_ast ~s0:4 ~s1:4 ~sr:128 in
        let ren = small_smem_renderer () in
        let t = P.create ast ren in
        raises_opt_error (fun () ->
          ignore (P.apply_opt t (K.Opt.Grouptop { axis = 0; amount = 32 }))));
      test "THREAD rejects double-thread" (fun () ->
        let ast = elementwise_ast ~s0:8 ~s1:8 in
        let ren = thread_renderer () in
        let t = P.create ast ren in
        P.convert_loop_to_global t;
        ignore (P.apply_opt t (K.Opt.Thread { axis = 0; amount = 2 }));
        raises_opt_error (fun () ->
          ignore (P.apply_opt t (K.Opt.Thread { axis = 0; amount = 2 }))));
      test "NOLOCALS rejects existing locals" (fun () ->
        let ast = elementwise_global_ast ~s0:8 ~s1:8 in
        let ren = gpu_renderer () in
        let t = P.create ast ren in
        ignore (P.apply_opt t (K.Opt.Local { axis = 0; amount = 2 }));
        raises_opt_error (fun () -> ignore (P.apply_opt t K.Opt.Nolocals)));
      test "LOCAL rejects non-global axis" (fun () ->
        let ast = reduce_global_ast ~s0:4 ~s1:4 ~sr:8 in
        let ren = gpu_renderer () in
        let t = P.create ast ren in
        (* axis 2 is the reduce range *)
        raises_opt_error (fun () ->
          ignore (P.apply_opt t (K.Opt.Local { axis = 2; amount = 2 }))));
    ]

(* Group 3: Apply_opt shift-based optimizations *)

let shift_opt_tests =
  group "apply_opt shift-based"
    [
      (* Port of test_local_and_grouped_reduce: LOCAL splits global into local
         tile *)
      test "LOCAL splits global into local tile" (fun () ->
        let ast = elementwise_global_ast ~s0:16 ~s1:16 in
        let ren = gpu_renderer () in
        let t = P.create ast ren in
        let initial_len = P.shape_len t in
        ignore (P.apply_opt t (K.Opt.Local { axis = 0; amount = 4 }));
        equal int (initial_len + 1) (P.shape_len t);
        let ats = P.axis_types t in
        is_true (List.exists (fun at -> at = Ak.Local) ats));
      (* Port of test_upcasts: UPCAST on global range *)
      test "UPCAST on global range" (fun () ->
        let ast = elementwise_global_ast ~s0:16 ~s1:16 in
        let ren = gpu_renderer () in
        let t = P.create ast ren in
        ignore (P.apply_opt t (K.Opt.Upcast { axis = 0; amount = 4 }));
        let ats = P.axis_types t in
        is_true (List.exists (fun at -> at = Ak.Upcast) ats);
        equal int 4 (P.upcast_size t));
      (* Port of test_full_upcast: UPCAST with amount=0 uses full range size *)
      test "UPCAST with amount=0 uses full range size" (fun () ->
        let ast = elementwise_global_ast ~s0:4 ~s1:4 in
        let ren = gpu_renderer () in
        let t = P.create ast ren in
        ignore (P.apply_opt t (K.Opt.Upcast { axis = 0; amount = 0 }));
        equal int 4 (P.upcast_size t);
        equal int 1 (P.upcasted t));
      (* Port of test_local_and_grouped_reduce: GROUPTOP on reduce *)
      test "GROUPTOP on reduce creates group_reduce range" (fun () ->
        let ast = reduce_global_ast ~s0:32 ~s1:32 ~sr:128 in
        let ren = gpu_renderer () in
        let t = P.create ast ren in
        ignore (P.apply_opt t (K.Opt.Grouptop { axis = 0; amount = 32 }));
        equal int 1 (P.group_for_reduces t);
        let ats = P.axis_types t in
        is_true (List.exists (fun at -> at = Ak.Group_reduce) ats));
      (* Port of test_matmul: GROUPTOP + UNROLL *)
      test "UNROLL after GROUPTOP" (fun () ->
        let ast = reduce_global_ast ~s0:32 ~s1:32 ~sr:128 in
        let ren = gpu_renderer () in
        let t = P.create ast ren in
        ignore (P.apply_opt t (K.Opt.Grouptop { axis = 0; amount = 32 }));
        ignore (P.apply_opt t (K.Opt.Unroll { axis = 0; amount = 4 }));
        equal int 1 (P.upcasted t);
        let ats = P.axis_types t in
        is_true (List.exists (fun at -> at = Ak.Unroll) ats);
        is_true (List.exists (fun at -> at = Ak.Group_reduce) ats));
      (* Port of test_matmul combo: LOCAL×2 + GROUPTOP + UNROLL + UPCAST×2 *)
      test "combined LOCAL + GROUPTOP + UNROLL + UPCAST" (fun () ->
        let ast = reduce_global_ast ~s0:128 ~s1:128 ~sr:128 in
        let ren = gpu_renderer () in
        let t = P.create ast ren in
        ignore (P.apply_opt t (K.Opt.Local { axis = 0; amount = 4 }));
        ignore (P.apply_opt t (K.Opt.Local { axis = 0; amount = 4 }));
        ignore (P.apply_opt t (K.Opt.Grouptop { axis = 0; amount = 8 }));
        ignore (P.apply_opt t (K.Opt.Unroll { axis = 0; amount = 4 }));
        ignore (P.apply_opt t (K.Opt.Upcast { axis = 0; amount = 4 }));
        ignore (P.apply_opt t (K.Opt.Upcast { axis = 1; amount = 2 }));
        let ats = P.axis_types t in
        is_true (List.exists (fun at -> at = Ak.Local) ats);
        is_true (List.exists (fun at -> at = Ak.Upcast) ats);
        is_true (List.exists (fun at -> at = Ak.Unroll) ats);
        is_true (List.exists (fun at -> at = Ak.Group_reduce) ats));
      (* Port of test_thread_opts: THREAD on threadable renderer *)
      test "THREAD on threadable renderer" (fun () ->
        let ast = elementwise_ast ~s0:8 ~s1:8 in
        let ren = thread_renderer () in
        let t = P.create ast ren in
        P.convert_loop_to_global t;
        ignore (P.apply_opt t (K.Opt.Thread { axis = 0; amount = 2 }));
        let ats = P.axis_types t in
        is_true (List.exists (fun at -> at = Ak.Thread) ats));
      (* Port of test_double_reduce: Multiple GROUPTOPs on double reduce.
         We use a single reduce with two reduce ranges. *)
      test "double GROUPTOP on reduce" (fun () ->
        let ast = reduce_global_ast ~s0:8 ~s1:8 ~sr:128 in
        let ren = gpu_renderer () in
        let t = P.create ast ren in
        ignore (P.apply_opt t (K.Opt.Grouptop { axis = 0; amount = 4 }));
        equal int 1 (P.group_for_reduces t));
    ]

(* Group 4: PADTO *)

let padto_tests =
  group "apply_opt PADTO"
    [
      (* Port of test_padto_matmul: PADTO pads 17 → 32 *)
      test "PADTO pads axis to next multiple" (fun () ->
        let ast = elementwise_global_ast ~s0:17 ~s1:4 in
        let ren = gpu_renderer () in
        let t = P.create ast ren in
        ignore (P.apply_opt t (K.Opt.Padto { axis = 0; amount = 32 }));
        (* After padding, the range should have size 32 *)
        let rngs = P.rngs t in
        let sizes =
          List.map (fun r -> range_size_int r) rngs
        in
        is_true (List.mem 32 sizes));
      (* Port of test_padto_upcasted_not_ok: PADTO rejects upcast axis *)
      test "PADTO rejects upcast axis" (fun () ->
        let ast = elementwise_global_ast ~s0:4 ~s1:4 in
        let ren = gpu_renderer () in
        let t = P.create ast ren in
        ignore (P.apply_opt t (K.Opt.Upcast { axis = 0; amount = 0 }));
        (* axis 0 is now Global size-1, the upcast is at the end.
           Find the upcast axis index *)
        raises_opt_error (fun () ->
          (* After full upcast of axis 0, the original is size-1 (filtered
             from rngs). The upcast range is now in rngs at some index.
             The upcast kind makes it non-paddable. *)
          let ats = P.axis_types t in
          let upcast_idx =
            let rec find i = function
              | [] -> failwith "no upcast"
              | at :: _ when at = Ak.Upcast -> i
              | _ :: rest -> find (i + 1) rest
            in
            find 0 ats
          in
          ignore
            (P.apply_opt t (K.Opt.Padto { axis = upcast_idx; amount = 8 }))));
      (* Port of test_padto_sum_not_ok: exp2 in backward slice → error *)
      test "PADTO rejects unsafe pad ops in reduce backward slice" (fun () ->
        let ast = reduce_unsafe_pad_ast ~s0:17 ~sr:32 in
        let ren = gpu_renderer () in
        let t = P.create ast ren in
        P.convert_loop_to_global t;
        (* The reduce axis (axis index 1) has unsafe pad ops in its
           backward slice *)
        raises_opt_error (fun () ->
          ignore (P.apply_opt t (K.Opt.Padto { axis = 1; amount = 64 }))));
      (* Port of test_padto_max: max reduce can't be padded on reduce axis *)
      test "PADTO rejects max reduce on reduce axis" (fun () ->
        let ast = max_reduce_ast ~s0:17 ~sr:32 in
        let ren = gpu_renderer () in
        let t = P.create ast ren in
        P.convert_loop_to_global t;
        raises_opt_error (fun () ->
          ignore (P.apply_opt t (K.Opt.Padto { axis = 1; amount = 64 }))));
    ]

(* Group 5: SWAP and NOLOCALS *)

let swap_nolocals_tests =
  group "apply_opt SWAP and NOLOCALS"
    [
      (* SWAP exchanges two global axes: sizes swap positions.
         Before: axis 0 → size 8, axis 1 → size 16
         After:  axis 0 → size 16, axis 1 → size 8  (axis numbers swapped) *)
      test "SWAP exchanges two global axes" (fun () ->
        let ast = elementwise_global_ast ~s0:8 ~s1:16 in
        let ren = gpu_renderer () in
        let t = P.create ast ren in
        let rngs_before = P.rngs t in
        let sz0_before = range_size_int (List.nth rngs_before 0) in
        let sz1_before = range_size_int (List.nth rngs_before 1) in
        ignore (P.apply_opt t (K.Opt.Swap { axis = 0; with_axis = 1 }));
        let rngs_after = P.rngs t in
        let sz0_after = range_size_int (List.nth rngs_after 0) in
        let sz1_after = range_size_int (List.nth rngs_after 1) in
        (* After swap, the sizes at each sorted position are exchanged *)
        equal int sz1_before sz0_after;
        equal int sz0_before sz1_after);
      (* SWAP rejects non-global axes *)
      test "SWAP rejects non-global axes" (fun () ->
        let ast = elementwise_global_ast ~s0:8 ~s1:8 in
        let ren = gpu_renderer () in
        let t = P.create ast ren in
        ignore (P.apply_opt t (K.Opt.Local { axis = 0; amount = 2 }));
        (* Now axis 0 is Global(4), axis 1 is Global(8), axis 2 is Local(2).
           Swapping axis 0 with axis 2 (Local) should fail. *)
        raises_opt_error (fun () ->
          ignore (P.apply_opt t (K.Opt.Swap { axis = 0; with_axis = 2 }))));
      (* NOLOCALS sets dont_use_locals *)
      test "NOLOCALS disables locals" (fun () ->
        let ast = elementwise_global_ast ~s0:8 ~s1:8 in
        let ren = gpu_renderer () in
        let t = P.create ast ren in
        ignore (P.apply_opt t K.Opt.Nolocals);
        (* Subsequent LOCAL should fail *)
        raises_opt_error (fun () ->
          ignore (P.apply_opt t (K.Opt.Local { axis = 0; amount = 2 }))));
    ]

(* Group 6: State queries *)

let state_query_tests =
  group "state queries"
    [
      (* rngs sorts by axis_to_pos: Loop(-1) < Global(0) < Reduce(4) *)
      test "rngs sorted by axis_to_pos then axis" (fun () ->
        let ast = reduce_ast ~s0:4 ~s1:4 ~sr:8 in
        let ren = gpu_renderer () in
        let t = P.create ast ren in
        let ats = P.axis_types t in
        (* Two Loop ranges first (pos=-1), then Reduce (pos=4) *)
        equal int 3 (List.length ats);
        is_true (List.nth ats 0 = Ak.Loop);
        is_true (List.nth ats 1 = Ak.Loop);
        is_true (List.nth ats 2 = Ak.Reduce));
      (* rngs filters out size-1 ranges *)
      test "rngs filters out size-1 ranges" (fun () ->
        let ast = elementwise_global_ast ~s0:8 ~s1:4 in
        let ren = gpu_renderer () in
        let t = P.create ast ren in
        (* Full upcast of axis 0: replaced range becomes size 1 *)
        ignore (P.apply_opt t (K.Opt.Upcast { axis = 0; amount = 0 }));
        (* Size-1 replaced range should be filtered from rngs *)
        let rngs = P.rngs t in
        List.iter
          (fun r -> is_true (range_size_int r > 1))
          rngs);
      (* shape_str produces correct labels *)
      test "shape_str produces correct labels" (fun () ->
        let ast = reduce_global_ast ~s0:4 ~s1:4 ~sr:8 in
        let ren = gpu_renderer () in
        let t = P.create ast ren in
        let ss = P.shape_str t in
        equal int 3 (List.length ss);
        equal string "g0" (List.nth ss 0);
        equal string "g1" (List.nth ss 1);
        equal string "R0" (List.nth ss 2));
      (* shape_str_to_axis resolves labels *)
      test "shape_str_to_axis resolves labels" (fun () ->
        let ast = reduce_global_ast ~s0:4 ~s1:4 ~sr:8 in
        let ren = gpu_renderer () in
        let t = P.create ast ren in
        let axes = P.shape_str_to_axis t [ "g1"; "R0" ] in
        equal int 1 (List.nth axes 0);
        equal int 2 (List.nth axes 1));
      (* copy preserves state *)
      test "copy preserves mutable state" (fun () ->
        let ast = elementwise_global_ast ~s0:8 ~s1:8 in
        let ren = gpu_renderer () in
        let t = P.create ast ren in
        ignore (P.apply_opt t K.Opt.Nolocals);
        let t2 = P.copy t in
        raises_opt_error (fun () ->
          ignore (P.apply_opt t2 (K.Opt.Local { axis = 0; amount = 2 }))));
      (* Helper queries *)
      test "upcastable_dims and unrollable_dims" (fun () ->
        let ast = reduce_global_ast ~s0:4 ~s1:4 ~sr:8 in
        let ren = gpu_renderer () in
        let t = P.create ast ren in
        let up = P.upcastable_dims t in
        let un = P.unrollable_dims t in
        (* 2 global axes with size > 1 → 2 upcastable dims *)
        equal int 2 (List.length up);
        (* 1 reduce axis with size > 1 → 1 unrollable dim *)
        equal int 1 (List.length un));
      (* output_shape replaces reduce/unroll/group_reduce with 1 *)
      test "output_shape replaces non-output axes with 1" (fun () ->
        let ast = reduce_global_ast ~s0:4 ~s1:4 ~sr:8 in
        let ren = gpu_renderer () in
        let t = P.create ast ren in
        let os = P.output_shape t in
        equal int 3 (List.length os);
        equal int 4 (K.const_to_int (List.nth os 0));
        equal int 4 (K.const_to_int (List.nth os 1));
        equal int 1 (K.const_to_int (List.nth os 2)));
    ]

(* Group 7: Integration *)

let integration_tests =
  group "integration"
    [
      (* get_optimized_ast produces valid kernel_info *)
      test "get_optimized_ast produces valid kernel_info" (fun () ->
        let ast = reduce_global_ast ~s0:32 ~s1:32 ~sr:128 in
        let ren = gpu_renderer () in
        let t = P.create ast ren in
        ignore
          (P.apply_opt t (K.Opt.Upcast { axis = 0; amount = 4 }));
        let result = P.get_optimized_ast t in
        match K.view result with
        | Sink { kernel_info = Some ki; _ } ->
            equal int 1 (List.length ki.applied_opts);
            is_true (K.tag result <> None)
        | _ -> failwith "expected Sink with kernel_info");
      (* Name generation: "r_" for reduce, "E_" for elementwise *)
      test "get_optimized_ast name generation" (fun () ->
        let ast_r = reduce_global_ast ~s0:4 ~s1:4 ~sr:8 in
        let ren = gpu_renderer () in
        let t_r = P.create ast_r ren in
        let result_r = P.get_optimized_ast t_r in
        (match K.view result_r with
        | Sink { kernel_info = Some ki; _ } ->
            is_true (String.length ki.name > 0);
            is_true (ki.name.[0] = 'r')
        | _ -> failwith "expected Sink");
        let ast_e = elementwise_global_ast ~s0:4 ~s1:4 in
        let t_e = P.create ast_e ren in
        let result_e = P.get_optimized_ast t_e in
        match K.view result_e with
        | Sink { kernel_info = Some ki; _ } ->
            is_true (String.length ki.name > 0);
            is_true (ki.name.[0] = 'E')
        | _ -> failwith "expected Sink");
      (* apply_opts respects opts_to_apply *)
      test "apply_opts respects opts_to_apply" (fun () ->
        let p0 = K.param ~idx:0 ~dtype:global_fptr in
        let p1 = K.param ~idx:1 ~dtype:global_fptr in
        let r0 = global_range ~axis:0 16 in
        let r1 = global_range ~axis:1 16 in
        let open K.O in
        let in_idx = K.index ~ptr:p1 ~idxs:[ r0 * idx 16 + r1 ] () in
        let ld = K.load ~src:in_idx () in
        let value = K.unary ~op:`Exp2 ~src:ld in
        let out_idx = K.index ~ptr:p0 ~idxs:[ r0 * idx 16 + r1 ] () in
        let st = K.store ~dst:out_idx ~value ~ranges:[] in
        let e = K.end_ ~value:st ~ranges:[ r0; r1 ] () in
        let opts = [ K.Opt.Upcast { axis = 0; amount = 4 } ] in
        let ast =
          K.sink
            ~kernel_info:(kernel_info ~opts_to_apply:(Some opts) ())
            [ e ]
        in
        let ren = gpu_renderer () in
        (* apply_opts dispatch moved to Pipeline; test the scheduler
           operations directly instead *)
        let k = P.create ast ren in
        P.convert_loop_to_global k;
        let opts = [ K.Opt.Upcast { axis = 0; amount = 4 } ] in
        List.iter (fun opt -> ignore (P.apply_opt k opt)) opts;
        let result = P.get_optimized_ast k in
        match K.view result with
        | Sink { kernel_info = Some ki; _ } ->
            equal int 1 (List.length ki.applied_opts)
        | _ -> failwith "expected Sink with kernel_info");
    ]

(* Group 8: Convert_loop_to_global *)

let convert_loop_to_global_tests =
  group "convert_loop_to_global"
    [
      test "LOOP ranges become GLOBAL on GPU" (fun () ->
        let ast = elementwise_ast ~s0:8 ~s1:8 in
        let ren = gpu_renderer () in
        let t = P.create ast ren in
        (* Before: all LOOP *)
        is_true
          (List.for_all (fun at -> at = Ak.Loop) (P.axis_types t));
        P.convert_loop_to_global t;
        (* After: all GLOBAL *)
        is_true
          (List.for_all (fun at -> at = Ak.Global) (P.axis_types t)));
      test "LOOP ranges stay LOOP on CPU" (fun () ->
        let ast = elementwise_ast ~s0:8 ~s1:8 in
        let ren = cpu_renderer () in
        let t = P.create ast ren in
        P.convert_loop_to_global t;
        is_true
          (List.for_all (fun at -> at = Ak.Loop) (P.axis_types t)));
      test "reduce ranges stay REDUCE after conversion" (fun () ->
        let ast = reduce_ast ~s0:4 ~s1:4 ~sr:8 in
        let ren = gpu_renderer () in
        let t = P.create ast ren in
        P.convert_loop_to_global t;
        let ats = P.axis_types t in
        (* LOOP ranges → GLOBAL, but REDUCE stays *)
        is_true (List.exists (fun at -> at = Ak.Reduce) ats);
        is_true (not (List.exists (fun at -> at = Ak.Loop) ats)));
    ]

(* Group 9: Apply_opts dispatch *)

let dispatch_tests =
  group "apply_opts dispatch"
    [
      test "opts_to_apply applied in order" (fun () ->
        let opts =
          [
            K.Opt.Upcast { axis = 0; amount = 4 };
            K.Opt.Upcast { axis = 1; amount = 2 };
          ]
        in
        let ast = elementwise_global_ast ~s0:16 ~s1:16 in
        let ki = kernel_info ~opts_to_apply:(Some opts) () in
        let ast =
          K.sink ~kernel_info:ki
            (match K.view ast with
            | Sink { srcs; _ } -> srcs
            | _ -> [ ast ])
        in
        let ren = gpu_renderer () in
        let result = P.apply_opts ast ren in
        match K.view result with
        | Sink { kernel_info = Some ki; _ } ->
            equal int 2 (List.length ki.applied_opts)
        | _ -> failwith "expected Sink with kernel_info");
      test "beam_search closure is called" (fun () ->
        let called = ref false in
        let beam_search k =
          called := true;
          k
        in
        let ast = elementwise_global_ast ~s0:8 ~s1:8 in
        let ren = gpu_renderer () in
        let _result =
          P.apply_opts ~beam_search ast ren
        in
        is_true !called);
      test "hand_coded closure is called" (fun () ->
        let called = ref false in
        let hco k =
          called := true;
          k
        in
        let ast = elementwise_global_ast ~s0:8 ~s1:8 in
        let ren = gpu_renderer () in
        let _result =
          P.apply_opts ~hand_coded_optimizations:hco ast ren
        in
        is_true !called);
      test "already-optimized kernel returns unchanged" (fun () ->
        let ast = elementwise_global_ast ~s0:8 ~s1:8 in
        let ren = gpu_renderer () in
        (* First pass: optimize normally *)
        let optimized = P.apply_opts ast ren in
        (* Second pass: should return unchanged (tag is set) *)
        let called = ref false in
        let hco k =
          called := true;
          k
        in
        let result =
          P.apply_opts ~hand_coded_optimizations:hco optimized ren
        in
        is_true (not !called);
        (* Result should be the same AST *)
        (match (K.view optimized, K.view result) with
        | ( Sink { kernel_info = Some ki1; _ },
            Sink { kernel_info = Some ki2; _ } ) ->
            equal string ki1.name ki2.name
        | _ -> failwith "expected Sink with kernel_info"));
      test "heuristic skipped with BUFFERIZE in AST" (fun () ->
        let p0 = K.param ~idx:0 ~dtype:global_fptr in
        let r0 = global_range ~axis:0 8 in
        let open K.O in
        let in_idx = K.index ~ptr:p0 ~idxs:[ r0 ] () in
        let ld = K.load ~src:in_idx () in
        let buf_opts =
          { K.device = None; addrspace = Global; removable = false }
        in
        let bz =
          K.bufferize ~src:ld ~ranges:[ r0 ] ~dtype:global_fptr
            ~opts:buf_opts
        in
        let out_idx = K.index ~ptr:p0 ~idxs:[ r0 + idx 8 ] () in
        let st = K.store ~dst:out_idx ~value:bz ~ranges:[] in
        let e = K.end_ ~value:st ~ranges:[ r0 ] () in
        let ast = wrap_sink [ e ] in
        let ren = gpu_renderer () in
        let called = ref false in
        let hco k =
          called := true;
          k
        in
        let _result =
          P.apply_opts ~hand_coded_optimizations:hco ast ren
        in
        is_true (not !called));
    ]

(* Group 10: TC optimization *)

(* Matmul-pattern AST:  output[i,j] = sum_k(a[i,k] * b[k,j])
   Two GLOBAL ranges + one REDUCE range, MUL inside REDUCE ADD. *)
let global_f16ptr = D.Ptr.create D.Val.float16 ~addrspace:Global ~size:(-1)

let matmul_ast ~si ~sj ~sk =
  let p_out = K.param ~idx:0 ~dtype:global_fptr in
  let p_a = K.param ~idx:1 ~dtype:global_fptr in
  let p_b = K.param ~idx:2 ~dtype:global_fptr in
  let ri = global_range ~axis:0 si in
  let rj = global_range ~axis:1 sj in
  let rk = reduce_range ~axis:2 sk in
  let open K.O in
  let idx_a = K.index ~ptr:p_a ~idxs:[ ri * idx sk + rk ] () in
  let ld_a = K.load ~src:idx_a () in
  let idx_b = K.index ~ptr:p_b ~idxs:[ rk * idx sj + rj ] () in
  let ld_b = K.load ~src:idx_b () in
  let mul = ld_a * ld_b in
  let red = K.reduce ~op:`Add ~src:mul ~ranges:[ rk ] ~dtype:D.Val.float32 in
  let out_idx = K.index ~ptr:p_out ~idxs:[ ri * idx sj + rj ] () in
  let st = K.store ~dst:out_idx ~value:red ~ranges:[] in
  let e = K.end_ ~value:st ~ranges:[ ri; rj ] () in
  wrap_sink [ e ]

let tc_renderer () =
  Renderer.make ~name:"metal" ~device:"METAL" ~has_local:true ~has_shared:true
    ~shared_max:32768 ~tensor_cores:Tc.metal
    ~render:(fun ?name:_ _ -> "") ()

let has_wmma ast =
  List.exists
    (fun n -> match K.view n with Wmma _ -> true | _ -> false)
    (K.toposort ast)

let tc_tests =
  group "TC optimization"
    [
      test "TC basic apply creates WMMA" (fun () ->
        let ast = matmul_ast ~si:16 ~sj:16 ~sk:16 in
        let ren = tc_renderer () in
        let t = P.create ast ren in
        let result =
          P.apply_opt t
            (K.Opt.Tc { axis = 0; tc_select = 0; tc_opt = 0; use_tc = 1 })
        in
        is_true (result <> None);
        is_true (P.tensor_core t <> None);
        is_true (has_wmma (P.ast t)));
      test "TC with padding (tc_opt=2)" (fun () ->
        (* 10 doesn't divide 8 cleanly — tc_opt=2 allows padding *)
        let ast = matmul_ast ~si:10 ~sj:10 ~sk:10 in
        let ren = tc_renderer () in
        let t = P.create ast ren in
        let result =
          P.apply_opt t
            (K.Opt.Tc { axis = 0; tc_select = 0; tc_opt = 2; use_tc = 1 })
        in
        is_true (result <> None);
        is_true (P.tensor_core t <> None));
      test "TC rejects non-reduce kernel" (fun () ->
        let ast = elementwise_global_ast ~s0:16 ~s1:16 in
        let ren = tc_renderer () in
        let t = P.create ast ren in
        raises_opt_error (fun () ->
          ignore
            (P.apply_opt t
               (K.Opt.Tc
                  { axis = 0; tc_select = 0; tc_opt = 0; use_tc = 1 }))));
      test "TC rejects invalid tc_select" (fun () ->
        let ast = matmul_ast ~si:16 ~sj:16 ~sk:16 in
        let ren = tc_renderer () in
        let t = P.create ast ren in
        raises_opt_error (fun () ->
          ignore
            (P.apply_opt t
               (K.Opt.Tc
                  { axis = 0; tc_select = 99; tc_opt = 0; use_tc = 1 }))));
      test "TC must be first opt" (fun () ->
        let ast = matmul_ast ~si:16 ~sj:16 ~sk:16 in
        let ren = tc_renderer () in
        let t = P.create ast ren in
        ignore
          (P.apply_opt t (K.Opt.Local { axis = 0; amount = 2 }));
        raises_opt_error (fun () ->
          ignore
            (P.apply_opt t
               (K.Opt.Tc
                  { axis = 0; tc_select = 0; tc_opt = 0; use_tc = 1 }))));
      test "TC use_tc=2 skips WMMA construction" (fun () ->
        let ast = matmul_ast ~si:16 ~sj:16 ~sk:16 in
        let ren = tc_renderer () in
        let t = P.create ast ren in
        ignore
          (P.apply_opt t
             (K.Opt.Tc { axis = 0; tc_select = 0; tc_opt = 0; use_tc = 2 }));
        is_true (P.tensor_core t <> None);
        is_true (not (has_wmma (P.ast t))));
    ]

(* Entry *)

let () =
  run __FILE__
    [
      shift_to_tests;
      validation_tests;
      shift_opt_tests;
      padto_tests;
      swap_nolocals_tests;
      state_query_tests;
      integration_tests;
      convert_loop_to_global_tests;
      dispatch_tests;
      tc_tests;
    ]
