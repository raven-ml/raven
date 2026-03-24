(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Unit tests for Search.

   Tests the beam search action list and get_kernel_actions filtering logic. *)

open Windtrap
open Tolk
open Tolk_ir
module K = Kernel
module D = Dtype
module C = Const
module Ak = Axis_kind
module P = Postrange
module S = Search

(* Helpers *)

let idx n = K.const (C.int D.index n)
let global_fptr = D.ptr_of D.float32 ~addrspace:Global ~size:(-1)

let kernel_info () =
  {
    K.name = "test";
    axis_kinds = [];
    dont_use_locals = false;
    applied_opts = [];
    opts_to_apply = None;
    estimates = None;
  }

let wrap_sink srcs = K.sink ~kernel_info:(kernel_info ()) srcs

let loop_range ~axis size =
  K.range ~size:(idx size) ~axis ~kind:Ak.Loop ~dtype:D.index ()

let reduce_range ~axis size =
  K.range ~size:(idx size) ~axis ~kind:Ak.Reduce ~dtype:D.index ()

let global_range ~axis size =
  K.range ~size:(idx size) ~axis ~kind:Ak.Global ~dtype:D.index ()

let gpu_renderer () =
  Renderer.make ~name:"test" ~device:"TEST" ~has_local:true ~has_shared:true
    ~shared_max:32768 ~render:(fun ?name:_ _ -> "") ()

let cpu_renderer () =
  Renderer.make ~name:"cpu" ~device:"CPU" ~has_local:false ~has_shared:false
    ~shared_max:0 ~render:(fun ?name:_ _ -> "") ()

(* AST Fixture Builders *)

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
  let red = K.reduce ~op:`Add ~src:ld ~ranges:[ rr ] ~dtype:D.float32 in
  let out_idx = K.index ~ptr:p0 ~idxs:[ r0 * idx s1 + r1 ] () in
  let st = K.store ~dst:out_idx ~value:red ~ranges:[] in
  let e = K.end_ ~value:st ~ranges:[ r0; r1 ] () in
  wrap_sink [ e ]

(* Analysis Helpers *)

let is_upcast = function K.Opt.Upcast _ -> true | _ -> false
let is_unroll = function K.Opt.Unroll _ -> true | _ -> false
let is_local = function K.Opt.Local _ -> true | _ -> false
let is_grouptop = function K.Opt.Grouptop _ -> true | _ -> false
let is_group = function K.Opt.Group _ -> true | _ -> false
let is_tc = function K.Opt.Tc _ -> true | _ -> false
let is_swap = function K.Opt.Swap _ -> true | _ -> false
let is_thread = function K.Opt.Thread _ -> true | _ -> false
let is_padto = function K.Opt.Padto _ -> true | _ -> false
let is_nolocals = function K.Opt.Nolocals -> true | _ -> false

let count pred xs = List.length (List.filter pred xs)

let has_opt_matching pred s = List.exists pred (P.applied_opts s)

let upcast_product s =
  let full = P.full_shape s in
  let types = P.axis_types s in
  List.fold_left2
    (fun acc x t ->
      match t with
      | Ak.Upcast | Ak.Unroll -> acc * K.const_to_int x
      | _ -> acc)
    1 full types

let local_product s =
  let full = P.full_shape s in
  let types = P.axis_types s in
  List.fold_left2
    (fun acc x t ->
      match t with
      | Ak.Warp | Ak.Local | Ak.Group_reduce -> acc * K.const_to_int x
      | _ -> acc)
    1 full types

(* Tests *)

(* Group 1: Actions list parity *)

let actions_tests =
  group "actions list parity"
    [
      test "total count is 193" (fun () ->
          equal int 193 (List.length S.actions));
      test "per-variant counts" (fun () ->
          equal int 48 (count is_upcast S.actions);
          equal int 15 (count is_unroll S.actions);
          equal int 44 (count is_local S.actions);
          equal int 24 (count is_grouptop S.actions);
          equal int 12 (count is_group S.actions);
          equal int 10 (count is_tc S.actions);
          equal int 10 (count is_swap S.actions);
          equal int 30 (count is_thread S.actions));
      test "no PADTO by default" (fun () ->
          equal int 0 (count is_padto S.actions));
      test "no NOLOCALS by default" (fun () ->
          equal int 0 (count is_nolocals S.actions));
      test "sentinel entries present" (fun () ->
          (* Extra LOCAL entries at boundary values *)
          is_true (List.mem (K.Opt.Local { axis = 0; amount = 32 }) S.actions);
          is_true (List.mem (K.Opt.Local { axis = 6; amount = 2 }) S.actions);
          (* TC with tc_opt=0 *)
          is_true
            (List.mem
               (K.Opt.Tc { axis = 0; tc_select = -1; tc_opt = 0; use_tc = 1 })
               S.actions);
          (* Upcast zero-variant *)
          is_true
            (List.mem (K.Opt.Upcast { axis = 0; amount = 0 }) S.actions);
          (* Swap entry *)
          is_true
            (List.mem (K.Opt.Swap { axis = 0; with_axis = 1 }) S.actions));
      test "TC sweep uses tc_opt=2 for axes 0..8" (fun () ->
          for axis = 0 to 8 do
            is_true
              ~msg:(Printf.sprintf "TC axis=%d tc_opt=2" axis)
              (List.mem
                 (K.Opt.Tc { axis; tc_select = -1; tc_opt = 2; use_tc = 1 })
                 S.actions)
          done);
    ]

(* Group 2: Get_kernel_actions basic behavior *)

let basic_behavior_tests =
  group "get_kernel_actions basic"
    [
      test "include_0 true includes identity" (fun () ->
          let ast = elementwise_global_ast ~s0:16 ~s1:16 in
          let s = P.create ast (gpu_renderer ()) in
          let actions = S.get_kernel_actions ~include_0:true s in
          is_true (List.exists (fun (id, _) -> id = 0) actions));
      test "include_0 false excludes identity" (fun () ->
          let ast = elementwise_global_ast ~s0:16 ~s1:16 in
          let s = P.create ast (gpu_renderer ()) in
          let actions = S.get_kernel_actions ~include_0:false s in
          is_false (List.exists (fun (id, _) -> id = 0) actions));
      test "returns non-empty for elementwise kernel" (fun () ->
          let ast = elementwise_global_ast ~s0:16 ~s1:16 in
          let s = P.create ast (gpu_renderer ()) in
          let actions = S.get_kernel_actions s in
          is_true (List.length actions > 1));
      test "action ids are valid" (fun () ->
          let ast = elementwise_global_ast ~s0:16 ~s1:16 in
          let s = P.create ast (gpu_renderer ()) in
          let actions = S.get_kernel_actions s in
          let max_id = List.length S.actions in
          List.iter
            (fun (id, _) ->
              is_true
                ~msg:(Printf.sprintf "id %d in [0, %d]" id max_id)
                (id >= 0 && id <= max_id))
            actions);
    ]

(* Group 3: Get_kernel_actions filtering *)

let filtering_tests =
  group "get_kernel_actions filtering"
    [
      test "all returned schedulers have applied opts" (fun () ->
          let ast = elementwise_global_ast ~s0:4 ~s1:4 in
          let s = P.create ast (gpu_renderer ()) in
          let actions = S.get_kernel_actions ~include_0:false s in
          List.iter
            (fun (_, s2) ->
              is_true (P.shape_len s2 >= 1);
              is_true (List.length (P.applied_opts s2) >= 1))
            actions);
      test "elementwise kernel has no unroll actions" (fun () ->
          let ast = elementwise_global_ast ~s0:16 ~s1:16 in
          let s = P.create ast (gpu_renderer ()) in
          let actions = S.get_kernel_actions ~include_0:false s in
          is_false
            (List.exists
               (fun (_, s2) -> has_opt_matching is_unroll s2)
               actions));
      test "noop upcast is filtered when shape equals amount" (fun () ->
          (* shape[0]=4, Upcast{axis=0,amount=4}: shape equals amount and
             Upcast{axis=0,amount=0} exists in actions → noop, filtered out *)
          let ast = elementwise_global_ast ~s0:4 ~s1:8 in
          let s = P.create ast (gpu_renderer ()) in
          let actions = S.get_kernel_actions ~include_0:false s in
          is_false
            (List.exists
               (fun (_, s2) ->
                 List.exists
                   (fun opt -> opt = K.Opt.Upcast { axis = 0; amount = 4 })
                   (P.applied_opts s2))
               actions));
    ]

(* Group 4: Budget enforcement *)

let budget_tests =
  group "budget enforcement"
    [
      test "max_up limits upcast product" (fun () ->
          let ast = elementwise_global_ast ~s0:16 ~s1:16 in
          let s = P.create ast (gpu_renderer ()) in
          List.iter
            (fun max_up ->
              let actions =
                S.get_kernel_actions ~include_0:false ~max_up s
              in
              List.iter
                (fun (_, s2) ->
                  if has_opt_matching is_upcast s2
                     || has_opt_matching is_unroll s2
                  then
                    is_true
                      ~msg:
                        (Printf.sprintf "upcast_product <= %d, got %d" max_up
                           (upcast_product s2))
                      (upcast_product s2 <= max_up))
                actions;
              is_true
                ~msg:
                  (Printf.sprintf
                     "expected upcast/unroll actions with max_up=%d" max_up)
                (List.exists
                   (fun (_, s2) ->
                     has_opt_matching is_upcast s2
                     || has_opt_matching is_unroll s2)
                   actions))
            [ 2; 4 ]);
      test "default max_lcl limits local product" (fun () ->
          let ast = elementwise_global_ast ~s0:16 ~s1:16 in
          let s = P.create ast (gpu_renderer ()) in
          let actions = S.get_kernel_actions ~include_0:false s in
          List.iter
            (fun (_, s2) ->
              if has_opt_matching is_local s2 then
                is_true
                  ~msg:
                    (Printf.sprintf "local_product <= 1024, got %d"
                       (local_product s2))
                  (local_product s2 <= 1024))
            actions);
    ]

(* Group 5: State isolation *)

let isolation_tests =
  group "state isolation"
    [
      test "input scheduler not mutated" (fun () ->
          let ast = elementwise_global_ast ~s0:16 ~s1:16 in
          let s = P.create ast (gpu_renderer ()) in
          let opts_before = List.length (P.applied_opts s) in
          let shape_before = P.shape_len s in
          ignore (S.get_kernel_actions s : (int * P.t) list);
          equal int shape_before (P.shape_len s);
          equal int opts_before (List.length (P.applied_opts s)));
      test "returned schedulers are independent copies" (fun () ->
          let ast = elementwise_global_ast ~s0:16 ~s1:16 in
          let s = P.create ast (gpu_renderer ()) in
          let actions = S.get_kernel_actions ~include_0:false s in
          match actions with
          | (_, s1) :: (_, s2) :: _ ->
              let opts1_before = List.length (P.applied_opts s1) in
              (match
                 P.apply_opt s2 (K.Opt.Upcast { axis = 0; amount = 2 })
               with
              | _ ->
                  equal int opts1_before (List.length (P.applied_opts s1))
              | exception P.Opt_error _ ->
                  equal int opts1_before (List.length (P.applied_opts s1)))
          | _ -> failwith "expected at least 2 actions");
    ]

(* Group 6: Reduce kernel and renderer *)

let reduce_and_renderer_tests =
  group "reduce kernel and renderer"
    [
      test "reduce kernel returns unroll actions" (fun () ->
          let ast = reduce_global_ast ~s0:4 ~s1:4 ~sr:8 in
          let s = P.create ast (gpu_renderer ()) in
          let actions = S.get_kernel_actions ~include_0:false s in
          is_true
            ~msg:"reduce kernel should have Unroll actions"
            (List.exists
               (fun (_, s2) -> has_opt_matching is_unroll s2)
               actions));
      test "GPU renderer returns local actions" (fun () ->
          let ast = elementwise_global_ast ~s0:16 ~s1:16 in
          let s = P.create ast (gpu_renderer ()) in
          let actions = S.get_kernel_actions ~include_0:false s in
          is_true
            ~msg:"GPU renderer should offer Local actions"
            (List.exists
               (fun (_, s2) -> has_opt_matching is_local s2)
               actions));
      test "CPU renderer returns no local actions" (fun () ->
          let ast = elementwise_ast ~s0:16 ~s1:16 in
          let s = P.create ast (cpu_renderer ()) in
          let actions = S.get_kernel_actions ~include_0:false s in
          is_false
            ~msg:"CPU renderer should not offer Local actions"
            (List.exists
               (fun (_, s2) -> has_opt_matching is_local s2)
               actions));
    ]

(* Group 7: Diskcache contract *)

let diskcache_tests =
  group "diskcache"
    [
      test "put then get returns same opts" (fun () ->
          let key = Printf.sprintf "test_%d" (Random.bits ()) in
          let opts =
            K.Opt.
              [
                Upcast { axis = 0; amount = 4 };
                Local { axis = 1; amount = 8 };
              ]
          in
          Diskcache.put ~table:"test_beam" ~key opts;
          match
            (Diskcache.get ~table:"test_beam" ~key : K.Opt.t list option)
          with
          | Some got -> equal int (List.length opts) (List.length got)
          | None -> fail "cache miss after put");
      test "missing key returns None" (fun () ->
          let result =
            (Diskcache.get ~table:"test_beam" ~key:"nonexistent_key_12345"
             : int option)
          in
          equal (option int) None result);
    ]

(* Group 8: Get_kernel_actions parity *)

let parity_tests =
  group "get_kernel_actions parity"
    [
      (* Verify max_up budget is enforced across different limits. *)
      test "max_up=2 enforces upcast budget" (fun () ->
          let ast = elementwise_global_ast ~s0:16 ~s1:16 in
          let s = P.create ast (gpu_renderer ()) in
          let actions = S.get_kernel_actions ~include_0:false ~max_up:2 s in
          List.iter
            (fun (_, s2) ->
              let up = upcast_product s2 in
              is_true
                ~msg:(Printf.sprintf "upcast_product %d should be <= 2" up)
                (up <= 2))
            actions);
      test "max_up=4 allows larger upcasts" (fun () ->
          let ast = elementwise_global_ast ~s0:16 ~s1:16 in
          let s = P.create ast (gpu_renderer ()) in
          let actions_2 =
            S.get_kernel_actions ~include_0:false ~max_up:2 s
          in
          let actions_4 =
            S.get_kernel_actions ~include_0:false ~max_up:4 s
          in
          is_true
            ~msg:"max_up=4 should allow at least as many actions as max_up=2"
            (List.length actions_4 >= List.length actions_2));
      (* Verify that actions with upcast matching the shape dim are filtered
         as no-ops when the zero-amount variant exists. *)
      test "noop actions are filtered" (fun () ->
          let ast = elementwise_global_ast ~s0:4 ~s1:4 in
          let s = P.create ast (gpu_renderer ()) in
          let actions = S.get_kernel_actions ~include_0:false s in
          (* Upcast(axis=0, amount=4) should be filtered because shape[0]=4
             and Upcast(axis=0, amount=0) exists in the actions list. *)
          let has_noop =
            List.exists
              (fun (_, s2) ->
                List.exists
                  (fun opt ->
                    match opt with
                    | K.Opt.Upcast { axis = 0; amount = 4 } -> true
                    | _ -> false)
                  (P.applied_opts s2))
              actions
          in
          is_false ~msg:"noop upcast(0,4) should be filtered" has_noop);
      (* Verify reduce kernels produce unroll actions on the reduce axis. *)
      test "reduce kernel has unroll on reduce axis" (fun () ->
          let ast = reduce_global_ast ~s0:4 ~s1:4 ~sr:8 in
          let s = P.create ast (gpu_renderer ()) in
          let actions = S.get_kernel_actions ~include_0:false s in
          is_true
            ~msg:"reduce kernel should have unroll actions"
            (List.exists
               (fun (_, s2) -> has_opt_matching is_unroll s2)
               actions));
    ]

(* Group 9: Get_test_global_size *)

let get_test_global_size_tests =
  group "get_test_global_size"
    [
      test "preserves size when under max" (fun () ->
          let result, factor = S.get_test_global_size [| 16; 16 |] 65536 in
          equal (array int) [| 16; 16 |] result;
          is_true ~msg:"factor should be 1.0" (factor = 1.0));
      test "scales down when product exceeds max" (fun () ->
          let result, factor =
            S.get_test_global_size [| 128; 128; 128 |] 65536
          in
          let product = Array.fold_left ( * ) 1 result in
          is_true
            ~msg:(Printf.sprintf "scaled product %d <= 65536" product)
            (product <= 65536);
          is_true ~msg:"factor >= 1.0" (factor >= 1.0));
      test "factor equals input / scaled product" (fun () ->
          let result, factor = S.get_test_global_size [| 256; 256 |] 65536 in
          let input_size = 256 * 256 in
          let scaled_size = Array.fold_left ( * ) 1 result in
          let expected =
            Float.of_int input_size /. Float.of_int scaled_size
          in
          is_true
            ~msg:
              (Printf.sprintf "factor %e should equal %e" factor expected)
            (Float.abs (factor -. expected) < 1e-10));
      test "halves from last axis first" (fun () ->
          (* Halving iterates from last axis to first *)
          let result, _ = S.get_test_global_size [| 32; 64 |] 1024 in
          (* 32*64=2048 > 1024. Axis 1 (64) halved to 32: 32*32=1024. *)
          equal (array int) [| 32; 32 |] result);
      test "stops halving at 16" (fun () ->
          let result, _ = S.get_test_global_size [| 32; 32 |] 100 in
          (* Both axes halve to 16, then loop exits (16*16=256 > 100 but
             no axis > 16 remains). *)
          equal (array int) [| 16; 16 |] result);
      test "single dimension" (fun () ->
          let result, factor = S.get_test_global_size [| 131072 |] 65536 in
          let product = Array.fold_left ( * ) 1 result in
          is_true
            ~msg:(Printf.sprintf "scaled %d <= 65536" product)
            (product <= 65536);
          is_true ~msg:"factor > 1" (factor > 1.0));
      test "already at max is unchanged" (fun () ->
          let result, factor = S.get_test_global_size [| 256; 256 |] 65536 in
          equal (array int) [| 256; 256 |] result;
          is_true ~msg:"factor should be 1.0" (factor = 1.0));
    ]

(* Entry *)

let () =
  run __FILE__
    [
      actions_tests;
      basic_behavior_tests;
      filtering_tests;
      budget_tests;
      isolation_tests;
      reduce_and_renderer_tests;
      diskcache_tests;
      parity_tests;
      get_test_global_size_tests;
    ]
