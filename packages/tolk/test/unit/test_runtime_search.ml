(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Runtime tests for Search.

   Tests that beam_search compiles and executes kernels correctly on real
   hardware (CPU via Clang). Complements the pure-logic unit tests in
   test_codegen_search.ml. *)

open Windtrap
open Tolk
open Tolk_uop
module U = Uop
module D = Dtype
module C = Const
module Ak = Axis_type
module P = Postrange

(* Helpers *)

let idx n = U.const_int n
let f32 x = U.const (C.float D.float32 x)
let ren = Cstyle.clang Gpu_target.X86_64
let index_ptr ptr idx = U.index ~ptr ~idxs:[idx] ()

let cpu name = Tolk_cpu.create ("CPU:" ^ name)

let f32_to_bytes values =
  let bytes = Bytes.create (List.length values * 4) in
  List.iteri
    (fun i v -> Bytes.set_int32_le bytes (i * 4) (Int32.bits_of_float v))
    values;
  bytes

let read_f32_buffer buf =
  let bytes = Device.Buffer.as_bytes buf in
  let n = Bytes.length bytes / 4 in
  List.init n (fun i -> Int32.float_of_bits (Bytes.get_int32_le bytes (i * 4)))

let create_f32_buffer device n values =
  let buf = Device.create_buffer ~size:n ~dtype:D.float32 device in
  Device.Buffer.ensure_allocated buf;
  Device.Buffer.copyin buf (f32_to_bytes values);
  buf

let create_bufs_for_kernel device ast =
  List.map
    (fun p ->
      let size = List.fold_left ( * ) 1 (U.max_shape p) in
      let buf = Device.create_buffer ~size ~dtype:(U.dtype p) device in
      Device.Buffer.ensure_allocated buf;
      buf)
    (P.bufs_from_ast ast)

(* AST Fixture Builders *)

(* Elementwise: output[i] = input[i] * 2, single flat loop.
   Avoids transcendental ops (exp2/sin/log2) because this runtime-search
   fixture is meant to exercise beam compilation on arithmetic kernels. *)
let elementwise_1d_ast ~n =
  let p0 = U.param ~slot:0 ~dtype:D.float32 ~shape:(idx n) () in
  let p1 = U.param ~slot:1 ~dtype:D.float32 ~shape:(idx n) () in
  let r0 = U.range ~size:(idx n) ~axis:0 ~kind:Ak.Loop ~dtype:D.index () in
  let in_idx = index_ptr p1 r0 in
  let ld = U.load ~src:in_idx () in
  let value = U.alu_binary ~op:Ops.Mul ~lhs:ld ~rhs:(f32 2.0) in
  let out_idx = index_ptr p0 r0 in
  let st = U.store ~dst:out_idx ~value () in
  let e = U.end_ ~value:st ~ranges:[ r0 ] in
  let ki =
    {
      U.name = "test";
      axis_types = [];
      dont_use_locals = false;
      applied_opts = [];
      opts_to_apply = None;
      estimates = None;
      beam = 0;
    }
  in
  U.sink ~kernel_info:ki [ e ]

let elementwise_1d_ast_with_params ~n ~ptr_n ~out_slot ~in_slot ?shape () =
  let shape = match shape with Some s -> s | None -> idx ptr_n in
  let p0 = U.param ~slot:out_slot ~dtype:D.float32 ~shape () in
  let p1 = U.param ~slot:in_slot ~dtype:D.float32 ~shape () in
  let r0 = U.range ~size:(idx n) ~axis:0 ~kind:Ak.Loop ~dtype:D.index () in
  let in_idx = index_ptr p1 r0 in
  let ld = U.load ~src:in_idx () in
  let value = U.alu_binary ~op:Ops.Mul ~lhs:ld ~rhs:(f32 2.0) in
  let out_idx = index_ptr p0 r0 in
  let st = U.store ~dst:out_idx ~value () in
  let e = U.end_ ~value:st ~ranges:[ r0 ] in
  let ki =
    {
      U.name = "test";
      axis_types = [];
      dont_use_locals = false;
      applied_opts = [];
      opts_to_apply = None;
      estimates = None;
      beam = 0;
    }
  in
  U.sink ~kernel_info:ki [ e ]

(* Elementwise 2D: output[r0,r1] = input[r0,r1] * 2 *)
let elementwise_2d_ast ~s0 ~s1 =
  let n = s0 * s1 in
  let p0 = U.param ~slot:0 ~dtype:D.float32 ~shape:(idx n) () in
  let p1 = U.param ~slot:1 ~dtype:D.float32 ~shape:(idx n) () in
  let r0 = U.range ~size:(idx s0) ~axis:0 ~kind:Ak.Loop ~dtype:D.index () in
  let r1 = U.range ~size:(idx s1) ~axis:1 ~kind:Ak.Loop ~dtype:D.index () in
  let open U.O in
  let linear_idx = (r0 * idx s1) + r1 in
  let in_idx = index_ptr p1 linear_idx in
  let ld = U.load ~src:in_idx () in
  let value = U.alu_binary ~op:Ops.Mul ~lhs:ld ~rhs:(f32 2.0) in
  let out_idx = index_ptr p0 linear_idx in
  let st = U.store ~dst:out_idx ~value () in
  let e = U.end_ ~value:st ~ranges:[ r0; r1 ] in
  let ki =
    {
      U.name = "test";
      axis_types = [];
      dont_use_locals = false;
      applied_opts = [];
      opts_to_apply = None;
      estimates = None;
      beam = 0;
    }
  in
  U.sink ~kernel_info:ki [ e ]

(* Tests *)

let beam_search_tests =
  group "beam_search on CPU"
    [
      slow "Lowering.compile produces correct output" (fun () ->
          let device = cpu "compile-test" in
          let n = 16 in
          let ast = elementwise_1d_ast ~n in
          let s = P.create ast ren in
          let opt_ast = P.get_optimized_ast (P.copy s) in
          let program = Device.compile_program device (Linearizer.linearize (Codegen_lower.lower ren opt_ast)) in
          let out_buf = create_f32_buffer device n (List.init n (fun _ -> 0.0)) in
          let in_buf =
            create_f32_buffer device n (List.init n (fun i -> Float.of_int i))
          in
          let car = Realize.Compiled_runner.create ~device program in
          ignore (Realize.Compiled_runner.call car [ out_buf; in_buf ] []
            ~wait:true ~timeout:None);
          Device.synchronize device;
          let output = read_f32_buffer out_buf in
          let expected =
            List.init n (fun i -> let x = Float.of_int i in x +. x)
          in
          List.iter2
            (fun exp act ->
              is_true
                ~msg:(Printf.sprintf "expected %.4f, got %.4f" exp act)
                (Float.abs (exp -. act) < 1e-4))
            expected output);
      slow "completes on 1D elementwise kernel" (fun () ->
          let device = cpu "beam-1d" in
          let n = 16 in
          let ast = elementwise_1d_ast ~n in
          let s = P.create ast ren in
          let rawbufs = create_bufs_for_kernel device ast in
          let input_data = List.init n (fun i -> Float.of_int i) in
          Device.Buffer.copyin (List.nth rawbufs 1) (f32_to_bytes input_data);
          let result = Search.beam_search s rawbufs 1 device in
          is_true (P.shape_len result >= 1));
      slow "completes on 2D elementwise kernel" (fun () ->
          let device = cpu "beam-2d" in
          let ast = elementwise_2d_ast ~s0:8 ~s1:8 in
          let s = P.create ast ren in
          let rawbufs = create_bufs_for_kernel device ast in
          let result = Search.beam_search s rawbufs 1 device in
          is_true (P.shape_len result >= 1));
      slow "accepts compact raw buffers for sparse parameter slots" (fun () ->
          let device = cpu "beam-sparse-slots" in
          let n = 16 in
          let ast =
            elementwise_1d_ast_with_params ~n ~ptr_n:n ~out_slot:0
              ~in_slot:2 ()
          in
          let s = P.create ast ren in
          let rawbufs = create_bufs_for_kernel device ast in
          let input_data = List.init n (fun i -> Float.of_int i) in
          Device.Buffer.copyin (List.nth rawbufs 1) (f32_to_bytes input_data);
          let result = Search.beam_search s rawbufs 1 device in
          is_true (P.shape_len result >= 1));
      slow "uses explicit max shape for beam buffers" (fun () ->
          let device = cpu "beam-explicit-shape" in
          let n = 16 in
          let ast =
            elementwise_1d_ast_with_params ~n ~ptr_n:1 ~out_slot:0
              ~in_slot:1 ~shape:(idx n) ()
          in
          let s = P.create ast ren in
          let rawbufs = create_bufs_for_kernel device ast in
          let result = Search.beam_search s rawbufs 1 device in
          is_true (P.shape_len result >= 1));
      slow "optimized kernel produces correct output" (fun () ->
          let device = cpu "beam-correct" in
          let n = 16 in
          let ast = elementwise_1d_ast ~n in
          let s = P.create ast ren in
          let rawbufs = create_bufs_for_kernel device ast in
          let input_data = List.init n (fun i -> Float.of_int i) in
          Device.Buffer.copyin (List.nth rawbufs 1) (f32_to_bytes input_data);
          let result = Search.beam_search s rawbufs 1 device in
          let out_buf = create_f32_buffer device n (List.init n (fun _ -> 0.0)) in
          let in_buf = create_f32_buffer device n input_data in
          let opt_ast = P.get_optimized_ast (P.copy result) in
          let program = Device.compile_program device (Linearizer.linearize (Codegen_lower.lower (P.ren result) opt_ast)) in
          let car = Realize.Compiled_runner.create ~device program in
          ignore (Realize.Compiled_runner.call car [ out_buf; in_buf ] []
            ~wait:true ~timeout:None);
          Device.synchronize device;
          let output = read_f32_buffer out_buf in
          let expected = List.map (fun x -> x +. x) input_data in
          List.iter2
            (fun exp act ->
              is_true
                ~msg:(Printf.sprintf "expected %.4f, got %.4f" exp act)
                (Float.abs (exp -. act) < 1e-4))
            expected output);
      (* Verify beam_search does not corrupt input buffer contents. *)
      slow "beam_search does not corrupt input buffers" (fun () ->
          let device = cpu "no-mutate" in
          let n = 16 in
          let ast = elementwise_1d_ast ~n in
          let s = P.create ast ren in
          let rawbufs = create_bufs_for_kernel device ast in
          let input_data = List.init n (fun i -> Float.of_int (i + 1)) in
          Device.Buffer.copyin (List.nth rawbufs 1) (f32_to_bytes input_data);
          let input_before = read_f32_buffer (List.nth rawbufs 1) in
          ignore (Search.beam_search s rawbufs 1 device : P.t);
          let input_after = read_f32_buffer (List.nth rawbufs 1) in
          List.iter2
            (fun before after ->
              is_true
                ~msg:
                  (Printf.sprintf "input buffer mutated: %.4f -> %.4f" before
                     after)
                (Float.abs (before -. after) < 1e-6))
            input_before input_after);
      (* Verify beam_search completes on a kernel with variable-sized range. *)
      slow "completes on variable-sized kernel" (fun () ->
          let device = cpu "beam-var" in
          let n = 16 in
          let p0 = U.param ~slot:0 ~dtype:D.float32 ~shape:(idx n) () in
          let p1 = U.param ~slot:1 ~dtype:D.float32 ~shape:(idx n) () in
          let var = U.variable ~name:"v" ~min_val:1 ~max_val:n () in
          let r0 =
            U.range ~size:var ~axis:0 ~kind:Ak.Loop ~dtype:D.index ()
          in
          let in_idx = index_ptr p1 r0 in
          let ld = U.load ~src:in_idx () in
          let value = U.alu_binary ~op:Ops.Mul ~lhs:ld ~rhs:(f32 2.0) in
          let out_idx = index_ptr p0 r0 in
          let st = U.store ~dst:out_idx ~value () in
          let e = U.end_ ~value:st ~ranges:[ r0 ] in
          let ki =
            {
              U.name = "test";
              axis_types = [];
              dont_use_locals = false;
              applied_opts = [];
              opts_to_apply = None;
              estimates = None;
              beam = 0;
            }
          in
          let ast = U.sink ~kernel_info:ki [ e ] in
          let s = P.create ast ren in
          let rawbufs = create_bufs_for_kernel device ast in
          let result = Search.beam_search s rawbufs 1 device in
          ignore (result : P.t));
      (* Verify disable_cache parameter works: running beam_search twice
         with disable_cache=true should both complete (no stale cache). *)
      slow "disable_cache bypasses cache" (fun () ->
          let device = cpu "beam-nocache" in
          let n = 16 in
          let ast = elementwise_1d_ast ~n in
          let s = P.create ast ren in
          let rawbufs = create_bufs_for_kernel device ast in
          let r1 =
            Search.beam_search ~disable_cache:true s rawbufs 1 device
          in
          let r2 =
            Search.beam_search ~disable_cache:true s rawbufs 1 device
          in
          is_true (P.shape_len r1 >= 1);
          is_true (P.shape_len r2 >= 1));
    ]

(* Search timing *)

(* Guards the CPU wait-timing contract BEAM ranking depends on.

   search.ml times every candidate through
   [Realize.Compiled_runner.call ~wait:true] and maps a [None] result to
   [infinity] (see time_program). If CPU timing regresses to [None], every
   candidate ties at [infinity] and BEAM can no longer order them. This drives
   the same construction the search timing loop uses — optimize, lower,
   linearize, compile, then time with [~wait:true] — and requires each
   measurement be finite and positive so candidates stay rankable. The raw
   runtime call is guarded in test_runtime_cpu; this pins the property for
   kernels built through the beam-search codegen pipeline. *)
let search_timing_tests =
  group "search timing on CPU"
    [
      slow "wait:true yields a finite, rankable time" (fun () ->
          let device = cpu "search-timing" in
          let n = 16 in
          let ast = elementwise_1d_ast ~n in
          let s = P.create ast ren in
          let opt_ast = P.get_optimized_ast (P.copy s) in
          let program =
            Device.compile_program device
              (Linearizer.linearize (Codegen_lower.lower ren opt_ast))
          in
          let out_buf =
            create_f32_buffer device n (List.init n (fun _ -> 0.0))
          in
          let in_buf =
            create_f32_buffer device n (List.init n (fun i -> Float.of_int i))
          in
          let car = Realize.Compiled_runner.create ~device program in
          (* Sample the cnt-style timing loop. A [None] would become [infinity]
             in search.ml and collapse ranking, so fail loudly on it. *)
          for _ = 1 to 3 do
            match
              Realize.Compiled_runner.call car [ out_buf; in_buf ] [] ~wait:true
                ~timeout:None
            with
            | None ->
                fail
                  "CPU wait:true timing returned None; BEAM would tie every \
                   candidate at infinity"
            | Some t ->
                is_true
                  ~msg:(Printf.sprintf "CPU wait:true timing not finite: %g" t)
                  (Float.is_finite t);
                is_true
                  ~msg:(Printf.sprintf "CPU wait:true timing not positive: %g" t)
                  (t > 0.)
          done);
    ]

(* Entry *)

let () = run __FILE__ [ beam_search_tests; search_timing_tests ]
