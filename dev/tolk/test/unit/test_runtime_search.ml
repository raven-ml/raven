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
open Tolk_ir
module K = Kernel
module D = Dtype
module C = Const
module Ak = Axis_kind
module P = Postrange

(* Helpers *)

let idx n = K.const (C.int D.index n)
let ren = Cstyle.clang

let f32_ptr n = D.ptr_of D.float32 ~addrspace:Global ~size:n

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
      match K.view p with
      | Param { dtype = pty; _ } ->
          let buf = Device.create_buffer ~size:(D.ptr_size pty) ~dtype:(D.base pty) device in
          Device.Buffer.ensure_allocated buf;
          buf
      | _ -> assert false)
    (P.bufs_from_ast ast)

(* AST Fixture Builders *)

(* Elementwise: output[i] = input[i] + input[i], single flat loop.
   Avoids transcendental ops (exp2/sin/log2) because the Clang freestanding
   backend compiles to ELF without libm — those ops require the transcendental
   decomposition pass which is not yet ported. *)
let elementwise_1d_ast ~n =
  let p0 = K.param ~idx:0 ~dtype:(f32_ptr n) in
  let p1 = K.param ~idx:1 ~dtype:(f32_ptr n) in
  let r0 = K.range ~size:(idx n) ~axis:0 ~kind:Ak.Loop ~dtype:D.index () in
  let in_idx = K.index ~ptr:p1 ~idxs:[ r0 ] () in
  let ld = K.load ~src:in_idx () in
  let value = K.binary ~op:`Add ~lhs:ld ~rhs:ld in
  let out_idx = K.index ~ptr:p0 ~idxs:[ r0 ] () in
  let st = K.store ~dst:out_idx ~value ~ranges:[] in
  let e = K.end_ ~value:st ~ranges:[ r0 ] () in
  let ki =
    {
      K.name = "test";
      axis_kinds = [];
      dont_use_locals = false;
      applied_opts = [];
      opts_to_apply = None;
      estimates = None;
    }
  in
  K.sink ~kernel_info:ki [ e ]

(* Elementwise 2D: output[r0,r1] = input[r0,r1] + input[r0,r1] *)
let elementwise_2d_ast ~s0 ~s1 =
  let n = s0 * s1 in
  let p0 = K.param ~idx:0 ~dtype:(f32_ptr n) in
  let p1 = K.param ~idx:1 ~dtype:(f32_ptr n) in
  let r0 = K.range ~size:(idx s0) ~axis:0 ~kind:Ak.Loop ~dtype:D.index () in
  let r1 = K.range ~size:(idx s1) ~axis:1 ~kind:Ak.Loop ~dtype:D.index () in
  let open K.O in
  let in_idx = K.index ~ptr:p1 ~idxs:[ r0 * idx s1 + r1 ] () in
  let ld = K.load ~src:in_idx () in
  let value = K.binary ~op:`Add ~lhs:ld ~rhs:ld in
  let out_idx = K.index ~ptr:p0 ~idxs:[ r0 * idx s1 + r1 ] () in
  let st = K.store ~dst:out_idx ~value ~ranges:[] in
  let e = K.end_ ~value:st ~ranges:[ r0; r1 ] () in
  let ki =
    {
      K.name = "test";
      axis_kinds = [];
      dont_use_locals = false;
      applied_opts = [];
      opts_to_apply = None;
      estimates = None;
    }
  in
  K.sink ~kernel_info:ki [ e ]

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
          let program = Lowering.compile device ren opt_ast in
          let out_buf = create_f32_buffer device n (List.init n (fun _ -> 0.0)) in
          let in_buf =
            create_f32_buffer device n (List.init n (fun i -> Float.of_int i))
          in
          let queue = Device.queue device in
          Device.Queue.exec queue program [ out_buf; in_buf ] [];
          Device.Queue.synchronize queue;
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
          let program = Lowering.compile device (P.ren result) opt_ast in
          let queue = Device.queue device in
          Device.Queue.exec queue program [ out_buf; in_buf ] [];
          Device.Queue.synchronize queue;
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
          let p0 = K.param ~idx:0 ~dtype:(f32_ptr n) in
          let p1 = K.param ~idx:1 ~dtype:(f32_ptr n) in
          let var = K.define_var ~name:"v" ~lo:1 ~hi:n () in
          let r0 =
            K.range ~size:var ~axis:0 ~kind:Ak.Loop ~dtype:D.index ()
          in
          let in_idx = K.index ~ptr:p1 ~idxs:[ r0 ] () in
          let ld = K.load ~src:in_idx () in
          let value = K.binary ~op:`Add ~lhs:ld ~rhs:ld in
          let out_idx = K.index ~ptr:p0 ~idxs:[ r0 ] () in
          let st = K.store ~dst:out_idx ~value ~ranges:[] in
          let e = K.end_ ~value:st ~ranges:[ r0 ] () in
          let ki =
            {
              K.name = "test";
              axis_kinds = [];
              dont_use_locals = false;
              applied_opts = [];
              opts_to_apply = None;
              estimates = None;
            }
          in
          let ast = K.sink ~kernel_info:ki [ e ] in
          let s = P.create ast ren in
          let rawbufs = create_bufs_for_kernel device ast in
          let result = Search.beam_search s rawbufs 1 device in
          is_true (P.shape_len result >= 1));
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

(* Entry *)

let () = run __FILE__ [ beam_search_tests ]
