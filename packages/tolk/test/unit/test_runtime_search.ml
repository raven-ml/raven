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
  let r0 = U.range ~size:(idx n) ~axis:0 ~kind:Ak.Weak ~dtype:D.weakint () in
  let in_idx = index_ptr p1 r0 in
  let ld = U.load ~src:in_idx () in
  let value = U.alu_binary ~op:Ops.Mul ~lhs:ld ~rhs:(f32 2.0) in
  let out_idx = index_ptr p0 r0 in
  let st = U.store ~dst:out_idx ~value () in
  let e = U.end_ ~value:st ~ranges:[ r0 ] in
  let ki =
    {
      U.name = "test";
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
  let r0 = U.range ~size:(idx n) ~axis:0 ~kind:Ak.Weak ~dtype:D.weakint () in
  let in_idx = index_ptr p1 r0 in
  let ld = U.load ~src:in_idx () in
  let value = U.alu_binary ~op:Ops.Mul ~lhs:ld ~rhs:(f32 2.0) in
  let out_idx = index_ptr p0 r0 in
  let st = U.store ~dst:out_idx ~value () in
  let e = U.end_ ~value:st ~ranges:[ r0 ] in
  let ki =
    {
      U.name = "test";
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
  let r0 = U.range ~size:(idx s0) ~axis:0 ~kind:Ak.Weak ~dtype:D.weakint () in
  let r1 = U.range ~size:(idx s1) ~axis:1 ~kind:Ak.Weak ~dtype:D.weakint () in
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
      applied_opts = [];
      opts_to_apply = None;
      estimates = None;
      beam = 0;
    }
  in
  U.sink ~kernel_info:ki [ e ]

(* Tests *)

let to_program device =
  Codegen.to_program ~optimize:false device (Device.renderer device)

let program_call program bufs =
  let info = Option.get (U.as_program_info program) in
  let selected = List.combine info.globals bufs in
  let args = List.init (1 + List.fold_left max (-1) info.globals) (fun slot ->
      match List.assoc_opt slot selected with
      | Some buf -> U.from_buffer buf | None -> U.noop ()) in
  U.call ~body:program ~args ~info:U.{grad_fxn = None; name = None;
    precompile = false; precompile_backward = false; dtype = D.void; aux = None}

let beam_search_tests =
  group "beam_search on CPU"
    [
      slow "selected kernel compilation produces correct output" (fun () ->
          let device = cpu "compile-test" in
          let n = 16 in
          let ast = elementwise_1d_ast ~n in
          let s = P.create ast ren in
          let opt_ast = P.get_optimized_ast (P.copy s) in
          let program = to_program device opt_ast in
          let out_buf = create_f32_buffer device n (List.init n (fun _ -> 0.0)) in
          let in_buf =
            create_f32_buffer device n (List.init n (fun i -> Float.of_int i))
          in
          Realize.run_linear ~device ~to_program ~wait:true
            (U.linear [program_call program [out_buf; in_buf]]);
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
          let result = Search.beam_search ~to_program s rawbufs ~var_vals:[] 1 device in
          is_true (P.shape_len result >= 1));
      slow "completes on 2D elementwise kernel" (fun () ->
          let device = cpu "beam-2d" in
          let ast = elementwise_2d_ast ~s0:8 ~s1:8 in
          let s = P.create ast ren in
          let rawbufs = create_bufs_for_kernel device ast in
          let result = Search.beam_search ~to_program s rawbufs ~var_vals:[] 1 device in
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
          let result = Search.beam_search ~to_program s rawbufs ~var_vals:[] 1 device in
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
          let result = Search.beam_search ~to_program s rawbufs ~var_vals:[] 1 device in
          is_true (P.shape_len result >= 1));
      slow "optimized kernel produces correct output" (fun () ->
          let device = cpu "beam-correct" in
          let n = 16 in
          let ast = elementwise_1d_ast ~n in
          let s = P.create ast ren in
          let rawbufs = create_bufs_for_kernel device ast in
          let input_data = List.init n (fun i -> Float.of_int i) in
          Device.Buffer.copyin (List.nth rawbufs 1) (f32_to_bytes input_data);
          let result = Search.beam_search ~to_program s rawbufs ~var_vals:[] 1 device in
          let out_buf = create_f32_buffer device n (List.init n (fun _ -> 0.0)) in
          let in_buf = create_f32_buffer device n input_data in
          let opt_ast = P.get_optimized_ast (P.copy result) in
          let program = to_program device opt_ast in
          Realize.run_linear ~device ~to_program ~wait:true
            (U.linear [program_call program [out_buf; in_buf]]);
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
          ignore (Search.beam_search ~to_program s rawbufs ~var_vals:[] 1 device : P.t);
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
            U.range ~size:var ~axis:0 ~kind:Ak.Weak ~dtype:D.weakint ()
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
              applied_opts = [];
              opts_to_apply = None;
              estimates = None;
              beam = 0;
            }
          in
          let ast = U.sink ~kernel_info:ki [ e ] in
          let s = P.create ast ren in
          let rawbufs = create_bufs_for_kernel device ast in
          let result = Search.beam_search ~to_program s rawbufs ~var_vals:[ "v", 8 ] 1 device in
          ignore (result : P.t));
      test "uses the supplied symbolic value during timing" (fun () ->
          let device = cpu "beam-explicit-variable" in
          let variable = U.variable ~name:"scale" ~min_val:(-4) ~max_val:(-1) () in
          let ast = U.substitute
              [ f32 2., U.cast ~src:variable ~dtype:D.float32 ]
              (elementwise_1d_ast ~n:4) in
          let output = create_f32_buffer device 4 [ 0.; 0.; 0.; 0. ] in
          let input = create_f32_buffer device 4 [ 1.; 2.; 3.; 4. ] in
          Fun.protect
            ~finally:(fun () -> List.iter Device.Buffer.deallocate [ output; input ])
            (fun () ->
              List.iter (fun var_vals ->
                  raises_match (function Invalid_argument _ -> true | _ -> false)
                    (fun () -> Search.beam_search ~to_program ~disable_cache:true
                      (P.create ast ren) [ output; input ] ~var_vals 1 device))
                [ []; [ "scale", -5 ]; [ "scale", 0 ] ];
              List.iter (fun scale ->
                  ignore (Search.beam_search ~to_program ~disable_cache:true
                    (P.create ast ren) [ output; input ] ~var_vals:[ "scale", scale ]
                    1 device : P.t);
                  equal (list (float 1e-6))
                    (List.map (fun x -> Float.of_int (scale * x)) [ 1; 2; 3; 4 ])
                    (read_f32_buffer output))
                [ -4; -1 ]));
      (* Verify disable_cache parameter works: running beam_search twice
         with disable_cache=true should both complete (no stale cache). *)
      slow "disable_cache bypasses cache" (fun () ->
          let device = cpu "beam-nocache" in
          let n = 16 in
          let ast = elementwise_1d_ast ~n in
          let s = P.create ast ren in
          let rawbufs = create_bufs_for_kernel device ast in
          let r1 =
            Search.beam_search ~to_program ~disable_cache:true s rawbufs ~var_vals:[] 1 device
          in
          let r2 =
            Search.beam_search ~to_program ~disable_cache:true s rawbufs ~var_vals:[] 1 device
          in
          is_true (P.shape_len r1 >= 1);
          is_true (P.shape_len r2 >= 1));
    ]

(* Search timing *)

(* Beam ranking needs finite positive samples through the same scoped compiled
   execution used by Search.time_program. Sample one selected candidate several
   times to catch missing or host-only timing results. *)
let search_timing_tests =
  group "search timing on CPU"
    [
      slow "wait:true yields a finite, rankable time" (fun () ->
          let device = cpu "search-timing" in
          let n = 16 in
          let ast = elementwise_1d_ast ~n in
          let s = P.create ast ren in
          let opt_ast = P.get_optimized_ast (P.copy s) in
          let program = to_program device opt_ast in
          let out_buf =
            create_f32_buffer device n (List.init n (fun _ -> 0.0))
          in
          let in_buf =
            create_f32_buffer device n (List.init n (fun i -> Float.of_int i))
          in
          Realize.time_call ~device ~to_program
            (program_call program [out_buf; in_buf]) (fun sample ->
              for _ = 1 to 3 do
                let elapsed = sample () in
                is_true ~msg:"CPU timing is finite" (Float.is_finite elapsed);
                is_true ~msg:"CPU timing is positive" (elapsed > 0.)
              done));
    ]

(* Runtime handles loaded only to time a candidate must not outlive timing. *)
let transient_program_lifetimes =
  let check failure () =
    let backing = cpu "beam-lifetime-buffers" in
    let sample = Device.create_buffer ~size:1 ~dtype:D.float32 backing in
    let allocator = Device.Buffer.allocator sample in
    let loaded = ref 0 and freed = ref 0 and pending = ref false in
    let runtime _ =
      incr loaded;
      let released = ref false in
      let call _ ~global:_ ~local:_ ~vals:_ ~wait:_ ~timeout:_ =
        is_false ~msg:"timing never reuses a released program" !released;
        pending := true;
        match failure with
        | Some exn -> raise exn
        | None -> pending := false; Some 1e-6
      in
      let free () =
        is_false ~msg:"queued work completes before program release" !pending;
        is_false ~msg:"program is released exactly once" !released;
        released := true;
        incr freed
      in
      Device.{ call; free; handle = 0n }
    in
    let compiler = Compiler.make ~name:"BEAM_LIFETIME"
        ~compile:Bytes.of_string () in
    let ren = Renderer.with_compiler compiler ren in
    let renderer_set = Device.Renderer_set.make ~device:"CPU" ~arch:"generic"
        [ "CLANG", (fun _ -> ren) ] in
    let device = Device.make ~name:"CPU:beam-lifetime" ~allocator ~renderer_set
        ~runtime ~synchronize:(fun timeout -> ignore timeout; pending := false) () in
    let ast = elementwise_1d_ast ~n:4 in
    let rawbufs = create_bufs_for_kernel device ast in
    let cachelevel = Sys.getenv_opt "CACHELEVEL" in
    Unix.putenv "CACHELEVEL" "0";
    Fun.protect
      ~finally:(fun () ->
        Unix.putenv "CACHELEVEL" (Option.value cachelevel ~default:"");
        List.iter Device.Buffer.deallocate rawbufs)
      (fun () ->
        let search () =
          ignore (Search.beam_search ~to_program ~disable_cache:true
            (P.create ast ren) rawbufs ~var_vals:[] 1 device : P.t)
        in
        (match failure with
         | Some Exit -> raises Exit search
         | _ -> search ());
        is_true ~msg:"the test timed at least one compiled candidate" (!loaded > 0);
        equal ~msg:"every transient program is released before search returns"
          int !loaded !freed)
  in
  group "transient program lifetimes"
    [ test "successful timings release programs" (check None);
      test "rejected timings release programs" (check (Some (Failure "timing failed")));
      test "interrupted search releases programs" (check (Some Exit)) ]

let codegen_midpoint_rounds_down () =
  let backing = cpu "beam-midpoint" in
  let sample = Device.create_buffer ~size:1 ~dtype:D.float32 backing in
  let observed = ref [] and evictions = ref 0 in
  let runtime obj =
    let prg = Device.runtime backing obj in
    let eviction = List.for_all (fun (arg : Tiny_elf.argument) -> arg.addrspace <> D.Alu)
        obj.signature in
    let call bufs ~global ~local ~vals ~wait ~timeout =
      if eviction then begin
        equal int 1 (Array.length bufs);
        equal int (1024 * 1024 * 4) (Device.Buffer.nbytes bufs.(0));
        equal (array int64) [||] vals;
        incr evictions
      end else observed := Array.to_list vals :: !observed;
      prg.call bufs ~global ~local ~vals ~wait ~timeout
    in
    { prg with call }
  in
  let ren = Device.renderer backing in
  let renderer_set = Device.Renderer_set.make ~device:"CPU" ~arch:"generic"
      [ "CLANG", (fun _ -> ren) ] in
  let device = Device.make ~name:"CPU:beam-midpoint-recording"
      ~allocator:(Device.Buffer.allocator sample) ~renderer_set ~runtime
      ~synchronize:(fun timeout -> Device.synchronize ?timeout backing) () in
  let variable = U.variable ~name:"scale" ~min_val:(-4) ~max_val:(-1) () in
  let ast = U.substitute [ f32 2., U.cast ~src:variable ~dtype:D.float32 ]
      (elementwise_1d_ast ~n:4) in
  let info = Option.get (U.as_kernel_info ast) in
  let ast = U.replace ast ~arg:(U.Arg.Kernel_info { info with beam = 1 }) () in
  let cachelevel = Sys.getenv_opt "CACHELEVEL" in
  Unix.putenv "CACHELEVEL" "0";
  Fun.protect
    ~finally:(fun () -> Unix.putenv "CACHELEVEL" (Option.value cachelevel ~default:""))
    (fun () -> ignore (Codegen.to_program ~beam_device:device device ren ast));
  is_true ~msg:"codegen clears the cache before timing" (!evictions > 0);
  is_true ~msg:"codegen benchmarks candidates" (!observed <> []);
  List.iter (equal (list int64) [ -3L ]) !observed

let parallel_failure_joins_workers failure () =
  let backing = cpu "beam-worker-ownership" in
  let sample = Device.create_buffer ~size:1 ~dtype:D.float32 backing in
  let started = Atomic.make 0 and finished = Atomic.make 0 in
  let returned = Atomic.make false and release = Atomic.make false in
  let timed_out = Atomic.make false and inherited_context = Atomic.make true in
  let await predicate =
    let deadline = Unix.gettimeofday () +. 5. in
    while not (predicate ()) && Unix.gettimeofday () < deadline do
      Unix.sleepf 0.001
    done;
    if not (predicate ()) then Atomic.set timed_out true
  in
  let compile src =
    ignore src;
    if Helpers.Context_var.get Search.beam_parallel <> 2
       || Helpers.Context_var.get Helpers.tc_opt <> 1 then
      Atomic.set inherited_context false;
    let worker = Atomic.fetch_and_add started 1 in
    Fun.protect
      ~finally:(fun () -> ignore (Atomic.fetch_and_add finished 1))
      (fun () ->
        if worker = 0 then await (fun () -> Atomic.get started >= 2)
        else await (fun () -> Atomic.get release);
        raise failure)
  in
  let ren = Renderer.with_compiler
      (Compiler.make ~name:"BEAM_WORKER_OWNERSHIP" ~compile ()) ren in
  let renderer_set = Device.Renderer_set.make ~device:"CPU" ~arch:"generic"
      ["CLANG", Fun.const ren] in
  let device = Device.make ~name:"CPU:beam-worker-ownership"
      ~allocator:(Device.Buffer.allocator sample) ~renderer_set
      ~runtime:(fun _ -> failwith "failed compilation must never execute")
      ~synchronize:(fun timeout -> ignore timeout) () in
  let ast = elementwise_1d_ast ~n:64 in
  let rawbufs = create_bufs_for_kernel device ast in
  (* The second worker cannot finish until either search has already returned
     (the bug), or the coordinator releases it while search is joining it. *)
  let coordinator = Domain.spawn (fun () ->
      await (fun () -> Atomic.get started >= 2);
      let deadline = Unix.gettimeofday () +. 0.1 in
      while not (Atomic.get returned) && Unix.gettimeofday () < deadline do
        Unix.sleepf 0.001
      done;
      Atomic.set release true) in
  let strict = Sys.getenv_opt "BEAM_STRICT_MODE" in
  Unix.putenv "BEAM_STRICT_MODE" "0";
  Fun.protect
    ~finally:(fun () ->
      Unix.putenv "BEAM_STRICT_MODE" (Option.value strict ~default:"");
      Atomic.set returned true;
      Atomic.set release true;
      Domain.join coordinator;
      await (fun () -> Atomic.get finished = Atomic.get started);
      List.iter Device.Buffer.deallocate rawbufs)
    (fun () ->
      let outcome =
        try
          Helpers.Context_var.with_context
            [B (Search.beam_parallel, 2); B (Helpers.tc_opt, 1)] (fun () ->
              ignore (Search.beam_search ~to_program ~disable_cache:true
                (P.create ast ren) rawbufs ~var_vals:[] 1 device));
          None
        with exn -> Some exn in
      let completed_at_return = Atomic.get finished in
      Atomic.set returned true;
      equal int 2 (Atomic.get started);
      is_true ~msg:"compiler workers receive the caller's immutable policy snapshot"
        (Atomic.get inherited_context);
      equal ~msg:"all started workers finish before search propagates failure"
        int 2 completed_at_return;
      is_false ~msg:"worker coordination completed within its deadline"
        (Atomic.get timed_out);
      match outcome with
      | Some exn when exn = failure -> ()
      | Some exn -> raise exn
      | None -> failwith "search swallowed the worker exception")

let sequential_compile_interrupt () =
  let device = cpu "beam-compile-interrupt" in
  let ast = elementwise_1d_ast ~n:64 in
  let rawbufs = create_bufs_for_kernel device ast in
  let compiled = ref 0 in
  let to_program device ast =
    ignore device;
    ignore ast;
    incr compiled;
    raise Sys.Break in
  let strict = Sys.getenv_opt "BEAM_STRICT_MODE" in
  Unix.putenv "BEAM_STRICT_MODE" "0";
  Fun.protect
    ~finally:(fun () ->
      Unix.putenv "BEAM_STRICT_MODE" (Option.value strict ~default:"");
      List.iter Device.Buffer.deallocate rawbufs)
    (fun () ->
      raises Sys.Break (fun () ->
          Helpers.Context_var.with_context [B (Search.beam_parallel, 0)] (fun () ->
              ignore (Search.beam_search ~to_program ~disable_cache:true
                (P.create ast ren) rawbufs ~var_vals:[] 1 device)));
      equal ~msg:"interruption stops candidate compilation immediately" int 1 !compiled)

let candidate_program_metadata () =
  let backing = cpu "beam-program-buffers" in
  let sample = Device.create_buffer ~size:1 ~dtype:D.float32 backing in
  let compiled = ref 0 and timed = ref 0 and expected = ref [] in
  let runtime (obj : Tiny_elf.t) =
    equal string "callback-arch" obj.target.arch;
    is_true ~msg:"timing retains the callback PROGRAM, changing only launch size"
      (List.mem obj.profile_key !expected);
    let call _ ~global ~local ~vals ~wait:_ ~timeout:_ =
      incr timed;
      equal (array int) [|65536; 1; 1|] global;
      equal (option (array int)) (Some [|2; 1; 1|]) local;
      equal (array int64) [||] vals;
      Some 1e-6 in
    Device.{call; free = (fun () -> ()); handle = 0n} in
  let renderer = Renderer.with_compiler
      (Compiler.make ~name:"BEAM_PROGRAM" ~compile:Bytes.of_string ()) ren in
  let renderer_set = Device.Renderer_set.make ~device:"CPU" ~arch:"generic"
      ["CLANG", Fun.const renderer] in
  let device = Device.make ~name:"CPU:beam-program"
      ~allocator:(Device.Buffer.allocator sample) ~renderer_set ~runtime
      ~synchronize:(fun timeout -> ignore timeout)
      ~invalidate_caches:(fun () -> ()) () in
  let compile_candidate device ast =
    incr compiled;
    let program = to_program device ast in
    let info = Option.get (U.as_program_info program) in
    let extent = U.variable ~name:"timing_extent" ~min_val:65536 ~max_val:131072 () in
    let info = {info with target = {info.target with arch = "callback-arch"};
      global_size = [U.Launch_sym extent; U.Launch_int 1; U.Launch_int 1];
      local_size = [U.Launch_int 2; U.Launch_int 1; U.Launch_int 1]} in
    let children = Array.copy (U.src program) in
    let kernel = Option.get (U.as_kernel_info children.(0)) in
    children.(0) <- U.replace children.(0)
        ~arg:(U.Arg.Kernel_info {kernel with estimates = None}) ();
    let program = U.replace program ~src:children ~arg:(U.Arg.Program_info info) () in
    let scaled = U.replace program ~arg:(U.Arg.Program_info {info with
        global_size = [U.Launch_int 65536; U.Launch_int 1; U.Launch_int 1]}) () in
    expected := Some (U.semantic_key scaled) :: !expected;
    program in
  let ast = elementwise_1d_ast ~n:4 in
  let rawbufs = create_bufs_for_kernel device ast in
  let cachelevel = Sys.getenv_opt "CACHELEVEL" and max_uops = Sys.getenv_opt "BEAM_UOPS_MAX" in
  Unix.putenv "CACHELEVEL" "0";
  Fun.protect
    ~finally:(fun () ->
      Unix.putenv "CACHELEVEL" (Option.value cachelevel ~default:"");
      Unix.putenv "BEAM_UOPS_MAX" (Option.value max_uops ~default:"");
      List.iter Device.Buffer.deallocate (sample :: rawbufs))
    (fun () ->
      let search () = ignore (Search.beam_search ~to_program:compile_candidate
          ~disable_cache:true (P.create ast renderer) rawbufs
          ~var_vals:["timing_extent", 131072] 1 device) in
      Unix.putenv "BEAM_UOPS_MAX" "1";
      search ();
      is_true ~msg:"candidate compilation uses the supplied constructor" (!compiled > 0);
      equal ~msg:"oversized PROGRAMs are rejected before timing" int 0 !timed;
      compiled := 0;
      Unix.putenv "BEAM_UOPS_MAX" "0";
      search ();
      is_true ~msg:"accepted candidates use the supplied constructor" (!compiled > 0);
      is_true ~msg:"PROGRAMs without estimates can still be timed" (!timed > 0))

(* Entry *)

let () = run __FILE__
    [ beam_search_tests; search_timing_tests; transient_program_lifetimes;
      test "beam retains candidate PROGRAM metadata and scales only its launch"
        candidate_program_metadata;
      test "parallel compilation joins workers before propagating failure"
        (parallel_failure_joins_workers Stack_overflow);
      test "parallel compilation joins workers before propagating interruption"
        (parallel_failure_joins_workers Sys.Break);
      test "sequential compilation propagates interruption"
        sequential_compile_interrupt;
      test "codegen rounds negative timing midpoints down" codegen_midpoint_rounds_down ]
