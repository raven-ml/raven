(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let bufferized_call sink =
  let sink, map = Tolk.Bufferize.run sink in
  Tolk.Callify.transform_to_call sink, map


open Windtrap
open Tolk
open Tolk_uop
module U = Uop

let i32_param ~slot =
  U.param ~slot ~dtype:Dtype.int32
    ~shape:(U.stack [ U.const_int 16 ]) ~addrspace:Dtype.Global ()

let int32_to_bytes values =
  let bytes = Bytes.create (List.length values * 4) in
  List.iteri
    (fun i value -> Bytes.set_int32_le bytes (i * 4) (Int32.of_int value))
    values;
  bytes

let int32_list_of_bytes bytes =
  let len = Bytes.length bytes / 4 in
  List.init len (fun i -> Int32.to_int (Bytes.get_int32_le bytes (i * 4)))

let cuda_device =
  let cached : Tolk.Device.t option ref = ref None in
  fun () ->
    match !cached with
    | Some device -> device
    | None -> (
        try
          let device = Tolk_cuda.create "CUDA" in
          cached := Some device;
          device
        with Failure msg -> skip ~reason:msg ())

let i32_buf device values =
  let buf =
    Device.create_buffer ~size:(List.length values) ~dtype:Dtype.int32 device
  in
  Device.Buffer.ensure_allocated buf;
  Device.Buffer.copyin buf (int32_to_bytes values);
  buf

let i32_view buf ~offset ~size =
  let view = Device.Buffer.view buf ~size ~dtype:Dtype.int32 ~offset in
  Device.Buffer.ensure_allocated view;
  view

let read_i32 buf = Device.Buffer.as_bytes buf |> int32_list_of_bytes

let increment_program () =
  let dt = Dtype.int32 in
  let p0 = i32_param ~slot:0 in
  let p1 = i32_param ~slot:1 in
  let c0 = U.const (Const.int Dtype.int32 0) in
  let idx_src = U.index ~ptr:p1 ~idxs:[ c0 ] () in
  let idx_dst = U.index ~ptr:p0 ~idxs:[ c0 ] () in
  let l0 = U.load ~src:idx_src () in
  let c1 = U.const (Const.int dt 1) in
  let sum = U.alu_binary ~op:Ops.Add ~lhs:l0 ~rhs:c1 in
  let store = U.store ~dst:idx_dst ~value:sum () in
  [ p0; p1; c0; idx_src; idx_dst; l0; c1; sum; store ]

let variable_program () =
  let dt = Dtype.int32 in
  let p0 = i32_param ~slot:0 in
  let c0 = U.const (Const.int dt 0) in
  let n = U.variable ~param:true ~name:"n" ~min_val:0 ~max_val:1024 ~dtype:dt () in
  let idx_dst = U.index ~ptr:p0 ~idxs:[ c0 ] () in
  let store = U.store ~dst:idx_dst ~value:n () in
  [ p0; c0; n; idx_dst; store ]

let compile_incr device name =
  Device.compile_program device ~name (increment_program ())

let compile_var device name =
  Device.compile_program device ~name (variable_program ())

let call_spec device spec bufs var_vals =
  let car = Realize.Compiled_runner.create ~device spec in
  let tm =
    Realize.Compiled_runner.call car bufs var_vals ~wait:true ~timeout:None
  in
  Device.synchronize device;
  tm

let run_spec device spec bufs = ignore (call_spec device spec bufs [])

let queue_call device spec slots =
  let info = Program_spec.program_info spec in
  let kernel_info = U.{name = Program_spec.name spec; applied_opts = [];
    opts_to_apply = None; estimates = None; beam = 0} in
  let program = U.program ~sink:(U.sink ~kernel_info (Program_spec.program spec))
      ~linear:(U.linear (Program_spec.program spec))
      ~source:(U.source (Program_spec.src spec))
      ~binary:(U.binary (Bytes.to_string (Option.get (Program_spec.lib spec)))) ~info () in
  let selected = List.map2 (fun global slot ->
      let formal = List.find (fun u -> match U.as_param u with
          | Some {param; _} -> param.slot = global | None -> false) (Program_spec.program spec) in
      global, U.param ~slot ~dtype:(U.dtype formal) ~shape:(U.const_int (U.max_numel formal))
        ~device:(U.Single (Device.name device)) ()) info.globals slots in
  let unused = U.param ~slot:999 ~dtype:Dtype.uint8 ~shape:(U.const_int 0)
      ~device:(U.Single (Device.name device)) () in
  let args = List.init (1 + List.fold_left max (-1) info.globals) (fun i ->
      Option.value (List.assoc_opt i selected) ~default:unused) in
  U.call ~body:program ~args
    ~info:{grad_fxn = None; name = None; precompile = false;
      precompile_backward = false; aux = None; dtype = Dtype.void}

let compile_queue device calls =
  let to_program device = Codegen.to_program device (Device.renderer device) in
  let compiled = Realize.compile_linear ~device ~to_program (U.linear calls) in
  is_true ~msg:"queue compilation produces a host submission" (List.exists (fun call ->
      match U.arg (U.without_after call) with
      | U.Arg.Call_info {aux = Some _; _} -> true | _ -> false) (U.children compiled));
  let linked = Realize.link_linear compiled in
  fun ?(wait = false) ?(vars = []) inputs ->
    Realize.run_linear ~device ~to_program ~jit:true ~wait ~var_vals:vars
      ~input_uops:(Array.map U.from_buffer inputs) linked

let schedule_queue_linear device ~to_program sink =
  let call, buffer_map = bufferized_call sink in
  let linear, var_vals =
    Schedule.create_linear_with_vars
      ~get_kernel_graph:Rangeify.get_kernel_graph call
  in
  let linear = Realize.compile_linear ~device ~to_program linear in
  (linear, var_vals, buffer_map)

let f32_to_bytes values =
  let bytes = Bytes.create (Array.length values * 4) in
  Array.iteri
    (fun i v -> Bytes.set_int32_le bytes (i * 4) (Int32.bits_of_float v))
    values;
  bytes

let read_f32 buf =
  let bytes = Device.Buffer.as_bytes buf in
  Array.init
    (Bytes.length bytes / 4)
    (fun i -> Int32.float_of_bits (Bytes.get_int32_le bytes (i * 4)))

let f32_buf device data =
  let buf =
    Device.create_buffer ~size:(Array.length data) ~dtype:Dtype.float32 device
  in
  Device.Buffer.ensure_allocated buf;
  Device.Buffer.copyin buf (f32_to_bytes data);
  buf

let f32_buffer_node device_name n =
  U.buffer ~slot:(U.fresh_buffer_slot ()) ~dtype:Dtype.float32
    ~shape:(U.const_int n) ~device:(U.Single device_name) ()

(* Half-precision helpers. [f16_bits] encodes a float that is exactly
   representable as a normal (or zero) IEEE 754 binary16 value. *)

let f16_bits x =
  if x = 0.0 then 0
  else
    let bits = Int32.bits_of_float x in
    let sign = Int32.to_int (Int32.shift_right_logical bits 31) lsl 15 in
    let exp =
      Int32.to_int (Int32.logand (Int32.shift_right_logical bits 23) 0xFFl)
      - 127 + 15
    in
    let mant = Int32.to_int (Int32.logand bits 0x7FFFFFl) lsr 13 in
    sign lor (exp lsl 10) lor mant

let f16_to_bytes values =
  let bytes = Bytes.create (Array.length values * 2) in
  Array.iteri
    (fun i v -> Bytes.set_uint16_le bytes (i * 2) (f16_bits v))
    values;
  bytes

let f16_buf device data =
  let buf =
    Device.create_buffer ~size:(Array.length data) ~dtype:Dtype.float16 device
  in
  Device.Buffer.ensure_allocated buf;
  Device.Buffer.copyin buf (f16_to_bytes data);
  buf

let mk_shape dims =
  match List.map (fun s -> U.const_int s) dims with
  | [ d ] -> d
  | ds -> U.stack ds

let output_buffer buffer_map out =
  match Hashtbl.find_opt buffer_map (U.tag out) with
  | Some node -> Realize.resolve (Realize.exec_context ()) (U.buf_uop node)
  | None -> fail "output was not scheduled to a buffer"

let test_mixed_scalar_widths () =
  let device = cuda_device () in
  let output = U.param ~slot:0 ~dtype:Dtype.int64 ~shape:(U.const_int 4) () in
  let cases = [ Dtype.int8, "small", -128, 127, -7;
                Dtype.int16, "halfword", -512, 512, 300;
                Dtype.int32, "word", 0, 65536, 12345;
                Dtype.int64, "wide", 0, 0x3_0000_0000, 0x1_0000_0002 ] in
  let vars = List.map (fun (dtype, name, min_val, max_val, _) ->
      U.variable ~param:true ~name ~min_val ~max_val ~dtype ()) cases in
  let stores = List.mapi (fun i var ->
      let offset = U.const (Const.int Dtype.int32 i) in
      let ptr = U.index ~ptr:output ~idxs:[ offset ] () in
      let value = U.cast ~src:var ~dtype:Dtype.int64 in
      [ offset; ptr; value; U.store ~dst:ptr ~value () ]) vars |> List.concat in
  let spec = Device.compile_program device ~name:"cuda_mixed_widths"
      (vars @ (output :: stores)) in
  let buffer = Device.create_buffer ~size:4 ~dtype:Dtype.int64 device in
  Device.Buffer.ensure_allocated buffer;
  Device.Buffer.copyin buffer (Bytes.make 32 '\000');
  let bindings = List.map (fun (_, name, _, _, value) -> name, value) cases in
  let read () =
    let bytes = Device.Buffer.as_bytes buffer in
    List.init 4 (fun i -> Bytes.get_int64_le bytes (8 * i)) in
  ignore (call_spec device spec [ buffer ] bindings);
  equal (list int64) [ -7L; 300L; 12345L; 0x1_0000_0002L ] (read ());
  let replay = compile_queue device [queue_call device spec [0]] in
  replay ~vars:bindings [|buffer|];
  replay ~wait:true ~vars:["small", 11; "halfword", 300; "word", 12345; "wide", 0x2_0000_0003] [|buffer|];
  equal (list int64) [11L; 300L; 12345L; 0x2_0000_0003L] (read ())

let () =
  run "Cuda_runtime"
    [
      group "Execution"
        [
          test "typed arguments preserve scalar widths in dispatch and replay"
            test_mixed_scalar_widths;
          test "compile and run one kernel" (fun () ->
              let device = cuda_device () in
              let spec = compile_incr device "cuda_add_one" in
              let dst = i32_buf device [ 0 ] in
              let src = i32_buf device [ 41 ] in
              run_spec device spec [ dst; src ];
              equal (list int) [ 42 ] (read_i32 dst));
          test "passes scalar variables" (fun () ->
              let device = cuda_device () in
              let spec = compile_var device "cuda_store_var" in
              let dst = i32_buf device [ 0 ] in
              ignore (call_spec device spec [ dst ] [ ("n", 37) ]);
              equal (list int) [ 37 ] (read_i32 dst));
          test "wait returns gpu time" (fun () ->
              let device = cuda_device () in
              let spec = compile_incr device "cuda_timed_add_one" in
              let dst = i32_buf device [ 0 ] in
              let src = i32_buf device [ 1 ] in
              match call_spec device spec [ dst; src ] [] with
              | Some tm -> is_true (tm >= 0.0)
              | None -> fail "expected CUDA wait timing");
          test "exec is ordered" (fun () ->
              let device = cuda_device () in
              let spec = compile_incr device "cuda_ordered_add_one" in
              let a = i32_buf device [ 0 ] in
              let b = i32_buf device [ 0 ] in
              run_spec device spec [ b; a ];
              run_spec device spec [ a; b ];
              equal (list int) [ 2 ] (read_i32 a);
              equal (list int) [ 1 ] (read_i32 b));
          test "buffer views copy at byte offsets" (fun () ->
              let device = cuda_device () in
              let base = i32_buf device [ 1; 2; 3; 4 ] in
              let view = i32_view base ~offset:4 ~size:2 in
              equal (list int) [ 2; 3 ] (read_i32 view);
              Device.Buffer.copyin view (int32_to_bytes [ 20; 30 ]);
              equal (list int) [ 1; 20; 30; 4 ] (read_i32 base));
          test "LRU-reused base buffers stay usable" (fun () ->
              let device = cuda_device () in
              let spec = compile_incr device "cuda_lru_reused_add_one" in
              let dst = i32_buf device [ 0 ] in
              let src = i32_buf device [ 41 ] in
              Device.Buffer.deallocate src;
              let src' = i32_buf device [ 41 ] in
              run_spec device spec [ dst; src' ];
              equal (list int) [ 42 ] (read_i32 dst));
          test "nested buffer views compose byte offsets" (fun () ->
              let device = cuda_device () in
              let base = i32_buf device [ 1; 2; 3; 4 ] in
              let mid = i32_view base ~offset:4 ~size:3 in
              let leaf = i32_view mid ~offset:4 ~size:1 in
              Device.Buffer.copyin leaf (int32_to_bytes [ 33 ]);
              equal (list int) [ 1; 2; 33; 4 ] (read_i32 base));
          test "kernel dispatch binds buffer view offsets" (fun () ->
              let device = cuda_device () in
              let spec = compile_incr device "cuda_view_add_one" in
              let dst_base = i32_buf device [ 0; 0; 0; 0 ] in
              let src_base = i32_buf device [ 10; 41; 99; 100 ] in
              let dst = i32_view dst_base ~offset:4 ~size:1 in
              let src = i32_view src_base ~offset:4 ~size:1 in
              run_spec device spec [ dst; src ];
              equal (list int) [ 0; 42; 0; 0 ] (read_i32 dst_base));
          test "pinned storage exposes its host mapping and view offsets" (fun () ->
              let device = cuda_device () in
              let spec = {Device.Buffer_spec.default with cpu_access = true} in
              let base = Device.create_buffer ~spec ~size:4 ~dtype:Dtype.int32 device in
              Device.Buffer.ensure_allocated base;
              Device.Buffer.copyin base (int32_to_bytes [1; 2; 3; 4]);
              let view = i32_view base ~offset:4 ~size:2 in
              let address = Option.get (Device.Buffer.host_addr base) in
              equal nativeint (Nativeint.add address 4n)
                (Option.get (Device.Buffer.host_addr view));
              let dst = i32_buf device [0; 0] in
              Device.Buffer.copy_from ~dst ~src:view;
              equal (list int) [2; 3] (read_i32 dst);
              Device.Buffer.copy_from ~dst:view ~src:(i32_buf device [20; 30]);
              equal (list int) [1; 20; 30; 4] (read_i32 base));
          test "cross-device copy preserves views with peer or host fallback" (fun () ->
              let first = cuda_device () in
              let second = try Tolk_cuda.create "CUDA:1" with
                | Failure msg -> skip ~reason:("Second CUDA device unavailable: " ^ msg) () in
              let src = i32_view (i32_buf first [1; 2; 3; 4]) ~offset:4 ~size:2 in
              let base = i32_buf second [0; 0; 0; 0] in
              let dst = i32_view base ~offset:8 ~size:2 in
              Device.Buffer.copy_from ~dst ~src;
              equal (list int) [0; 0; 2; 3] (read_i32 base);
              Device.Buffer.copy_from ~dst:src ~src:dst;
              equal (list int) [2; 3] (read_i32 src));
          test "transfer respects buffer view offsets" (fun () ->
              let device = cuda_device () in
              let dst_base = i32_buf device [ 0; 0; 0; 0 ] in
              let src_base = i32_buf device [ 1; 2; 3; 4 ] in
              let dst = i32_view dst_base ~offset:4 ~size:2 in
              let src = i32_view src_base ~offset:8 ~size:2 in
              is_true (Device.Buffer.transfer ~dst ~src);
              equal (list int) [ 0; 3; 4; 0 ] (read_i32 dst_base));
        ];
      group "Queues"
        [
          test "queues mapped host copies and falls back for unaligned imports" (fun () ->
              let device = cuda_device () in
              let host = Tolk_cpu.create "CPU" in
              let pinned () = Device.create_buffer ~size:16 ~dtype:Dtype.int32
                  ~spec:{Device.Buffer_spec.default with host = true} device in
              let source_owner = pinned () and output_owner = pinned () in
              let source_addr = Option.get (Device.Buffer.host_addr source_owner)
              and output_addr = Option.get (Device.Buffer.host_addr output_owner) in
              let wrap address = Device.create_buffer ~size:1 ~dtype:Dtype.int32
                  ~spec:{Device.Buffer_spec.default with external_ptr = Some address} host in
              let p slot dev = U.param ~slot ~dtype:Dtype.int32 ~shape:(U.const_int 1)
                  ~device:(U.Single dev) () in
              let input = p 0 "CPU" and middle = p 1 (Device.name device) and output = p 2 "CPU" in
              let replay = compile_queue device [U.store_call ~dst:middle ~src:input;
                  U.store_call ~dst:output ~src:middle] in
              List.iter (fun offset ->
                  let source = wrap (Nativeint.add source_addr (Nativeint.of_int offset))
                  and output = wrap output_addr in
                  Device.Buffer.copyin source (int32_to_bytes [347]);
                  let before = !(Realize.queue_submissions) in
                  replay ~wait:true [|source; i32_buf device [0]; output|];
                  equal (list int) [347] (read_i32 output);
                  equal int (before + if offset = 0 then 1 else 0) !(Realize.queue_submissions)) [0; 1];
              ignore (Sys.opaque_identity (source_owner, output_owner)));
          test "queued peer copies preserve views with mapping fallback" (fun () ->
              let first = cuda_device () in
              let second = try Tolk_cuda.create "CUDA:1" with
                | Failure msg -> skip ~reason:("Second CUDA device unavailable: " ^ msg) () in
              let source = i32_view (i32_buf first [1; 2; 3; 4]) ~offset:4 ~size:2 in
              let base = i32_buf second [0; 0; 0; 0] in
              let output = i32_view base ~offset:8 ~size:2 in
              let p slot dev = U.param ~slot ~dtype:Dtype.int32 ~shape:(U.const_int 2)
                  ~device:(U.Single (Device.name dev)) () in
              let replay = compile_queue first [U.store_call ~dst:(p 1 second) ~src:(p 0 first)] in
              replay ~wait:true [|source; output|];
              equal (list int) [0; 0; 2; 3] (read_i32 base));
          test "replays a multi-kernel chain" (fun () ->
              let device = cuda_device () in
              let spec = compile_incr device "cuda_queue_chain" in
              let replay = compile_queue device
                  [queue_call device spec [1; 0]; queue_call device spec [2; 1]] in
              let a = i32_buf device [41] and b = i32_buf device [0] and c = i32_buf device [0] in
              replay [|a; b; c|];
              equal (list int) [42] (read_i32 b);
              equal (list int) [43] (read_i32 c));
          test "patches scalar values between launches" (fun () ->
              let device = cuda_device () in
              let replay = compile_queue device [queue_call device (compile_var device "cuda_queue_var") [0]] in
              let dst = i32_buf device [0] in
              replay ~vars:["n", 5] [|dst|];
              equal (list int) [5] (read_i32 dst);
              replay ~vars:["n", 9] [|dst|];
              equal (list int) [9] (read_i32 dst));
          test "rebinds buffers through repeated asynchronous launches" (fun () ->
              let device = cuda_device () in
              let replay = compile_queue device [queue_call device (compile_var device "cuda_queue_rebind") [0]] in
              let outputs = Array.init 128 (fun _ -> i32_buf device [0]) in
              Array.iteri (fun i dst -> replay ~vars:["n", i + 1] [|dst|]) outputs;
              Device.synchronize device;
              Array.iteri (fun i dst -> equal (list int) [i + 1] (read_i32 dst)) outputs);
          test "copies feed dependent kernels and later copies" (fun () ->
              let device = cuda_device () in
              let p slot = U.param ~slot ~dtype:Dtype.int32 ~shape:(U.const_int 1)
                  ~device:(U.Single "CUDA") () in
              let spec = compile_incr device "cuda_queue_copy" in
              let replay = compile_queue device [U.store_call ~dst:(p 1) ~src:(p 0);
                  queue_call device spec [2; 1]; U.store_call ~dst:(p 3) ~src:(p 2)] in
              let src = i32_buf device [7] and tmp = i32_buf device [0]
              and mid = i32_buf device [0] and dst = i32_buf device [0] in
              replay [|src; tmp; mid; dst|];
              equal (list int) [8] (read_i32 dst);
              Device.Buffer.copyin src (int32_to_bytes [11]);
              replay [|src; tmp; mid; dst|];
              run_spec device spec [src; dst];
              equal (list int) [13] (read_i32 src));
        ];
      group "Tensor core"
        [
          (* C = A @ B with half inputs and float32 accumulate at the
             16x8x16 shape the heuristic optimizer maps onto the half
             tensor core: the compiled kernel must contain a WMMA
             (mma.sync) call and produce exact results for inputs whose
             products and partial sums are exactly representable. *)
          test "f16 tensor-core matmul" (fun () ->
              let device = cuda_device () in
              let to_program device body =
                Codegen.to_program device (Device.renderer device) body
              in
              let m, n, k = (16, 8, 16) in
              let a_data =
                Array.init (m * k) (fun i ->
                    let r, c = (i / k, i mod k) in
                    float_of_int (((r + (2 * c)) mod 7) - 3) *. 0.25)
              in
              let b_data =
                Array.init (k * n) (fun i ->
                    let r, c = (i / n, i mod n) in
                    float_of_int ((((3 * r) + c) mod 5) - 2) *. 0.5)
              in
              let a_node = U.from_buffer (f16_buf device a_data) in
              let b_node = U.from_buffer (f16_buf device b_data) in
              (* dot: a.reshape(M,1,K) * b.permute(1,0).reshape(1,N,K),
                 summed over K. *)
              let ae =
                U.broadcast_to
                  ~src:(U.reshape ~src:a_node ~shape:(mk_shape [ m; 1; k ]))
                  ~shape:(mk_shape [ m; n; k ])
              in
              let be =
                U.broadcast_to
                  ~src:
                    (U.reshape
                       ~src:
                         (U.permute
                            ~src:
                              (U.reshape ~src:b_node
                                 ~shape:(mk_shape [ k; n ]))
                            ~order:[ 1; 0 ])
                       ~shape:(mk_shape [ 1; n; k ]))
                  ~shape:(mk_shape [ m; n; k ])
              in
              let mul = U.alu_binary ~op:Ops.Mul ~lhs:ae ~rhs:be in
              let mulf = U.cast ~src:mul ~dtype:Dtype.float32 in
              let red = U.reduce_axis ~src:mulf ~op:Ops.Add ~axes:[ 2 ] in
              let out = U.contiguous ~src:red () in
              let linear, _var_vals, buffer_map =
                schedule_queue_linear device ~to_program (U.sink [ out ])
              in
              let sources =
                List.filter_map
                  (fun node ->
                    match U.as_param node with
                    | Some {param = {allocation = Some ("cuda_function", data); _}; _} ->
                        let object_ = (Marshal.from_string data 0 : Tiny_elf.t) in
                        Some (Bytes.to_string object_.lib)
                    | _ -> None)
                  (U.toposort ~enter_calls:true linear)
              in
              let contains hay needle =
                let nlen = String.length needle in
                let rec at i =
                  i + nlen <= String.length hay
                  && (String.equal (String.sub hay i nlen) needle || at (i + 1))
                in
                at 0
              in
              is_true ~msg:"kernel uses the tensor core"
                (List.exists (fun src -> contains src "mma.sync") sources);
              let linear = Realize.link_linear linear in
              Realize.run_linear ~device ~to_program linear;
              Device.synchronize device;
              let expected =
                Array.init (m * n) (fun i ->
                    let r, c = (i / n, i mod n) in
                    let acc = ref 0.0 in
                    for j = 0 to k - 1 do
                      acc :=
                        !acc +. (a_data.((r * k) + j) *. b_data.((j * n) + c))
                    done;
                    !acc)
              in
              let buf = output_buffer buffer_map out in
              equal (array (float 1e-6)) expected (read_f32 buf));
        ];
      group "Queue engine"
        [
          (* A symbolic kernel inside a queue call: the launch geometry and
             the scalar argument both depend on [start_pos], patched into the
             compiled queue on every replay. *)
          test "queue call replays with updated variables" (fun () ->
              let device = cuda_device () in
              let to_program device body =
                Codegen.to_program device (Device.renderer device) body
              in
              let data = [| 1.0; 2.0; 4.0; 8.0; 16.0; 32.0; 64.0; 128.0 |] in
              let n = Array.length data in
              let buf_node = U.from_buffer (f32_buf device data) in
              let v =
                U.variable ~name:"start_pos" ~min_val:1 ~max_val:(n - 1) ()
              in
              let bound = U.bind ~var:v ~value:(U.const_int 2) in
              let size =
                U.alu_binary ~op:Ops.Add ~lhs:bound ~rhs:(U.const_int 1)
              in
              let shr = U.shrink ~src:buf_node ~offset:(U.const_int 0) ~size in
              let out =
                U.contiguous ~src:(U.alu_unary ~op:Ops.Neg ~src:shr) ()
              in
              let linear, _var_vals, buffer_map =
                schedule_queue_linear device ~to_program (U.sink [ out ])
              in
              let linear = Realize.link_linear linear in
              let check value =
                Realize.run_linear ~device ~to_program
                  ~var_vals:[ ("start_pos", value) ]
                  ~jit:true linear;
                Device.synchronize device;
                let buf = output_buffer buffer_map out in
                let got = Array.sub (read_f32 buf) 0 (value + 1) in
                let expected =
                  Array.map (fun x -> -.x) (Array.sub data 0 (value + 1))
                in
                equal
                  ~msg:(Printf.sprintf "neg prefix for start_pos=%d" value)
                  (array (float 1e-6)) expected got
              in
              (* Every replay patches values and launch dimensions. *)
              check 2;
              check 6;
              check 4);
          test "queue call replays with rebound inputs" (fun () ->
              let device = cuda_device () in
              let to_program device body =
                Codegen.to_program device (Device.renderer device) body
              in
              let data1 = [| 1.0; 2.0; 3.0; 4.0 |] in
              let data2 = [| 10.0; 20.0; 30.0; 40.0 |] in
              let n = Array.length data1 in
              let buf_node = f32_buffer_node "CUDA" n in
              let out =
                U.contiguous ~src:(U.alu_unary ~op:Ops.Neg ~src:buf_node) ()
              in
              let linear, _var_vals, buffer_map =
                schedule_queue_linear device ~to_program (U.sink [ out ])
              in
              (* Substitute the input buffer with a slotted PARAM, as
                 [Jit.jit_lower] does, so replays resolve it through
                 [input_uops]. *)
              let param =
                U.param ~slot:0 ~dtype:(U.dtype buf_node)
                  ?device:(U.device_of buf_node) ()
              in
              let linear =
                U.substitute ~walk:true [ (buf_node, param) ] linear
              in
              let linear = Realize.link_linear linear in
              let check data =
                let node = U.from_buffer (f32_buf device data) in
                Realize.run_linear ~device ~to_program
                  ~input_uops:[| node |] ~jit:true linear;
                Device.synchronize device;
                let buf = output_buffer buffer_map out in
                equal (array (float 1e-6))
                  (Array.map (fun x -> -.x) data)
                  (read_f32 buf)
              in
              check data1;
              check data2);
          test "queue call replays with rebound input and output slots" (fun () ->
              let device = cuda_device () in
              let to_program device body =
                Codegen.to_program device (Device.renderer device) body
              in
              let data1 = [| 1.0; 2.0; 3.0; 4.0 |] in
              let data2 = [| 10.0; 20.0; 30.0; 40.0 |] in
              let n = Array.length data1 in
              let in_node = f32_buffer_node "CUDA" n in
              let out =
                U.contiguous ~src:(U.alu_unary ~op:Ops.Neg ~src:in_node) ()
              in
              let linear, _var_vals, buffer_map =
                schedule_queue_linear device ~to_program (U.sink [ out ])
              in
              let out_node =
                match Hashtbl.find_opt buffer_map (U.tag out) with
                | Some node -> U.buf_uop node
                | None -> fail "output was not scheduled to a buffer"
              in
              let param slot node = U.param ~slot ~dtype:(U.dtype node)
                  ?device:(U.device_of node) () in
              let linear = U.substitute ~walk:true
                  [in_node, param 0 in_node; out_node, param 1 out_node] linear
                  |> Realize.link_linear in
              let run in_buf out_buf =
                Realize.run_linear ~device ~to_program ~jit:true
                  ~input_uops:(Array.map U.from_buffer [|in_buf; out_buf|]) linear;
                Device.synchronize device
              in
              let neg = Array.map (fun x -> -.x) in
              let in1 = f32_buf device data1 in
              let out1 = f32_buf device (Array.make n 0.0) in
              (* First submission binds [in1]/[out1]. *)
              run in1 out1;
              equal (array (float 1e-6)) (neg data1) (read_f32 out1);
              (* Rebinding both slots must repatch the recorded addresses:
                 the second run reads [in2] and writes [out2], leaving [out1]
                 untouched. *)
              let in2 = f32_buf device data2 in
              let out2 = f32_buf device (Array.make n 0.0) in
              run in2 out2;
              equal (array (float 1e-6)) (neg data2) (read_f32 out2);
              equal ~msg:"first output buffer is untouched"
                (array (float 1e-6))
                (neg data1) (read_f32 out1));
        ];
    ]
