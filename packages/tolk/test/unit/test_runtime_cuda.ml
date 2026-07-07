(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tolk
open Tolk_uop
module U = Uop

let global_ptr dt = Dtype.Ptr.create dt ~addrspace:Global ~size:16

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
  let dt = Dtype.Val.int32 in
  let ptr = Dtype.Ptr (global_ptr dt) in
  let p0 = U.param ~slot:0 ~dtype:ptr () in
  let p1 = U.param ~slot:1 ~dtype:ptr () in
  let c0 = U.const (Const.int Dtype.Val.int32 0) in
  let idx_src = U.index ~ptr:p1 ~idxs:[ c0 ] ~as_ptr:true () in
  let idx_dst = U.index ~ptr:p0 ~idxs:[ c0 ] ~as_ptr:true () in
  let l0 = U.load ~src:idx_src () in
  let c1 = U.const (Const.int dt 1) in
  let sum = U.alu_binary ~op:Ops.Add ~lhs:l0 ~rhs:c1 in
  let store = U.store ~dst:idx_dst ~value:sum () in
  [ p0; p1; c0; idx_src; idx_dst; l0; c1; sum; store ]

let variable_program () =
  let dt = Dtype.Val.int32 in
  let ptr = Dtype.Ptr (global_ptr dt) in
  let p0 = U.param ~slot:0 ~dtype:ptr () in
  let c0 = U.const (Const.int dt 0) in
  let n = U.variable ~name:"n" ~min_val:0 ~max_val:1024 ~dtype:dt () in
  let idx_dst = U.index ~ptr:p0 ~idxs:[ c0 ] ~as_ptr:true () in
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

(* Graph helpers *)

let device_graph device =
  match Device.graph device with
  | Some g -> g
  | None -> fail "CUDA device has no graph capability"

let prog_of_spec device spec =
  let lib =
    match Program_spec.lib spec with
    | Some lib -> lib
    | None ->
        let comp = Option.get (Renderer.compiler (Device.renderer device)) in
        Compiler.compile_cached comp (Program_spec.src spec)
  in
  Device.runtime device
    (U.sanitize_function_name (Program_spec.name spec))
    lib ~runtimevars:[]

let ones3 = [| 1; 1; 1 |]

let kernel_node handle bufs ?(vals = [||]) ?(deps = [||]) () =
  Device.Graph.Kernel
    {
      handle;
      global = ones3;
      local = ones3;
      bufs = Array.map Device.Buffer.addr bufs;
      vals;
      deps;
    }

(* Engine-level graph helpers: schedule a tensor sink, compile it, and wrap
   every call into one CUSTOM_FUNCTION "graph" call, as the JIT's graph
   batching does. *)

let graph_call_info : U.call_info =
  {
    grad_fxn = None;
    metadata = [];
    name = None;
    precompile = false;
    precompile_backward = false;
    aux = None;
  }

let wrap_graph linear =
  let cf =
    U.custom_function ~name:"graph" ~srcs:[ U.linear (U.children linear) ]
  in
  let call =
    U.call ~body:(U.custom_function ~name:"graph" ~srcs:[]) ~args:[]
      ~info:graph_call_info
  in
  U.linear [ U.replace call ~src:[| cf |] () ]

let schedule_graph_linear device ~to_program sink =
  let call, buffer_map = Allocations.transform_to_call sink in
  let linear, var_vals =
    Schedule.create_linear_with_vars
      ~get_kernel_graph:Rangeify.get_kernel_graph call
  in
  let linear = Realize.pm_compile ~device ~to_program linear in
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

let output_buffer binding buffer_map out =
  match Hashtbl.find_opt buffer_map (U.tag out) with
  | Some node -> (
      match Realize.Buffers.find_opt binding (U.buf_uop node) with
      | Some buf -> buf
      | None -> fail "output buffer was not bound")
  | None -> fail "output was not scheduled to a buffer"

let () =
  run "Cuda_runtime"
    [
      group "Execution"
        [
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
          test "transfer respects buffer view offsets" (fun () ->
              let device = cuda_device () in
              let dst_base = i32_buf device [ 0; 0; 0; 0 ] in
              let src_base = i32_buf device [ 1; 2; 3; 4 ] in
              let dst = i32_view dst_base ~offset:4 ~size:2 in
              let src = i32_view src_base ~offset:8 ~size:2 in
              is_true (Device.Buffer.transfer ~dst ~src);
              equal (list int) [ 0; 3; 4; 0 ] (read_i32 dst_base));
        ];
      group "Graph"
        [
          test "replays a multi-kernel chain" (fun () ->
              let device = cuda_device () in
              let prog =
                prog_of_spec device (compile_incr device "cuda_graph_chain")
              in
              let a = i32_buf device [ 41 ] in
              let b = i32_buf device [ 0 ] in
              let c = i32_buf device [ 0 ] in
              let exec =
                (device_graph device).Device.Graph.build
                  [|
                    kernel_node prog.Device.handle [| b; a |] ();
                    kernel_node prog.Device.handle [| c; b |] ~deps:[| 0 |] ();
                  |]
              in
              ignore (exec.Device.Graph.launch ~wait:false : float option);
              Device.synchronize device;
              equal (list int) [ 42 ] (read_i32 b);
              equal (list int) [ 43 ] (read_i32 c));
          test "patches scalar values between launches" (fun () ->
              let device = cuda_device () in
              let prog =
                prog_of_spec device (compile_var device "cuda_graph_var")
              in
              let dst = i32_buf device [ 0 ] in
              let exec =
                (device_graph device).Device.Graph.build
                  [| kernel_node prog.Device.handle [| dst |] ~vals:[| 5 |] () |]
              in
              ignore (exec.Device.Graph.launch ~wait:false : float option);
              Device.synchronize device;
              equal (list int) [ 5 ] (read_i32 dst);
              exec.Device.Graph.set_val 0 0 9;
              exec.Device.Graph.set_params 0;
              ignore (exec.Device.Graph.launch ~wait:false : float option);
              Device.synchronize device;
              equal (list int) [ 9 ] (read_i32 dst));
          test "rebinds buffer arguments between launches" (fun () ->
              let device = cuda_device () in
              let prog =
                prog_of_spec device (compile_incr device "cuda_graph_rebind")
              in
              let dst1 = i32_buf device [ 0 ] in
              let src1 = i32_buf device [ 41 ] in
              let dst2 = i32_buf device [ 0 ] in
              let src2 = i32_buf device [ 10 ] in
              let exec =
                (device_graph device).Device.Graph.build
                  [| kernel_node prog.Device.handle [| dst1; src1 |] () |]
              in
              ignore (exec.Device.Graph.launch ~wait:false : float option);
              Device.synchronize device;
              equal (list int) [ 42 ] (read_i32 dst1);
              exec.Device.Graph.set_buf 0 0 (Device.Buffer.addr dst2);
              exec.Device.Graph.set_buf 0 1 (Device.Buffer.addr src2);
              exec.Device.Graph.set_params 0;
              ignore (exec.Device.Graph.launch ~wait:false : float option);
              Device.synchronize device;
              equal (list int) [ 42 ] (read_i32 dst1);
              equal (list int) [ 11 ] (read_i32 dst2));
          test "copies feed dependent kernels" (fun () ->
              let device = cuda_device () in
              let graph = device_graph device in
              is_true ~msg:"CUDA graphs support copies"
                graph.Device.Graph.supports_copy;
              let prog =
                prog_of_spec device (compile_incr device "cuda_graph_copy")
              in
              let src = i32_buf device [ 7 ] in
              let tmp = i32_buf device [ 0 ] in
              let dst = i32_buf device [ 0 ] in
              let exec =
                graph.Device.Graph.build
                  [|
                    Device.Graph.Copy
                      {
                        dest = Device.Buffer.addr tmp;
                        src = Device.Buffer.addr src;
                        nbytes = Device.Buffer.nbytes src;
                        deps = [||];
                      };
                    kernel_node prog.Device.handle [| dst; tmp |]
                      ~deps:[| 0 |] ();
                  |]
              in
              ignore (exec.Device.Graph.launch ~wait:false : float option);
              Device.synchronize device;
              equal (list int) [ 8 ] (read_i32 dst));
        ];
      group "Graph engine"
        [
          (* A symbolic kernel inside a graph call: the launch geometry and
             the scalar argument both depend on [start_pos], patched into the
             instantiated graph on every replay. *)
          test "graph call replays with updated variables" (fun () ->
              let device = cuda_device () in
              let to_program body =
                Codegen.to_program device (Device.renderer device) body
              in
              let data = [| 1.0; 2.0; 4.0; 8.0; 16.0; 32.0; 64.0; 128.0 |] in
              let n = Array.length data in
              let buf_node = f32_buffer_node "CUDA" n in
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
                schedule_graph_linear device ~to_program (U.sink [ out ])
              in
              let linear = wrap_graph linear in
              let binding = Realize.Buffers.create ~device in
              Realize.Buffers.seed binding buf_node (f32_buf device data);
              let check value =
                Realize.run_linear ~device ~to_program binding
                  ~var_vals:[ ("start_pos", value) ]
                  ~jit:true linear;
                Device.synchronize device;
                let buf = output_buffer binding buffer_map out in
                let got = Array.sub (read_f32 buf) 0 (value + 1) in
                let expected =
                  Array.map (fun x -> -.x) (Array.sub data 0 (value + 1))
                in
                equal
                  ~msg:(Printf.sprintf "neg prefix for start_pos=%d" value)
                  (array (float 1e-6)) expected got
              in
              (* The first run records the graph; later runs replay it with
                 patched values and launch dimensions. *)
              check 2;
              check 6;
              check 4);
          test "graph call replays with rebound inputs" (fun () ->
              let device = cuda_device () in
              let to_program body =
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
                schedule_graph_linear device ~to_program (U.sink [ out ])
              in
              (* Substitute the input buffer with a slotted PARAM, as
                 [Jit.jit_lower] does, so replays resolve it through
                 [input_uops]. *)
              let param =
                U.param ~slot:0 ~dtype:(U.dtype buf_node)
                  ?device:(U.device_of buf_node) ()
              in
              let linear =
                wrap_graph (U.substitute ~walk:true [ (buf_node, param) ] linear)
              in
              let binding = Realize.Buffers.create ~device in
              let check node data =
                Realize.Buffers.seed binding node (f32_buf device data);
                Realize.run_linear ~device ~to_program binding
                  ~input_uops:[| node |] ~jit:true linear;
                Device.synchronize device;
                Realize.Buffers.remove binding node;
                let buf = output_buffer binding buffer_map out in
                equal (array (float 1e-6))
                  (Array.map (fun x -> -.x) data)
                  (read_f32 buf)
              in
              let node1 = f32_buffer_node "CUDA" n in
              let node2 = f32_buffer_node "CUDA" n in
              check node1 data1;
              check node2 data2);
          (* Buffer nodes reseeded in the binding between replays (no PARAM
             slots): rune's jit reseeds its input nodes and binds a fresh
             output buffer on every call, so the graph must repatch both. *)
          test "graph call replays with reseeded buffer nodes" (fun () ->
              let device = cuda_device () in
              let to_program body =
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
                schedule_graph_linear device ~to_program (U.sink [ out ])
              in
              let linear = wrap_graph linear in
              let out_node =
                match Hashtbl.find_opt buffer_map (U.tag out) with
                | Some node -> U.buf_uop node
                | None -> fail "output was not scheduled to a buffer"
              in
              let binding = Realize.Buffers.create ~device in
              let run in_buf out_buf =
                Realize.Buffers.seed binding in_node in_buf;
                Realize.Buffers.seed binding out_node out_buf;
                Realize.run_linear ~device ~to_program binding ~jit:true linear;
                Device.synchronize device
              in
              let neg = Array.map (fun x -> -.x) in
              let in1 = f32_buf device data1 in
              let out1 = f32_buf device (Array.make n 0.0) in
              (* First run records the graph against [in1]/[out1]. *)
              run in1 out1;
              equal (array (float 1e-6)) (neg data1) (read_f32 out1);
              (* Reseeding both nodes must repatch the recorded addresses:
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
