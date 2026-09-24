(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tolk
open Tolk_uop
module U = Uop

let i32_param ~slot =
  U.param ~slot ~dtype:Dtype.int32
    ~shape:(U.stack [ U.const_int 16 ]) ~addrspace:Dtype.Global ()

let int32_to_bytes values =
  let bytes = Bytes.create (List.length values * 4) in
  let set off value =
    let open Int32 in
    Bytes.set bytes off (Char.chr (to_int (logand value 0xFFl)));
    Bytes.set bytes (off + 1)
      (Char.chr (to_int (logand (shift_right_logical value 8) 0xFFl)));
    Bytes.set bytes (off + 2)
      (Char.chr (to_int (logand (shift_right_logical value 16) 0xFFl)));
    Bytes.set bytes (off + 3)
      (Char.chr (to_int (logand (shift_right_logical value 24) 0xFFl)))
  in
  List.iteri (fun i value -> set (i * 4) (Int32.of_int value)) values;
  bytes

let int32_list_of_bytes bytes =
  let len = Bytes.length bytes / 4 in
  let get off =
    let open Int32 in
    logor
      (of_int (Char.code (Bytes.get bytes off)))
      (logor
         (shift_left (of_int (Char.code (Bytes.get bytes (off + 1)))) 8)
         (logor
            (shift_left (of_int (Char.code (Bytes.get bytes (off + 2)))) 16)
            (shift_left (of_int (Char.code (Bytes.get bytes (off + 3)))) 24)))
  in
  List.init len (fun i -> Int32.to_int (get (i * 4)))

let metal_device =
  let cached : Tolk.Device.t option ref = ref None in
  fun () ->
    match !cached with
    | Some device -> device
    | None -> (
        try
          let device = Tolk_metal.create "METAL:test" in
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
  let idx_src = U.index ~ptr:p1 ~idxs:[c0] () in
  let idx_dst = U.index ~ptr:p0 ~idxs:[c0] () in
  let l0 = U.load ~src:idx_src () in
  let c1 = U.const (Const.int dt 1) in
  let sum = U.alu_binary ~op:Ops.Add ~lhs:l0 ~rhs:c1 in
  let store = U.store ~dst:idx_dst ~value:sum () in
  [ p0; p1; c0; idx_src; idx_dst; l0; c1; sum; store ]

let variable_program () =
  let dt = Dtype.int32 in
  let p0 = i32_param ~slot:0 in
  let c0 = U.const (Const.int dt 0) in
  let n = U.variable ~name:"n" ~min_val:0 ~max_val:1024 ~dtype:dt () in
  let idx_dst = U.index ~ptr:p0 ~idxs:[c0] () in
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

let device_graph device =
  match Device.graph device with
  | Some g -> g
  | None -> fail "Metal device has no graph capability"

let prog_of_spec device spec =
  let lib =
    match Program_spec.lib spec with
    | Some lib -> lib
    | None ->
        let comp = Option.get (Renderer.compiler (Device.renderer device)) in
        Compiler.compile_cached comp (Program_spec.src spec)
  in
  Device.runtime device (Program_spec.to_elf (Program_spec.with_lib lib spec))

let ones3 = [| 1; 1; 1 |]

let kernel_node handle bufs ?(vals = [||]) () =
  Device.Graph.Kernel
    {
      handle;
      global = ones3;
      local = ones3;
      bufs = Array.map Device.Buffer.addr bufs;
      vals;
      deps = [||];
    }

let test_mixed_scalar_widths () =
  let device = metal_device () in
  let output = U.param ~slot:0 ~dtype:Dtype.int64 ~shape:(U.const_int 4) () in
  let cases = [ Dtype.int8, "small", -128, 127, -7;
                Dtype.int16, "halfword", -512, 512, 300;
                Dtype.int32, "word", 0, 65536, 12345;
                Dtype.int64, "wide", 0, 0x3_0000_0000, 0x1_0000_0002 ] in
  let vars = List.map (fun (dtype, name, min_val, max_val, _) ->
      U.variable ~name ~min_val ~max_val ~dtype ()) cases in
  let stores = List.mapi (fun i var ->
      let offset = U.const (Const.int Dtype.int32 i) in
      let ptr = U.index ~ptr:output ~idxs:[ offset ] () in
      let value = U.cast ~src:var ~dtype:Dtype.int64 in
      [ offset; ptr; value; U.store ~dst:ptr ~value () ]) vars |> List.concat in
  let spec = Device.compile_program device ~name:"metal_mixed_widths"
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
  let prg = prog_of_spec device spec in
  let vals = Array.of_list (List.map snd bindings) in
  let graph = (device_graph device).build [| kernel_node prg.handle [| buffer |] ~vals () |] in
  ignore (graph.launch ~wait:false);
  graph.set_val 0 0 11;
  graph.set_val 0 3 0x2_0000_0003;
  graph.set_params 0;
  ignore (graph.launch ~wait:true);
  equal (list int64) [ 11L; 300L; 12345L; 0x2_0000_0003L ] (read ())

let test_many_buffer_arguments () =
  let device = metal_device () in
  let count = 33 in
  let params = List.init count (fun slot ->
      U.param ~slot ~dtype:Dtype.int32 ~shape:(U.const_int 1) ()) in
  let output = List.hd params in
  let zero = U.const (Const.int Dtype.int32 0) in
  let nodes = ref [] and sum = ref zero in
  List.iter (fun ptr ->
      let index = U.index ~ptr ~idxs:[ zero ] () in
      let value = U.load ~src:index () in
      let next = U.alu_binary ~op:Ops.Add ~lhs:!sum ~rhs:value in
      nodes := next :: value :: index :: !nodes;
      sum := next) (List.tl params);
  let dst = U.index ~ptr:output ~idxs:[ zero ] () in
  let linear = params @ [ zero ] @ List.rev !nodes @
      [ dst; U.store ~dst ~value:!sum () ] in
  let spec = Device.compile_program device ~name:"metal_many_arguments" linear in
  let buffers = Array.init count (fun i -> i32_buf device [ i ]) in
  run_spec device spec (Array.to_list buffers);
  equal (list int) [ 528 ] (read_i32 buffers.(0));
  let second = i32_buf device [ 0 ] in
  let second_buffers = Array.copy buffers in
  second_buffers.(0) <- second;
  let prg = prog_of_spec device spec in
  let graph = (device_graph device).build
      [| kernel_node prg.handle buffers (); kernel_node prg.handle second_buffers () |] in
  ignore (graph.launch ~wait:false);
  let replacement = i32_buf device [ 100 ] in
  graph.set_buf 0 1 (Device.Buffer.addr replacement);
  graph.set_params 0;
  ignore (graph.launch ~wait:true);
  equal (list int) [ 627 ] (read_i32 buffers.(0));
  equal (list int) [ 528 ] (read_i32 second)

let test_thread_reduction kind width expected () =
  let device = metal_device () in
  let param slot size = U.param ~slot ~dtype:Dtype.int32 ~shape:(U.const_int size) () in
  let count = 16 * width in
  let output = param 0 2 and input = param 1 count in
  let range axis size kind = U.range ~size:(U.const_int size) ~axis ~kind () in
  let row = range 0 2 Axis_type.Local in
  let col = range 1 width kind in
  let seq = range 2 8 Axis_type.Reduce in
  let stride = 8 * width in
  let idx = U.O.(row * int_ stride + col * int_ 8 + seq) in
  let loaded = U.load ~src:(U.index ~ptr:input ~idxs:[ idx ] ()) () in
  let value = U.reduce ~src:loaded ~ranges:[ seq; col ] ~op:Ops.Add ~dtype:Dtype.int32 in
  let store = U.store ~dst:(U.index ~ptr:output ~idxs:[ row ] ()) ~value () in
  let kernel_info : U.kernel_info =
    { name = "metal_thread_reduce"; axis_types = [];
      applied_opts = []; opts_to_apply = None; estimates = None; beam = 0 } in
  let sink = U.sink ~kernel_info [ U.end_ ~value:store ~ranges:[ row ] ] in
  let linear = Codegen.full_rewrite_to_sink ~optimize:false (Device.renderer device) sink
      |> Linearizer.linearize in
  let spec = Device.compile_program device ~name:"metal_thread_reduce" linear in
  let output = i32_buf device [ 0; 0 ] and input = i32_buf device (List.init count Fun.id) in
  run_spec device spec [ output; input ];
  equal (list int) expected (read_i32 output)

let test_tensor_core_matmul ~m ~n ~k ~locals () =
  let device = metal_device () in
  let ren = Device.renderer device in
  if Renderer.tensor_cores ren = [] then skip ~reason:"Metal tensor cores unavailable" ();
  let param slot size = U.param ~slot ~dtype:Dtype.float32 ~shape:(U.const_int size) () in
  let output = param 0 (m * n) and a = param 1 (m * k) and b = param 2 (k * n) in
  let range axis size kind = U.range ~size:(U.const_int size) ~axis ~kind () in
  let row = range 0 m Axis_type.Global and col = range 1 n Axis_type.Global in
  let red = range 2 k Axis_type.Reduce in
  let load ptr index = U.load ~src:(U.index ~ptr ~idxs:[ index ] ()) () in
  let av = load a U.O.(row * int_ k + red) in
  let bv = load b U.O.(red * int_ n + col) in
  let product = U.alu_binary ~op:Ops.Mul ~lhs:av ~rhs:bv in
  let value = U.reduce ~src:product ~ranges:[ red ] ~op:Ops.Add ~dtype:Dtype.float32 in
  let dst = U.index ~ptr:output ~idxs:[ U.O.(row * int_ n + col) ] () in
  let store = U.store ~dst ~value () in
  let kernel_info : U.kernel_info =
    { name = "metal_warp_grouping"; axis_types = []; applied_opts = [];
      opts_to_apply = None; estimates = None; beam = 0 } in
  let sink = U.sink ~kernel_info [ U.end_ ~value:store ~ranges:[ row; col ] ] in
  let scheduler = Postrange.create sink ren in
  ignore (Postrange.apply_opt scheduler
    (U.Opt.Tc { axis = 0; tc_select = -1; tc_opt = 2; use_tc = 1 }));
  for step = 0 to locals - 1 do
    let axis = List.hd (Postrange.axes_of scheduler [ Axis_type.Global ]) in
    ignore (Postrange.apply_opt scheduler
      (U.Opt.Split { axis; amount = 2; kind = Axis_type.Local; top = false }));
    equal int (step + 1) (List.length (Postrange.axes_of scheduler [ Axis_type.Local ]))
  done;
  let linear = Codegen.full_rewrite_to_sink ~optimize:false ren (Postrange.ast scheduler)
      |> Linearizer.linearize in
  let spec = Device.compile_program device ~name:"metal_warp_grouping" linear in
  let buffer values =
    let buf = Device.create_buffer ~size:(Array.length values) ~dtype:Dtype.float32 device in
    let bytes = Bytes.create (Array.length values * 4) in
    Array.iteri (fun i v -> Bytes.set_int32_le bytes (i * 4) (Int32.bits_of_float v)) values;
    Device.Buffer.ensure_allocated buf;
    Device.Buffer.copyin buf bytes;
    buf in
  let a = Array.init (m * k) (fun i -> float_of_int ((i * 13 mod 7) - 2)) in
  let b = Array.init (k * n) (fun i -> float_of_int ((i * 11 mod 9) - 3)) in
  let output = buffer (Array.make (m * n) nan) in
  run_spec device spec [ output; buffer a; buffer b ];
  let bytes = Device.Buffer.as_bytes output in
  for i = 0 to m - 1 do
    for j = 0 to n - 1 do
      let expected = ref 0. in
      for r = 0 to k - 1 do expected := !expected +. a.(i * k + r) *. b.(r * n + j) done;
      let actual = Int32.float_of_bits (Bytes.get_int32_le bytes ((i * n + j) * 4)) in
      is_true ~msg:(Printf.sprintf "matmul[%d,%d]: expected %g, got %g" i j !expected actual)
        (Float.abs (actual -. !expected) < 1.e-5)
    done
  done

let () =
  run "Metal_runtime"
    [
      group "Execution"
        [
          test "tensor cores retain warp lanes across four local dimensions"
            (test_tensor_core_matmul ~m:32 ~n:64 ~k:8 ~locals:3);
          test "tensor cores preserve padded output and contraction lanes"
            (test_tensor_core_matmul ~m:9 ~n:11 ~k:13 ~locals:0);
          test "local reductions preserve independent output threads"
            (test_thread_reduction Axis_type.Local 4 [ 496; 1520 ]);
          test "warp reductions preserve independent output threads"
            (test_thread_reduction Axis_type.Warp 32 [ 32640; 98176 ]);
          test "typed argument structures preserve scalar widths in dispatch and replay"
            test_mixed_scalar_widths;
          test "argument structures support more than 31 buffers and rebinding"
            test_many_buffer_arguments;
          test "compile and run one kernel" (fun () ->
            let device = metal_device () in
            let spec = compile_incr device "metal_add_one" in
            let dst = i32_buf device [ 0 ] in
            let src = i32_buf device [ 41 ] in
            run_spec device spec [ dst; src ];
            equal (list int) [ 42 ] (read_i32 dst));
          test "passes scalar variables" (fun () ->
            let device = metal_device () in
            let spec = compile_var device "metal_store_var" in
            let dst = i32_buf device [ 0 ] in
            ignore (call_spec device spec [ dst ] [ "n", 37 ]);
            equal (list int) [ 37 ] (read_i32 dst));
          test "wait returns gpu time" (fun () ->
            let device = metal_device () in
            let spec = compile_incr device "metal_timed_add_one" in
            let dst = i32_buf device [ 0 ] in
            let src = i32_buf device [ 1 ] in
            match call_spec device spec [ dst; src ] [] with
            | Some tm -> is_true (tm >= 0.0)
            | None -> fail "expected Metal wait timing");
          test "exec is ordered" (fun () ->
            let device = metal_device () in
            let spec = compile_incr device "metal_ordered_add_one" in
            let a = i32_buf device [ 0 ] in
            let b = i32_buf device [ 0 ] in
            run_spec device spec [ b; a ];
            run_spec device spec [ a; b ];
            equal (list int) [ 2 ] (read_i32 a);
            equal (list int) [ 1 ] (read_i32 b));
          test "buffer views copy at byte offsets" (fun () ->
            let device = metal_device () in
            let base = i32_buf device [ 1; 2; 3; 4 ] in
            let view = i32_view base ~offset:4 ~size:2 in
            equal (list int) [ 2; 3 ] (read_i32 view);
            Device.Buffer.copyin view (int32_to_bytes [ 20; 30 ]);
            equal (list int) [ 1; 20; 30; 4 ] (read_i32 base));
          test "as_buffer aliases a view's bytes" (fun () ->
            let device = metal_device () in
            let base = i32_buf device [ 1; 2; 3; 4 ] in
            let view = i32_view base ~offset:4 ~size:2 in
            match Device.Buffer.as_buffer view with
            | None -> fail "Metal memory is host-visible"
            | Some mem ->
                equal int 8 (Bigarray.Array1.dim mem);
                equal int 2 (Bigarray.Array1.get mem 0);
                Bigarray.Array1.set mem 4 30;
                equal (list int) [ 1; 2; 30; 4 ] (read_i32 base));
          test "LRU-reused base buffers keep valid tokens" (fun () ->
            let device = metal_device () in
            let spec = compile_incr device "metal_lru_reused_add_one" in
            let dst = i32_buf device [ 0 ] in
            let src = i32_buf device [ 41 ] in
            Device.Buffer.deallocate src;
            let src' = i32_buf device [ 41 ] in
            run_spec device spec [ dst; src' ];
            equal (list int) [ 42 ] (read_i32 dst));
          test "nested buffer views compose byte offsets" (fun () ->
            let device = metal_device () in
            let base = i32_buf device [ 1; 2; 3; 4 ] in
            let mid = i32_view base ~offset:4 ~size:3 in
            let leaf = i32_view mid ~offset:4 ~size:1 in
            Device.Buffer.copyin leaf (int32_to_bytes [ 33 ]);
            equal (list int) [ 1; 2; 33; 4 ] (read_i32 base));
          test "collects dropped views while creating new ones" (fun () ->
            let device = metal_device () in
            let base = i32_buf device [ 1; 2; 3; 4 ] in
            (* A small, eager heap runs the finalisers of dropped views inside
               the allocations that register new ones. *)
            let gc = Gc.get () in
            Gc.set { gc with minor_heap_size = 4096; space_overhead = 20 };
            Fun.protect
              ~finally:(fun () -> Gc.set gc)
              (fun () ->
                for i = 0 to 99_999 do
                  ignore (i32_view base ~offset:(4 * (i mod 4)) ~size:1)
                done);
            let view = i32_view base ~offset:8 ~size:1 in
            equal (list int) [ 3 ] (read_i32 view));
          test "kernel dispatch binds buffer view offsets" (fun () ->
            let device = metal_device () in
            let spec = compile_incr device "metal_view_add_one" in
            let dst_base = i32_buf device [ 0; 0; 0; 0 ] in
            let src_base = i32_buf device [ 10; 41; 99; 100 ] in
            let dst = i32_view dst_base ~offset:4 ~size:1 in
            let src = i32_view src_base ~offset:4 ~size:1 in
            run_spec device spec [ dst; src ];
            equal (list int) [ 0; 42; 0; 0 ] (read_i32 dst_base));
          test "blit transfer respects buffer view offsets" (fun () ->
            let device = metal_device () in
            let dst_base = i32_buf device [ 0; 0; 0; 0 ] in
            let src_base = i32_buf device [ 1; 2; 3; 4 ] in
            let dst = i32_view dst_base ~offset:4 ~size:2 in
            let src = i32_view src_base ~offset:8 ~size:2 in
            is_true (Device.Buffer.transfer ~dst ~src);
            equal (list int) [ 0; 3; 4; 0 ] (read_i32 dst_base));
        ];
      group "Graph"
        [
          test "replays a multi-kernel chain in order" (fun () ->
            let device = metal_device () in
            let prog =
              prog_of_spec device (compile_incr device "metal_graph_chain")
            in
            let a = i32_buf device [ 41 ] in
            let b = i32_buf device [ 0 ] in
            let c = i32_buf device [ 0 ] in
            let exec =
              (device_graph device).Device.Graph.build
                [|
                  kernel_node prog.Device.handle [| b; a |] ();
                  kernel_node prog.Device.handle [| c; b |] ();
                  kernel_node prog.Device.handle [| a; c |] ();
                |]
            in
            ignore (exec.Device.Graph.launch ~wait:false : float option);
            Device.synchronize device;
            equal (list int) [ 44 ] (read_i32 a);
            equal (list int) [ 42 ] (read_i32 b);
            equal (list int) [ 43 ] (read_i32 c));
          test "relaunches without an intervening synchronize" (fun () ->
            let device = metal_device () in
            let prog =
              prog_of_spec device (compile_incr device "metal_graph_relaunch")
            in
            let a = i32_buf device [ 0 ] in
            let b = i32_buf device [ 0 ] in
            let exec =
              (device_graph device).Device.Graph.build
                [|
                  kernel_node prog.Device.handle [| b; a |] ();
                  kernel_node prog.Device.handle [| a; b |] ();
                |]
            in
            for _ = 1 to 10 do
              ignore (exec.Device.Graph.launch ~wait:false : float option)
            done;
            Device.synchronize device;
            equal (list int) [ 20 ] (read_i32 a);
            equal (list int) [ 19 ] (read_i32 b));
          test "patches scalar values between launches" (fun () ->
            let device = metal_device () in
            let prog =
              prog_of_spec device (compile_var device "metal_graph_var")
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
          test "patches a launch still in flight" (fun () ->
            let device = metal_device () in
            let prog =
              prog_of_spec device (compile_var device "metal_graph_inflight")
            in
            let first = i32_buf device [ 0 ] in
            let second = i32_buf device [ 0 ] in
            let exec =
              (device_graph device).Device.Graph.build
                [|
                  kernel_node prog.Device.handle [| first |] ~vals:[| 5 |] ();
                |]
            in
            ignore (exec.Device.Graph.launch ~wait:false : float option);
            exec.Device.Graph.set_val 0 0 9;
            exec.Device.Graph.set_buf 0 0 (Device.Buffer.addr second);
            exec.Device.Graph.set_params 0;
            ignore (exec.Device.Graph.launch ~wait:false : float option);
            Device.synchronize device;
            equal (list int) [ 5 ] (read_i32 first);
            equal (list int) [ 9 ] (read_i32 second));
          test "rebinds buffer views between launches" (fun () ->
            let device = metal_device () in
            let prog =
              prog_of_spec device (compile_incr device "metal_graph_rebind")
            in
            let dst1 = i32_buf device [ 0 ] in
            let src1 = i32_buf device [ 41 ] in
            let dst_base = i32_buf device [ 0; 0; 0; 0 ] in
            let src_base = i32_buf device [ 1; 2; 10; 4 ] in
            let dst2 = i32_view dst_base ~offset:4 ~size:1 in
            let src2 = i32_view src_base ~offset:8 ~size:1 in
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
            equal (list int) [ 0; 11; 0; 0 ] (read_i32 dst_base));
          test "wait returns gpu time" (fun () ->
            let device = metal_device () in
            let prog =
              prog_of_spec device (compile_incr device "metal_graph_timed")
            in
            let dst = i32_buf device [ 0 ] in
            let src = i32_buf device [ 1 ] in
            let exec =
              (device_graph device).Device.Graph.build
                [| kernel_node prog.Device.handle [| dst; src |] () |]
            in
            (match exec.Device.Graph.launch ~wait:true with
            | Some tm -> is_true (tm >= 0.0)
            | None -> fail "expected Metal graph wait timing");
            Device.synchronize device;
            equal (list int) [ 2 ] (read_i32 dst));
          test "collects a dropped graph while another settles" (fun () ->
            let device = metal_device () in
            let prog =
              prog_of_spec device (compile_incr device "metal_graph_dropped")
            in
            let a = i32_buf device [ 0 ] in
            let build () =
              (device_graph device).Device.Graph.build
                [| kernel_node prog.Device.handle [| a; a |] () |]
            in
            let launch exec =
              ignore (exec.Device.Graph.launch ~wait:false : float option)
            in
            (* Collect a dropped graph at the [n]th allocation that patching a
               live one makes, for each [n], so that one collection lands while
               the patch prunes the in-flight list. *)
            let countdown = ref 0 in
            let tracker =
              {
                Gc.Memprof.null_tracker with
                alloc_minor =
                  (fun _ ->
                    if !countdown > 0 then begin
                      decr countdown;
                      if !countdown = 0 then Gc.full_major ()
                    end;
                    None);
              }
            in
            (match
               Gc.Memprof.start ~sampling_rate:1.0 ~callstack_size:0 tracker
             with
            | exception Failure reason -> skip ~reason ()
            | _ -> ());
            Fun.protect ~finally:Gc.Memprof.stop (fun () ->
                for n = 1 to 8 do
                  launch (build ());
                  let exec = build () in
                  launch exec;
                  countdown := n;
                  exec.Device.Graph.set_launch_dims 0 ~global:ones3
                    ~local:ones3;
                  countdown := 0;
                  launch exec;
                  Device.synchronize device
                done);
            equal (list int) [ 24 ] (read_i32 a));
          test "copies stay out of graphs" (fun () ->
            let device = metal_device () in
            is_false (device_graph device).Device.Graph.supports_copy);
        ];
    ]
