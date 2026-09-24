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
  Device.runtime device
    (U.sanitize_function_name (Program_spec.name spec))
    lib ~runtimevars:[]

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

let () =
  run "Metal_runtime"
    [
      group "Execution"
        [
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
          test "copies stay out of graphs" (fun () ->
            let device = metal_device () in
            is_false (device_graph device).Device.Graph.supports_copy);
        ];
    ]
