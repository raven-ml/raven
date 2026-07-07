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
  let dt = Dtype.Val.int32 in
  let ptr = Dtype.Ptr (global_ptr dt) in
  let p0 = U.param ~slot:0 ~dtype:ptr () in
  let p1 = U.param ~slot:1 ~dtype:ptr () in
  let c0 = U.const (Const.int Dtype.Val.int32 0) in
  let idx_src = U.index ~ptr:p1 ~idxs:[c0] ~as_ptr:true () in
  let idx_dst = U.index ~ptr:p0 ~idxs:[c0] ~as_ptr:true () in
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
  let idx_dst = U.index ~ptr:p0 ~idxs:[c0] ~as_ptr:true () in
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
    ]
