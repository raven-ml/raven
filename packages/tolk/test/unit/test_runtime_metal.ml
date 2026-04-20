(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tolk
open Tolk_ir
module P = Program

let global_ptr dt = Dtype.Ptr.create dt ~addrspace:Global ~size:(-1)

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

let read_i32 buf = Device.Buffer.as_bytes buf |> int32_list_of_bytes

let increment_program () =
  let dt = Dtype.Val.int32 in
  let ptr = global_ptr dt in
  let b = P.create () in
  let p0 = P.emit b (Param { idx = 0; dtype = ptr }) in
  let p1 = P.emit b (Param { idx = 1; dtype = ptr }) in
  let c0 = P.emit b (Const { value = Const.int Dtype.Val.int32 0; dtype = Dtype.Val.int32 }) in
  let idx_src = P.emit b (Index { ptr = p1; idxs = [ c0 ]; gate = None; dtype = ptr }) in
  let idx_dst = P.emit b (Index { ptr = p0; idxs = [ c0 ]; gate = None; dtype = ptr }) in
  let l0 = P.emit b (Load { src = idx_src; alt = None; dtype = dt }) in
  let c1 = P.emit b (Const { value = Const.int dt 1; dtype = dt }) in
  let sum = P.emit b (Binary { op = `Add; lhs = l0; rhs = c1; dtype = dt }) in
  let _ = P.emit b (Store { dst = idx_dst; value = sum }) in
  P.finish b

let compile_incr device name =
  Device.compile_program device ~name (increment_program ())

let run_spec device spec bufs =
  let car = Realize.Compiled_runner.create ~device spec in
  ignore (Realize.Compiled_runner.call car bufs [] ~wait:true ~timeout:None);
  Device.synchronize device

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
          test "exec is ordered" (fun () ->
            let device = metal_device () in
            let spec = compile_incr device "metal_ordered_add_one" in
            let a = i32_buf device [ 0 ] in
            let b = i32_buf device [ 0 ] in
            run_spec device spec [ b; a ];
            run_spec device spec [ a; b ];
            equal (list int) [ 2 ] (read_i32 a);
            equal (list int) [ 1 ] (read_i32 b));
        ];
    ]
