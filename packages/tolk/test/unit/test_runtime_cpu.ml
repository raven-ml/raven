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

let cpu name = Tolk_cpu.create ("CPU:" ^ name)

let create_i32_buffer device values =
  let buf =
    Device.create_buffer ~size:(List.length values) ~dtype:Dtype.int32 device
  in
  Device.Buffer.ensure_allocated buf;
  Device.Buffer.copyin buf (int32_to_bytes values);
  buf

let read_i32_buffer buf = Device.Buffer.as_bytes buf |> int32_list_of_bytes

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

let core_id_program ~threads =
  let dt = Dtype.Val.int32 in
  let ptr = global_ptr dt in
  let b = P.create () in
  let p0 = P.emit b (Param { idx = 0; dtype = ptr }) in
  let dv = P.emit b (Define_var { name = "core_id"; lo = 0; hi = threads - 1; dtype = dt }) in
  let idx = P.emit b (Index { ptr = p0; idxs = [ dv ]; gate = None; dtype = ptr }) in
  let _ = P.emit b (Store { dst = idx; value = dv }) in
  P.finish b

let run_spec device spec bufs =
  let car = Realize.Compiled_runner.create ~device spec in
  ignore (Realize.Compiled_runner.call car bufs [] ~wait:true ~timeout:None);
  Device.synchronize device

let () =
  run "Cpu_runtime"
    [
      group "Execution"
        [
          test "compile and run one kernel" (fun () ->
            let device = cpu "run-one" in
            let spec =
              Device.compile_program device ~name:"add_one"
                (increment_program ())
            in
            let dst = create_i32_buffer device [ 0 ] in
            let src = create_i32_buffer device [ 41 ] in
            run_spec device spec [ dst; src ];
            equal (list int) [ 42 ] (read_i32_buffer dst));
          test "exec is ordered" (fun () ->
            let device = cpu "ordered" in
            let spec =
              Device.compile_program device ~name:"ordered_add_one"
                (increment_program ())
            in
            let a = create_i32_buffer device [ 0 ] in
            let b = create_i32_buffer device [ 0 ] in
            run_spec device spec [ b; a ];
            run_spec device spec [ a; b ];
            equal (list int) [ 2 ] (read_i32_buffer a);
            equal (list int) [ 1 ] (read_i32_buffer b));
          test "core_id drives parallel execution" (fun () ->
            let device = cpu "core-id" in
            let threads = 4 in
            let spec =
              Device.compile_program device ~name:"write_core_id"
                (core_id_program ~threads)
            in
            let dst = create_i32_buffer device [ 0; 0; 0; 0 ] in
            run_spec device spec [ dst ];
            equal (list int) [ 0; 1; 2; 3 ] (read_i32_buffer dst));
        ];
    ]
