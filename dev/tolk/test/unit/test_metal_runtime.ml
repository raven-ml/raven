(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tolk
module P = Ir.Program

let global_ptr dt = Dtype.Ptr.create dt ~addrspace:Global ()

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

let create_i32_buffer device values =
  let buf =
    Device.create_buffer ~size:(List.length values) ~dtype:Dtype.int32 device
  in
  Device.Buffer.ensure_allocated buf;
  Device.Buffer.copyin buf (int32_to_bytes values);
  buf

let read_i32_buffer buf = Device.Buffer.as_bytes buf |> int32_list_of_bytes
let compile device ~name program = Device.compile_program device ~name program

let increment_program () =
  let dt = Dtype.int32 in
  let ptr = global_ptr dt in
  [|
    P.Param { idx = 0; dtype = ptr };
    P.Param { idx = 1; dtype = ptr };
    P.Const { value = Int 0; dtype = Dtype.int32 };
    P.Index { ptr = 1; idxs = [ 2 ]; gate = None; dtype = ptr };
    P.Index { ptr = 0; idxs = [ 2 ]; gate = None; dtype = ptr };
    P.Load { src = 3; alt = None; dtype = dt };
    P.Const { value = Int 1; dtype = dt };
    P.Add { lhs = 5; rhs = 6; dtype = dt };
    P.Store { dst = 4; value = 7 };
  |]

let test_compile_and_run_one_kernel () =
  let device = metal_device () in
  let program = compile device ~name:"metal_add_one" (increment_program ()) in
  let dst = create_i32_buffer device [ 0 ] in
  let src = create_i32_buffer device [ 41 ] in
  let queue = Device.queue device in
  Device.Queue.exec queue program [ dst; src ] [];
  Device.Queue.synchronize queue;
  equal (list int) [ 42 ] (read_i32_buffer dst)

let test_queue_exec_is_ordered () =
  let device = metal_device () in
  let program =
    compile device ~name:"metal_ordered_add_one" (increment_program ())
  in
  let a = create_i32_buffer device [ 0 ] in
  let b = create_i32_buffer device [ 0 ] in
  let queue = Device.queue device in
  Device.Queue.exec queue program [ b; a ] [];
  Device.Queue.exec queue program [ a; b ] [];
  Device.Queue.synchronize queue;
  equal (list int) [ 2 ] (read_i32_buffer a);
  equal (list int) [ 1 ] (read_i32_buffer b)

let test_synchronize_propagates_unprepared_program_error () =
  let device = metal_device () in
  let program =
    compile device ~name:"metal_released_program" (increment_program ())
  in
  let dst = create_i32_buffer device [ 0 ] in
  let src = create_i32_buffer device [ 1 ] in
  is_true (Option.is_some (Device.Program.entry_addr program));
  Device.Program.release program;
  is_true (Option.is_none (Device.Program.entry_addr program));
  let queue = Device.queue device in
  raises_invalid_arg "metal program not prepared" (fun () ->
      Device.Queue.exec queue program [ dst; src ] [];
      Device.Queue.synchronize queue)

let () =
  run "Metal_runtime"
    [
      group "Execution"
        [
          test "compile and run one kernel" test_compile_and_run_one_kernel;
          test "queue exec is ordered" test_queue_exec_is_ordered;
          test "synchronize propagates unprepared program error"
            test_synchronize_propagates_unprepared_program_error;
        ];
    ]
