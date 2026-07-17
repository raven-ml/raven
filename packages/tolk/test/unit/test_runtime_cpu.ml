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

let cpu name = Tolk_cpu.create ("CPU:" ^ name)

let create_i32_buffer device values =
  let buf =
    Device.create_buffer ~size:(List.length values) ~dtype:Dtype.int32 device
  in
  Device.Buffer.ensure_allocated buf;
  Device.Buffer.copyin buf (int32_to_bytes values);
  buf

let read_i32_buffer buf = Device.Buffer.as_bytes buf |> int32_list_of_bytes

let i32_view buf ~offset ~size =
  let view = Device.Buffer.view buf ~size ~dtype:Dtype.int32 ~offset in
  Device.Buffer.ensure_allocated view;
  view

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

let core_id_program ~threads =
  let dt = Dtype.int32 in
  let p0 = i32_param ~slot:0 in
  let core_id =
    U.variable ~name:"core_id" ~min_val:0 ~max_val:(threads - 1) ~dtype:dt ()
  in
  let idx = U.index ~ptr:p0 ~idxs:[core_id] () in
  let store = U.store ~dst:idx ~value:core_id () in
  [ p0; core_id; idx; store ]

let run_spec device spec bufs =
  let car = Realize.Compiled_runner.create ~device spec in
  ignore (Realize.Compiled_runner.call car bufs [] ~wait:true ~timeout:None);
  Device.synchronize device

(* Like [increment_program] but subtracting, so no other test builds this
   graph: nodes exported by the forked child below are genuinely foreign to
   the parent process until imported. *)
let decrement_program () =
  let dt = Dtype.int32 in
  let p0 = i32_param ~slot:0 in
  let p1 = i32_param ~slot:1 in
  let c0 = U.const (Const.int Dtype.int32 0) in
  let idx_src = U.index ~ptr:p1 ~idxs:[ c0 ] () in
  let idx_dst = U.index ~ptr:p0 ~idxs:[ c0 ] () in
  let l0 = U.load ~src:idx_src () in
  let c1 = U.const (Const.int dt 1) in
  let diff = U.alu_binary ~op:Ops.Sub ~lhs:l0 ~rhs:c1 in
  let store = U.store ~dst:idx_dst ~value:diff () in
  [ p0; p1; c0; idx_src; idx_dst; l0; c1; diff; store ]

let read_file path =
  let ic = open_in_bin path in
  Fun.protect
    ~finally:(fun () -> close_in ic)
    (fun () -> really_input_string ic (in_channel_length ic))

(* Export the program in a child process (re-running this executable with
   [TOLK_EXPORT_BLOB] set; the CPU runtime spawns domains, which forbids
   forking), then import and execute it here. *)
let export_blob_var = "TOLK_EXPORT_BLOB"

let export_child path =
  let oc = open_out_bin path in
  output_string oc (U.export (U.sink (decrement_program ())));
  close_out oc

let imported_program_runs () =
  let blob_file = Filename.temp_file "tolk-import-exec" ".blob" in
  Fun.protect
    ~finally:(fun () -> Sys.remove blob_file)
    (fun () ->
      let env =
        Array.append (Unix.environment ())
          [| export_blob_var ^ "=" ^ blob_file |]
      in
      let pid =
        Unix.create_process_env Sys.executable_name
          [| Sys.executable_name |]
          env Unix.stdin Unix.stdout Unix.stderr
      in
      let _, status = Unix.waitpid [] pid in
      (match status with
       | Unix.WEXITED 0 -> ()
       | _ -> fail "exporting child failed");
      let imported = U.import (read_file blob_file) in
      let device = cpu "imported" in
      let spec =
        Device.compile_program device ~name:"sub_one" (U.children imported)
      in
      let dst = create_i32_buffer device [ 0 ] in
      let src = create_i32_buffer device [ 42 ] in
      run_spec device spec [ dst; src ];
      equal (list int) [ 41 ] (read_i32_buffer dst))

(* Coverage gaps accepted as not unit-testable at this level:
   - the bfloat16 compiler probe (Compiler_cpu.supports_bf16) is gated on the
     host clang version;
   - the AArch64 CALL26/JUMP26 trampoline only fires for branch targets beyond
     +/-128 MiB, unreachable with kernel-sized images;
   - the no-core_id/global>1 dispatch guard (threads forced to 1) is not
     observable through the compile path, which emits global>1 only alongside a
     core_id variable. *)
let main () =
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
          test "wait returns positive elapsed time" (fun () ->
            (* BEAM search selects kernels by this timing; [None] would collapse
               every candidate to infinity. Assert a real, positive measurement
               reaches the caller through the same path search.ml uses. *)
            let device = cpu "wait-timing" in
            let spec =
              Device.compile_program device ~name:"timed_add_one"
                (increment_program ())
            in
            let dst = create_i32_buffer device [ 0 ] in
            let src = create_i32_buffer device [ 41 ] in
            let car = Realize.Compiled_runner.create ~device spec in
            (match
               Realize.Compiled_runner.call car [ dst; src ] [] ~wait:true
                 ~timeout:None
             with
            | Some t -> is_true ~msg:"elapsed time is positive" (t > 0.)
            | None -> fail "CPU call ~wait:true returned no timing");
            Device.synchronize device;
            equal (list int) [ 42 ] (read_i32_buffer dst));
          test "external_ptr wraps caller memory zero-copy" (fun () ->
            let device = cpu "external-ptr" in
            (* Caller-owned backing storage: a normal buffer's memory, whose
               address stands in for memory tolk does not own (e.g. an nx
               buffer wrapped zero-copy by rune's jit). *)
            let backing = create_i32_buffer device [ 41 ] in
            let ptr = Device.Buffer.addr backing in
            let spec =
              { Device.Buffer_spec.default with external_ptr = Some ptr }
            in
            let external_ =
              Device.create_buffer ~size:1 ~dtype:Dtype.int32 ~spec device
            in
            Device.Buffer.ensure_allocated external_;
            (* alloc hands [ptr] back verbatim: no copy, no fresh allocation. *)
            is_true ~msg:"external buffer aliases caller memory"
              (Nativeint.equal ptr (Device.Buffer.addr external_));
            (* An in-place kernel over the external buffer mutates the caller's
               memory directly, visible through the backing buffer. *)
            let prog =
              Device.compile_program device ~name:"cpu_external_add_one"
                (increment_program ())
            in
            run_spec device prog [ external_; external_ ];
            equal (list int) [ 42 ] (read_i32_buffer backing);
            (* Freeing the external buffer must neither free nor cache the caller
               memory (LRU skip): the backing buffer stays valid afterwards. *)
            Device.Buffer.deallocate external_;
            equal (list int) [ 42 ] (read_i32_buffer backing));
          test "buffer views copy at byte offsets" (fun () ->
            let device = cpu "views-copy" in
            let base = create_i32_buffer device [ 1; 2; 3; 4 ] in
            let view = i32_view base ~offset:4 ~size:2 in
            equal (list int) [ 2; 3 ] (read_i32_buffer view);
            Device.Buffer.copyin view (int32_to_bytes [ 20; 30 ]);
            equal (list int) [ 1; 20; 30; 4 ] (read_i32_buffer base));
          test "nested buffer views compose byte offsets" (fun () ->
            let device = cpu "nested-views" in
            let base = create_i32_buffer device [ 1; 2; 3; 4 ] in
            let mid = i32_view base ~offset:4 ~size:3 in
            let leaf = i32_view mid ~offset:4 ~size:1 in
            Device.Buffer.copyin leaf (int32_to_bytes [ 33 ]);
            equal (list int) [ 1; 2; 33; 4 ] (read_i32_buffer base));
          test "kernel dispatch binds buffer view offsets" (fun () ->
            let device = cpu "views-dispatch" in
            let spec =
              Device.compile_program device ~name:"cpu_view_add_one"
                (increment_program ())
            in
            let dst_base = create_i32_buffer device [ 0; 0; 0; 0 ] in
            let src_base = create_i32_buffer device [ 10; 41; 99; 100 ] in
            let dst = i32_view dst_base ~offset:4 ~size:1 in
            let src = i32_view src_base ~offset:4 ~size:1 in
            run_spec device spec [ dst; src ];
            equal (list int) [ 0; 42; 0; 0 ] (read_i32_buffer dst_base));
          test "oversized views are rejected" (fun () ->
            let device = cpu "view-bounds" in
            let base = create_i32_buffer device [ 1; 2; 3; 4 ] in
            raises (Invalid_argument "buffer view exceeds base buffer")
              (fun () ->
                ignore
                  (Device.Buffer.view base ~size:2 ~dtype:Dtype.int32
                     ~offset:12)));
          test "compile and run a cross-process imported kernel"
            imported_program_runs;
        ];
    ]

let () =
  match Sys.getenv_opt export_blob_var with
  | Some path -> export_child path
  | None -> main ()
