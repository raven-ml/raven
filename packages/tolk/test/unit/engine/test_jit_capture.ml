(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* End-to-end JIT capture/replay over the real CPU (clang) backend, asserting
   numeric results. The jitted function schedules its kernels through
   [Schedule.create_linear_with_vars] and executes the returned linear, the
   same path the frontend uses; during capture the schedule flows to the JIT
   through the capture registry and the returned linear is empty. Each test
   warms up, captures on the second call, and replays on the third with fresh
   inputs. Kernels are built directly at the scheduled sink level and compiled
   through [Codegen.to_program]. *)

open Windtrap
open Tolk
open Tolk_uop
module U = Uop

let device_name = "CPU:jit-capture"
let device = Tolk_cpu.create device_name
let renderer = Device.renderer device
let to_program body = Codegen.to_program device renderer body

(* Little-endian int32 <-> bytes for buffer payloads. *)
let int32_to_bytes values =
  let bytes = Bytes.create (List.length values * 4) in
  List.iteri
    (fun i v ->
      let v = Int32.of_int v in
      let off = i * 4 in
      let byte k =
        Char.chr
          (Int32.to_int
             (Int32.logand (Int32.shift_right_logical v (k * 8)) 0xFFl))
      in
      Bytes.set bytes off (byte 0);
      Bytes.set bytes (off + 1) (byte 1);
      Bytes.set bytes (off + 2) (byte 2);
      Bytes.set bytes (off + 3) (byte 3))
    values;
  bytes

let int32_list_of_bytes bytes =
  let len = Bytes.length bytes / 4 in
  List.init len (fun i ->
      let off = i * 4 in
      let byte k = Int32.of_int (Char.code (Bytes.get bytes (off + k))) in
      Int32.to_int
        (Int32.logor (byte 0)
           (Int32.logor
              (Int32.shift_left (byte 1) 8)
              (Int32.logor
                 (Int32.shift_left (byte 2) 16)
                 (Int32.shift_left (byte 3) 24)))))

let make_buffer values =
  let buf =
    Device.create_buffer ~size:(List.length values) ~dtype:Dtype.int32 device
  in
  Device.Buffer.ensure_allocated buf;
  Device.Buffer.copyin buf (int32_to_bytes values);
  buf

let zero_buffer size = make_buffer (List.init size (fun _ -> 0))
let read_buffer buf = Device.Buffer.as_bytes buf |> int32_list_of_bytes

(* Kernel-building helpers, mirroring the scheduled sinks the rangeifier emits. *)
let idx n = U.const (Const.int Dtype.weakint n)
let ci n = U.const (Const.int Dtype.int32 n)
let iparam ~slot size = U.param ~slot ~dtype:Dtype.int32 ~shape:(idx size) ()

let kernel_info name axis_types : U.kernel_info =
  {
    name;
    axis_types;
    dont_use_locals = false;
    applied_opts = [];
    opts_to_apply = Some [];
    estimates = None;
    beam = 0;
  }

let call_info name : U.call_info =
  {
    grad_fxn = None;
    name = Some name;
    precompile = false;
    precompile_backward = false;
    aux = None;
  }

let buffer_node ~size () =
  U.buffer ~slot:(U.fresh_buffer_slot ()) ~dtype:Dtype.int32 ~shape:(idx size)
    ~device:(U.Single device_name) ()

(* out[i] = in[i] + in[i] over a Global range of [size] elements. *)
let double_kernel name ~size =
  let p_out = iparam ~slot:0 size in
  let p_in = iparam ~slot:1 size in
  let r = U.range ~size:(idx size) ~axis:0 ~kind:Axis_type.Global () in
  let ld = U.load ~src:(U.index ~ptr:p_in ~idxs:[ r ] ()) () in
  let v = U.alu_binary ~op:Ops.Add ~lhs:ld ~rhs:ld in
  let st = U.store ~dst:(U.index ~ptr:p_out ~idxs:[ r ] ()) ~value:v () in
  U.sink
    ~kernel_info:(kernel_info name [ Axis_type.Global ])
    [ U.end_ ~value:st ~ranges:[ r ] ]

(* out[i] = in[i] + [addend] over a Global range of [size] elements. *)
let add_const_kernel name ~size ~addend =
  let p_out = iparam ~slot:0 size in
  let p_in = iparam ~slot:1 size in
  let r = U.range ~size:(idx size) ~axis:0 ~kind:Axis_type.Global () in
  let ld = U.load ~src:(U.index ~ptr:p_in ~idxs:[ r ] ()) () in
  let v = U.alu_binary ~op:Ops.Add ~lhs:ld ~rhs:(ci addend) in
  let st = U.store ~dst:(U.index ~ptr:p_out ~idxs:[ r ] ()) ~value:v () in
  U.sink
    ~kernel_info:(kernel_info name [ Axis_type.Global ])
    [ U.end_ ~value:st ~ranges:[ r ] ]

(* out[i] = sum_{j<=i} in[j] — a triangular reduce (cumsum). *)
let running_sum_kernel name ~size =
  let p_out = iparam ~slot:0 size in
  let p_in = iparam ~slot:1 size in
  let ri = U.range ~size:(idx size) ~axis:0 ~kind:Axis_type.Global () in
  let rj = U.range ~size:(idx size) ~axis:1 ~kind:Axis_type.Reduce () in
  let ld = U.load ~src:(U.index ~ptr:p_in ~idxs:[ rj ] ()) () in
  let masked = U.O.where U.O.(ri < rj) (ci 0) ld in
  let red = U.reduce ~op:Ops.Add ~src:masked ~ranges:[ rj ] ~dtype:Dtype.int32 in
  let st = U.store ~dst:(U.index ~ptr:p_out ~idxs:[ ri ] ()) ~value:red () in
  U.sink
    ~kernel_info:(kernel_info name [ Axis_type.Global; Axis_type.Reduce ])
    [ U.end_ ~value:st ~ranges:[ ri ] ]

(* out[0] = sum_i in[i] over a single Reduce range — the scalar-output path. *)
let sum_to_scalar_kernel name ~size =
  let p_out = iparam ~slot:0 1 in
  let p_in = iparam ~slot:1 size in
  let r = U.range ~size:(idx size) ~axis:0 ~kind:Axis_type.Reduce () in
  let ld = U.load ~src:(U.index ~ptr:p_in ~idxs:[ r ] ()) () in
  let red = U.reduce ~op:Ops.Add ~src:ld ~ranges:[ r ] ~dtype:Dtype.int32 in
  let st = U.store ~dst:(U.index ~ptr:p_out ~idxs:[ idx 0 ] ()) ~value:red () in
  U.sink ~kernel_info:(kernel_info name [ Axis_type.Reduce ]) [ st ]

(* JIT driver. The function builds the CALL(LINEAR) form allocations emits —
   scheduled kernels whose call-level PARAM slots index the outer buffer
   arguments — schedules it, seeds every registered buffer argument, and
   executes. [body_calls] builds the per-kernel calls; the outer arguments are
   the output node followed by the current input node. Returns a driver that
   registers a fresh input buffer and runs the JIT on it. *)
let schedule_jit ~registry ~out_node ~body_calls =
  let buffers node = Hashtbl.find_opt registry (U.tag node) in
  let fxn input_uops _var_vals =
    let big =
      U.call
        ~body:(U.linear body_calls)
        ~args:[ out_node; input_uops.(0) ]
        ~info:(call_info "jit")
    in
    let linear, var_vals =
      Schedule.create_linear_with_vars ~get_kernel_graph:Fun.id big
    in
    let binding = Realize.Buffers.create ~device in
    List.iter
      (fun call ->
        match U.as_call call with
        | Some { args; _ } ->
            List.iter
              (fun arg ->
                match buffers arg with
                | Some buf -> Realize.Buffers.seed binding arg buf
                | None -> ())
              args
        | None -> ())
      (U.children linear);
    Realize.run_linear ~device ~to_program binding ~var_vals ~wait:true linear
  in
  let tjit = Jit.create ~device ~to_program ~fxn () in
  fun values ->
    let node = buffer_node ~size:(List.length values) () in
    Hashtbl.replace registry (U.tag node) (make_buffer values);
    Jit.call tjit [| node |] [] ~wait:true
      ~held_buffers:(fun () -> [ out_node ])
      ~buffers

(* A single-kernel JIT: one output, one external input. *)
let single_kernel_jit ~sink ~out_buf ~out_node =
  let registry : (int, Device.Buffer.t) Hashtbl.t = Hashtbl.create 8 in
  Hashtbl.replace registry (U.tag out_node) out_buf;
  let cp_out = U.param ~slot:0 ~dtype:Dtype.int32 () in
  let cp_in = U.param ~slot:1 ~dtype:Dtype.int32 () in
  schedule_jit ~registry ~out_node
    ~body_calls:
      [ U.call ~body:sink ~args:[ cp_out; cp_in ] ~info:(call_info "k") ]

let () =
  run "Engine_jit_capture"
    [
      group "Capture and replay (numeric)"
        [
          test "elementwise double: capture computes, replay recomputes"
            (fun () ->
              let size = 8 in
              let out_buf = zero_buffer size in
              let run =
                single_kernel_jit
                  ~sink:(double_kernel "mul_two" ~size)
                  ~out_buf
                  ~out_node:(buffer_node ~size ())
              in
              (* warmup executes eagerly *)
              run [ 1; 2; 3; 4; 5; 6; 7; 8 ];
              equal (list int) [ 2; 4; 6; 8; 10; 12; 14; 16 ]
                (read_buffer out_buf);
              (* capture: runs the recorded schedule on the current input *)
              run [ 1; 2; 3; 4; 5; 6; 7; 8 ];
              equal (list int) [ 2; 4; 6; 8; 10; 12; 14; 16 ]
                (read_buffer out_buf);
              (* replay with fresh values recomputes into the same output *)
              run [ 10; 20; 30; 40; 50; 60; 70; 80 ];
              equal (list int) [ 20; 40; 60; 80; 100; 120; 140; 160 ]
                (read_buffer out_buf));
          test "running sum (cumsum) reduces a triangular window" (fun () ->
            let size = 8 in
            let out_buf = zero_buffer size in
            let run =
              single_kernel_jit
                ~sink:(running_sum_kernel "running_sum" ~size)
                ~out_buf
                ~out_node:(buffer_node ~size ())
            in
            run [ 1; 2; 3; 4; 5; 6; 7; 8 ];
            run [ 1; 2; 3; 4; 5; 6; 7; 8 ];
            equal (list int) [ 1; 3; 6; 10; 15; 21; 28; 36 ]
              (read_buffer out_buf);
            run [ 2; 0; 4; 0; 6; 0; 8; 0 ];
            equal (list int) [ 2; 2; 6; 6; 12; 12; 20; 20 ]
              (read_buffer out_buf));
          test "sum to scalar hits the shape () output path" (fun () ->
            let size = 8 in
            let out_buf = zero_buffer 1 in
            let run =
              single_kernel_jit
                ~sink:(sum_to_scalar_kernel "sum_scalar" ~size)
                ~out_buf
                ~out_node:(buffer_node ~size:1 ())
            in
            run [ 1; 2; 3; 4; 5; 6; 7; 8 ];
            run [ 1; 2; 3; 4; 5; 6; 7; 8 ];
            equal (list int) [ 36 ] (read_buffer out_buf);
            run [ 10; 10; 10; 10; 10; 10; 10; 10 ];
            equal (list int) [ 80 ] (read_buffer out_buf));
        ];
      group "Multi-kernel program"
        [
          test "two chained kernels: double then add-ten through a planned \
                intermediate" (fun () ->
            let size = 4 in
            let out_node = buffer_node ~size () in
            let tmp_node = buffer_node ~size () in
            let out_buf = zero_buffer size in
            let registry : (int, Device.Buffer.t) Hashtbl.t =
              Hashtbl.create 8
            in
            Hashtbl.replace registry (U.tag out_node) out_buf;
            let k1 = double_kernel "mul_two" ~size in
            let k2 = add_const_kernel "add_ten" ~size ~addend:10 in
            let cp_out = U.param ~slot:0 ~dtype:Dtype.int32 () in
            let cp_in = U.param ~slot:1 ~dtype:Dtype.int32 () in
            (* [tmp_node] is internal to the schedule: it is never registered
               nor held, so the capture folds it into an arena. *)
            let run =
              schedule_jit ~registry ~out_node
                ~body_calls:
                  [
                    U.call ~body:k1 ~args:[ tmp_node; cp_in ]
                      ~info:(call_info "k1");
                    U.call ~body:k2 ~args:[ cp_out; tmp_node ]
                      ~info:(call_info "k2");
                  ]
            in
            run [ 1; 2; 3; 4 ];
            run [ 1; 2; 3; 4 ];
            equal (list int) [ 12; 14; 16; 18 ] (read_buffer out_buf);
            run [ 5; 6; 7; 8 ];
            equal (list int) [ 20; 22; 24; 26 ] (read_buffer out_buf));
        ];
    ]
