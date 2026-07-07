(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* End-to-end JIT capture/replay over the real CPU (clang) backend, asserting
   numeric results. Each test warms up, captures the kernel graph on the second
   call, and replays it on the third with fresh inputs — the same three-phase
   path the engine uses at runtime. Kernels are built directly at the scheduled
   sink level and compiled through [Codegen.to_program]. *)

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
let idx n = U.const (Const.int Dtype.Val.weakint n)
let ci n = U.const (Const.int Dtype.Val.int32 n)
let iptr size = Dtype.Ptr (Dtype.Ptr.create Dtype.Val.int32 ~addrspace:Global ~size)

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
    metadata = [];
    name = Some name;
    precompile = false;
    precompile_backward = false;
    aux = None;
  }

let buffer_node ~slot ~size () =
  U.buffer ~slot ~dtype:Dtype.int32 ~shape:(idx size)
    ~device:(U.Single device_name) ()

(* out[i] = in[i] + in[i] over a Global range of [size] elements. *)
let double_kernel name ~size =
  let p_out = U.param ~slot:0 ~dtype:(iptr size) () in
  let p_in = U.param ~slot:1 ~dtype:(iptr size) () in
  let r = U.range ~size:(idx size) ~axis:0 ~kind:Axis_type.Global () in
  let ld = U.load ~src:(U.index ~ptr:p_in ~idxs:[ r ] ~as_ptr:true ()) () in
  let v = U.alu_binary ~op:Ops.Add ~lhs:ld ~rhs:ld in
  let st = U.store ~dst:(U.index ~ptr:p_out ~idxs:[ r ] ~as_ptr:true ()) ~value:v () in
  U.sink
    ~kernel_info:(kernel_info name [ Axis_type.Global ])
    [ U.end_ ~value:st ~ranges:[ r ] ]

(* out[i] = in[i] + [addend] over a Global range of [size] elements. *)
let add_const_kernel name ~size ~addend =
  let p_out = U.param ~slot:0 ~dtype:(iptr size) () in
  let p_in = U.param ~slot:1 ~dtype:(iptr size) () in
  let r = U.range ~size:(idx size) ~axis:0 ~kind:Axis_type.Global () in
  let ld = U.load ~src:(U.index ~ptr:p_in ~idxs:[ r ] ~as_ptr:true ()) () in
  let v = U.alu_binary ~op:Ops.Add ~lhs:ld ~rhs:(ci addend) in
  let st = U.store ~dst:(U.index ~ptr:p_out ~idxs:[ r ] ~as_ptr:true ()) ~value:v () in
  U.sink
    ~kernel_info:(kernel_info name [ Axis_type.Global ])
    [ U.end_ ~value:st ~ranges:[ r ] ]

(* out[i] = sum_{j<=i} in[j] — a triangular reduce (cumsum). *)
let running_sum_kernel name ~size =
  let p_out = U.param ~slot:0 ~dtype:(iptr size) () in
  let p_in = U.param ~slot:1 ~dtype:(iptr size) () in
  let ri = U.range ~size:(idx size) ~axis:0 ~kind:Axis_type.Global () in
  let rj = U.range ~size:(idx size) ~axis:1 ~kind:Axis_type.Reduce () in
  let ld = U.load ~src:(U.index ~ptr:p_in ~idxs:[ rj ] ~as_ptr:true ()) () in
  let masked = U.O.where U.O.(ri < rj) (ci 0) ld in
  let red = U.reduce ~op:Ops.Add ~src:masked ~ranges:[ rj ] ~dtype:Dtype.Val.int32 in
  let st = U.store ~dst:(U.index ~ptr:p_out ~idxs:[ ri ] ~as_ptr:true ()) ~value:red () in
  U.sink
    ~kernel_info:(kernel_info name [ Axis_type.Global; Axis_type.Reduce ])
    [ U.end_ ~value:st ~ranges:[ ri ] ]

(* out[0] = sum_i in[i] over a single Reduce range — the scalar-output path. *)
let sum_to_scalar_kernel name ~size =
  let p_out = U.param ~slot:0 ~dtype:(iptr 1) () in
  let p_in = U.param ~slot:1 ~dtype:(iptr size) () in
  let r = U.range ~size:(idx size) ~axis:0 ~kind:Axis_type.Reduce () in
  let ld = U.load ~src:(U.index ~ptr:p_in ~idxs:[ r ] ~as_ptr:true ()) () in
  let red = U.reduce ~op:Ops.Add ~src:ld ~ranges:[ r ] ~dtype:Dtype.Val.int32 in
  let st = U.store ~dst:(U.index ~ptr:p_out ~idxs:[ idx 0 ] ~as_ptr:true ()) ~value:red () in
  U.sink ~kernel_info:(kernel_info name [ Axis_type.Reduce ]) [ st ]

(* A single-kernel JIT: one output, one external input. [buffers] resolves the
   argument nodes; the input node is mapped to the same buffer passed in
   [input_bufs], so replay re-seeds it. *)
let single_kernel_jit ~sink ~out_buf ~out_node ~in_node =
  let resolver = ref (fun _ -> None) in
  let fxn input_bufs _ =
    if Jit.is_capturing () then begin
      let cap_in = input_bufs.(0) in
      Jit.add_linear
        (U.linear
           [ U.call ~body:sink ~args:[ out_node; in_node ]
               ~info:(call_info "k") ]);
      resolver :=
        (fun node ->
          if U.tag node = U.tag out_node then Some out_buf
          else if U.tag node = U.tag in_node then Some cap_in
          else None)
    end
  in
  let tjit = Jit.create ~device ~to_program ~fxn () in
  fun input ->
    Jit.call tjit [| input |] [] ~buffers:(fun n -> !resolver n) ~wait:true

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
                  ~out_node:(buffer_node ~slot:0 ~size ())
                  ~in_node:(buffer_node ~slot:1 ~size ())
              in
              (* warmup: eager, no execution yet *)
              run (make_buffer [ 1; 2; 3; 4; 5; 6; 7; 8 ]);
              (* capture: runs the kernel on the current input *)
              run (make_buffer [ 1; 2; 3; 4; 5; 6; 7; 8 ]);
              equal (list int) [ 2; 4; 6; 8; 10; 12; 14; 16 ]
                (read_buffer out_buf);
              (* replay with fresh values recomputes into the same output *)
              run (make_buffer [ 10; 20; 30; 40; 50; 60; 70; 80 ]);
              equal (list int) [ 20; 40; 60; 80; 100; 120; 140; 160 ]
                (read_buffer out_buf));
          test "running sum (cumsum) reduces a triangular window" (fun () ->
            let size = 8 in
            let out_buf = zero_buffer size in
            let run =
              single_kernel_jit
                ~sink:(running_sum_kernel "running_sum" ~size)
                ~out_buf
                ~out_node:(buffer_node ~slot:0 ~size ())
                ~in_node:(buffer_node ~slot:1 ~size ())
            in
            run (make_buffer [ 1; 2; 3; 4; 5; 6; 7; 8 ]);
            run (make_buffer [ 1; 2; 3; 4; 5; 6; 7; 8 ]);
            equal (list int) [ 1; 3; 6; 10; 15; 21; 28; 36 ]
              (read_buffer out_buf);
            run (make_buffer [ 2; 0; 4; 0; 6; 0; 8; 0 ]);
            equal (list int) [ 2; 2; 6; 6; 12; 12; 20; 20 ]
              (read_buffer out_buf));
          test "sum to scalar hits the shape () output path" (fun () ->
            let size = 8 in
            let out_buf = zero_buffer 1 in
            let run =
              single_kernel_jit
                ~sink:(sum_to_scalar_kernel "sum_scalar" ~size)
                ~out_buf
                ~out_node:(buffer_node ~slot:0 ~size:1 ())
                ~in_node:(buffer_node ~slot:1 ~size ())
            in
            run (make_buffer [ 1; 2; 3; 4; 5; 6; 7; 8 ]);
            run (make_buffer [ 1; 2; 3; 4; 5; 6; 7; 8 ]);
            equal (list int) [ 36 ] (read_buffer out_buf);
            run (make_buffer [ 10; 10; 10; 10; 10; 10; 10; 10 ]);
            equal (list int) [ 80 ] (read_buffer out_buf));
        ];
      group "Multi-kernel program"
        [
          test "two chained kernels: double then add-ten" (fun () ->
            let size = 4 in
            let in_node = buffer_node ~slot:1 ~size () in
            let tmp_node = buffer_node ~slot:2 ~size () in
            let out_node = buffer_node ~slot:3 ~size () in
            let tmp_buf = zero_buffer size in
            let out_buf = zero_buffer size in
            let k1 = double_kernel "mul_two" ~size in
            let k2 = add_const_kernel "add_ten" ~size ~addend:10 in
            let resolver = ref (fun _ -> None) in
            let fxn input_bufs _ =
              if Jit.is_capturing () then begin
                let cap_in = input_bufs.(0) in
                Jit.add_linear
                  (U.linear
                     [
                       U.call ~body:k1 ~args:[ tmp_node; in_node ]
                         ~info:(call_info "k1");
                       U.call ~body:k2 ~args:[ out_node; tmp_node ]
                         ~info:(call_info "k2");
                     ]);
                resolver :=
                  (fun node ->
                    if U.tag node = U.tag out_node then Some out_buf
                    else if U.tag node = U.tag tmp_node then Some tmp_buf
                    else if U.tag node = U.tag in_node then Some cap_in
                    else None)
              end
            in
            let tjit = Jit.create ~device ~to_program ~fxn () in
            let run input =
              Jit.call tjit [| input |] []
                ~buffers:(fun n -> !resolver n) ~wait:true
            in
            run (make_buffer [ 1; 2; 3; 4 ]);
            run (make_buffer [ 1; 2; 3; 4 ]);
            equal (list int) [ 2; 4; 6; 8 ] (read_buffer tmp_buf);
            equal (list int) [ 12; 14; 16; 18 ] (read_buffer out_buf);
            run (make_buffer [ 5; 6; 7; 8 ]);
            equal (list int) [ 20; 22; 24; 26 ] (read_buffer out_buf));
        ];
    ]
