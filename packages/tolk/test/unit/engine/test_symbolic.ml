(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* End-to-end symbolic execution: a uop graph shaped by a bound variable is
   scheduled once and launched for several bind values, with results checked
   against the equivalent concrete-shape computation. *)

open Windtrap
open Tolk
open Tolk_uop
module U = Uop

let float_data = [| 1.0; 2.0; 4.0; 8.0; 16.0; 32.0; 64.0; 128.0 |]

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

let input_buffer device data =
  let buf =
    Device.create_buffer ~size:(Array.length data) ~dtype:Dtype.float32 device
  in
  Device.Buffer.ensure_allocated buf;
  Device.Buffer.copyin buf (f32_to_bytes data);
  buf

(* SUM(SHRINK(buf, (0, start_pos+1))) with start_pos bound to [value]. The
   input buffer node is shared across calls so only the bind value differs. *)
let symbolic_sum_sink ~buf_node ~value =
  let n = Array.length float_data in
  let v =
    U.variable ~name:"start_pos" ~min_val:1 ~max_val:(n - 1) ()
  in
  let bound = U.bind ~var:v ~value:(U.const_int value) in
  let size = U.alu_binary ~op:Ops.Add ~lhs:bound ~rhs:(U.const_int 1) in
  let shr = U.shrink ~src:buf_node ~offset:(U.const_int 0) ~size in
  let red = U.reduce_axis ~src:shr ~op:Ops.Add ~axes:[ 0 ] in
  let out = U.contiguous ~src:red () in
  (U.sink [ out ], out)

let concrete_sum value =
  Array.fold_left ( +. ) 0.0 (Array.sub float_data 0 (value + 1))

let realize_output device ~to_program ~buf_node ~value =
  let sink, out = symbolic_sum_sink ~buf_node ~value in
  let call, buffer_map = Callify.transform_to_call sink in
  let linear, var_vals =
    Schedule.create_linear_with_vars
      ~get_kernel_graph:Rangeify.get_kernel_graph call
  in
  equal ~msg:"extracted bind value" (list (pair string int))
    [ ("start_pos", value) ]
    var_vals;
  let binding = Realize.Buffers.create ~device in
  Realize.Buffers.seed binding buf_node (input_buffer device float_data);
  Realize.run_linear ~device ~to_program binding ~var_vals linear;
  Device.synchronize device;
  match Hashtbl.find_opt buffer_map (U.tag out) with
  | Some node -> (
      match
        Realize.Buffers.find_opt binding (U.buf_uop node)
      with
      | Some buf -> (read_f32 buf).(0)
      | None -> fail "output buffer was not bound")
  | None -> fail "output was not scheduled to a buffer"

(* NEG(SHRINK(buf, (0, start_pos+1))): an elementwise kernel whose launch
   geometry is the symbolic size itself. *)
let symbolic_neg_sink ~buf_node ~value =
  let n = Array.length float_data in
  let v = U.variable ~name:"start_pos" ~min_val:1 ~max_val:(n - 1) () in
  let bound = U.bind ~var:v ~value:(U.const_int value) in
  let size = U.alu_binary ~op:Ops.Add ~lhs:bound ~rhs:(U.const_int 1) in
  let shr = U.shrink ~src:buf_node ~offset:(U.const_int 0) ~size in
  let out = U.contiguous ~src:(U.alu_unary ~op:Ops.Neg ~src:shr) () in
  (U.sink [ out ], out)

let realize_neg_output device ~to_program ~buf_node ~value =
  let sink, out = symbolic_neg_sink ~buf_node ~value in
  let call, buffer_map = Callify.transform_to_call sink in
  let linear, var_vals =
    Schedule.create_linear_with_vars
      ~get_kernel_graph:Rangeify.get_kernel_graph call
  in
  let binding = Realize.Buffers.create ~device in
  Realize.Buffers.seed binding buf_node (input_buffer device float_data);
  Realize.run_linear ~device ~to_program binding ~var_vals linear;
  Device.synchronize device;
  match Hashtbl.find_opt buffer_map (U.tag out) with
  | Some node -> (
      match Realize.Buffers.find_opt binding (U.buf_uop node) with
      | Some buf -> Array.sub (read_f32 buf) 0 (value + 1)
      | None -> fail "output buffer was not bound")
  | None -> fail "output was not scheduled to a buffer"

let symbolic_launch_matches_concrete device_name device =
  let to_program body =
    Codegen.to_program device (Device.renderer device) body
  in
  let buf_node =
    U.buffer ~slot:(U.fresh_buffer_slot ()) ~dtype:Dtype.float32
      ~shape:(U.const_int (Array.length float_data))
      ~device:(U.Single device_name) ()
  in
  List.iter
    (fun value ->
      let got = realize_neg_output device ~to_program ~buf_node ~value in
      let expected =
        Array.map (fun x -> -.x) (Array.sub float_data 0 (value + 1))
      in
      equal ~msg:(Printf.sprintf "neg prefix for start_pos=%d" value)
        (array (float 1e-6)) expected got)
    [ 2; 6 ]

let symbolic_reduce_matches_concrete device_name device =
  let compiles = ref 0 in
  let to_program body =
    incr compiles;
    Codegen.to_program device (Device.renderer device) body
  in
  let buf_node =
    U.buffer ~slot:(U.fresh_buffer_slot ()) ~dtype:Dtype.float32
      ~shape:(U.const_int (Array.length float_data))
      ~device:(U.Single device_name) ()
  in
  List.iter
    (fun value ->
      let got = realize_output device ~to_program ~buf_node ~value in
      equal ~msg:(Printf.sprintf "sum for start_pos=%d" value) (float 1e-6)
        (concrete_sum value) got)
    [ 3; 5; 7 ];
  (* The kernel AST is bind-value independent: one compilation serves all
     launches. *)
  equal ~msg:"single compilation across bind values" int 1 !compiles

let () =
  run "Engine_symbolic"
    [
      test "symbolic shrink+sum runs for several bind values on CPU" (fun () ->
          symbolic_reduce_matches_concrete "CPU" (Tolk_cpu.create "CPU"));
      test "symbolic shrink+sum runs for several bind values on CUDA"
        (fun () ->
          let device =
            try Tolk_cuda.create "CUDA"
            with Failure msg -> skip ~reason:msg ()
          in
          symbolic_reduce_matches_concrete "CUDA" device);
      test "symbolic launch dims run on CPU" (fun () ->
          symbolic_launch_matches_concrete "CPU" (Tolk_cpu.create "CPU"));
      test "symbolic launch dims run on CUDA" (fun () ->
          let device =
            try Tolk_cuda.create "CUDA"
            with Failure msg -> skip ~reason:msg ()
          in
          symbolic_launch_matches_concrete "CUDA" device);
    ]
