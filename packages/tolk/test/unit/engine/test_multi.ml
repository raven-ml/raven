(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Multi-device execution: sharded and replicated schedules running on
   several CPU device instances (and, when available, a duplicated CUDA
   device tuple), exercising the engine's multi-buffer resolution, per-device
   kernel launches, and cross-device copies. *)

open Windtrap
open Tolk
open Tolk_uop
module U = Uop

let () = Device.register "CPU" Tolk_cpu.create
let () = Device.register "CUDA" Tolk_cuda.create

let cpu = lazy (Device.get "CPU")

let cuda_available =
  lazy (match Device.get "CUDA" with _ -> true | exception _ -> false)

(* Data helpers *)

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

let f32_buf device data =
  let buf =
    Device.create_buffer ~size:(Array.length data) ~dtype:Dtype.float32 device
  in
  Device.Buffer.ensure_allocated buf;
  Device.Buffer.copyin buf (f32_to_bytes data);
  buf

let shape_node dims =
  match List.map U.const_int dims with [ d ] -> d | ds -> U.stack ds

let f32_buffer_node device_name dims =
  U.buffer ~slot:(U.fresh_buffer_slot ()) ~dtype:Dtype.float32
    ~shape:(shape_node dims) ~device:(U.Single device_name) ()

(* Sharding: copy to the device tuple, then take each device's slice of the
   shard axis through the symbolic [_device_num] variable. The frontend shard
   API builds the same graph. *)

let int_ = U.const_int
let emit = function [ d ] -> d | ds -> U.stack ds

let shard_shrink shape ndev src axis =
  let dim = List.nth shape axis in
  let sz = dim / ndev in
  let dnum =
    U.variable ~name:"_device_num" ~min_val:0 ~max_val:(ndev - 1) ()
  in
  let off = U.alu_binary ~op:Ops.Mul ~lhs:dnum ~rhs:(int_ sz) in
  let before =
    List.mapi (fun i _ -> if i <> axis then int_ 0 else off) shape
  in
  let size =
    List.mapi (fun i s -> if i <> axis then int_ s else int_ sz) shape
  in
  U.shrink ~src ~offset:(emit before) ~size:(emit size)

let sharded x shape devices axis =
  let copied = U.copy ~src:x ~device:(U.Multi devices) () in
  U.multi ~src:(shard_shrink shape (List.length devices) copied axis) ~axis

(* Schedule and execute a sink, as the frontend realize does. *)

let realize ~device ~binding sink =
  let to_program = Codegen.to_program device (Device.renderer device) in
  let call, buffer_map = Allocations.transform_to_call sink in
  let linear, var_vals =
    Schedule.create_linear_with_vars
      ~get_kernel_graph:Rangeify.get_kernel_graph call
  in
  Realize.run_linear ~device ~to_program binding ~var_vals linear;
  buffer_map

let output_node buffer_map out =
  match Hashtbl.find_opt buffer_map (U.tag out) with
  | Some node -> node
  | None -> fail "output was not scheduled to a buffer"

let output_f32 binding buffer_map out =
  let node = output_node buffer_map out in
  match Realize.Buffers.find_opt binding (U.buf_uop node) with
  | Some buf -> read_f32 buf
  | None -> fail "output buffer was not bound"

let devs2 = [ "CPU:1"; "CPU:2" ]
let devs4 = [ "CPU:1"; "CPU:2"; "CPU:3"; "CPU:4" ]

(* Shard [data] reshaped to [shape] over the devices, run [op] on the sharded
   value, copy the result back to the host device, and return the realized
   array. [host] names the device holding the input and gathered output; the
   shard devices must share its backend, as one schedule compiles with one
   renderer. *)
let run_sharded ?(host = "CPU") ~devices ~shape ~axis data op =
  let device = Device.get host in
  let x = f32_buffer_node host [ Array.length data ] in
  let xs =
    sharded (U.reshape ~src:x ~shape:(shape_node shape)) shape devices axis
  in
  let out =
    U.contiguous ~src:(U.copy ~src:(op xs) ~device:(U.Single host) ()) ()
  in
  let binding = Realize.Buffers.create ~device in
  Realize.Buffers.seed binding x (f32_buf device data);
  let buffer_map = realize ~device ~binding (U.sink [ out ]) in
  output_f32 binding buffer_map out

let iota n = Array.init n (fun i -> float_of_int (i + 1))

let () =
  run "Multi_device"
    [
      group "Resolution"
        [
          test "mstack joins and mselect indexes seeded shards" (fun () ->
              let device = Lazy.force cpu in
              let data1 = [| 1.; 2.; 3.; 4. |] in
              let data2 = [| 10.; 20.; 30.; 40. |] in
              let a = f32_buffer_node "CPU:1" [ 4 ] in
              let b = f32_buffer_node "CPU:2" [ 4 ] in
              let binding = Realize.Buffers.create ~device in
              Realize.Buffers.seed binding a
                (f32_buf (Device.get "CPU:1") data1);
              Realize.Buffers.seed binding b
                (f32_buf (Device.get "CPU:2") data2);
              let ctx = Realize.exec_context () in
              let ms = U.mstack [ a; b ] in
              (match Realize.resolve_buffer binding ctx ms with
              | Realize.Multi m ->
                  equal ~msg:"mstack shard devices" (list string)
                    [ "CPU:1"; "CPU:2" ]
                    (List.map Device.Buffer.device
                       (Device.Multi_buffer.bufs m))
              | Realize.Single _ -> fail "MSTACK resolved to a single buffer");
              let second =
                Realize.resolve binding ctx (U.mselect ~src:ms ~index:1)
              in
              equal ~msg:"mselect shard contents" (array (float 1e-6)) data2
                (read_f32 second);
              raises_match
                (function Invalid_argument _ -> true | _ -> false)
                (fun () ->
                  ignore
                    (Realize.resolve binding ctx (U.mselect ~src:a ~index:0))));
          test "multi buffer node allocates one shard per device" (fun () ->
              let device = Lazy.force cpu in
              let node =
                U.buffer ~slot:(U.fresh_buffer_slot ()) ~dtype:Dtype.float32
                  ~shape:(shape_node [ 4 ]) ~device:(U.Multi devs2) ()
              in
              let binding = Realize.Buffers.create ~device in
              let ctx = Realize.exec_context () in
              match Realize.resolve_buffer binding ctx node with
              | Realize.Multi m ->
                  equal ~msg:"shard devices" (list string) devs2
                    (List.map Device.Buffer.device
                       (Device.Multi_buffer.bufs m));
                  (* SLICE of a multi buffer views every shard. *)
                  let sliced =
                    U.slice ~src:node ~offset:(U.const_int 1) ~size:2
                      ~dtype:Dtype.float32
                  in
                  (match Realize.resolve_buffer binding ctx sliced with
                  | Realize.Multi v ->
                      List.iter2
                        (fun view base ->
                          is_true ~msg:"view shares its shard's base"
                            (Device.Buffer.base_id view
                            = Device.Buffer.base_id base);
                          equal ~msg:"view size" int 2
                            (Device.Buffer.size view);
                          equal ~msg:"view offset" int 4
                            (Device.Buffer.offset view))
                        (Device.Multi_buffer.bufs v)
                        (Device.Multi_buffer.bufs m)
                  | Realize.Single _ ->
                      fail "SLICE of a multi buffer resolved to single")
              | Realize.Single _ ->
                  fail "multi-device BUFFER resolved to a single buffer");
        ];
      group "Execution"
        [
          test "shard and gather round-trips" (fun () ->
              let data = iota 8 in
              let got =
                run_sharded ~devices:devs2 ~shape:[ 8 ] ~axis:0 data Fun.id
              in
              equal (array (float 1e-6)) data got);
          test "copy to device tuple replicates" (fun () ->
              let device = Lazy.force cpu in
              let data = iota 8 in
              let x = f32_buffer_node "CPU" [ 8 ] in
              let out =
                U.contiguous ~src:(U.copy ~src:x ~device:(U.Multi devs2) ()) ()
              in
              let binding = Realize.Buffers.create ~device in
              Realize.Buffers.seed binding x (f32_buf device data);
              let buffer_map = realize ~device ~binding (U.sink [ out ]) in
              let node = output_node buffer_map out in
              match
                Realize.resolve_buffer binding (Realize.exec_context ())
                  (U.buf_uop node)
              with
              | Realize.Multi m ->
                  let bufs = Device.Multi_buffer.bufs m in
                  equal ~msg:"shard devices" (list string) devs2
                    (List.map Device.Buffer.device bufs);
                  List.iter
                    (fun buf ->
                      equal ~msg:"replicated shard contents"
                        (array (float 1e-6)) data (read_f32 buf))
                    bufs
              | Realize.Single _ ->
                  fail "replicated output is not a multi buffer");
          test "elementwise on sharded tensors" (fun () ->
              let data = iota 8 in
              let got =
                run_sharded ~devices:devs2 ~shape:[ 8 ] ~axis:0 data (fun xs ->
                    U.alu_binary ~op:Ops.Add ~lhs:xs ~rhs:xs)
              in
              equal (array (float 1e-6))
                (Array.map (fun v -> v +. v) data)
                got);
          test "elementwise with a broadcast operand" (fun () ->
              let data = iota 8 in
              let ones =
                U.expand
                  ~src:(U.const (Const.float Dtype.float32 1.0))
                  ~dims:(shape_node [ 8 ])
              in
              let got =
                run_sharded ~devices:devs2 ~shape:[ 8 ] ~axis:0 data (fun xs ->
                    U.alu_binary ~op:Ops.Add ~lhs:xs ~rhs:ones)
              in
              equal (array (float 1e-6)) (Array.map (fun v -> v +. 1.) data)
                got);
          test "reduce over the sharded axis allreduces on 2 devices"
            (fun () ->
              let data = iota 8 in
              let got =
                run_sharded ~devices:devs2 ~shape:[ 8 ] ~axis:0 data (fun xs ->
                    U.reduce_axis ~src:xs ~op:Ops.Add ~axes:[ 0 ])
              in
              equal (array (float 1e-6)) [| 36. |] got);
          test "reduce over the sharded axis allreduces on 4 devices"
            (fun () ->
              let data = iota 8 in
              let got =
                run_sharded ~devices:devs4 ~shape:[ 8 ] ~axis:0 data (fun xs ->
                    U.reduce_axis ~src:xs ~op:Ops.Add ~axes:[ 0 ])
              in
              equal (array (float 1e-6)) [| 36. |] got);
          test "reduce over a non-sharded axis stays sharded" (fun () ->
              let data = iota 8 in
              let got =
                run_sharded ~devices:devs2 ~shape:[ 2; 4 ] ~axis:0 data
                  (fun xs -> U.reduce_axis ~src:xs ~op:Ops.Add ~axes:[ 1 ])
              in
              (* Row sums of [[1..4]; [5..8]]. *)
              equal (array (float 1e-6)) [| 10.; 26. |] got);
        ];
      group "Cuda"
        [
          test "duplicated device tuple runs on one CUDA device" (fun () ->
              if not (Lazy.force cuda_available) then
                skip ~reason:"no CUDA device" ();
              let data = iota 8 in
              let got =
                run_sharded ~host:"CUDA" ~devices:[ "CUDA:0"; "CUDA:0" ]
                  ~shape:[ 8 ] ~axis:0 data (fun xs ->
                    U.reduce_axis ~src:xs ~op:Ops.Add ~axes:[ 0 ])
              in
              equal (array (float 1e-6)) [| 36. |] got);
        ];
    ]
