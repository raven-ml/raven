(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_uop
module U = Uop
module D = Dtype
module T = Tensor

(* Force [Op] to be linked so its initialiser installs the broadcasting hook
   that the element-wise operations rely on. Realizing a graph is meaningless
   without the composed-op surface, so the execution entry pulls it in. *)
let _op_linked = Sys.opaque_identity Op.broadcasted

(* One process-wide CPU device backs every realization. This mirrors
   tinygrad, whose [Tensor.realize] resolves the device internally rather than
   taking it as an argument. *)
let default_device = lazy (Tolk_cpu.create "CPU:0")
let device () = Lazy.force default_device
let device_name = "CPU:0"

(* Host-data inputs: a fresh buffer slot per source tensor, its seeded device
   buffer kept keyed by the [Ops.Buffer] node's tag so realization can bind it.
   Realized outputs are cached the same way so a second read does not recompute
   an already-materialised value. *)
let next_slot = ref 0

let fresh_slot () =
  let s = !next_slot in
  incr next_slot;
  s

let host_buffers : (int, Tolk.Device.Buffer.t) Hashtbl.t = Hashtbl.create 64
let output_buffers : (int, Tolk.Device.Buffer.t) Hashtbl.t = Hashtbl.create 64

let make_input ~dtype ~shape n fill =
  let dev = device () in
  let buf = Tolk.Device.create_buffer ~size:n ~dtype dev in
  Tolk.Device.Buffer.ensure_allocated buf;
  let bytes = Bytes.create (Tolk.Device.Buffer.nbytes buf) in
  fill bytes;
  Tolk.Device.Buffer.copyin buf bytes;
  let node =
    U.buffer ~slot:(fresh_slot ()) ~dtype ~shape:(T.shape_uop [ n ])
      ~device:(U.Single device_name) ()
  in
  Hashtbl.replace host_buffers (U.tag node) buf;
  Movement.reshape (T.of_uop node) shape

let of_float_array ~shape data =
  make_input ~dtype:D.float32 ~shape (Array.length data) (fun bytes ->
      Array.iteri
        (fun i x -> Bytes.set_int32_le bytes (i * 4) (Int32.bits_of_float x))
        data)

let of_int_array ~shape data =
  make_input ~dtype:D.int32 ~shape (Array.length data) (fun bytes ->
      Array.iteri
        (fun i x -> Bytes.set_int32_le bytes (i * 4) (Int32.of_int x))
        data)

(* Realize a batch of tensors: lower the shared graph to a linear schedule,
   seed the host inputs, execute, then rebind each tensor onto its output
   buffer. Only the passed tensors are rebound (shared subgraphs in other live
   tensors will recompute on a later realize; see the deferral note in the
   frontend changelog). *)
let realize_buffers ts =
  let dev = device () in
  let to_program = Tolk.Codegen.to_program dev (Tolk.Device.renderer dev) in
  (* Force each output into a materialised buffer: an unrealized ALU/movement
     expression has no store target for the scheduler to write. *)
  let outs = List.map (fun t -> U.contiguous ~src:(T.uop t) ()) ts in
  let sink = U.sink outs in
  let call, buffer_map = Tolk.Allocations.transform_to_call sink in
  let linear, var_vals =
    Tolk.Schedule.create_linear_with_vars
      ~get_kernel_graph:Tolk.Rangeify.get_kernel_graph call
  in
  let binding = Tolk.Realize.Buffers.create ~device:dev in
  List.iter
    (fun node ->
      match Hashtbl.find_opt host_buffers (U.tag node) with
      | Some buf -> Tolk.Realize.Buffers.seed binding node buf
      | None -> ())
    (U.toposort sink);
  Tolk.Realize.run_linear ~device:dev ~to_program binding ~var_vals linear;
  List.map2
    (fun t out ->
      match Hashtbl.find_opt buffer_map (U.tag out) with
      | Some node ->
          let buf = Tolk.Realize.Buffers.of_buffer_node binding node in
          T.set_uop t node;
          Hashtbl.replace output_buffers (U.tag node) buf;
          buf
      | None -> failwith "Run.realize: tensor was not scheduled to a buffer")
    ts outs

let realize_many ts = ignore (realize_buffers ts)

let realize t =
  ignore (realize_buffers [ t ]);
  t

let buffer_of t =
  let tag = U.tag (T.uop t) in
  match Hashtbl.find_opt output_buffers tag with
  | Some buf -> buf
  | None -> (
      match Hashtbl.find_opt host_buffers tag with
      | Some buf -> buf
      | None -> List.hd (realize_buffers [ t ]))

let data t = Tolk.Device.Buffer.as_bytes (buffer_of t)

let to_float_array t =
  let n = T.numel t in
  let bytes = data t in
  Array.init n (fun i -> Int32.float_of_bits (Bytes.get_int32_le bytes (i * 4)))

let to_int_array t =
  let n = T.numel t in
  let bytes = data t in
  Array.init n (fun i -> Int32.to_int (Bytes.get_int32_le bytes (i * 4)))

let item_float t =
  if T.numel t <> 1 then invalid_arg "Run.item_float: tensor is not a scalar";
  (to_float_array t).(0)

let item_int t =
  if T.numel t <> 1 then invalid_arg "Run.item_int: tensor is not a scalar";
  (to_int_array t).(0)

(* Boolean selection with a data-dependent length. The graph-level
   [Op.masked_select]/[Op.nonzero] need the result length up front to keep a
   static shape; here the length is instead discovered by realizing the count
   of kept elements, then the fixed-length form is built with it. *)

let masked_select ?fill_value ?size t mask =
  match size with
  | Some size -> Op.masked_select ?fill_value t mask ~size
  | None ->
      let size =
        item_int (Reduce.sum (Movement.broadcast_to (Dtype_ops.bool mask) (T.shape t)))
      in
      Op.masked_select ?fill_value t mask ~size

let nonzero ?fill_value ?size t =
  match size with
  | Some size -> Op.nonzero ?fill_value t ~size
  | None ->
      let size = item_int (Reduce.sum (Elementwise.ne t (T.i 0))) in
      Op.nonzero ?fill_value t ~size
