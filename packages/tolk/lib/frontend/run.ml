(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Capture before [open Tolk_uop] shadows it with the uop movement module. *)
module Movement_ops = Movement
open Tolk_uop
module Movement = Movement_ops
module U = Uop
module D = Dtype
module T = Tensor

let device = Backend.device
let device_name = Backend.device_name

let owned_buffer node =
  match U.op node, U.Arg.as_param_arg (U.arg node) with
  | Ops.Buffer, Some { buffer = Some [buf]; _ }
    when Storage.is_allocated buf -> Some buf
  | _ -> None

(* Resolve a node that is a contiguous view of a realized buffer to that buffer
   viewed at its element offset — an alias, with no copy. Returns [None] when
   the view is non-contiguous, its offset is not statically known, or its base
   buffer has not been realized. A whole-buffer view resolves to the base buffer
   itself, so an already-materialised node keeps its exact identity. *)
let view_buffer node =
  let rec pending_effect node =
    if Option.is_some (owned_buffer node) then false
    else
      match U.op node with
      | Ops.After -> true
      | Ops.Buffer | Ops.Param -> false
      | _ ->
          let src = U.src node in
          Array.length src > 0 && pending_effect src.(0)
  in
  match U.contiguous_view_offset node with
  | None -> None
  | Some _ when pending_effect node -> None
  | Some offset ->
      Option.map
        (fun src ->
          let numel = List.fold_left ( * ) 1 (U.max_shape node) in
          let dtype = U.dtype node in
          if
            offset = 0
            && numel = Tolk.Device.Buffer.size src
            && D.equal dtype (Tolk.Device.Buffer.dtype src)
          then src
          else
            let v =
              Tolk.Device.Buffer.view src ~size:numel ~dtype
                ~offset:(offset * D.itemsize dtype)
            in
            Tolk.Device.Buffer.ensure_allocated v;
            v)
        (owned_buffer (U.buf_uop node))

let buffer_of_node = view_buffer

let make_input ~dtype ~shape n fill =
  let dev = device () in
  let buf = Tolk.Device.create_buffer ~size:n ~dtype dev in
  Tolk.Device.Buffer.ensure_allocated buf;
  let bytes = Bytes.create (Tolk.Device.Buffer.nbytes buf) in
  fill bytes;
  Tolk.Device.Buffer.copyin buf bytes;
  let node = U.from_buffer buf in
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

let of_bytes ~dtype ~shape data =
  if D.is_weak dtype then invalid_arg "Run.of_bytes: dtype must be concrete";
  let n = List.fold_left ( * ) 1 shape in
  let nbytes = n * D.itemsize dtype in
  if Bytes.length data <> nbytes then
    invalid_arg
      (Printf.sprintf "Run.of_bytes: expected %d bytes for shape, got %d"
         nbytes (Bytes.length data));
  make_input ~dtype ~shape n (fun bytes -> Bytes.blit data 0 bytes 0 nbytes)

(* Realize a batch of tensors: lower the shared graph to a linear schedule,
   execute using graph-owned storage, then rebind each tensor onto its output
   buffer. Scheduled nodes appearing in other live
   tensors — in particular the write effects embedded by [Op.assign] — are
   rebound onto their storage through [Tensor.apply_map], so an assignment
   executes once and later reads reuse the written buffer. Other shared
   subgraphs recompute on a later realize (see the deferral note in the
   frontend changelog). *)
let realize_buffers ts =
  let dev = device () in
  let to_program = Tolk.Codegen.to_program dev (Tolk.Device.renderer dev) in
  (* Force each output into a materialised buffer: an unrealized ALU/movement
     expression has no store target for the scheduler to write. *)
  let outs = List.map (fun t -> U.contiguous ~src:(T.uop t) ()) ts in
  let tensor_sink = U.sink outs in
  let sink, buffer_map = Tolk.Bufferize.run tensor_sink in
  let call = Tolk.Callify.transform_to_call sink in
  (* Rebind every scheduled node still referenced by a live tensor onto its
     final storage before executing, as the reference frontend does. *)
  let mappings =
    List.filter_map
      (fun n ->
        match Hashtbl.find_opt buffer_map (U.tag n) with
        | Some v when v != n -> Some (n, v)
        | _ -> None)
      (U.toposort tensor_sink @ U.toposort sink)
  in
  T.apply_map mappings;
  let linear, var_vals =
    Tolk.Schedule.create_linear_with_vars
      ~get_kernel_graph:Tolk.Rangeify.get_kernel_graph call
  in
  let binding = Tolk.Realize.Buffers.create ~device:dev in
  Tolk.Realize.run_linear ~device:dev ~to_program binding ~var_vals linear;
  List.map2 (fun t out ->
      (match Hashtbl.find_opt buffer_map (U.tag out) with
       | Some node -> T.set_uop t node
       | None -> ());
      match view_buffer (T.uop t) with
      | Some buffer -> Some buffer
      | None ->
          let node = U.buf_uop (T.uop t) in
          if U.op node = Ops.Buffer then
            Some (Tolk.Realize.Buffers.of_buffer_node binding node)
          else None) ts outs

let has_empty_shape t =
  List.exists (fun dim -> U.const_int_value dim = Some 0) (T.symbolic_shape t)

let realize_many ts =
  let ts = List.filter (fun t -> not (has_empty_shape t)) ts in
  if ts <> [] then ignore (realize_buffers ts)

let realize t =
  realize_many [ t ];
  t

(* Materialize [t] into a fresh buffer on the default device, written by a
   store effect. A graph that folds to a pure constant expression is placed on
   no device and owns no storage; reading its bytes needs one. *)
let materialize t = Creation.clone ~device:(U.Single (device_name ())) t

let buffer_of t =
  match buffer_of_node (T.uop t) with
  | Some buf -> buf
  | None -> (
      (* A contiguous view of a realized buffer aliases it directly, so resolve
         it without a round-trip through the scheduler. *)
      match view_buffer (T.uop t) with
      | Some buf -> buf
      | None -> (
          let t = if T.device t = None then materialize t else t in
          match List.hd (realize_buffers [ t ]) with
          | Some buf -> buf
          | None ->
              failwith
                "Run.buffer_of: tensor folded to a constant expression with no \
                 storage"))

let data t =
  if has_empty_shape t then Bytes.empty
  else begin
    if not (List.for_all
              (fun d -> Option.is_some (U.const_int_value d))
              (T.symbolic_shape t)) then
      invalid_arg "Run.data: tensor shape is symbolic";
    Tolk.Device.Buffer.as_bytes (buffer_of t)
  end

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
