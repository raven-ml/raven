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

(* One process-wide device backs every realization. This mirrors tinygrad,
   whose [Tensor.realize] resolves the device internally rather than taking it
   as an argument. The backend is chosen the same way tinygrad selects
   [Device.DEFAULT]: the [DEV] environment variable picks a backend by name,
   otherwise backends are scanned in priority order and the first one that
   opens wins, falling back to CPU. *)
let all_devices : (string * (string -> Tolk.Device.t)) list =
  (match Device_metal.opener with
  | Some create -> [ ("METAL", create) ]
  | None -> [])
  @ [ ("CUDA", Tolk_cuda.create); ("CPU", Tolk_cpu.create) ]

let canonicalize device =
  let device =
    match String.index_opt device ':' with
    | Some i ->
        String.uppercase_ascii (String.sub device 0 i)
        ^ String.sub device i (String.length device - i)
    | None -> String.uppercase_ascii device
  in
  let len = String.length device in
  if len >= 2 && String.equal (String.sub device (len - 2) 2) ":0" then
    String.sub device 0 (len - 2)
  else device

let open_device device =
  let backend =
    match String.index_opt device ':' with
    | Some i -> String.sub device 0 i
    | None -> device
  in
  match List.assoc_opt backend all_devices with
  | Some create -> create device
  | None -> failwith (Printf.sprintf "unknown device %S" device)

let default_device =
  lazy
    (match Sys.getenv_opt "DEV" with
    | Some dev when String.trim dev <> "" ->
        open_device (canonicalize (String.trim dev))
    | _ ->
        let rec first_available = function
          | [] -> failwith "no usable devices"
          | (name, create) :: rest -> (
              try create name with _ -> first_available rest)
        in
        first_available all_devices)

let device () = Lazy.force default_device
let device_name () = Tolk.Device.name (device ())

(* Storage registry: every node known to be backed by a concrete device
   buffer — host inputs, realized outputs, assignment targets — keyed by node
   tag and carrying the node itself so the backing buffers can be enumerated
   (JIT capture holds them all). A node with buffer identity distinct from
   itself is registered under both tags, so a lookup succeeds whether it
   starts from the tensor's node or from the underlying [Ops.Buffer]. *)
let storage : (int, U.t * Tolk.Device.Buffer.t) Hashtbl.t = Hashtbl.create 64

let register node buf =
  Hashtbl.replace storage (U.tag node) (node, buf);
  let b = U.buf_uop node in
  if b != node && Ops.equal (U.op b) Ops.Buffer then
    Hashtbl.replace storage (U.tag b) (b, buf)

let buffer_of_node node =
  Option.map snd (Hashtbl.find_opt storage (U.tag node))

let buffer_nodes () =
  Hashtbl.fold
    (fun _ (node, _) acc ->
      if Ops.equal (U.op node) Ops.Buffer then node :: acc else acc)
    storage []

let make_input ~dtype ~shape n fill =
  let dev = device () in
  let buf = Tolk.Device.create_buffer ~size:n ~dtype dev in
  Tolk.Device.Buffer.ensure_allocated buf;
  let bytes = Bytes.create (Tolk.Device.Buffer.nbytes buf) in
  fill bytes;
  Tolk.Device.Buffer.copyin buf bytes;
  let node =
    U.buffer ~slot:(U.fresh_buffer_slot ()) ~dtype ~shape:(T.shape_uop [ n ])
      ~device:(U.Single (device_name ())) ()
  in
  register node buf;
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
   seed the host inputs and previously realized buffers, execute, then rebind
   each tensor onto its output buffer. Scheduled nodes appearing in other live
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
  let sink = U.sink outs in
  let call, buffer_map = Tolk.Allocations.transform_to_call sink in
  (* Rebind every scheduled node still referenced by a live tensor onto its
     final storage before executing, as the reference frontend does. *)
  let mappings =
    List.filter_map
      (fun n ->
        match Hashtbl.find_opt buffer_map (U.tag n) with
        | Some v when v != n -> Some (n, v)
        | _ -> None)
      (U.toposort sink)
  in
  T.apply_map mappings;
  let linear, var_vals =
    Tolk.Schedule.create_linear_with_vars
      ~get_kernel_graph:Tolk.Rangeify.get_kernel_graph call
  in
  let binding = Tolk.Realize.Buffers.create ~device:dev in
  List.iter
    (fun node ->
      match buffer_of_node node with
      | Some buf -> Tolk.Realize.Buffers.seed binding node buf
      | None -> ())
    (U.toposort sink);
  Tolk.Realize.run_linear ~device:dev ~to_program binding ~var_vals linear;
  (* Persist the storage of every rebound node so the next realize seeds it
     instead of recomputing (or, worse, reallocating it empty). *)
  List.iter
    (fun (_, v) ->
      let b = U.buf_uop v in
      match Tolk.Realize.Buffers.find_opt binding b with
      | Some buf -> register b buf
      | None -> ())
    mappings;
  List.map2
    (fun t out ->
      match Hashtbl.find_opt buffer_map (U.tag out) with
      | Some node ->
          let buf =
            Tolk.Realize.Buffers.of_buffer_node binding (U.buf_uop node)
          in
          T.set_uop t node;
          register node buf;
          buf
      | None -> (
          (* Nothing was scheduled: the tensor is already materialised (its
             node has buffer identity), so resolve its backing buffer. *)
          match buffer_of_node (U.buf_uop out) with
          | Some buf -> buf
          | None ->
              failwith "Run.realize: tensor was not scheduled to a buffer"))
    ts outs

let realize_many ts = ignore (realize_buffers ts)

let realize t =
  ignore (realize_buffers [ t ]);
  t

let buffer_of t =
  match buffer_of_node (T.uop t) with
  | Some buf -> buf
  | None -> List.hd (realize_buffers [ t ])

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
