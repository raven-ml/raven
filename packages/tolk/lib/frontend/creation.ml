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

let dtype_of_fill = function
  | T.Sint _ | T.Sint64 _ -> D.weakint
  | T.Sfloat _ -> D.weakfloat
  | T.Sbool _ -> D.bool

let broadcast_scalar dt fill shape =
  let v = T.of_uop (U.const (T.scalar_const dt fill)) in
  Movement.expand (Movement.reshape v (List.map (fun _ -> 1) shape)) shape

let empty ?(dtype = D.default_float) ?device shape =
  if D.is_weak dtype then invalid_arg "Creation.empty: dtype must be concrete";
  let device = Option.value device ~default:(U.Single (Backend.device_name ())) in
  let n = List.fold_left ( * ) 1 shape in
  let buf =
    U.alloc ~bind_on_realize:true ~slot:(U.fresh_buffer_slot ()) ~dtype ~shape:(T.shape_uop [ n ])
      ~device ()
  in
  Movement.reshape (T.of_uop buf) shape

(* Clone [t] into a fresh buffer: an unallocated flat buffer viewed at [t]'s
   shape, written by a store effect. Realization allocates the storage and
   runs the fill, and in-place assignment then writes into it. *)
let clone ?device t =
  let device = match device, T.device t with
    | Some device, _ | None, Some device -> device
    | None, None -> U.Single (Backend.device_name ())
  in
  let canonicalize = Tolk.Helpers.canonicalize_device_name in
  let device = match device with
    | U.Single device | U.Multi [device] -> U.Single (canonicalize device)
    | U.Multi devices -> U.Multi (List.map canonicalize devices)
    | U.Index _ -> device in
  let disk = String.starts_with ~prefix:"DISK" in
  if (match device with U.Single d -> disk d
      | U.Multi devices -> List.exists disk devices | U.Index _ -> false) then
    invalid_arg "Creation.clone: cannot clone DISK storage; use an explicit store";
  let axis = match device with U.Multi _ -> U.axis (T.uop t) | _ -> None in
  let shape, max_shape, n =
    match axis with
    | Some _ -> U.shard_shape (T.uop t), U.max_shard_shape (T.uop t), U.max_shard_numel (T.uop t)
    | None -> T.symbolic_shape t, U.max_shape (T.uop t), U.max_numel (T.uop t) in
  let dtype = U.commit_dtype (T.uop t) in
  let buf =
    U.alloc ~bind_on_realize:true ~slot:(U.fresh_buffer_slot ()) ~dtype
      ~shape:(T.shape_uop [ n ]) ~device ()
  in
  let dst = U.reshape ~src:buf ~shape:(T.shape_uop max_shape) in
  let dst =
    if List.equal U.equal (U.shape dst) shape then dst
    else U.shrink ~src:dst
        ~offset:(T.shape_uop (List.map (fun _ -> 0) shape))
        ~size:(T.symbolic_shape_uop shape)
  in
  let dst = match axis with
    | Some axis -> U.unshard ~src:dst ~axes:[axis] ()
    | None -> dst in
  let value =
    match (T.device t, device) with
    | Some from, device when from <> device ->
        U.copy ~src:(T.uop t) ~device ()
    | _ -> T.uop t
  in
  let value = U.cast ~src:value ~dtype in
  T.of_uop (U.after ~src:dst ~deps:[ U.store ~dst ~value () ])

(* Each device's own shard along [axis] of [value], a value present in full on
   every device of [devices]. *)
let partition t value ~devices axis =
  let axis = T.resolve_dim t axis in
  let shape = T.symbolic_shape t in
  let count = U.const_int (List.length devices) in
  let size = List.nth shape axis in
  let remainder = U.simplify (U.alu_binary ~op:Ops.Floormod ~lhs:size ~rhs:count) in
  if U.const_int_value remainder <> Some 0 then
    invalid_arg "Creation.shard: axis size must be divisible by the device count";
  let size = U.simplify (U.alu_binary ~op:Ops.Floordiv ~lhs:size ~rhs:count) in
  let range = U.range ~size:count ~axis:(-1) ~kind:Axis_type.Device () in
  let offset = U.simplify (U.alu_binary ~op:Ops.Mul ~lhs:range ~rhs:size) in
  let local = U.shrink ~src:value
      ~offset:(T.symbolic_shape_uop (List.mapi (fun i _ -> if i = axis then offset else U.const_int 0) shape))
      ~size:(T.symbolic_shape_uop (List.mapi (fun i dim -> if i = axis then size else dim) shape)) in
  T.of_uop (U.unshard ~src:local ~axes:[axis] ~ranges:[range] ())

let replicated t =
  match U.axis (T.uop t) with
  | None -> true
  | Some _ | exception Invalid_argument _ -> false

let shard ?axis ~devices t =
  let devices = List.map Tolk.Helpers.canonicalize_device_name devices in
  if devices = [] then invalid_arg "Creation.shard: empty device group";
  match T.device t with
  | None -> t
  | Some (U.Multi on) when on = devices && replicated t ->
      (* The tinygrad counterpart rejects every multi-device source. A value
         replicated on [devices] is split where it lives: each device keeps
         its own shard of its replica, so resharding from replicated to split
         moves no data. *)
      (match axis with
       | None -> t
       | Some axis -> partition t (T.uop t) ~devices axis)
  | Some (U.Multi _ | U.Index _) ->
      invalid_arg "Creation.shard: source must be on one device or replicated on [devices]"
  | Some (U.Single source) ->
      match devices with
      | [device] ->
          if source = device then t
          else T.of_uop (U.copy ~src:(T.uop t) ~device:(U.Single device) ())
      | _ ->
          let copied = U.copy ~src:(T.uop t) ~device:(U.Multi devices) () in
          match axis with
          | None -> T.of_uop copied
          | Some axis -> partition t copied ~devices axis

let full ?dtype ?(buffer = true) shape fill =
  let dt = match dtype with Some d -> d | None -> dtype_of_fill fill in
  let v = broadcast_scalar dt fill shape in
  if buffer then clone v else v

let zeros ?dtype ?buffer shape = full ?dtype ?buffer shape (T.Sfloat 0.0)
let ones ?dtype ?buffer shape = full ?dtype ?buffer shape (T.Sfloat 1.0)
let const_like ?dtype t fill =
  let dt = match dtype with Some d -> d | None -> T.val_dtype t in
  broadcast_scalar dt fill (T.shape t)

let full_like ?dtype ?(buffer = true) t fill =
  let dt = match dtype with Some d -> d | None -> T.val_dtype t in
  Like.create t (fun shape device ->
      let scalar = T.of_uop (U.const (T.scalar_const dt fill)) in
      let value = Movement.symbolic_broadcast_to scalar shape in
      if buffer then clone ?device:(Option.map (fun d -> U.Single d) device) value
      else value)

let zeros_like ?dtype ?buffer t = full_like ?dtype ?buffer t (T.Sint 0)
let ones_like ?dtype ?buffer t = full_like ?dtype ?buffer t (T.Sint 1)
