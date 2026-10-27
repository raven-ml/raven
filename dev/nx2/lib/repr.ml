(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module A = Nx_array
module L = Nx_array.Layout
module M = Nx_array.Move

let invalid_argf fmt = Format.kasprintf invalid_arg fmt

let pp_shape = Prim.pp_shape

let unsafe_of_array ~by s a =
  match Devices.position s (A.device a) with
  | Some k -> Value.Array { at = Devices.one s k; a; dead = Value.Live }
  | None ->
      invalid_argf "%s: the array lies on %s, which is not a device of %a" by
        (Rig.name (A.device a))
        Devices.pp s

let of_array ~by s a =
  let x = unsafe_of_array ~by s a in
  (* Its caller keeps the array: no donation writes this memory. *)
  Rig.Claim.share (A.buffer a);
  x

let of_shards ~by p arrays =
  let set = Devices.set p and devices = Grid.devices (Devices.grid p) in
  let n = Array.length devices in
  if Iarray.length arrays <> n then
    invalid_argf "%s: %d arrays for a placement of %d devices" by
      (Iarray.length arrays) n;
  Array.iteri
    (fun j k ->
      let d = Devices.rig set k and a = Iarray.get arrays j in
      if not (Rig.equal (A.device a) d) then
        invalid_argf "%s: array %d lies on %s, where the placement puts %s" by j
          (Rig.name (A.device a))
          (Rig.name d))
    devices;
  let shapes =
    Array.init n (fun j ->
        Iarray.to_array (L.shape (A.layout (Iarray.get arrays j))))
  in
  let rank = Array.length shapes.(0) in
  Array.iteri
    (fun j s ->
      if Array.length s <> rank then
        invalid_argf "%s: array %d's rank differs from array 0's" by j)
    shapes;
  let cuts = Grid.cuts (Devices.grid p) in
  Array.iter
    (fun (axis, _) ->
      if axis >= rank then
        invalid_argf "%s: the placement cuts axis %d of arrays of rank %d" by
          axis rank)
    cuts;
  (* Each of a cut's [t] tiles lies on [n / t] devices, so the whole's extent
     along it is the sum of every array's extent along it, times [t / n]. The
     windows of that shape then check each array. *)
  let shape = Array.copy shapes.(0) in
  Array.iter
    (fun (axis, t) ->
      let sum = Array.fold_left (fun sum s -> sum + s.(axis)) 0 shapes in
      shape.(axis) <- sum * t / n)
    cuts;
  Array.iteri
    (fun j s ->
      let w = Devices.window ~by p shape j in
      if Array.exists2 (fun (r : M.range) e -> r.count <> e) w s then
        invalid_argf "%s: array %d of shape %a is not its window of %a" by j
          pp_shape s pp_shape shape)
    shapes;
  Iarray.iter (fun a -> Rig.Claim.share (A.buffer a)) arrays;
  Prim.of_arrays p ~shape arrays

(* [x]'s arrays, handed to a caller: their memory leaves the claims for good, so
   that no donation writes what the caller reads. A traced value has no bytes to
   read; a formula answers [None]. *)
let crossing (type v s d) ~by (x : (v, s, d) Value.t) :
    (v, s) A.t iarray option =
  Prim.alive ~by x;
  match x with
  | Value.Deferred _ -> None
  | Value.Traced { owner; _ } ->
      invalid_argf "%s: a value traced by %s has no bytes" by owner.name
  | Value.Array _ | Value.Shards _ | Value.Donated _ ->
      let arrays = Prim.arrays x in
      Iarray.iter (fun a -> Rig.Claim.share (A.buffer a)) arrays;
      Some arrays

let array x =
  match crossing ~by:"Nx.Repr.array" x with
  | Some arrays when Iarray.length arrays = 1 -> Some (Iarray.get arrays 0)
  | Some _ | None -> None

let shards x = crossing ~by:"Nx.Repr.shards" x
