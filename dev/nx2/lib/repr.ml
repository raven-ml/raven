(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module A = Nx_array
module L = Nx_array.Layout

let invalid_argf fmt = Format.kasprintf invalid_arg fmt

let of_array ~by s a =
  match Devices.position s (A.device a) with
  | Some k ->
      (* Its caller keeps the array: no donation writes this memory. *)
      Rig.Claim.share (A.buffer a);
      Value.Array { at = Devices.one s k; a; dead = Prim.live }
  | None ->
      invalid_argf "%s: the array lies on %s, which is not a device of %a" by
        (Rig.name (A.device a))
        Devices.pp s

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
  let shape = L.shape (A.layout (Iarray.get arrays 0)) in
  Iarray.iteri
    (fun j a ->
      if L.shape (A.layout a) <> shape then
        invalid_argf "%s: array %d's shape differs from array 0's" by j)
    arrays;
  Array.iter
    (fun (axis, _) ->
      if axis >= Array.length shape then
        invalid_argf "%s: the placement cuts axis %d of arrays of rank %d" by
          axis (Array.length shape))
    (Grid.cuts (Devices.grid p));
  Iarray.iter (fun a -> Rig.Claim.share (A.buffer a)) arrays;
  Prim.of_arrays p arrays

(* [x]'s arrays, handed to a caller: their memory leaves the claims for good, so
   that no donation writes what the caller reads. A traced value has no bytes to
   read; a formula answers [None]. *)
let crossing (type v s d) ~by (x : (v, s, d) Value.t) :
    (v, s) A.t iarray option =
  Prim.alive ~by 0 x;
  let out arrays =
    Iarray.iter (fun a -> Rig.Claim.share (A.buffer a)) arrays;
    Some arrays
  in
  match x with
  | Value.Deferred _ -> None
  | Value.Traced { owner; _ } ->
      invalid_argf "%s: a value traced by %s has no bytes" by owner.name
  | Value.Array { a; _ } -> out (Iarray.of_list [ a ])
  | Value.Shards { arrays; _ } | Value.Donated { arrays; _ } -> out arrays

let array x =
  match crossing ~by:"Nx.Repr.array" x with
  | Some arrays when Iarray.length arrays = 1 -> Some (Iarray.get arrays 0)
  | Some _ | None -> None

let shards x = crossing ~by:"Nx.Repr.shards" x
