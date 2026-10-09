(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module A = Nx_array
module L = Nx_array.Layout

let invalid_argf fmt = Format.kasprintf invalid_arg fmt

let of_array ~by s a =
  match Devices.position s (A.device a) with
  | Some k -> Value.Array { at = Devices.one s k; a }
  | None ->
      invalid_argf "%s: the array lies on %s, which is not a device of %a" by
        (Rig.name (A.device a))
        Devices.pp s

let of_shards ~by p arrays =
  let set = Devices.set p and devices = Grid.devices (Devices.grid p) in
  let n = Array.length devices in
  if Array.length arrays <> n then
    invalid_argf "%s: %d arrays for a placement of %d devices" by
      (Array.length arrays) n;
  Array.iteri
    (fun j k ->
      let d = Devices.rig set k and a = arrays.(j) in
      if not (Rig.equal (A.device a) d) then
        invalid_argf "%s: array %d lies on %s, where the placement puts %s" by j
          (Rig.name (A.device a))
          (Rig.name d))
    devices;
  let shape = L.shape (A.layout arrays.(0)) in
  Array.iteri
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
  if n = 1 then Value.Array { at = p; a = arrays.(0) }
  else Value.Shards { at = p; arrays = Array.copy arrays }

let array (type v s d) (x : (v, s, d) Value.t) =
  match x with Value.Array { a; _ } -> Some a | Value.Shards _ -> None

let shards (type v s d) (x : (v, s, d) Value.t) =
  match x with
  | Value.Array { a; _ } -> Some [| a |]
  | Value.Shards { arrays; _ } -> Some (Array.copy arrays)
