(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

type kind = Measurement | Bitfield

type t = {
  name : string option;
  common_name : string option;
  unit : string option;
  kind : kind;
  scale : float;
  offset : float;
  nodata : float option;
  saturated : float option;
}

(* [x] in float64, after refusing the dtypes that hold no real numbers. *)
let dns (type v s) fn (x : (v, s, _) Nx.t) =
  match Nx.Dtype.kind (Nx.dtype x) with
  | Float | Signed | Unsigned -> Nx.cast Nx.float64 x
  | Complex | Boolean ->
      let name = Nx.Dtype.name (Nx.dtype x) in
      invalid_arg (strf "%s: %s is not a real dtype" fn name)

(* Where the DNs [x] equal the special value [s], a NaN [s] matching NaN. *)
let equal x s = if Float.is_nan s then Nx.isnan x else Nx.equal_s x s

let none x = Nx.full_at (Nx.placement x) Nx.bool (Nx.shape x) false

let decode b dtype x =
  if b.kind = Bitfield then
    invalid_arg "Jord.Band.decode: a bitfield band is read as bits";
  let x = dns "Jord.Band.decode" x in
  let special =
    List.fold_left
      (fun m s -> Nx.logical_or m (equal x s))
      (none x)
      (List.filter_map Fun.id [ b.nodata; b.saturated ])
  in
  let valid = Nx.logical_not special in
  let v = Nx.add_s (Nx.mul_s x b.scale) b.offset in
  (Nx.cast dtype (Nx.where valid v (Nx.zeros_like v)), valid)

let saturated b x =
  let x = dns "Jord.Band.saturated" x in
  match b.saturated with None -> none x | Some s -> equal x s
