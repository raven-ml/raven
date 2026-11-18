(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type class_ = {
  value : int;
  name : string option;
  title : string option;
  description : string option;
  nodata : bool;
}

type kind = Measurement | Bitfield | Classes of class_ list

type t = {
  name : string option;
  common_name : string option;
  center_wavelength : float option;
  full_width_half_max : float option;
  unit : string option;
  kind : kind;
  scale : float;
  offset : float;
  nodata : float option;
  saturated : float option;
}

let decode b dtype x =
  (match b.kind with
  | Measurement -> ()
  | Bitfield ->
      invalid_arg
        "Jord.Band.decode: a bitfield band is read as bits; mask its fill \
         with its product module's fill, or with Band.nodata where the \
         module has none, then read its flags with the product module or \
         Nx's bitwise operations"
  | Classes _ ->
      invalid_arg
        "Jord.Band.decode: a classified band's codes are classes; mask its \
         fill with Band.nodata, then compare its codes with its classes'");
  let specials = List.filter_map Fun.id [ b.nodata; b.saturated ] in
  let valid = Nx.logical_not (Dn.mem "Jord.Band.decode" specials x) in
  let v = Nx.add_s (Nx.mul_s (Nx.cast Nx.float64 x) b.scale) b.offset in
  (Nx.cast dtype (Nx.where valid v (Nx.zeros_like v)), valid)

let nodata b x = Dn.mem "Jord.Band.nodata" (Option.to_list b.nodata) x
let saturated b x = Dn.mem "Jord.Band.saturated" (Option.to_list b.saturated) x
