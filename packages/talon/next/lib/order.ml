(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = { name : string; desc : bool; nulls_first : bool }

let asc name = { name; desc = false; nulls_first = false }
let desc name = { name; desc = true; nulls_first = false }
let nulls_first k = { k with nulls_first = true }

let equal k0 k1 =
  String.equal k0.name k1.name
  && Bool.equal k0.desc k1.desc
  && Bool.equal k0.nulls_first k1.nulls_first

let pp ppf k =
  let dir = if k.desc then "desc" else "asc" in
  if k.nulls_first then Format.fprintf ppf "nulls_first (%s %S)" dir k.name
  else Format.fprintf ppf "%s %S" dir k.name
