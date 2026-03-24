(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

type t = Group_id of int | Local_id of int | Global_idx of int

let axis = function Group_id a | Local_id a | Global_idx a -> a
let equal = ( = )
let compare = Stdlib.compare

let pp fmt t =
  let s = match t with Group_id _ -> "gid" | Local_id _ -> "lid" | Global_idx _ -> "idx" in
  Format.fprintf fmt "%s%d" s (axis t)
