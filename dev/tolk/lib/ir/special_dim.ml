(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

type t = Group_id of int | Local_id of int | Global_idx of int

let axis = function Group_id a | Local_id a | Global_idx a -> a
let equal = ( = )
let compare = Stdlib.compare

let pp fmt = function
  | Group_id i -> Format.fprintf fmt "gid%d" i
  | Local_id i -> Format.fprintf fmt "lid%d" i
  | Global_idx i -> Format.fprintf fmt "idx%d" i
