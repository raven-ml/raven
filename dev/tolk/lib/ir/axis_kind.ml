(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

type t =
  | Global
  | Thread
  | Local
  | Warp
  | Loop
  | Group_reduce
  | Reduce
  | Upcast
  | Unroll
  | Outer
  | Placeholder

let equal = ( = )
let compare = Stdlib.compare

let pp fmt kind =
  Format.pp_print_string fmt
    (match kind with
    | Global -> "global"
    | Thread -> "thread"
    | Local -> "local"
    | Warp -> "warp"
    | Loop -> "loop"
    | Group_reduce -> "group_reduce"
    | Reduce -> "reduce"
    | Upcast -> "upcast"
    | Unroll -> "unroll"
    | Outer -> "outer"
    | Placeholder -> "placeholder")
