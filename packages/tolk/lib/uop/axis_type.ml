(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Declaration order is load-bearing: compare and sort stability depend
   on it. Do not reorder. *)
type t =
  | Global
  | Warp
  | Local
  | Loop
  | Group_reduce
  | Reduce
  | Upcast
  | Unroll
  | Thread
  | Placeholder

let equal = ( = )
let compare = Stdlib.compare

let to_string = function
  | Global -> "global"
  | Warp -> "warp"
  | Local -> "local"
  | Loop -> "loop"
  | Group_reduce -> "group_reduce"
  | Reduce -> "reduce"
  | Upcast -> "upcast"
  | Unroll -> "unroll"
  | Thread -> "thread"
  | Placeholder -> "placeholder"

let pp fmt t = Format.pp_print_string fmt (to_string t)

let to_pos = function
  | Loop -> -1
  | Thread | Global -> 0
  | Warp -> 1
  | Local | Group_reduce -> 2
  | Upcast -> 3
  | Reduce -> 4
  | Unroll -> 5
  | Placeholder -> 6

let letter = function
  | Global -> "g"
  | Thread -> "t"
  | Local -> "l"
  | Warp -> "w"
  | Loop -> "L"
  | Upcast -> "u"
  | Group_reduce -> "G"
  | Reduce -> "R"
  | Unroll -> "r"
  | Placeholder -> "?"
