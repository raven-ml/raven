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
  | Placeholder

let equal = ( = )
let compare = Stdlib.compare

let to_string = function
  | Global -> "global"
  | Thread -> "thread"
  | Local -> "local"
  | Warp -> "warp"
  | Loop -> "loop"
  | Group_reduce -> "group_reduce"
  | Reduce -> "reduce"
  | Upcast -> "upcast"
  | Unroll -> "unroll"
  | Placeholder -> "placeholder"

let pp fmt kind = Format.pp_print_string fmt (to_string kind)

(* Sorting priority: Loop=-1, Thread=Global=0, Warp=1,
   Local=Group_reduce=2, Upcast=3, Reduce=4, Unroll=5. *)
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

let color = function
  | Global -> "blue"
  | Thread -> "BLUE"
  | Local -> "cyan"
  | Warp -> "CYAN"
  | Loop -> "WHITE"
  | Upcast -> "yellow"
  | Group_reduce -> "RED"
  | Reduce -> "red"
  | Unroll -> "magenta"
  | Placeholder -> "white"
