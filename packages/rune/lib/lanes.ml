(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Named maps and lane gathers.

   [axis ()] is a fresh name for a map, and [Rune.vmap ~axis:a] gives its map
   the name [a]. Inside it, [Rune.lanes a x] is every lane's [x] stacked on a
   new leading axis, as data: the map that owns the name answers the call
   itself, so the value it returns is a constant of the map, while every other
   map, and every differentiation, passes the call on or transforms it.

   The gather travels as an effect so that each handler treats it as the
   transformation of its own state: jvp gathers the tangent, vmap swaps the
   physical lane axis, and reverse refuses a transpose no consumer needs. When
   no map named [a] encloses the call — the effect reaches jit or no handler —
   [lanes a x] is one lane, [unsqueeze ~axes:[0] x], as [axis_index] is one lane
   outside a map. *)

type axis = Axis of unit ref

let axis () = Axis (ref ())
let same (Axis a) (Axis b) = a == b

type _ Effect.t +=
  | E_lanes : { axis : axis; t_in : ('a, 'b) Nx.t } -> ('a, 'b) Nx.t Effect.t

let lanes a x =
  match Effect.perform (E_lanes { axis = a; t_in = x }) with
  | y -> y
  | exception Effect.Unhandled _ -> Nx.unsqueeze ~axes:[ 0 ] x
