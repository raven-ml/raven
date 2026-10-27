(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Forward mode: an interpretation whose values carry a tangent.

    A {e dual} is a value of the interpretation holding a primal and a tangent,
    of the primal's form; a value with no tangent is its primal alone, so a zero
    tangent is no dual. Only float and complex values have tangents. The rule of
    each operation over duals issues its primal and its tangent's operations
    outward, under the interpretations around this one: a map's in one map of
    both ({!Scalar}). {!Nx.item} of a dual reads its primal where nx allows it
    ([Nx.Prim.twinned]).

    The interpretation's handler of constructs ({!Program.construct}) lies just
    outside it and answers those over its duals: [Detach] is the primal, an
    addition adds the primal outward, and the others run their default where
    they were performed ({!Program.Evaluate}), so that their programs'
    operations reach the rules. *)

val run : name:string -> (Nx.Prim.interpretation -> 'a) -> 'a
(** [run ~name f] is [f i], [i] a new forward-mode interpretation named [name]
    in messages. Raises what [f] raises. *)

val seed :
  Nx.Prim.interpretation ->
  ('v, 's, 'd) Nx.t ->
  ('v, 's, 'd) Nx.t ->
  ('v, 's, 'd) Nx.t
(** [seed i x t] is the dual of [i] of primal [x] and tangent [t]. *)

val primal : Nx.Prim.interpretation -> ('v, 's, 'd) Nx.t -> ('v, 's, 'd) Nx.t
(** [primal i x] is [x]'s primal: [x] itself if [i] does not own it. *)

val tangent :
  Nx.Prim.interpretation -> ('v, 's, 'd) Nx.t -> ('v, 's, 'd) Nx.t option
(** [tangent i x] is [x]'s tangent as its own memory, in C order: a copy where
    the tangent is a seed's or a view. [None] if [i] does not own [x]. *)
