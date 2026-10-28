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
    addition adds the primal outward, [Lanes] is the dual of the primal's and
    the tangent's lanes, a custom rule or a linear call over an argument it
    tracks applies its rule, a scan, a remat and a call are performed outward
    with their programs' derivatives over the primals and the tangents (a scan's
    carry has a tangent where its init has one or a step gives it one, a zero
    until then; a call's derivative is derived once per derivation at its site,
    {!Program.derive}), an iterate, a switch, a root and a solve are performed
    outward with their derivatives too, an operator's tangent is the operator
    of the tangents, and [Lane_index] and a switch whose index no
    interpretation traces run their default where they were performed
    ({!Program.Evaluate}), so that their programs' operations reach the
    rules.

    A custom rule ({!Program.Custom}) is performed outward on the arguments'
    primals, so that each differentiation around applies it too. Its results'
    tangents are its tangent map, evaluated outward over the arguments' tangents
    (zeros for one without) and the residuals, or, under reverse mode, a linear
    call of its pullback over the tangents the tape holds; so a differentiation
    around differentiates the derivative's code. The residuals carry the
    derivatives around: the call performed outward is asked for them, and a call
    asked for its own derives them again, the residuals' slice of the primal
    evaluated under a differentiation seeded from this one's tangents. Forward
    mode has no rule for a pullback's results: where a reverse mode inside
    applies the pullback ([applied]), it takes their tangents from the rule's
    code; otherwise a float or complex result raises. A linear call
    ({!Program.Linear}) is the call on the primals and the call on the tangents,
    which a tape records as one operation, plus the derivative of its function's
    code along the captures.

    Under reverse mode a tangent map is not evaluated where no result has a
    float or complex array. The handler raises [Invalid_argument] for a rule
    that captures a value this differentiation tracks beside one of its
    arguments, and under reverse mode for a tangent map that adds a tangent to a
    total. *)

val run :
  name:string ->
  ?tape:Nx.Prim.interpretation ->
  (Nx.Prim.interpretation -> 'a) ->
  'a
(** [run ~name ~tape f] is [f i], [i] a new forward-mode interpretation named
    [name] in messages. [tape], where given, is the tape tracer ({!Trace.Tape})
    whose variables are the tangents [f] seeds, as reverse mode's are. Raises
    what [f] raises. *)

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

val tangent_of :
  Nx.Prim.interpretation -> ('v, 's, 'd) Nx.t -> ('v, 's, 'd) Nx.t option
(** [tangent_of i x] is [x]'s tangent as its rule made it, which may be a view
    of another value. [None] if [i] does not own [x]. *)
