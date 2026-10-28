(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The forward-mode interpretation, whose values carry tangents.

    A {e dual} is a value of the interpretation holding a primal and a tangent
    of the primal's dtype and shape. A value that is not a dual is its own
    primal, and the rules take its tangent as zero. Only float and complex
    values have tangents. The rule of each operation over duals issues its
    primal's and its tangent's operations {e outward}, to the interpretations
    around this one ([Nx.Prim]). The rule of a [Map] issues one [Map] that
    computes both ({!Scalar}). {!Nx.item} of a dual reads its primal where nx
    allows it ([Nx.Prim.twinned]). {e Reverse mode} ({!Linear}) runs this
    interpretation over a tape tracer ({!Trace.Tape}), whose variables are the
    tangents it seeds. Forward mode runs {e alone} where no tape records its
    tangents.

    The interpretation's handler of constructs ({!Program.construct}) lies just
    outside it and answers each construct that reads a dual by the construct's
    derivative. [Detach] and an addition act on the primal, so a total receives
    primals; an [Accumulate]'s result also carries the sum of its running
    value's and its rows' tangents, which a collect inside the differentiation
    holds. [Lanes] of a dual is the dual of the primal's and the tangent's
    lanes, and an operator's tangent ({!Program.Operator}) is the operator of
    the tangents. A scan, an iterate, a switch, a remat, a call, a root and a
    solve are performed outward with their programs' derivatives over the
    primals and the tangents, a call's derived at its site ({!Program.derive}).
    A switch whose index no interpretation traces runs its default where it was
    performed ({!Program.Evaluate}), so that its branch's operations reach the
    rules.

    A custom rule's call ({!Program.Rule}) asks the innermost interpretation
    around it. This one answers with [None] unless it tracks one of the
    arguments and no interpretation traces their primals. Then the rule runs on
    the primals where the call was performed, and no interpretation inside this
    one holds a value it captures. A tangent map then runs there on the
    arguments' tangents: values under forward mode, the tape's variables under
    reverse mode, where an addition of one raises at the addition. A pullback
    under reverse mode is a linear call ({!Program.Linear}) whose transpose is
    the pullback itself ({!Program.Host}), which the tape records and each
    backward pass runs on that pass's cotangents. Under reverse mode neither
    runs where no result has a float or complex array, and a pullback never runs
    where forward mode runs alone.

    A traced rule ({!Program.Custom}) over an argument this differentiation
    tracks is performed outward on the arguments' primals, so that each
    differentiation around applies it too. A rule over values it captures alone
    runs its default where it was performed if no argument is traced, so that
    its code is differentiated. If one is, it is performed outward on the
    captures' primals, so that a differentiation around tracking the argument
    applies the rule, and its results' tangents are the derivative of its
    primal's code along the captures. For a rule with a tangent map, its
    results' tangents are the map evaluated outward over the arguments'
    tangents (zeros for one without) and the residuals. A differentiation
    around thus differentiates the map's code. Under reverse mode, the map is
    not evaluated where no result has a float or complex array. For a rule with
    a pullback under reverse mode, the results' tangents are a linear call of
    the pullback over the arguments' tangents, which the tape records. Where
    forward mode runs alone and a reverse mode inside applies the pullback
    ([applied]), the tangents are the derivative of the rule's primal code. The
    rule of a linear call ({!Program.Linear}) issues the call on the primals and
    the call on the tangents, which a tape records as one operation. Its tangent
    adds the derivative of its function's code along the captures.

    The interpretation raises [Invalid_argument] for a contraction whose scale
    has a tangent, and for a rule that captures a value this differentiation
    tracks beside one of its arguments, where the rule is traced or its
    results or tangent map's results are values of this differentiation. A
    pullback run on values that uses such a value raises in the backward pass,
    as a value used after its interpretation returned. Running alone, forward
    mode raises for a rule with a pullback that no reverse mode inside applies,
    where a result is float or complex. Under reverse mode it raises for a
    tangent map that adds a tangent to a total. *)

val run :
  name:string ->
  ?tape:Nx.Prim.interpretation ->
  (Nx.Prim.interpretation -> 'a) ->
  'a
(** [run ~name ~tape f] is [f i], [i] a new forward-mode interpretation named
    [name] in messages. [tape], where given, must be a tape tracer
    ({!Trace.Tape}), and the tangents [f] seeds are its variables. Raises what
    [f] raises. *)

val seed :
  Nx.Prim.interpretation ->
  ('v, 's, 'd) Nx.t ->
  ('v, 's, 'd) Nx.t ->
  ('v, 's, 'd) Nx.t
(** [seed i x t] is the dual of [i] with primal [x] and tangent [t]. [t] must
    have [x]'s dtype and shape. *)

val primal : Nx.Prim.interpretation -> ('v, 's, 'd) Nx.t -> ('v, 's, 'd) Nx.t
(** [primal i x] is [x]'s primal, or [x] where [x] is not a dual of [i]. *)

val tangent :
  Nx.Prim.interpretation -> ('v, 's, 'd) Nx.t -> ('v, 's, 'd) Nx.t option
(** [tangent i x] is [x]'s tangent as its own memory, in C order. It is a copy
    where the tangent is a seed's or a view. [None] where [x] is not a dual of
    [i]. *)

val tangent_of :
  Nx.Prim.interpretation -> ('v, 's, 'd) Nx.t -> ('v, 's, 'd) Nx.t option
(** [tangent_of i x] is [x]'s tangent as its rule made it, which may be a view
    of another value. [None] where [x] is not a dual of [i]. *)
