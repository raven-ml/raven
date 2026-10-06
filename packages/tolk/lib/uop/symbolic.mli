(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Symbolic simplification for code generation.

    {!sym} extends {!Shape.symbolic} with the rewrites of loads, stores and
    reductions that code generation relies on, and with reasoning on the
    conditions that gate values ({!pm_simplify_valid}). Its rewrites keep values
    as {!Shape.symbolic}'s do, except that it computes every power as
    [exp2 (y * log2 x)] ({!Transcendental.xpow}). *)

(** {1:valid Conditions} *)

val uop_given_valid : ?try_simplex:bool -> Ops.t -> Ops.t -> Ops.t
(** [uop_given_valid valid u] is [u] simplified with {!Shape.symbolic}, knowing
    that the boolean [valid] holds. Each clause of the conjunction [valid] that
    bounds an integer expression, [e < c], [c < e] or [not (e < c)], is assumed
    by replacing [e] with a variable of those bounds. With [try_simplex]
    (default [true]), a clause stating that a sum of irreducible terms is at
    least [1] also tries each term at least [1] in turn, and keeps the result
    they all agree on. The result has [u]'s values wherever [valid] holds. *)

val simplify_valid : Ops.t -> Ops.t option
(** [simplify_valid valid] is the conjunction [valid] with each clause
    simplified knowing that the clauses before it hold ({!uop_given_valid}),
    after removing duplicates and ordering the clauses so that one whose
    expression others depend on, then one of tighter bounds, comes first. It is
    [None] if nothing changes or [valid] involves an {!Op.Index}. *)

val pm_simplify_valid : (unit, Ops.t) Ops.Pattern_matcher.t
(** [pm_simplify_valid] simplifies each boolean conjunction with
    {!simplify_valid}, and the weak integer value of each gate knowing its
    condition holds ({!uop_given_valid}, without [try_simplex]) when it involves
    no {!Op.Index}. *)

val pm_drop_and_clauses : (unit, Ops.t) Ops.Pattern_matcher.t
(** [pm_drop_and_clauses] drops from each gate's condition the clauses that run
    in none of the gated value's ranges. *)

val pm_move_where_on_load : (unit, Ops.t) Ops.Pattern_matcher.t
(** [pm_move_where_on_load] turns a selection between a load's index and [0]
    into a load gated by the selection's condition: each clause that the index
    does not already require, that is not constant, that runs in no range the
    index lacks and that reads no other {!Op.Index} moves into the index's gate;
    the others stay in the selection. A gated load reads [+0.] where its gate
    fails, so a selection of [-0.] is left as it is. *)

val pm_clean_up_group_sink : (unit, Ops.t) Ops.Pattern_matcher.t
(** [pm_clean_up_group_sink] replaces a group of one node by that node, and
    splices into each sink and group the sources of its {!Op.Noop}, {!Op.Stack},
    {!Op.Sink} and {!Op.Group} sources. *)

val sym : (unit, Ops.t) Ops.Pattern_matcher.t
(** [sym] is {!Shape.symbolic} and {!pm_simplify_valid}, followed by:

    - {b powers}: every {!Op.Pow} is computed from [exp2] and [log2]
      ({!Transcendental.xpow});
    - {b stores}: storing to an index what a load from that index reads does
      nothing; storing [where g alt (load index)] stores [alt] where [g] holds;
      storing {!Ops.invalid} does nothing, and storing a gated value stores it
      where its gate holds;
    - {b reductions}: the factors of an integer reduced product that do not
      depend on the reduction's ranges move out of a sum, and out of a maximum
      when they are non-negative;
    - {b terms}: for integers, [-(x + y)] is [-x + -y], and [(x + y) * c] is
      [x * c + y * c] for weak integers;

    then cleans up groups and sinks ({!pm_clean_up_group_sink}). *)
