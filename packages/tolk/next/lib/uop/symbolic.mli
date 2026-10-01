(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Symbolic simplification.

    Rewrites that replace a graph by a simpler one with the same values:
    constant folding, algebraic identities, reasoning on the bounds of integers
    ({!Ops.vmin}, {!Ops.vmax}) and on the conditions that guard them. They come
    in three matchers, each containing the one before: {!symbolic_simple} folds
    one node at a time, {!symbolic} matches deeper and canonicalises index
    arithmetic, and {!sym} adds the rewrites of loads, stores and reductions
    that code generation relies on.

    {b Values.} Every rewrite keeps each value bit for bit, IEEE's signed zeros,
    infinities, NaN and subnormals included, and wraps integers at their type,
    as compiled code computes them. The exceptions are the rounding of powers: a
    constant power is computed by products and square roots, and {!sym} computes
    every other one as [exp2 (y * log2 x)] ({!Transcendental.xpow}), as is
    [c ** x] for a positive finite constant [c]. So the rewrites that
    reassociate or distribute arithmetic, or rely on it never overflowing, apply
    to integers only, and to a committed integer only where no value it computes
    wraps ({!Ops.exact}); a weak integer, such as an index, never wraps.

    {b Invalid values.} An index that is {!Ops.invalid} where a condition fails,
    [where cond x invalid], is a {e gated} value ({!invalid_gate}). The rewrites
    keep the gate outermost, so that it reaches the load or store that reads the
    index, which then does nothing where the gate fails.

    {b Installation.} Initialising the module makes {!symbolic} the rules of
    {!Ops.simplify}, and so of {!Ops.resolve} and of every shape computation. *)

(** {1:invalid Invalid values} *)

val invalid_gate : Ops.Upat.t
(** [invalid_gate] matches [where cond x i] with [i] the constant
    {!Ops.invalid}, naming its sources ["cond"], ["x"] and ["i"]. *)

val pm_remove_invalid : (unit, Ops.t) Ops.Pattern_matcher.t
(** [pm_remove_invalid] replaces each {!Ops.invalid} that a gate or a stack
    holds by [0] of the holder's type, once the gates have been read. *)

(** {1:matchers Simplification} *)

val symbolic_simple : (unit, Ops.t) Ops.Pattern_matcher.t
(** [symbolic_simple] folds nodes whose value follows from their operation and
    their sources alone:

    - {b invalid values}: an operation on a gated value moves inside the gate;
      an operation on {!Ops.invalid} is invalid; a gate on a reduction's operand
      that does not depend on its ranges moves out of the reduction; a store to
      an invalid index does nothing, and a load from one is its alternative
      value, or [0];
    - {b identities}: [x + 0], [x lxor 0], [x lor 0], [x lsl 0], [x lsr 0],
      [x * 1] and [x // 1] are [x], a float [x + 0] only for [-0.]; [x // -1] is
      [-x]; [x // x] is [1]; [(x lxor y) lxor y] is [x]; [(x % y) % y] is
      [x % y]; a boolean [x land c] and [x lor c] with [c] constant are [x] or
      [c]; [x <> false], a double negation, [where x true false] and an
      idempotent operation of [x] with itself are [x]; [where x false true] is
      the negation of [x]; a boolean cast to an integer and compared to [0] or
      [1] is the boolean or its negation, and to any other integer [true]; the
      truncation of an integer is itself;
    - {b recombination}: in a weak integer sum, [(b % d) * m] and a term
      [q * (d * m)], where [q] is [b' // d] for some [b'] congruent to [b]
      modulo [d] up to constants, recombine into [b' * m]; where [q] is
      [(b' // d) % k] with [k] positive, into [(b' % (d * k)) * m];
    - {b zeros}: [x < x] is [false], [x <> x] is [false] for integers and
      booleans, [x % x], [x lxor x] and [x land 0] are [0], and so is [x * 0]
      for integers and booleans; a mask that clears only bits that a right shift
      or a division by a power of two drops is removed;
    - {b constants}: an arithmetic operation on constants is its value
      ({!Ops.exec_alu}), except {!Op.Threefry}; weak constants keep their
      mathematical value, a committed integer holds its type's value (a weak
      operand of an operation on one is read, and the result written, at its
      width), and an operation mixing weak and committed constants commits the
      weak ones to the promoted type; a cast of a constant is the constant of
      the cast's type; [0 / 0] is NaN;
    - {b booleans}: a boolean [*] is [land], and a boolean [+] or maximum is
      [lor];
    - {b casts}: a cast or bitcast to its operand's type is its operand; a
      bitcast of a constant is the constant with the same bits; a cast through a
      type that holds every value of the result's type, back to that type, is
      its operand; two bitcasts are one; a cast to a boolean is [x <> 0];
    - {b powers}: [x ** c], for a constant [c] that is an integer or a half
      integer, [0] or at least [1] in magnitude, is a product of powers of [x],
      its reciprocal and its square root, a float half-integer power being [+0.]
      at [-0.] and [+inf] at [-inf]; [c ** x] is [c] if [c = 1], and
      [exp2 (x * log2 c)] for positive finite [c]; a 64-bit integer packed from
      two 32-bit halves and unpacked again is the half read;
    - {b selections}: a selection between equal values is that value, and a
      selection by a constant is the branch it picks, keeping the selection's
      type;

    then cleans up movements ({!Movement.mop_cleanup}). *)

val commutative : (unit, Ops.t) Ops.Pattern_matcher.t
(** [commutative] orders the two operands of each commutative weak integer
    operation by {!Ops.compare_structure}, least first, so that two sums of the
    same terms become the same node. *)

val symbolic : (unit, Ops.t) Ops.Pattern_matcher.t
(** [symbolic] is {!symbolic_simple} and {!commutative}, followed by rewrites
    that match deeper:

    - {b terms}: [x lor not x] is [true]; [x + x] is [x * 2]; for integers, like
      terms combine, [x * c0 + x * c1] into [x * (c0 + c1)] and [y + x + x] into
      [y + x * 2], also as the last two terms of a longer sum, and [-(x + c)] is
      [-x + -c]; [c * (x + c')] is [c * x + c * c'] for a weak integer [x];
    - {b selections}: a selection by a negation swaps its branches; within
      [where c t f], [c] is [true] in [t] and [false] in [f], unless an
      {!Op.Index} is involved; [where g x 0 <> 0] is [g land (x <> 0)]; nested
      selections sharing a branch merge their conditions with [land] or [lor];
      an operation on two selections by the same condition, one of whose branch
      pairs is constant, selects between the operations on the branches, also as
      the last two terms of an integer sum; for integers,
      [where c t 0 + where c 0 f] is [where c t f];
    - {b bounds}: a comparison, division, remainder, variable, {!Op.After},
      {!Op.Special} or range with a constant end whose bounds are equal is that
      constant; an integer maximum of two operands whose bounds, as it reads
      them ({!Ops.operand_bounds}), do not overlap is the greater, committed to
      its type; an integer selection that computes a maximum,
      [where (a < b) b a] with [a] or [b] a constant, is {!Ops.maximum};
    - {b constants}: two applications of an associative operation to constants
      fold the constants together, sums, products and maxima for integers only;
      [(x // c1) // c2] is [x // (c1 * c2)] for positive [c2] where [c1 * c2]
      does not wrap ({!Ops.exact}); constants move to the end of integer sums
      and products;
    - {b comparisons}, on integers: [c0 + x < c1] is [x < c1 - c0] where neither
      side wraps; [c0 * x < c1] divides both sides by [c0], rounding up, and
      flips [x]'s sign if [c0] is negative; [x // d < c] is [x < c * d] for
      positive [d] and [c * d < x] for negative [d]; in [x < c] with [c]
      positive, a divisor [d] common to [c] and the coefficients of some terms
      of [x], whose other terms stay within [0] and [d - 1], divides out;
      [-x < -y] is [y < x]; [not (x < 1)] with [x] a sum of non-negative terms
      with positive coefficients drops the coefficients;
    - {b ranges}: a range modulo its end is the range, and divided by its end is
      [0];
    - {b casts}: a cast through a type that holds every value of the operand is
      one cast; a cast of an integer that fits the intermediate integer type is
      one cast; a binary operation on 64-bit or weak integers whose values fit
      32 bits computes in {!Dtype.Int32} and casts back; a cast of [x + c] to a
      signed integer is the cast of [x] plus [c];
    - {b ordering}: an {!Op.After} waits only on ranges, stores, calls,
      barriers, ends, backedges, linear programs and stages, and on the sources
      of any other node it names; an {!Op.After} or {!Op.End} of nothing is its
      value, and an {!Op.End} drops the ranges that became constants;

    then simplifies division and remainder ({!Divandmod.div_and_mod_symbolic}),
    and restores bare constant operands ({!Uop_weak.pm_uncast_const}). *)

(** {1:valid Conditions} *)

val uop_given_valid : ?try_simplex:bool -> Ops.t -> Ops.t -> Ops.t
(** [uop_given_valid valid u] is [u] simplified with {!symbolic}, knowing that
    the boolean [valid] holds. Each clause of the conjunction [valid] that
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
(** [sym] is {!symbolic} and {!pm_simplify_valid}, followed by:

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
