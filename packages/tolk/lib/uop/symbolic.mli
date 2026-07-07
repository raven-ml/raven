(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Symbolic simplification of the {!Uop} IR.

    Pattern-based rewriter that folds algebraic identities, propagates
    constants and {!Const.Invalid} sentinels, and canonicalises term
    order. Rules are grouped into three cumulative layers; each layer is
    a {!Upat.Pattern_matcher.t} value that can be composed with
    {{!Upat.Pattern_matcher.val-(++)}[++]}.

    {1:layers Layers}

    Each layer extends the previous one. Installing {!sym} (or invoking
    {!simplify}) exercises all three. *)

val symbolic_simple : Upat.Pattern_matcher.t
(** [symbolic_simple] is the base layer of local, context-free folds:
    identity laws ([x + 0], [x * 1], [cdiv x 1], [x // 1],
    [cdiv x x], [x // x], [cmod x x], [x mod x], [x < x]),
    Python-floor constant folding for {!Ops.Floordiv} and
    {!Ops.Floormod}, boolean algebra and [where] identities, idempotent
    ALU collapsing, cast and bitcast folding over constants, [pow] of
    small integer exponents, [bool]-typed arithmetic rewriting into
    logical connectives, and propagation of {!Const.Invalid} through
    unary and binary ALU ops. *)

val pm_fold_lane_stack : Upat.Pattern_matcher.t
(** [pm_fold_lane_stack] folds a [Stack] of per-lane [Index] extracts of a
    single vector value back into that value when the shapes agree.
    Applied after memory coalescing, where such lane re-stacks appear. *)

val symbolic : Upat.Pattern_matcher.t
(** [symbolic] extends {!symbolic_simple} with rewrites that reshape
    arithmetic trees:

    - two-stage associative folding for every op in
      {!Ops.Group.associative}, e.g. [(x op c0) op c1 -> x op (c0 op c1)];
    - linear-combination folding ([x * c0 + x * c1 -> x * (c0 + c1)],
      [x + x -> x * 2], [x + x * c -> x * (c + 1)]);
    - nested division collapsing ([(x / c1) / c2 -> x / (c1 * c2)],
      including {!Ops.Floordiv} when the outer divisor is positive);
    - const-tail canonicalisation for [+] and [*] so constants float
      to the right;
    - [Cmplt] with a shifted constant sum ([(c0 + x) < c1 -> x < c1 - c0]);
    - range-bound specialisations (a {!Ops.Range} modulo or divided by
      its own upper bound, and any node whose analysis bounds coincide
      folds to that constant);
    - non-narrowing cast chains;
    - the div/mod simplification rules of {!Divandmod.div_and_mod_symbolic},
      appended last so that constant-folding and canonicalisation run first.

    This is the pattern matcher installed into {!Uop.simplify_ref} (via
    {!simplify}). *)

val index_pushing : Upat.Pattern_matcher.t
(** [index_pushing] is the tinygrad-compatible subset for value-lane
    {!Ops.Index}: constant indexes into {!Ops.Stack} are projected and
    identity stacks of indexes collapse back to their source. It is kept
    separate because late reduction lowering composes only
    [pm_reduce + index_pushing], not the full symbolic matcher. *)

val pm_simplify_valid : Upat.Pattern_matcher.t
(** [pm_simplify_valid] simplifies validity predicates and gated weakint
    indices. *)

val pm_drop_and_clauses : Upat.Pattern_matcher.t
(** [pm_drop_and_clauses] removes invalid-gate clauses that do not depend
    on ranges used by the gated index. *)

val pm_remove_invalid : Upat.Pattern_matcher.t
(** [pm_remove_invalid] replaces non-index {!Const.Invalid} constants with
    zero after load insertion. Weak-index invalid sentinels are preserved so
    masked indexes still carry their gate. *)

val pm_move_where_on_load : Upat.Pattern_matcher.t
(** [pm_move_where_on_load] moves eligible [where] guards around value-typed
    indexes into the index validity predicate. *)

val pm_lower_index_dtype : Upat.Pattern_matcher.t
(** [pm_lower_index_dtype] lowers weak integer indexing expressions to
    concrete integer math while preserving the weak integer interface with
    outer casts. *)

val parse_valid : Uop.t -> (Uop.t * bool * int) option
(** [parse_valid v] parses a validity clause. It returns
    [(expr, is_upper, c)] for [expr < c + 1] when [is_upper] is [true],
    or [expr >= c] when [is_upper] is [false]. *)

val uop_given_valid : ?try_simplex:bool -> Uop.t -> Uop.t -> Uop.t
(** [uop_given_valid valid u] simplifies [u] under the assumption that
    every AND-clause in [valid] is true. *)

val sym : Upat.Pattern_matcher.t
(** [sym] extends {!symbolic} with higher-level cleanups used once the
    IR is closer to its final shape:

    - self-store elimination ([store(i, load(i)) -> noop]);
    - invalid-index load/store folding, using a load's [alt] branch when
      present;
    - store-of-gated-load folding
      ([store(i, gate.where(alt, load(i))) -> store(gated_i, alt)]);
    - reciprocal-of-square splitting ([(x * x).recip -> x.recip * x.recip]);
    - collapsing a single-source {!Ops.Group} into its child;
    - flattening nested {!Ops.Sink} and {!Ops.Group} nodes. *)

(** {1:simplify Driver} *)

val simplify : Uop.t -> Uop.t
(** [simplify u] rewrites [u] by applying {!symbolic} with
    {!Uop.graph_rewrite} until a fixed point is reached, i.e. until a full
    pass leaves the graph structurally unchanged. Terminates because every
    rewrite reduces a well-founded measure on the IR.

    It runs {!symbolic}, not {!sym}: the higher-level [sym] cleanups
    (notably validity-predicate simplification) call back into
    {!Uop.simplify}, so exercising them here would make simplification
    mutually recursive.

    This function is installed into {!Uop.simplify_ref} at module
    initialisation, so {!Uop.simplify} delegates here. *)
