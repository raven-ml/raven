(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Committing weak dtypes to concrete widths.

    A weak dtype ({!Dtype.weakint}, {!Dtype.weakfloat}) is a mathematical value
    with no committed bit width. Literals, loop ranges, hardware indices, shape
    expressions, and symbolic variables all start weak, so index arithmetic
    never forces a width before one is known. Weak computations resolve before rendering. A literal retains a weak
    CONST payload beneath a CAST that states its concrete width.

    Three demands commit a weak value, in decreasing priority:

    - a {b peer}: a broadcastable node with both weak and concrete sources
      commits its weak sources at the promotion of all of them, and a store
      commits its value at the destination's dtype ({!pm_commit_weak});
    - a {b consumer's cast}: a concrete cast over a weak ALU node states the
      width the value will live at. That width is a floor, never a narrowing
      ({!pm_cast_weak});
    - {b nothing}: a value nobody constrains takes its default width —
      {!Dtype.default_float} for {!Dtype.weakfloat}; integers select the first
      of [int32], [int64], [uint64] containing their exact bounds. An exact
      scalar outside these types raises [Invalid_argument]. An unresolved
      interval that fits none of them defaults to [int64].

    Each committed node is re-wrapped in a cast back to its weak dtype, so a
    consumer that has not yet been visited still sees a weak edge; the cast is
    absorbed when that consumer commits in turn. *)

val pm_commit_weak : Upat.Pattern_matcher.t
(** [pm_commit_weak] commits weak sources under demand from a peer: a
    broadcastable node whose sources mix weak and concrete dtypes rebuilds its
    weak sources at the promotion of all of them, and a {!Ops.Store} rebuilds a
    weak value at the destination's dtype. Derivable literal edges retain bare
    weak payloads; other edges receive a cast that states their width.

    It runs before {!pm_lower_index_dtype} and in the dtype decompositions, so
    that a rule which mints a weak constant commits it within the same
    rewrite. *)

val pm_cast_weak : Upat.Pattern_matcher.t
(** [pm_cast_weak] commits a weak ALU node under demand from a concrete cast
    over it. The committed width is the promotion of the cast's dtype with the
    node's and each weak operand's default widths. A cast can widen the
    computation but never narrows it below the ranges its operands require. *)

val pm_lower_index_dtype : unit -> Upat.Pattern_matcher.t
(** [pm_lower_index_dtype ()] lowers weak expressions once peer and cast demands
    have reached a fixed point. Derivable literal edges stay weak; unresolved
    literals take their default width. Run separately from symbolic folding. *)

val pm_uncast_const : Upat.Pattern_matcher.t
(** [pm_uncast_const] removes a literal's committed cast only when both operand
    promotion and result dtype remain unchanged and the literal keeps its value
    through the cast. *)

val pm_cast_const : Upat.Pattern_matcher.t
(** [pm_cast_const] states every remaining literal's concrete width at its
    consumer edge before rendering, including bool literals. Invalid stays bare. *)
