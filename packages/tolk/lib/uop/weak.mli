(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Committing weak dtypes to concrete widths.

    A weak dtype ({!Dtype.weakint}, {!Dtype.weakfloat}) is a mathematical value
    with no committed bit width. Literals, loop ranges, hardware indices, shape
    expressions, and symbolic variables all start weak, so index arithmetic
    never forces a width before one is known. Nothing weak may reach a
    renderer: these rules resolve every weak node to a concrete dtype.

    Three demands commit a weak value, in decreasing priority:

    - a {b peer}: a broadcastable node with both weak and concrete sources
      commits its weak sources at the promotion of all of them, and a store
      commits its value at the destination's dtype ({!pm_commit_weak});
    - a {b consumer's cast}: a concrete cast over a weak ALU node states the
      width the value will live at. That width is a floor, never a narrowing
      ({!pm_cast_weak});
    - {b nothing}: a value nobody constrains takes its default width —
      {!Dtype.default_float} for {!Dtype.weakfloat}, otherwise [int32], or
      [int64] when the node's value range overflows [int32].

    Each committed node is re-wrapped in a cast back to its weak dtype, so a
    consumer that has not yet been visited still sees a weak edge; the cast is
    absorbed when that consumer commits in turn. *)

val pm_commit_weak : Upat.Pattern_matcher.t
(** [pm_commit_weak] commits weak sources under demand from a peer: a
    broadcastable node whose sources mix weak and concrete dtypes rebuilds its
    weak sources at the promotion of all of them, and a {!Ops.Store} rebuilds a
    weak value at the destination's dtype. A weak constant is rebuilt in place;
    any other weak node takes a cast.

    It runs both in {!pm_lower_index_dtype} and in the dtype decompositions, so
    that a rule which mints a weak constant commits it within the same
    rewrite. *)

val pm_cast_weak : Upat.Pattern_matcher.t
(** [pm_cast_weak] commits a weak ALU node under demand from a concrete cast
    over it. The committed width is the promotion of the cast's dtype with the
    node's own default width, so a cast can widen the computation but never
    narrows it below what the node's value range requires. *)

val pm_lower_index_dtype : unit -> Upat.Pattern_matcher.t
(** [pm_lower_index_dtype ()] is the full weak-lowering pass: {!pm_commit_weak}
    and {!pm_cast_weak}, then a catch-all that resolves any remaining weak
    source of a concrete node at its default width, then a narrowing rule — a
    gated [int64] index into a buffer whose element count fits [int32] narrows
    to [int32], since out-of-gate values are discarded by the gate.

    Each call allocates its own memo table for the catch-all, which lowers each
    weak subgraph once per pass rather than once per consumer. Call it per
    rewrite, not once at module initialisation. *)
