(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Committing weak data types to widths.

    A node of a weak type ({!Dtype.Weak_int}, {!Dtype.Weak_float}) holds a value
    whose width is not decided yet: literals, loop variables and index
    arithmetic start weak and take the width of what they meet. Before a kernel
    is rendered, every weak node gets a width, decided in this order:

    - by a {b peer}: an operation whose sources mix weak and committed types
      commits the weak ones at their least upper type, and a store commits its
      value at its destination's type ({!pm_commit_weak});
    - by a {b cast}: a cast to a committed type over a weak expression of the
      same kind states the width the expression computes at, widened to what its
      bounds and its sources' bounds need, never narrowed ({!pm_commit_weak});
    - by {b default}, for the rest ({!pm_lower_weak}): a weak float takes
      {!Dtype.default_float}, and a weak integer the first of {!Dtype.Int32},
      {!Dtype.Int64} and {!Dtype.Uint64} that holds its bounds
      ({!Ops.commit_dtype}).

    A weak constant whose consumer derives its type from its other sources stays
    a bare literal, so rules keyed on constant values keep matching it;
    {!pm_cast_const} states the width of every remaining literal last. *)

val commit_weak_consts : Ops.t -> Dtype.t option -> Ops.t
(** [commit_weak_consts u dt] is [u] with each of its sources that is a weak
    constant converted to [dt] ({!Ops.ccast}), and [u] if [dt] is [None]. A
    rewrite that builds a node from weak literals next to an operand of type
    [dt] states their width with it. *)

val pm_commit_weak : (unit, Ops.t) Ops.Pattern_matcher.t
(** [pm_commit_weak] commits weak sources whose width something states:

    - a broadcastable node ({!Op.Set.broadcastable}) with a weak source, whose
      sources' least upper type [dt] is committed, converts its weak sources to
      [dt]. A weak constant stays a weak literal, its value converted to [dt]
      ({!Dtype.const}), when the node's type does not depend on it: the other
      sources already derive a committed type for the node;
    - a {!Op.Store} of a weak value converts it to the destination's type;
    - a cast to a committed type [dt] over a weak arithmetic node [u] of [dt]'s
      kind commits [u]'s weak sources at the least upper type of [dt], [u]'s
      committed type and each weak source's committed type, and casts the result
      to [dt].

    It runs with every rewrite that can build a weak constant, and must reach
    its fixed point before {!pm_lower_weak} gives defaults. *)

val pm_lower_weak : (unit, Ops.t) Ops.Pattern_matcher.t
(** [pm_lower_weak] gives every weak node that nothing committed its default
    width ({!Ops.commit_dtype}, with {!Dtype.Int32} as the default integer):

    - a node's sources lose the weak casts over them, which only restated a
      width the node restates; a weak integer cast of a boolean or a float is a
      conversion, and commits to its default instead;
    - a weak constant whose consumer derives no committed type from its sources
      takes its default;
    - once no source is a weak expression, a unary, binary, {!Op.Where},
      {!Op.Range}, {!Op.Stack} or {!Op.Special} node computes at one committed
      type: for a binary node, the least upper type of its sources and of its
      own default, and otherwise the type it derives. Its sources are converted
      to it, and the node is cast back to its own type, which its consumers
      absorb in turn;
    - two stacked weak casts over a committed value are two conversions, each at
      the default of its kind;
    - a weak integer scalar parameter or buffer is stored at its default, cast
      back to {!Dtype.Weak_int};
    - an {!Op.Index} or {!Op.Shrink} whose index is an {!Dtype.Int64} guarded by
      a condition, and {!Ops.invalid} elsewhere, indexes through an
      {!Dtype.Int32} when the storage's greatest index fits one: the values
      outside the guard are discarded. *)

val pm_uncast_const : (unit, Ops.t) Ops.Pattern_matcher.t
(** [pm_uncast_const] removes the cast from each committed constant source of a
    broadcastable node, leaving the bare weak literal, when the node's sources
    still derive the same least upper type and the node the same type. Rules
    keyed on constant values then match the literal. *)

val pm_cast_const : (unit, Ops.t) Ops.Pattern_matcher.t
(** [pm_cast_const] states the width of every constant still weak, at each
    consumer: a broadcastable consumer that derives a committed type from its
    sources converts its weak constants to their least upper type, and every
    remaining bare constant becomes a cast of the literal to its committed type
    ({!Ops.cconst}), booleans included. {!Ops.invalid} stays bare. *)
