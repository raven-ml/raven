(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Range and reduction simplification.

    Rewrites of a kernel's loops ({!Op.Range}) and of the reductions and ends
    that close them: listing the ranges a reduction or end closes, merging and
    splitting ranges when that simplifies their indices, shrinking ranges to the
    bound every access guards them with, and computing reductions whose value
    has a closed form. *)

(** {1:ranges Ranges} *)

val pm_flatten_range : (unit, Ops.t) Ops.Pattern_matcher.t
(** [pm_flatten_range] replaces the sources of an {!Op.Reduce} or {!Op.End}
    after its value by the ranges they are or run inside, in order, each once.
*)

val pm_simplify_ranges : (Ops.t Ops.Tbl.t, Ops.t) Ops.Pattern_matcher.t
(** [pm_simplify_ranges] simplifies a kernel's ranges, with a context that
    starts empty:
    - at a {!Op.Sink}, two ranges [r0] and [r1] of one axis type, neither of
      whose sizes reads the other, merge into one range [n] of the product of
      their sizes, [r0] becoming [n / size r1] and [r1] [n mod size r1], when
      that quotient and remainder fold away, when every node that closes
      either may merge them, and when every node that is not pure runs inside
      both or neither. The pure nodes are {!Op.Set.elementwise},
      {!Op.Set.irreducible}, {!Op.Index}, {!Op.Load} and {!Op.Stack}. A
      reduction ({!Op.Reduce}) whose value reads only pure nodes and
      reductions merges any two of its ranges; an {!Op.End}, and any other
      reduction, run their ranges as loops nested by axis type, then
      identity, so they merge only ranges next to each other in that order;
      no other node merges the ranges it closes. One pair merges per
      rewrite;
    - a range that every index guards with [r < c] for constants [c], that
      no effect runs inside unguarded by its own index, and that no reduction
      closes, is shrunk to the least of its size and the greatest such [c] when
      the kernel's {!Op.Sink} is reached.

    Only a sink carrying kernel information is shrunk. The context keeps the
    guards recorded until the rewrite ends. *)

val pm_split_ranges : (Ops.t Ops.Tbl.t, Ops.t) Ops.Pattern_matcher.t
(** [pm_split_ranges] splits a range of constant size [n] whose remainder by a
    constant [c] dividing [n] is taken, and that is neither a warp nor a device
    axis, into an outer range of size [n / c] and an inner one of size [c], with
    a context that starts empty. The split is applied, and the result
    simplified, when the kernel's {!Op.Sink} is reached; only a sink carrying
    kernel information is rewritten. *)

(** {1:effects Effects} *)

val pin_effects : Ops.t -> Ops.t
(** [pin_effects sink] is the kernel [sink] with each effect bound to the
    ranges it runs inside. An effect runs once per step of the ranges
    ({!Ops.val-ranges}) it depends on, and a rewrite that keeps every value
    may drop a range from what it reads: [r / 8] over a range [r] of size [8]
    is [0] at every step. An effect's index is a {!Op.Store}'s destination, or
    the first {!Op.Index} a {!Op.Call} takes. Each
    {!Op.Index} of storage that any {!Op.Index} an effect takes reads
    ({!Ops.buf_uop}) reads it through an
    {!Op.After} of the ranges its offsets depend on, and an effect that
    depends on ranges its index does not reads its index's storage through
    an {!Op.After} of those too. An effect without an index is left as it
    is, as are calls' bodies. *)

val pm_unpin_effects : (unit, Ops.t) Ops.Pattern_matcher.t
(** [pm_unpin_effects] undoes {!pin_effects} where it is redundant: an
    {!Op.Index}, or an effect of the index, whose storage is read through an
    {!Op.After} of ranges alone reads it directly where it depends on those
    ranges without it. Run before vector axes become lanes, once opts and
    range shrinking are done, it leaves the {!Op.After} only where a
    rewrite removed a range. *)

(** {1:reductions Reductions} *)

val pm_reduce_unparented : (unit, Ops.t) Ops.Pattern_matcher.t
(** [pm_reduce_unparented] removes from a sum, product or maximum ({!Op.Reduce})
    the ranges its value does not run inside: a sum over such ranges is
    multiplied by the product of their sizes, and a product raised to it, where
    that product is positive; where it is not, the reduction is its operation's
    identity.

    Raises [Invalid_argument] if a source of the reduction after its value is
    not a range. *)

val pm_reduce_collapse : (unit, Ops.t) Ops.Pattern_matcher.t
(** [pm_reduce_collapse] is the rewrites that compute a sum over a range in
    closed form: {!pm_reduce_unparented}; integer comparisons [x + y < c] and
    [x * y < c] solved for [x] when [y] and [c] do not depend on a range and no
    value wraps ({!Ops.exact}); the sum of a value over the part of a range
    where it is selected by bounds on the range, as the size of that part times
    the value, and [+0.] for a float over an empty part; sums distributed over
    additions; an integer product by a boolean cast as a selection; a sum over a
    selection whose condition has a conjunct free of the range, as that conjunct
    selecting the sum over the rest, [0] where it is false; and
    {!Shape.val-symbolic}. A float sum so computed is its terms added in another
    order, which rounds differently.

    Raises [Invalid_argument] on a sum over a range of size [1]: the range folds
    to [0], which {!pm_reduce_unparented} refuses. *)

val pm_reduce_simplify : (unit, Ops.t) Ops.Pattern_matcher.t
(** [pm_reduce_simplify] is {!pm_reduce_unparented}, and replaces a sum over
    ranges ({!Op.Reduce} of {!Op.Add}) that {!pm_reduce_collapse} computes
    without any range by its closed form. Each range is collapsed in turn, with
    the values its part of the graph reads from outside standing for variables
    bounded as those values are; a sum whose part holds an effect
    ({!Op.Set.effectful}) or another reduction is left as it is.

    Raises [Invalid_argument] as {!pm_reduce_collapse} does. *)

val load_collapse : Ops.t -> Ops.t option
(** [load_collapse red] is the sum [red] over one range ({!Op.Reduce} of
    {!Op.Add}) computed without the range, by {!pm_reduce_collapse}'s closed
    forms and by taking a value selected where an integer index that does not
    depend on the range equals the range, cast to a type that holds its values
    or not, at that index, as when indexing a tensor with another one: the value
    there where the index is within the range, and [0] elsewhere. It is [None]
    when no such form removes the range, and when [red] is no sum or sums over
    several ranges.

    Raises [Invalid_argument] as {!pm_reduce_collapse} does. *)

val pm_load_collapse : (unit, Ops.t) Ops.Pattern_matcher.t
(** [pm_load_collapse] replaces a sum by {!load_collapse}'s, where it is one;
    and solves [x + y < c] for [x] when [x] is a {!Dtype.Weak_int} that reads
    memory and [y] and [c] do not. *)
