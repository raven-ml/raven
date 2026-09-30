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
    - two ranges that an {!Op.End} closes one after the other, or any two that
      an {!Op.Reduce} closes, of one axis type and in the same reductions, are
      merged into one of the product of their sizes, when that leaves no more
      floor divisions and remainders than before;
    - a range that every index guards with [r < c] for constants [c], and no
      reduction closes, is shrunk to the greatest such [c] when the kernel's
      {!Op.Sink} is reached. Only a sink carrying kernel information is
      rewritten.

    The guards must be simplified first ({!Symbolic.symbolic}): a guard [r < c]
    with [c] at least the range's size sets the size to [c]. *)

val pm_split_ranges : (Ops.t Ops.Tbl.t, Ops.t) Ops.Pattern_matcher.t
(** [pm_split_ranges] splits a range of constant size [n] whose remainder by a
    constant [c] dividing [n] is taken, and that is neither a warp nor a device
    axis, into an outer range of size [n / c] and an inner one of size [c], with
    a context that starts empty. The split is applied, and the result
    simplified, when the kernel's {!Op.Sink} is reached; only a sink carrying
    kernel information is rewritten. *)

(** {1:reductions Reductions} *)

val pm_reduce_unparented : (unit, Ops.t) Ops.Pattern_matcher.t
(** [pm_reduce_unparented] removes from a sum, product or maximum ({!Op.Reduce})
    the ranges its value does not run inside: a sum over such a range is
    multiplied by its size, and a product raised to it.

    Raises [Invalid_argument] if a source of the reduction after its value is
    not a range. *)

val pm_reduce_collapse : (unit, Ops.t) Ops.Pattern_matcher.t
(** [pm_reduce_collapse] is the rewrites that compute a sum over a range in
    closed form: {!pm_reduce_unparented}; comparisons [x + y < c] and
    [x * y < c] solved for [x] when [y] and [c] do not depend on a range; the
    sum of a value over the part of a range where it is selected by bounds on
    the range, as the size of that part times the value; sums distributed over
    additions; and {!Symbolic.symbolic}.

    Raises [Invalid_argument] on a sum over a range of size [1]: the range folds
    to [0], which {!pm_reduce_unparented} refuses. *)

val pm_reduce_simplify : (unit, Ops.t) Ops.Pattern_matcher.t
(** [pm_reduce_simplify] is {!pm_reduce_unparented}, and replaces a sum over
    ranges ({!Op.Reduce} of {!Op.Add}) that {!pm_reduce_collapse} computes
    without any range by its closed form. Each range is collapsed in turn, with
    the values its part of the graph reads from outside standing for variables
    bounded as those values are; a sum whose part holds a store or another
    reduction is left as it is.

    Raises [Invalid_argument] as {!pm_reduce_collapse} does. *)

val pm_load_collapse : (unit, Ops.t) Ops.Pattern_matcher.t
(** [pm_load_collapse] replaces a sum over one range of a value selected where
    an index equals the range, as when indexing a tensor with another one, by
    the value at that index where it is within the range, and [0] elsewhere; and
    solves [x + y < c] for [x] when [x] is a {!Dtype.Weak_int} that reads memory
    and [y] and [c] do not. *)
