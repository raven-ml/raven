(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Reductions, scans, arg-reductions and sorts as UOps.

    Each function takes the node of an operation's operand, of the shape and
    dtype nx gives it, and is the node that computes its result as nx documents
    it:

    - {b sums and products} accumulate in the type {!Tolk.Dtype.sum_acc} gives,
      unsigned for the signed integers, and are converted once to the operand's
      dtype: integers wrap, and floats round once. A float sum is [+0.] plus its
      terms in an unspecified association, so a sum that is exactly zero is
      [+0.];
    - {b extremes} follow IEEE 754-2019: NaN propagates, and [-0.] is less than
      [+0.];
    - {b positions} are [int64], as nx's indices are: of the first element equal
      to the extreme, or of the elements in their sorted order. *)

open Tolk

val accumulator : Dtype.t -> Dtype.t
(** [accumulator dt] is the type in which sums and products of [dt] elements
    accumulate: {!Dtype.sum_acc}'s, unsigned for the signed integers. *)

val reduce : Nx_backend.reduce -> axes:int list -> Ops.t -> Ops.t
(** [reduce k ~axes x] is [x] reduced by [k] over [axes], which the result
    drops. A sum over no element is [0], and a product over none is [1]. *)

val scan : Nx_backend.reduce -> axis:int -> Ops.t -> Ops.t
(** [scan k ~axis x] is the inclusive running [k] of [x] along [axis], of [x]'s
    shape. *)

val arg_reduce : Nx_backend.arg_reduce -> axis:int -> Ops.t -> Ops.t
(** [arg_reduce k ~axis x] is the position along [axis], which the result drops,
    of the first element of [x] that is the extreme {!reduce} gives: of the
    first NaN if there is one. [axis] must not be empty. *)

val along : int -> int -> Ops.t -> Ops.t
(** [along rank axis v] is the vector [v] as the axis [axis] of a node of [rank]
    axes, the others of one element. *)

val bits : Ops.t -> Ops.t
(** [bits u] is [u]'s bit patterns as the unsigned integers of its width, a
    boolean's as [uint8]. *)

val of_bits : Dtype.t -> Ops.t -> Ops.t
(** [of_bits dt b] is the bit patterns [b] read back as [dt], {!bits}'s inverse.
*)

val pick : Ops.t -> Ops.t -> Ops.t
(** [pick mask b] is the bits [b] where [mask] holds, summed over the last axis:
    the element [mask] selects, or zero where it selects none. *)

val varies : Ops.t -> int list
(** [varies u] is the axes, in order, along which the elements of [u] may
    differ: through [u]'s movements, those whose position the index into the
    node they move reads. A value that is not a movement varies along every
    axis. *)

(** The rows of an indexed access to [x] along [axis] at positions [p]: the axes
    of [x] along which [p] {!varies}, a sharded one first, then [axis], are
    flattened into rows, and the other axes are accessed whole. *)
type rows = {
  view : Ops.t -> Ops.t;
      (** [view y] is [y], of [x]'s shape, as its rows followed by the whole
          axes. *)
  on_rows : Ops.t -> Ops.t;
      (** [on_rows u] is [u], of [p]'s shape and constant along the whole
          axes, one element per row access: of [p]'s shape along the flattened
          axes, in their order. *)
  at : Ops.t -> Ops.t;
      (** [at i] is the row of [view x] each row access reads, given its
          position [i] along [axis], a [weakint] node of [on_rows p]'s shape. *)
  count : int;  (** The number of rows of [view x]. *)
  laid : Ops.t -> Ops.t;
      (** [laid u] is [u], of [p]'s shape, laid out as an access reads it: its
          flattened axes, then the whole ones. *)
  unlaid : Ops.t -> Ops.t;  (** [unlaid] is [laid]'s inverse permutation. *)
}

val rows : Ops.t -> int -> Ops.t -> rows
(** [rows x axis p] is the rows of an access to [x] along [axis] at [p]. Raises
    [Invalid_argument] if a flattened axis has a symbolic size. *)

val clamped : int -> Ops.t -> Ops.t
(** [clamped n i] is the [int64] positions [i] clamped into \[[0], [n]) as
    [weakint]. *)

val take : Ops.t -> int -> Ops.t -> Ops.t
(** [take x axis p] is the elements of [x] along [axis] at the positions [p],
    each with its bits, of [p]'s shape: [x] and [p] agree on every other axis.
    [p] is [int64], and a position out of range takes [0].

    It reads [x] with one {!Op.Index} by a value with axes, the axes along which
    [p] {!varies} flattened with [axis] into its rows, a sharded one first, and
    [p] clamped into [axis]. Raises [Invalid_argument] if one of those axes has
    a symbolic size. *)

val argsort : descending:bool -> axis:int -> Ops.t -> Ops.t
(** [argsort ~descending ~axis x] is the positions that sort [x] along [axis],
    in nx's sort order, or its exact reverse when [descending]. The sort is
    stable in both directions, [-0.] sorts before [+0.] ascending, and every NaN
    sorts above every number: last ascending, first descending. *)

val sort : descending:bool -> axis:int -> Ops.t -> Ops.t
(** [sort ~descending ~axis x] is the elements of [x] along [axis] in the order
    of [argsort ~descending ~axis x], each with its bits. *)
