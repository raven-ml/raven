(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Layouts.

    A layout maps an index [(i{_0}, …, i{_k-1})], [0 <= i{_j} < d{_j}], to the
    element position [offset + Σ i{_j}·s{_j}], counted in elements from a
    buffer's first byte: [d{_j}] are its extents, its {e shape}, and [s{_j}] its
    strides. Positions are non-negative. An element at position [p] of a dtype
    of [b] bits occupies bits [p·b] to [p·b + b - 1] of the buffer, where bit
    [8q + r] is bit [r] of byte [q], counted from the least significant: element
    [p] of a 4-bit dtype is the low half of byte [p / 2] for an even [p].

    A layout is immutable and in one {e canonical form}: an axis of extent 1 has
    stride 0, and a layout with no element has offset 0 and every stride 0. Two
    layouts of one shape that map every index to the same position are then
    {!equal}.

    No OCaml array a function takes is kept, and every array a function returns
    is fresh. *)

type t
(** The type for layouts. *)

val max_rank : int
(** [max_rank] is [32], the most axes a layout has. *)

val contiguous : int array -> t
(** [contiguous s] is the layout of shape [s] in C order at offset 0: element
    [k] in C order is at position [k].

    Raises [Invalid_argument] if [s] has more than {!max_rank} axes, a negative
    extent, or a number of elements that does not fit in an [int]. *)

val v : ?offset:int -> strides:int array -> int array -> t
(** [v ~offset ~strides s] is the layout of shape [s] with [strides] and
    [offset] (defaults to [0]), in canonical form.

    Raises [Invalid_argument] as {!contiguous} does, if [strides] does not have
    [s]'s length, if [(d - 1)·|t|] does not fit in an [int] for an axis of
    extent [d] and stride [t], if a position is negative or does not fit in an
    [int], or if the end of its span ({!span}) does not fit in an [int]. *)

(** {1:queries Queries}

    {!rank}, {!dim}, {!stride}, {!offset}, {!numel}, {!is_contiguous} and
    {!is_distinct} allocate nothing. *)

val rank : t -> int
(** [rank l] is [l]'s number of axes. *)

val dim : t -> int -> int
(** [dim l i] is the extent of [l]'s axis [i].

    Raises [Invalid_argument] unless [0 <= i < rank l]. *)

val stride : t -> int -> int
(** [stride l i] is the stride of [l]'s axis [i].

    Raises [Invalid_argument] unless [0 <= i < rank l]. *)

val offset : t -> int
(** [offset l] is the position of [l]'s index [(0, …, 0)]. *)

val numel : t -> int
(** [numel l] is [l]'s number of indices, the product of its extents. *)

val shape : t -> int array
(** [shape l] is [l]'s extents. *)

val strides : t -> int array
(** [strides l] is [l]'s strides. *)

val span : t -> int * int
(** [span l] is [(lo, hi)]: every position [l] reaches lies in [\[lo, hi)], with
    [0 <= lo], and [(0, 0)] if [l] has no element. *)

val is_contiguous : t -> bool
(** [is_contiguous l] is [true] iff element [k] of [l] in C order is at position
    [offset l + k]. *)

val is_distinct : t -> bool
(** [is_distinct l] is [true] iff [l] has no element or, with [l]'s axes of
    extent above 1 ordered by [|stride|], ties by axis, each stride exceeds the
    reach [Σ (d{_j} - 1)·|s{_j}|] of the axes before it. Then no two indices of
    [l] reach one position. It is [true] of every layout {!contiguous} reaches
    by [Permute], [Slice] and windows whose step is at least their extent
    [dilation·(size - 1) + 1], and [false] of every broadcast and overlapping
    window. *)

(** {1:moving Moving} *)

val move : Move.t -> t -> t option
(** [move m l] is the layout of [m]'s result over [l]'s positions: its index [i]
    reaches the position [l] reaches at [m]'s map of [i]. It is [None] iff [m]
    is a [Reshape] that no strides express.

    Raises [Invalid_argument] as {!Move.shape} does on [shape l]. *)

val coalesce : t array -> t array
(** [coalesce ls] is layouts of one shape that reach the positions [ls] reach,
    in the same C order of indices, with their axes of extent 1 dropped and
    adjacent axes merged where every layout lays them out as one run. Each has
    at least one axis.

    Raises [Invalid_argument] unless [ls] has 1 to 4 layouts, the most operands
    a kernel of this library takes, all of one shape. *)

(** {1:eq Equality} *)

val equal : t -> t -> bool
(** [equal l l'] is [true] iff [l] and [l'] have the same shape, strides and
    offset: by the canonical form, iff they have one shape and map every index
    to the same position. It allocates nothing. *)

val hash : t -> int
(** [hash l] is a hash of [l] such that [equal l l'] implies [hash l = hash l'].
    It allocates nothing. *)

val pp : Format.formatter -> t -> unit
(** [pp] formats a layout's shape, strides and offset. *)
