(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Movements.

    A movement maps the indices of a result to the indices of its argument: the
    result's element at an index is the argument's element at the mapped index.
    Movements are plain data, so code that records or lowers them shares one
    vocabulary, and equal movements, by structural equality, have equal maps. A
    movement holds the arrays it is given: code that keeps a movement keeps its
    own copy. *)

type range = { start : int; count : int; step : int }
(** The type for the elements [start + j·step], [j < count], of an axis. *)

type window = { axis : int; size : int; step : int; dilation : int }
(** The type for windows along [axis]: window [w]'s element [j] is the axis's
    element [w·step + j·dilation], [j < size]. *)

(** The type for movements. *)
type t =
  | Reshape of int array
      (** [Reshape s'] has shape [s'] and the argument's elements in the same C
          order of indices: element [k] in C order is the argument's element [k]
          in C order. *)
  | Broadcast of int array
      (** [Broadcast s'] has shape [s'], which has at least the argument's rank.
          Aligned from the right, each extent of the argument is [1] or [s']'s;
          a new axis or an extent-1 one repeats the argument's elements along
          it. *)
  | Permute of int array
      (** [Permute p] has the argument's axis [p.(i)] as its axis [i]. *)
  | Slice of range array
      (** [Slice rs] keeps the elements [rs.(i)] of each axis [i]. A negative
          step reverses the axis. *)
  | Window of window array
      (** [Window ws] has, for each window of [ws], on strictly increasing axes,
          its axis replaced by the number of windows,
          [(d - 1 - dilation·(size - 1)) / step + 1] of an axis of extent [d],
          and a trailing axis of extent [size] appended, in axis order. The
          result's element at [(…, w, …, j)] is the argument's element at
          [(…, w·step + j·dilation, …)]. *)

val shape : t -> int array -> int array
(** [shape m s] is the shape of the result of [m] on an argument of shape [s].

    Raises [Invalid_argument] if an extent of [s] is negative, if [s] or the
    result has extents other than [0] whose product exceeds
    {!Nx_array.Layout.max_numel}, if the result has more than
    {!Nx_array.Layout.max_rank} axes, or unless:
    - [Reshape s']: [s']'s extents are non-negative, their product [s]'s.
    - [Broadcast s']: [s']'s extents are non-negative; it has at least [s]'s
      rank, and each extent of [s] aligned from the right is [1] or [s']'s.
    - [Permute p]: [p] is a permutation of [s]'s axes.
    - [Slice rs]: [rs] has a range per axis, each with [step <> 0] and
      [count >= 0]; if [count > 0], [start] and [start + (count - 1)·step] lie
      in [\[0, d)] for an axis of extent [d].
    - [Window ws]: each window's axis is an axis of [s], after the previous
      window's; [size], [step] and [dilation] are at least [1] and
      [dilation·(size - 1) <= d - 1] for an axis of extent [d]. *)
