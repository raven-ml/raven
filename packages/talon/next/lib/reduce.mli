(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Reductions and frame operations.

    The kernels behind [aggregate], [Expr.over], [shift], [rank] and the
    reductions. They compute over the rows of one batch cut into {e segments},
    the frames that [Talon_next.Expr] describes: a row belongs to one segment,
    and each segment orders its rows. A reduction gives one value per segment;
    [shift] and [rank] one per row. *)

type segments
(** The type for rows cut into segments, each ordered. *)

val one : int -> segments
(** [one n] is [n] rows in one segment, in row order. Over no row it is one
    segment without rows. *)

val group : Column.t list -> segments
(** [group cs] cuts the rows of [cs] into segments of equal keys
    ({!Key.identity}), numbered in order of first appearance, each in row order.

    Raises [Invalid_argument] if [cs] is empty. *)

val count : segments -> int
(** [count s] is the number of segments of [s]. *)

val first : segments -> Nx.int64_t
(** [first s] is the first row of each segment of [s], ascending: [0] for a
    segment without rows. *)

val refine :
  segments -> by:Column.t list -> order:(Column.t * Order.t) list -> segments
(** [refine s ~by ~order] cuts each segment of [s] by the keys [by] and orders
    each by [order] ({!Key.order}), ties keeping their order in [s]. *)

val broadcast : segments -> Column.t -> Column.t
(** [broadcast s c] is the value of [c], one per segment of [s], on each row of
    its segment. *)

val rows : segments -> Column.t
(** [rows s] is the number of rows of each segment of [s], as [int64]. *)

val reduce :
  ('a, 'b) Expr.reduction ->
  Type.any ->
  segments ->
  Column.t ->
  Column.t * (int * string) option
(** [reduce r ty s c] is [r] of the values of [c] in each segment of [s], of
    type [ty], with the first segment where [r] fails and the reason: [only]
    over two values, a [duration] sum that overflows. [sum] takes integers at
    [int64], and floats and durations at [ty]. Over no values [count], [sum] and
    [n_unique] are [0], and the others null. *)

val shift : segments -> int -> Column.t -> Column.t
(** [shift s n c] is, on each row, [c] at the row [n] places earlier in its
    segment's order, or null past the segment's edges. *)

val rank : segments -> Column.t -> Column.t
(** [rank s c] is, on each row, [1] plus the number of non-null values of its
    segment that order before its value, as [int64]; null where [c] is. *)
