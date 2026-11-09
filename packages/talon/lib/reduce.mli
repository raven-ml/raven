(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Reductions and frame operations.

    The kernels behind [aggregate], [Expr.over], [shift], [rank] and the
    reductions. They compute over the rows of one batch cut into {e segments},
    the frames that [Talon.Expr] describes: a row belongs to one segment, and
    each segment orders its rows. A reduction gives one value per segment;
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

val of_ids :
  segments:int -> sizes:Nx.host Nx.int64_t -> Nx.host Nx.int64_t -> segments
(** [of_ids ~segments ~sizes ids] is the rows [i] cut into [segments] segments,
    row [i] in segment [ids.{i}] and segment [s] of [sizes.{s}] rows, each in
    row order. [ids] ascend. *)

val count : segments -> int
(** [count s] is the number of segments of [s]. *)

val ids : segments -> Nx.host Nx.int64_t
(** [ids s] is the segment of each row of [s]. *)

val view : segments -> (int64, Nx.int64_elt, Nx.host) Nx_ragged.t
(** [view s] is the rows of each segment of [s], in its order. *)

val first : segments -> Nx.host Nx.int64_t
(** [first s] is the first row of each segment of [s], ascending: for a segment
    without rows, the place where the next segment's rows start. *)

val refine :
  segments -> by:Column.t list -> order:(Column.t * Order.t) list -> segments
(** [refine s ~by ~order] cuts each segment of [s] by the keys [by] and orders
    each by [order] ({!Key.order}), ties keeping their order in [s]. *)

val broadcast : segments -> Column.t -> Column.t
(** [broadcast s c] is the value of [c], one per segment of [s], on each row of
    its segment. *)

(** {1:reductions Reductions}

    A reduction reads its frames through a {!frame}: the segments of {!frame},
    or the windows that [Window] makes. A {e holder} is a row of the reduced
    column, [-1] where a frame has none. *)

(** The type for frames: [sizes ()] is each frame's number of rows, [values
    held] the primitives below over the rows where [held], every row for
    [None], and [place h] the place of each holder in its frame, [-1] for
    none. *)
type frame = {
  sizes : unit -> Nx.host Nx.int64_t;
  values : Nx.host Nx.bool_t option -> values;
  place : Nx.host Nx.int64_t -> Nx.host Nx.int64_t;
}

(** The type for a frame's primitives over the rows where a column holds
    values:
    - [fold op x] folds by [op] each frame's values [x] at those rows, the
      identity of [op] over none;
    - [pick op] is the holder of each frame's first ([Min]) or last ([Max])
      value in frame order;
    - [extreme ~desc words] is the holder of each frame's first least word of
      [words], or greatest with [~desc];
    - [variance p] is each frame's sample variance of [p]'s values, as
      {!moments}; [None] leaves [var] and [std] to [holistic];
    - [holistic r ty p] is [r], a [median], [quantile], [n_unique] or a
      [Moments] state's compaction, or a [var] or [std] without [variance], as
      {!reduce} gives it. *)
and values = {
  fold :
    'v 'e. Nx.combine -> ('v, 'e, Nx.host) Nx.t -> ('v, 'e, Nx.host) Nx.t;
  pick : Nx.combine -> Nx.host Nx.int64_t;
  extreme : desc:bool -> Nx.host Nx.uint64_t -> Nx.host Nx.int64_t;
  variance : (parts -> moments) option;
  holistic :
    'a 'b.
    ('a, 'b) Expr.reduction ->
    Type.any ->
    parts ->
    Column.t * (int * string) option;
}

(** The type for a frame's sample variances. [up] is a power of two for each
    frame, [1] for [None], and [var] the sample variance of each frame's
    values divided by [up], so that the variance is [var] times [up] squared
    and the standard deviation the root of [var] times [up]. [n] is the number
    of the frame's values. *)
and moments = {
  var : Nx.host Nx.float64_t;
  up : Nx.host Nx.float64_t option;
  n : Nx.host Nx.int64_t;
}

and parts
(** The type for a column's operands over a frame that its reductions share:
    the frame's {!values} over its values, their counts, sums, mean and
    variance, its order words and the holders of their extremes, each computed
    once, when a reduction first reads it. *)

val frame : segments -> frame
(** [frame s] is the segments [s] as frames. *)

val overflows : Column.t -> bool
(** [overflows c] is [true] when [c] holds float64 values, the only ones
    whose sums or sums of squares can pass max_float. *)

val tiny : float
(** [tiny] is the sample variance below which a square's underflow can show
    in it, [2{^-969}]. *)

val parts : frame -> Column.t -> parts
(** [parts f c] is the parts of [c] over [f]. *)

val column : parts -> Column.t
(** [column p] is the column of [p]. *)

val reduce :
  ('a, 'b) Expr.reduction ->
  Type.any ->
  parts ->
  Column.t * (int * string) option
(** [reduce r ty (parts f c)] is [r] of the values of [c] in each frame of
    [f], of type [ty], with the first frame where [r] fails and the reason:
    [only] over two values, a [duration] sum that overflows. [sum] takes
    integers at [int64], and floats and durations at [ty]. Over no values
    [count], [sum] and [n_unique] are [0], and the others null. *)

val exact_sum :
  'a Type.t ->
  (Nx.host Nx.int64_t -> Nx.host Nx.int64_t) ->
  Nx.host Nx.int64_t ->
  Column.t * (int * string) option
(** [exact_sum ty sum x] is each frame's sum, by [sum], of the [int64] values
    [x] of the duration type [ty], exact over fewer than 2{^42} values a frame,
    and null where it leaves [int64], with the first such frame and why. [sum]
    adds each frame's values of a column of integers. *)

val fixed :
  Type.any -> ?valid:Nx.host Nx.bool_t -> ('a, 'b, Nx.host) Nx.t -> Column.t
(** [fixed ty ~valid x] is the column of [ty] stored as [x], null where [valid]
    is [false]. *)

val rows_at : Nx.host Nx.int64_t -> Column.t -> Column.t
(** [rows_at rows c] is [c]'s rows at [rows], null where a row is [-1]. *)

val shift : segments -> int -> Column.t -> Column.t
(** [shift s n c] is, on each row, [c] at the row [n] places earlier in its
    segment's order, or null past the segment's edges. *)

val rank : segments -> Column.t -> Column.t
(** [rank s c] is, on each row, [1] plus the number of non-null values of its
    segment that order before its value, as [int64]; null where [c] is. *)
