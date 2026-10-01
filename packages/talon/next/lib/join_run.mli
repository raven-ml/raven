(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Running joins.

    The programs of [Query.join] steps over batches; {!Run} streams them. A join
    pairs left row indices with right row indices, [-1] where a side has no row,
    then takes every column at them ({!Column.take} makes the nulls).

    {b Assertions.} A join checks [each_left] on its left rows in order, then
    [each_right] on its right rows, and fails at the first row whose number of
    matches the count does not allow. The reason names the side, the row's keys
    and the number: [the left row whose "k" is 3 matches 2 rows, not one]. The
    failure's row is a row of that side's input. *)

(** The type for compiled joins. *)
type t =
  | Blocking of (Table.t -> Table.t -> (Table.t, Eval.failure) result)
      (** [Blocking f]: [f l r] is the join of all the left rows [l] and all the
          right rows [r]. Equality and {!Join.position} joins. *)
  | Streaming of {
      batch : Table.t -> Table.t -> (Table.t, Eval.failure) result;
          (** [batch r b] is the join of the left batch [b] with all the right
              rows [r]. *)
      last : Table.t -> Table.t -> int -> (Table.t, Eval.failure) result;
          (** [last r e n] is the rows that follow the left's [n] rows, [e]
              being the left with no row: [r]'s rows in a {!Join.Full} join when
              [n] is [0]. *)
    }  (** A join on {!Join.all}, which streams its left. *)

val compile : Query.t -> t option
(** [compile q] compiles the join step [q], or is [None] if no unit lowers its
    condition yet. Raises as {!Eval.widen} does when the columns of an equality
    atom meet through a conversion not lowered yet. *)
