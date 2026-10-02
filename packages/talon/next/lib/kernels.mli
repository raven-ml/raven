(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Kernels of casts, text and temporal operations.

    Each function analyses its types when applied to them; the function it
    returns maps columns of those types, of the frame's rows or of one row that
    nx broadcasts. A kernel that can fail returns the failure at the first row
    where it fails, with its column, which is computed as if that row had not
    failed and is null where it does. *)

type checked = Column.t * (int * Error.t) option
(** The type for a column and the failure, if any: the row and the error. *)

val cast : Type.any -> Type.any -> Column.t -> checked
(** [cast from ty c] is [Talon_next.Expr.cast ty] of [c], a column of [from],
    which binding lets [cast] convert to [ty]. A value that [ty] cannot hold
    fails, as [cannot cast 3.5 to int32].

    Raises [Invalid_argument] if binding does not let [cast] convert [from] to
    [ty]. *)

val text : 'a Expr.text_op -> Column.t -> checked
(** [text op c] is [op] of the [string] column [c]. A text that [Str.parse] does
    not read fails with the text. *)

val add : Type.any -> Type.any -> Column.t -> Column.t -> checked
(** [add ta td a d] is [Temporal.add a d], [a] of [ta] and [d] of [td]. A span
    that is not whole days for a date, and a result out of range, fail. *)

val diff : Type.any -> Column.t -> Column.t -> checked
(** [diff t a b] is [Temporal.diff a b], [a] and [b] of [t]. A result out of
    range fails. *)

val field : Expr.Temporal.field -> Type.any -> Column.t -> Column.t
(** [field f t c] is the field [f] of [c], a [date], a clock or a datetime
    without a zone, of [t]. *)

val floor : Time.step -> Type.any -> Column.t -> checked
(** [floor step t c] is [Temporal.floor step c], [c] a [date] or a datetime
    without a zone, of [t]. A period that starts out of range fails. *)

val offset : Time.step -> Type.any -> Column.t -> checked
(** [offset step t c] is [Temporal.offset step c], [c] a [date] or a datetime
    without a zone, of [t]. A result out of range fails. *)

val parse_with : string -> Type.any -> Column.t -> checked
(** [parse_with fmt ty c] is [Temporal.parse fmt ty c] of the [string] column
    [c]. A text that does not read fails with the text. *)
