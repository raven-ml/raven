(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Windows: the rows around a row.

    [Talon_next.Window] documents the windows that users write. This interface
    adds their representation, equality and formatting. *)

(** The type for windows. *)
type t = private
  | Rows of { before : int; after : int }
      (** Row i's window holds the rows j of its frame with
          [i - before <= j <= i + after]. *)
  | Times of { on : string; before : Time.span; after : Time.span }
      (** Row i's window holds the rows j of its frame with
          [t(i) - before < t(j) <= t(i) + after], where [t] is the column [on].
      *)

val rows : before:int -> after:int -> t
val time : ?after:Time.span -> before:Time.span -> string -> t

(** {1:comparing Comparing and formatting} *)

val equal : t -> t -> bool
(** [equal w0 w1] is [true] iff [w0] and [w1] are the same window. *)

val pp : Format.formatter -> t -> unit
(** [pp ppf w] formats [w] as it is written: [rows ~before:6 ~after:0],
    [time ~before:168h "ts"], with [~after] when it is not zero. A bound of
    [max_int] formats as [max_int], as a window that grows from the frame's
    first row is written: [rows ~before:max_int ~after:0]. Spans format with
    {!Time.Span.pp}. *)
