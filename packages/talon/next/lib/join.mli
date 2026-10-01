(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Joins: conditions, kinds and counts.

    [Talon_next.Join] documents the conditions that users write. This interface
    adds their representation, their check against the two inputs' schemas, the
    columns of a join, and their equality and formatting. *)

(** {1:conditions Conditions} *)

type order = [ `Lt | `Le | `Gt | `Ge ]
(** The type for the orders of inequality atoms: [`Lt] is "the left column is
    less than the right column", and so on. *)

(** The type for atoms. [within] is unbound in a built condition, and bound by
    {!check}. *)
type atom =
  | Eq of string * string  (** [Eq (l, r)]: [l] and [r] are the same key. *)
  | Compare of order * string * string
      (** [Compare (o, l, r)]: [l] is ordered [o] against [r]. *)
  | Closest of {
      order : order;
      left : string;
      right : string;
      within : Expr.packed option;
    }  (** See {!closest}. *)
  | Nearest of { left : string; right : string; within : Expr.packed option }
      (** See {!nearest}. *)
  | Position  (** See {!position}. *)

type cond = private atom list
(** The type for conditions: the conjunction of the atoms, in the order they are
    written. The empty conjunction is {!all}. A condition is one that an
    algorithm runs: it is [[Position]], or it holds at most one {!Closest} or
    {!Nearest} atom and then no {!Compare} atom, and its {!Compare} atoms have
    one right column. *)

val keys : string list -> cond
val eq : string -> string -> cond
val lt : string -> string -> cond
val le : string -> string -> cond
val gt : string -> string -> cond
val ge : string -> string -> cond
val closest : ?within:('a, Expr.row) Expr.t -> cond -> cond
val nearest : ?within:('a, Expr.row) Expr.t -> string -> string -> cond
val position : cond
val all : cond
val ( && ) : cond -> cond -> cond

(** {1:kinds Kinds and counts} *)

type kind = Inner | Left | Full | Semi | Anti
type count = Any | At_most_one | One | At_least_one

(** {1:checking Checking} *)

val check :
  kind -> Schema.t -> Schema.t -> cond -> (cond, Problem.t list) result
(** [check kind left right c] is [Ok b] with [b] the condition [c] with each
    [within] bound, or [Error ps] with every problem of the [kind] join on [c]
    of rows of the columns [left] and [right], those of [c] in the order of its
    atoms, then the names on both sides. The problems are:
    - a column missing on its side, with suggestions from that side's names;
    - the columns of an equality atom that do not meet, and extension columns
      that are not of one type;
    - the columns of another atom that do not meet or do not order;
    - a [within] or a {!nearest} over columns that have no difference, and a
      [within] that is not a literal, is negative, or that the difference's type
      does not hold;
    - in a {!Full} join, a left column that is the left of two equality atoms,
      which makes its coalesced value ambiguous;
    - except for {!Semi} and {!Anti}, the joined columns that have the same
      name, all listed in one problem. *)

val columns : kind -> Schema.t -> Schema.t -> cond -> Schema.t
(** [columns kind left right c] is the columns of the [kind] join on [c] of rows
    of the columns [left] and [right], as [Talon_next.Join] describes them. [c]
    has passed {!check} against [left] and [right], or against schemas of which
    they hold the columns that [c] names, with the same types. [Query.make]
    computes with it the columns of a join, whose inputs the optimizer may
    narrow. *)

val common_type : Type.any -> Type.any -> Type.any option
(** [common_type l r] is the type at which a left column of type [l] meets a
    right column of type [r] in an equality atom: their common type
    ({!Type.common}), or their one type if either holds an extension type. A
    {!Full} join's key has it. *)

val equal : cond -> cond -> bool
(** [equal c0 c1] is [true] iff [c0] and [c1] have equal atoms in the same
    order, [within] expressions compared by their identity ({!Expr.same}). *)

val pp : Format.formatter -> cond -> unit
(** [pp ppf c] formats [c] as it is written inside [Join.( … )], its atoms
    joined by [&&]: each run of [Eq (n, n)] atoms as one [keys ["a"; "b"]],
    other atoms as [eq "l" "r"], [ge "ts" "quote_ts"],
    [closest ~within:5s (ge "ts" "quote_ts")], [nearest "x" "y"] and [position],
    and the condition with no atom as [all]. *)
