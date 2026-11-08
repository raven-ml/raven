(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Joins: conditions, kinds and counts.

    [Talon.Join] documents the conditions that users write. This interface adds
    their representation, their check against the two inputs' schemas, the
    columns of a join, and their equality and formatting. *)

(** {1:conditions Conditions} *)

type compare = [ `Lt | `Le | `Gt | `Ge ]
(** The type for the comparisons of atoms. *)

(** The type for atoms. *)
type atom =
  | Eq of string * string  (** [Eq (l, r)]: [l] and [r] are the same key. *)
  | Cmp of compare * string * string
      (** [Cmp (op, l, r)]: the left [l] compares to the right [r] by [op]. *)
  | Overlaps of (string * string) * (string * string)  (** See {!overlaps}. *)
  | Position  (** See {!position}. *)

type cond = private atom list
(** The type for conditions: the conjunction of the atoms, in the order they are
    written. The empty conjunction is {!all}. [Position] is alone, and no atom
    appears twice. *)

val keys : string list -> cond
val eq : string -> string -> cond
val lt : string -> string -> cond
val le : string -> string -> cond
val gt : string -> string -> cond
val ge : string -> string -> cond
val inside : string -> string * string -> cond
val overlaps : string * string -> string * string -> cond
val position : cond
val all : cond
val ( && ) : cond -> cond -> cond

val reads : cond -> string list * string list
(** [reads c] is the left columns and the right columns that [c]'s atoms name,
    in any order, a column named twice twice. *)

(** {1:kinds Kinds and counts} *)

type kind = Inner | Left | Full | Semi | Anti
type count = Any | At_most_one | One | At_least_one

(** {1:checking Checking} *)

val check : kind -> Schema.t -> Schema.t -> cond -> Problem.t list
(** [check kind left right c] is every problem of the [kind] join on [c] of rows
    of the columns [left] and [right], those of [c] in the order of its atoms,
    then the names on both sides. The problems are:
    - a column missing on its side, with suggestions from that side's names;
    - the columns of an equality atom that do not meet, and extension columns
      that are not of one type;
    - the columns of a comparison or an [Overlaps] that do not meet at one type,
      or meet at one that is or holds an extension type, which has no order;
    - in a {!Full} join, a left column that is the left of two equality atoms,
      which makes its coalesced value ambiguous;
    - except for {!Semi} and {!Anti}, the joined columns that have the same
      name, all listed in one problem. *)

val columns : kind -> Schema.t -> Schema.t -> cond -> Schema.t
(** [columns kind left right c] is the columns of the [kind] join on [c] of rows
    of the columns [left] and [right], as [Talon.Join] describes them. [c] has
    passed {!check} against [left] and [right], or against schemas of which they
    hold the columns that [c] names, with the same types. [Query.make] computes
    with it the columns of a join, whose inputs the optimizer may narrow. *)

val common_type : Type.any -> Type.any -> Type.any option
(** [common_type l r] is the type at which a left column of type [l] meets a
    right column of type [r] in an equality atom: their common type
    ({!Type.common}), or their one type if either holds an extension type. A
    {!Full} join's key has it. *)

val equal : cond -> cond -> bool
(** [equal c0 c1] is [true] iff [c0] and [c1] have equal atoms in the same
    order. *)

val pp : Format.formatter -> cond -> unit
(** [pp ppf c] formats [c] as it is written inside [Join.( … )], its atoms
    joined by [&&]: each run of [Eq (n, n)] atoms as one [keys ["a"; "b"]],
    other atoms as [eq "l" "r"], [lt "l" "r"], [overlaps ("a", "b") ("c", "d")]
    and [position], and the condition with no atom as [all]. *)
