(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Locales.

    A locale holds the strings a culture writes numbers and dates with: the
    decimal separator, the separator and sizes of digit groups, the minus sign
    and the names of the months. No locale is global: every function that writes
    a number or a date takes one as an optional argument, which defaults to
    {!default}. *)

(** {1:locales Locales} *)

type t
(** The type for locales. *)

val v :
  ?decimal:string ->
  ?group:string ->
  ?grouping:int list ->
  ?minus:string ->
  ?months:string array ->
  unit ->
  t
(** [v ~decimal ~group ~grouping ~minus ~months ()] is the locale with:
    - [decimal], the decimal separator. Defaults to ["."].
    - [group], the separator between groups of integer digits. Defaults to
      [","]; [""] separates groups with nothing.
    - [grouping], the sizes of the groups of integer digits from the decimal
      separator leftwards, the last size repeating. Defaults to [[3]]; [[3; 2]]
      groups [123456789] as [12,34,56,789].
    - [minus], the minus sign. Defaults to ["\u{2212}"], U+2212 MINUS SIGN, as
      wide as a plus sign; ["-"] is the ASCII hyphen-minus.
    - [months], the short names of the months from January. Defaults to [Jan],
      [Feb], [Mar], [Apr], [May], [Jun], [Jul], [Aug], [Sep], [Oct], [Nov] and
      [Dec]. The array is copied.

    Raises [Invalid_argument] if [grouping] is empty or holds a size below [1],
    [months] does not have twelve elements, [decimal], [minus] or a month name
    is empty, [group] is [decimal], or a string is not valid UTF-8. *)

val default : t
(** [default] is [v ()]. *)

(** {1:accessors Accessors} *)

val decimal : t -> string
(** [decimal l] is the decimal separator of [l]. *)

val group : t -> string
(** [group l] is the digit group separator of [l]. *)

val grouping : t -> int list
(** [grouping l] is the sizes of the digit groups of [l], from the decimal
    separator leftwards, the last size repeating, without the repeats of the
    last size that end the list it was given: [grouping (v ~grouping:[3; 3] ())]
    is [[3]]. *)

val minus : t -> string
(** [minus l] is the minus sign of [l]. *)

val month : t -> int -> string
(** [month l m] is the short name of month [m] in [l], [1] for January.

    Raises [Invalid_argument] if [m] is not in \[[1];[12]\]. *)

(** {1:comparing Comparing and formatting} *)

val equal : t -> t -> bool
(** [equal l l'] is [true] iff [l] and [l'] have equal strings and group sizes.
*)

val pp : Format.formatter -> t -> unit
(** [pp ppf l] formats [l] for debugging. *)
