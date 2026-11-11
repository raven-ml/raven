(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** How the dashboard prints numbers, steps, durations, times and keys: the
    Numbers section of {!Munin_tui}. A width is in terminal cells. *)

val width : string -> int
(** [width s] is the cells [s] takes on a terminal. *)

val cut : int -> string -> string
(** [cut w s] is [s] in at most [w] cells, ending in […] when it is cut, and
    [""] when [w <= 0]. *)

val elide : int -> string -> string
(** [elide w key] is [key] in at most [w] cells. A key that does not fit keeps
    its last [/] segment whole when it can and loses the middle, as
    [val/acc…/top5]: two keys that differ only at their end stay apart. *)

val non_finite : float -> string
(** [non_finite x] is [NaN], [inf] or [-inf]. *)

val decimal : int -> int -> float
(** [decimal m k] is the float nearest [m] × 10{^ [k]}, the same on every
    system: an infinity past the largest float, a zero below the least. *)

val decade : float -> int
(** [decade x] is the [k] with [decimal 1 k <= |x| < decimal 1 (k + 1)],
    from the exponent of [x]'s shortest decimal, for a finite [x] other
    than zero. *)

val value : float -> string
(** [value x] is [x] as it reads back, or with a mark that it is rounded.

    A whole value of six digits or fewer prints in full, as [2150] or
    [348513]. Any other is rounded to four significant digits and printed
    with its trailing zeros, as [0.2540] or [10.00] (for [9.9996]): with the
    suffix [k] from a thousand up, as [348.5k] (for [348513.5]); in
    scientific notation when the rounded value is below [1e-4] or from [1e6]
    in magnitude, as [5.116e-05]. NaN and the infinities print [NaN], [inf]
    and [-inf], and [-0.] prints [0]. So a printed value either reads back as
    [x], NaN and the infinities as their words and [-0.] as [0.], or holds a
    decimal point, a [k] or an exponent, and a value that is not zero never
    prints as [0]. *)

val metric : string -> float -> string
(** [metric key x] is [x] as {!value} prints it, or, for an accuracy in
    [[0;1]], a percentage with two decimals. An accuracy is a metric whose key
    holds [acc]. *)

val given : float -> string
(** [given x] is [x] in the fewest digits that read back as [x], for a value
    a program gave, such as a param. *)

val difference : string -> value:float -> other:float -> string option
(** [difference key ~value ~other] is [value] against [other]: relative, as
    [+60%], or in points for two accuracies in [[0;1]], as [+0.14 pt]. It is
    [None] when they are the same in the digits shown, or one is not
    finite. *)

val count : int -> string
(** [count n] is [n] with separators, as [12,400]. *)

val compact : float -> string
(** [compact x] is [x] as an axis prints a step: [500], [1.5k], [12.5k],
    [3M], in the fewest digits that give it exactly, at most four. *)

val tick : float list -> (float -> string) option
(** [tick xs] labels the ticks [xs] of a y axis, each in the fewest digits
    that give it exactly, all in one notation: plain, as [0.0125], [-250] or
    [12500], if every tick so fits six cells; else with the suffix [k], [M]
    or [G] from a thousand up, as [500k] and [1.5M]; else with an exponent,
    as [1e-05] and [2e-05]. [0] is [0] in each. It is [None] when no
    notation fits every tick, as for [0.51234], or when one is not finite. *)

val percent_of : int -> int -> string
(** [percent_of s t] is [s] of [t] as a whole percentage, rounded down, as
    [99%] for [19950] of [20000]: [100%] only once [s] reaches [t]. It is [""]
    for a [t] that is not above 0. *)

val percent : float -> string
(** [percent x] is the percentage [x] with one decimal below 10, as [7.5%]
    or [71%]. *)

val gib : float -> string
(** [gib b] is [b] bytes in GiB with one decimal, without the unit. *)

val span : float -> string
(** [span s] is [s] seconds in two units, as [4m 10s]. *)

val clock : now:float -> float -> string
(** [clock ~now t] is [t]'s local time of day, as [14:22], with its date
    unless it is [now]'s day. *)

val date : float -> string
(** [date t] is [t]'s local date and time to the second. *)
