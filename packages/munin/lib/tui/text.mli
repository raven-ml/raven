(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Numbers, counts, durations and times as the screens print them (the Numbers
   section of [Munin_tui]), and text cut to a width. A width is in cells: every
   glyph the screens print is one cell wide. *)

val width : string -> int

val pad : int -> string -> string
(** [pad w s] is [s] with spaces after it to [w] cells. *)

val pad_left : int -> string -> string

val cut : int -> string -> string
(** [cut w s] is [s] in at most [w] cells, its last an ellipsis when cut. *)

val fit : int -> string -> string
(** [fit w s] is [s] cut and padded to [w] cells. *)

val elide : int -> string -> string
(** [elide w key] is [key] in at most [w] cells, keeping its last segment whole
    when it can. *)

val non_finite : float -> string
(** [non_finite x] is [NaN], [inf] or [-inf]. *)

val value : float -> string
(** [value x] is [x] in four significant digits with trailing zeros, as
    [0.2540], [2150] or [10.00]; in scientific notation below [1e-4] and from
    [1e6], as [5.116e-05]; or [NaN], [inf], [-inf]. A value that is not zero
    never prints as [0]. *)

val percent_of : int -> int -> string
(** [percent_of s t] is [s] of [t] as a whole percentage, rounded down, as [99%]
    for [19950] of [20000]: [100%] only once [s] reaches [t]. It is [""] for a
    [t] that is not above 0. *)

val signed : float -> string
(** [signed x] is [value x], with [+] before a positive [x]. *)

val given : float -> string
(** [given x] is [x] in the fewest digits that read back as [x], for a value a
    program gave, such as a param. *)

val count : int -> string
(** [count n] is [n] with separators, as [12,400]. *)

val duration : float -> string
(** [duration s] is [s] seconds in at most two units, as [1h37m]. *)

val clock : now:float -> float -> string
(** [clock ~now t] is [t]'s local time of day, with its date unless it is
    [now]'s day. *)

val date : float -> string
(** [date t] is [t]'s local date and time to the second. *)

val gib : float -> string
val percent : float -> string
val mbs : float -> string
