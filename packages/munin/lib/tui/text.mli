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
(** [value x] is [x] in 4 significant digits, or [NaN], [inf], [-inf]. *)

val values : float list -> string list
(** [values xs] is [xs] in the fewest significant digits from 4 to 17 that print
    every two values that differ differently. *)

val count : int -> string
(** [count n] is [n] with separators, as [12,400]. *)

val short : int -> string
(** [short n] is [n] in three significant digits and a suffix, as [12.4k]. *)

val duration : float -> string
(** [duration s] is [s] seconds in at most two units, as [1h37m]. *)

val age : float -> string
(** [age s] is [s] seconds in one unit, as [2h]. *)

val clock : now:float -> float -> string
(** [clock ~now t] is [t]'s local time of day, with its date unless it is
    [now]'s day. *)

val date : float -> string
(** [date t] is [t]'s local date and time to the second. *)

val gib : float -> string
val percent : float -> string
val mbs : float -> string

val bar : int -> float -> string
(** [bar w p] is a bar of [w] eighth-block cells filled to [p] in [[0;1]],
    between two edges. *)
