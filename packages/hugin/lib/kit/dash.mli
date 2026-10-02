(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Dash patterns.

    A dash pattern says how a line is broken: the lengths of its dashes and of
    the gaps between them, in multiples of the line's width, starting with a
    dash, so that a pattern keeps its look at every width. A pattern of an odd
    number of lengths repeats once to make it even, as {!Hugin_gg.Stroke.v}
    repeats it. The lengths are those of the dashes themselves, whatever the
    caps that end them: a round or square cap adds half the width beyond each
    end of a dash, so a dash of length [0.] draws a dot.

    {!all} is the ordered set of patterns that tell the series of a figure
    apart. *)

(** {1:patterns Patterns} *)

type t
(** The type for dash patterns. *)

val v : float list -> t
(** [v lengths] is the pattern of [lengths], dash first. [v []] is {!solid}.

    Raises [Invalid_argument] if a length is negative or not finite, or if
    [lengths] is not empty and sums to [0.]. *)

val solid : t
(** [solid] is an unbroken line, [v []]. *)

val dashed : t
(** [dashed] is [v [ 4.; 2. ]]. *)

val dotted : t
(** [dotted] is [v [ 1.; 2. ]]. *)

val dash_dot : t
(** [dash_dot] is [v [ 4.; 2.; 1.; 2. ]]. *)

val all : t list
(** [all] is {!solid}, {!dashed}, {!dotted} and {!dash_dot}, in this order: the
    patterns of the categories of a band scale that sets none. *)

(** {1:observing Observing} *)

val lengths : t -> float list
(** [lengths d] is the lengths of [d] as given to {!v}, [[]] if solid. *)

(** {1:comparing Comparing and formatting} *)

val equal : t -> t -> bool
(** [equal d d'] is [true] iff [d] and [d'] have equal lengths, compared by
    [Float.equal]. *)

val pp : Format.formatter -> t -> unit
(** [pp ppf d] formats [d] for debugging: [solid], or its lengths, such as
    [4 2]. *)
