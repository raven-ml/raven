(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Stroke styles.

    A stroke style says how a path is stroked: the width of the line, how open
    ends are capped, how corners are joined and whether the line is dashed. Its
    lengths are in the units of the coordinates the path is stroked in. A map
    applied to a picture that strokes a path scales them. Mapping the path
    itself with {!Hugin_next_gg.Path.transform} does not. *)

(** {1:types Types} *)

type cap = [ `Butt | `Round | `Square ]
(** The type for line caps. [`Butt] ends a line exactly at its end point;
    [`Round] adds a half disc and [`Square] a half square of the line's width
    beyond it. *)

type join = [ `Miter | `Round | `Bevel ]
(** The type for line joins. [`Miter] extends the outer edges of a corner until
    they meet, unless the miter would exceed the {!miter_limit}, in which case
    the corner is bevelled; [`Round] fills the corner with a disc and [`Bevel]
    with a triangle. *)

type t
(** The type for stroke styles. Invariants: the width is finite and
    non-negative, the miter limit finite and at least [1.], and the dash pattern
    empty or made of finite, non-negative lengths whose sum, counting a pattern
    of odd length twice, is positive and finite. *)

(** {1:constructors Constructors} *)

val v :
  ?cap:cap ->
  ?join:join ->
  ?miter_limit:float ->
  ?dash:float list ->
  ?dash_offset:float ->
  float ->
  t
(** [v ~cap ~join ~miter_limit ~dash ~dash_offset width] is the stroke style of
    line width [width] with:
    - [cap], the line cap. Defaults to [`Round].
    - [join], the line join. Defaults to [`Round].
    - [miter_limit], the ratio of a miter's length to the line width beyond
      which a [`Miter] join is bevelled. Defaults to [4.].
    - [dash], the lengths of the dashes and of the gaps between them, starting
      with a dash. A pattern with an odd number of lengths is repeated once to
      make it even. A zero-length dash draws a dot with a [`Round] cap and a
      square with a [`Square] one. Defaults to [[]], a solid line.
    - [dash_offset], the distance into the pattern at which each subpath starts.
      It is reduced modulo the length of the pattern, made even, so any finite
      offset, negative ones included, is valid. Defaults to [0.]; a solid line
      ignores it.

    A stroke of width [0.] draws nothing.

    Raises [Invalid_argument] if an invariant of {!type-t} would not hold or if
    [dash_offset] is not finite. *)

(** {1:accessors Accessors} *)

val width : t -> float
(** [width s] is the line width of [s]. *)

val cap : t -> cap
(** [cap s] is the line cap of [s]. *)

val join : t -> join
(** [join s] is the line join of [s]. *)

val miter_limit : t -> float
(** [miter_limit s] is the miter limit of [s]. *)

val dash : t -> float list
(** [dash s] is the dash pattern of [s] as given to {!v}, [[]] if solid. *)

val dash_offset : t -> float
(** [dash_offset s] is the dash offset of [s], reduced into \[[0];[l]\[ for the
    length [l] of the even pattern, and [0.] if solid. *)

(** {1:comparing Comparing and formatting} *)

val equal : t -> t -> bool
(** [equal s s'] is [true] iff [s] and [s'] have equal widths, caps, joins,
    miter limits, dash patterns and dash offsets. *)

val compare : t -> t -> int
(** [compare s s'] is a total order on stroke styles compatible with {!equal}.
*)

val pp : Format.formatter -> t -> unit
(** [pp ppf s] formats [s] for debugging. *)
