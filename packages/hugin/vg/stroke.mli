(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Stroke styles.

    A stroke style says how {!Picture.stroke} draws the pen along a path: its
    width, how open ends are capped, how corners are joined and whether the line
    is dashed. All lengths are in the path's units and follow the transforms
    applied to the picture. *)

(** {1:types Types} *)

type cap = [ `Butt | `Round | `Square ]
(** The type for line caps. [`Butt] ends the line exactly at its endpoint,
    [`Round] adds a half disc and [`Square] a half square of the line's width.
*)

type join = [ `Miter | `Round | `Bevel ]
(** The type for line joins. [`Miter] extends the outer edges to a point unless
    the {!field-miter_limit} is exceeded, in which case the corner is bevelled;
    [`Round] adds a disc and [`Bevel] a triangle. *)

type t = private {
  width : float;
  cap : cap;
  join : join;
  dash : float array;
  miter_limit : float;
}
(** The type for stroke styles. [dash] alternates drawn and skipped lengths,
    starting drawn, and restarts at each subpath; empty means solid. Invariants:
    [width >= 0.], [miter_limit >= 1.], and [dash] is empty or has non-negative
    entries and a positive sum. *)

(** {1:constructors Constructors} *)

val v :
  ?cap:cap ->
  ?join:join ->
  ?dash:float array ->
  ?miter_limit:float ->
  float ->
  t
(** [v ~cap ~join ~dash ~miter_limit width] is the stroke style of line width
    [width] with:
    - [cap], the line cap. Defaults to [`Round].
    - [join], the line join. Defaults to [`Round].
    - [dash], the dash pattern. Defaults to solid. An odd number of entries is
      repeated to make the pattern even, and a zero drawn length with a round
      cap draws a dot.
    - [miter_limit], the ratio of miter length to line width beyond which a
      miter join is bevelled. Defaults to [4.].

    A zero [width] draws nothing.

    Raises [Invalid_argument] if an invariant of {!type:t} is violated. *)
