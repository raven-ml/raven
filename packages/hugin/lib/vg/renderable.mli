(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Renderables.

    A renderable is a picture on a page: the rectangle of the plane from
    [(0, 0)] to [(w, h)], in points, 1/72 inch. It gives the picture a physical
    size, the same in every output: a page 360 points wide is five inches wide
    as PDF and as SVG, and [360 *. d] pixels wide drawn at a density of [d]
    pixels per point, which PNG output records so that the file prints five
    inches wide.

    What the picture paints outside the page is not shown, and the page is
    transparent where the picture paints nothing: a picture brings its own
    background. *)

(** {1:types Types} *)

type t
(** The type for renderables. Invariant: the width and height are finite and
    positive. *)

(** {1:constructors Constructors and accessors} *)

val v : float -> float -> Picture.t -> t
(** [v w h p] is [p] on a page [w] points wide and [h] points high.

    Raises [Invalid_argument] if [w] or [h] is not finite and positive. *)

val w : t -> float
(** [w r] is the width of the page of [r], in points. *)

val h : t -> float
(** [h r] is the height of the page of [r], in points. *)

val picture : t -> Picture.t
(** [picture r] is the picture of [r]. *)

(** {1:comparing Comparing and formatting} *)

val equal : t -> t -> bool
(** [equal r r'] is [true] iff [r] and [r'] have equal widths and heights and
    {!Picture.equal} pictures. *)

val pp : Format.formatter -> t -> unit
(** [pp ppf r] formats the size and the picture of [r] for debugging. *)
