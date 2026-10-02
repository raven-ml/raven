(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the SVG and PDF renderers share: the walk of a picture, numbers written
    to an accuracy in points, the maps they write, and paths mapped to a frame
    and cut at a margin around the page. A renderer gives the walk its syntax as
    a {!target}.

    A {e frame} is the coordinate system a renderer writes numbers in: the page
    (y-down, in points), or the definition of a stamp, which its instances
    place. Both renderers compile this module from this directory with
    [copy_files]. *)

open Hugin_gg
open Hugin_font
open Hugin_vg

(** {1:numbers Numbers} *)

val decimals : float -> int
(** [decimals k] is the number of decimals that keeps a number of a frame the
    page magnifies [k] times within 0.0005 point once rounded: [3] for
    [k <= 1.], and one more per power of ten above, at most [17]. *)

val finer : int -> int -> int
(** [finer k d] is [k] decimals more than [d], at most [17]. Numbers whose
    errors add up get finer decimals than their frame's, so that the sum stays
    within the accuracy. *)

val add_fixed : Buffer.t -> int -> float -> unit
(** [add_fixed b d v] adds [v] rounded to [d] decimals, without trailing zeros
    or point, and [0] for a number that rounds to zero, whatever its sign. [v]
    is taken within [±1e15], NaN as [1e15], so that what is added is a number of
    the order of the page's coordinates. *)

val add_exact : Buffer.t -> float -> unit
(** [add_exact b v] adds the shortest decimal that reads back as [v], in [%g]
    notation, and [0] for zero, whatever its sign. [v] is finite. *)

val rounded : int -> float -> float
(** [rounded d v] is [v] rounded to [d] decimals, as {!add_fixed} writes it. *)

(** {1:maps Maps} *)

val unit : Affine.t -> Affine.t
(** [unit m] is the linear part of [m] divided by [Affine.stretch m], which
    scales no length by more than [1.]. *)

val is_even : Affine.t -> bool
(** [is_even m] is [true] iff the linear part of [m] is a positive multiple of
    the identity, up to a relative [1e-9]. *)

val is_axial : Affine.t -> bool
(** [is_axial m] is [true] iff the linear part of [m] scales the axes by
    positive factors and turns nothing, up to a relative [1e-9]. *)

(** {1:paths Paths and leaves}

    A frame's {e cut} is the box its geometry is cut at: on the page, a margin
    around it; in a stamp's definition, {!Instances.everywhere}. Coordinates
    mapped beyond [1e15] are taken at [1e15]. *)

type sink = {
  move : float -> float -> unit;
  line : float -> float -> unit;
  cubic : float -> float -> float -> float -> float -> float -> unit;
  close : unit -> unit;
}
(** The type for what receives a path. Each subpath starts with [move]. *)

val area : Affine.t -> Box2.t -> Path.t -> sink -> unit
(** [area m cut q sink] gives [sink] the subpaths of [q] mapped through [m] into
    a frame, each closed, as filling and clipping see them, as
    {!Hugin_gg.Path.crop} crops them to [cut]. *)

val overlaps : Box2.t -> float -> float -> float -> float -> bool
(** [overlaps cut minx miny maxx maxy] is [true] iff the box of these bounds
    meets [cut], and [false] if a bound is NaN. *)

val run_meets : Box2.t -> Affine.t -> Run.t -> bool
(** [run_meets cut m r] is [true] iff the ink of [r] mapped through [m], or its
    origin if it has none, is finite and meets [cut]. *)

val ems : Font.t -> float
(** [ems f] is how far, in ems, the outline of a glyph of [f] reaches from its
    origin in any direction. *)

(** {1:pens Pens} *)

type pen = { width : float; dash : float list; offset : float }
(** The type for pens as written: width, dash lengths and dash offset, in a
    frame. *)

val scale_pen : float -> pen -> pen
(** [scale_pen k pen] is [pen] with its lengths multiplied by [k]. *)

val dash_decimals : int -> int
(** [dash_decimals d] is the decimals of dash lengths in a frame of [d]
    decimals: three more, since the errors of the lengths add up along a
    subpath, so that a thousand of them stay within the accuracy. *)

val is_dashed : int -> pen -> bool
(** [is_dashed d pen] is [true] iff a dash length of [pen] is written non-zero
    with {!dash_decimals}[ d] decimals. A pattern that is not is written solid.
*)

val phase : pen -> float -> float
(** [phase pen o] is the dash offset of [pen] moved by [o], within the length of
    its pattern made even. *)

(** {1:walk The walk} *)

(** What paints a leaf of a kind: its own colour, the colour an enclosing
    instance sets and the leaf inherits, or a colour an instance drawn in full
    gives it. *)
type paint = Own | Inherit | Fixed of Color.t

type ctx = {
  m : Affine.t;  (** From the picture's coordinates to the frame's. *)
  cut : Box2.t;  (** The frame's cut. *)
  mag : float;  (** How much the page magnifies the frame's numbers. *)
  d : int;  (** Decimals of the frame's numbers. *)
  fills : paint;  (** What paints fills and glyphs. *)
  strokes : paint;  (** What paints strokes. *)
  fill_set : bool;  (** An enclosing instance sets the fill colour. *)
  stroke_set : bool;  (** An enclosing instance sets the stroke colour. *)
  pen : float;  (** What the lengths of pens are multiplied by. *)
  pen_set : bool;  (** An enclosing instance sets the pen's lengths. *)
}
(** The type for where the walk is: its frame and what enclosing instances set.
*)

val page : Renderable.t -> ctx
(** [page r] is the context of the page of [r], cut at a margin of the page's
    larger side around it. *)

val box : ctx -> Picture.t -> Box2.t option
(** [box ctx p] is a box of the frame holding what [p] paints there, grown by
    one point and cut by [ctx.cut], or [None] if [p] paints nothing there. *)

type stroke = {
  style : Stroke.t;
  pen : pen;  (** The pen as written in the frame. *)
  frame : Affine.t option;
      (** The matrix the pen is written under, if the frame stretches it
          unevenly: the frame's linear part scaled to stretch no more than the
          page does, its coefficients rounded to 15 decimals. *)
  dashed : bool;  (** {!is_dashed} of the pen in the frame. *)
}
(** The type for strokes as written. *)

type image = {
  pixels : Nx.uint8_t;  (** All the pixels of the image. *)
  window : int * int * int * int;
      (** The columns [c0] to [c1] and rows [r0] to [r1], exclusive, whose cells
          may meet the cut. *)
  x : float;
  y : float;
  w : float;
  h : float;  (** The box of these cells, in the picture's coordinates. *)
}
(** The type for images cropped to the cut. *)

type stamp = {
  picture : Picture.t;
  extent : Box2.t;  (** The picture's box under the frame's linear part. *)
  pens : pen list;
      (** The distinct pens of the picture as written, if [scaled]; [[]]
          otherwise. *)
  decimals : int;
      (** The decimals that keep the picture's points within the accuracy once
          an instance's scale is rounded to them. *)
  translucent : bool;  (** An instance sets a translucent colour. *)
  scaled : bool;  (** The stamp scales its instances. *)
  fills : Color.t array option;
  strokes : Color.t array option;
  rows : int array option;  (** The rows of its instances, from a tag. *)
}
(** The type for stamps as the walk writes them. *)

val scale_decimals : ctx -> float -> int
(** [scale_decimals ctx s] is the decimals of the pen an instance of scale [s]
    sets in the frame of [ctx]. *)

type ('b, 'd) target = {
  fill : 'b -> ctx -> Picture.rule -> Color.t -> Path.t -> unit;
  stroke :
    'b ->
    ctx ->
    stroke ->
    Color.t ->
    (piece:(float -> unit) -> sink -> unit) ->
    unit;
      (** [stroke b ctx s c outline] writes the stroke [s].
          [outline ~piece sink] gives [sink] its subpaths in the frame, cut at
          the cut grown by the pen's reach, mapped through the inverse of
          [s.frame]. [piece d] precedes each, [d] the length before it along the
          subpath it was cut from: [0.] unless [s.dashed] and the subpath was
          cut, so that the dashes keep their phase. *)
  glyphs : 'b -> ctx -> Color.t -> P2.t -> Run.t -> unit;
  image : 'b -> ctx -> image -> unit;
  clip : 'b -> ctx -> Picture.rule -> Path.t -> ('b -> unit) -> unit;
      (** [clip b ctx rule q k] writes what [k] writes, clipped. *)
  opacity : 'b -> ctx -> float -> Picture.t -> ('b -> unit) -> unit;
      (** [opacity b ctx a p k] writes what [k] writes of [p], faded. *)
  tag : 'b -> ctx -> Picture.tag -> ('b -> unit) -> unit;
      (** [tag b ctx t k] writes what [k] writes, tagged. A tag of rows on a
          stamp gives its rows to the stamp's instances and reaches [tag]
          without rows. *)
  carry : ctx -> stamp -> float -> float;
      (** [carry ctx st s] is the scale an instance of scale [s] writes, or NaN
          if the definition cannot carry it and the instance is drawn in full.
          It is applied to [ctx] and [st] once per stamp. *)
  define : ctx -> Box2.t -> ('b -> unit) -> 'd;
      (** [define dctx box k] is the definition of what [k] writes in the frame
          of [dctx], within [box]. *)
  use : 'b -> ctx -> stamp -> 'd -> int -> P2.t -> float -> unit;
      (** [use b ctx st def i at s] writes instance [i] of [def] at [at], of
          scale [s] as carried. *)
  instance : 'b -> stamp -> int -> ('b -> unit) -> unit;
      (** [instance b st i k] writes instance [i] drawn in full by [k]. *)
}
(** The type for targets: what a renderer writes for each case of a picture,
    into a ['b], a stamp's definition being a ['d]. *)

val walk : ('b, 'd) target -> ctx -> 'b -> Picture.t -> unit
(** [walk t ctx b p] writes [p] into [b] with [t]. Transforms compose into
    [ctx], and one with no inverse leaves nothing to write. A stamp is written
    as one definition that its instances use, except the instances whose scale
    [t.carry] does not carry, and all of them if they scale and the picture
    holds a scaling stamp or strokes of differing pens: those are drawn in full.
    An instance at a position or scale that is not finite, or of scale [0.],
    paints nothing, and so does one that {!Instances.shows} does not show within
    [ctx.cut], with the pens it keeps. *)
