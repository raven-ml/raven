(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the SVG and PDF renderers share: numbers written to an accuracy in
    points, the maps they write, and paths mapped to a frame and cut at a margin
    around the page.

    A {e frame} is the coordinate system a renderer writes numbers in: the page
    (y-down, in points), or the definition of a stamp, which its instances
    place. Both renderers compile this module from this directory with
    [copy_files]. *)

open Hugin_next_gg
open Hugin_next_font
open Hugin_next_vg

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

(** {1:maps Maps} *)

val linear : Affine.t -> Affine.t
(** [linear m] is [m] without its translation. *)

val stretch : Affine.t -> float
(** [stretch m] is the largest factor by which [m] scales a length, the largest
    singular value of its linear part, computed so that it neither overflows nor
    underflows where floats hold it: [stretch (Affine.scale 1e-200 1e-200)] is
    [1e-200]. The linear part of [m] is finite and not zero. *)

val unit : Affine.t -> Affine.t
(** [unit m] is the linear part of [m] divided by [stretch m], which scales no
    length by more than [1.]. *)

val is_similar : Affine.t -> bool
(** [is_similar m] is [true] iff the linear part of [m] scales lengths evenly,
    turning and mirroring allowed, up to a relative [1e-9]: a round pen stays
    round under it. *)

val is_even : Affine.t -> bool
(** [is_even m] is [true] iff the linear part of [m] is a positive multiple of
    the identity, up to a relative [1e-9]. *)

val is_axial : Affine.t -> bool
(** [is_axial m] is [true] iff the linear part of [m] scales the axes by
    positive factors and turns nothing, up to a relative [1e-9]. *)

(** {1:paths Paths} *)

type sink = {
  move : float -> float -> unit;
  line : float -> float -> unit;
  cubic : float -> float -> float -> float -> float -> float -> unit;
  close : unit -> unit;
}
(** The type for what receives a path. Each subpath starts with [move]. *)

val area : ?out:Affine.t -> Affine.t -> Box2.t -> Path.t -> sink -> unit
(** [area ~out m cut q sink] gives [sink] the subpaths of [q] mapped through [m]
    into a frame, each closed, as filling and clipping see them, cut at [cut], a
    box of the frame: a subpath within it is given as it is; any other is
    clipped to it, its curves that do not lie within it first flattened to
    within 0.0005, so that the area within the box is unchanged and nothing lies
    outside it. The points given are mapped through [out], a linear map, which
    defaults to the identity. Coordinates mapped beyond [1e15] are taken at
    [1e15]. *)

val outline :
  ?out:Affine.t ->
  Affine.t ->
  Box2.t ->
  points:bool ->
  dashed:bool ->
  Path.t ->
  piece:(float -> unit) ->
  sink ->
  unit
(** [outline ~out m cut ~points ~dashed q ~piece sink] gives [sink] the subpaths
    of [q] mapped through [m] into a frame, as stroking sees them: closed only
    where [q] closes them, and those of zero length only if [points], a start
    alone given a line of zero length, cut at [cut], a box of the frame grown by
    the reach of the pen: a subpath within it is given as it is; the parts of
    any other that lie within it are given as open subpaths, curves first
    flattened as {!area} does, except that a closed subpath that is not [dashed]
    keeps its join at its start. [piece d] precedes each subpath given, [d]
    being the length along the subpath of [q] it was cut from, in [out]'s
    coordinates, before its start: [0.] for a whole subpath. Lengths are
    measured only if [dashed], and are [0.] otherwise. *)

(** {1:leaves Leaves}

    A frame's {e cut} is the box its geometry is cut at: on the page, a margin
    around it; in a stamp's definition, {!all}. What does not meet it is left
    out, and so is what lies beyond the range of floats. *)

val all : Box2.t
(** [all] is the box from [-1e15] to [1e15] along both axes, beyond which
    coordinates are taken at its edges: the cut of a frame that has no other. *)

val grown : Box2.t -> float -> Box2.t
(** [grown r k] is [r] grown by [k] on each side, [k] taken at [1e15] if it is
    larger or NaN. *)

val overlaps : Box2.t -> float -> float -> float -> float -> bool
(** [overlaps cut minx miny maxx maxy] is [true] iff the box of these bounds
    meets [cut], and [false] if a bound is NaN. *)

val meets : Box2.t -> Box2.t -> bool
(** [meets cut b] is [true] iff [b] meets [cut]. *)

val run_meets : Box2.t -> Affine.t -> Run.t -> bool
(** [run_meets cut m r] is [true] iff the ink of [r] mapped through [m], or its
    origin if it has none, is finite and meets [cut]. *)

val crop :
  Box2.t -> Affine.t -> Box2.t -> int -> int -> (int * int * int * int) option
(** [crop cut m b w h] is [Some (c0, r0, c1, r1)], the columns from [c0] to [c1]
    and rows from [r0] to [r1], exclusive, of an image of [w] by [h] pixels over
    [b] whose cells, mapped through [m], may meet [cut], or [None] if none does
    or [m] maps them beyond the range of floats. *)

(** {1:stamps Stamps} *)

type pen = { width : float; dash : float list; offset : float }
(** The type for pens as written: width, dash lengths and dash offset. *)

val pen : Affine.t -> float -> Stroke.t -> pen
(** [pen m k s] is the pen of [s], its lengths multiplied by [k], as written
    under [m]: multiplied by [stretch m] too, and taken at [1e15] beyond it. *)

val pens : Affine.t -> float -> Picture.t -> pen list
(** [pens m k p] is the distinct pens of the strokes of [p], as {!pen} writes
    them under [m] and the transforms within [p]. *)

val scales_within : Picture.t -> bool
(** [scales_within p] is [true] iff [p] holds a stamp that scales its instances.
*)

val reach : pen -> Stroke.t -> float
(** [reach pen s] is how far the pen [pen] of the style [s] reaches beyond its
    path: half its width, multiplied by the miter limit if [s] miters its joins,
    or by [sqrt 2.] if its caps are square and that is larger. *)
