(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The rules of stamps that every renderer follows: which instances show and
    which colours they paint with, and the walks of a stamp's picture that
    decide them.

    An instance of a stamp paints the stamp's picture scaled and moved to its
    position, with the picture's pens: whatever its scale, its strokes reach as
    far beyond their paths as the picture's do. *)

open Hugin_gg
open Hugin_vg

val everywhere : Box2.t
(** [everywhere] is the box from [-1e15] to [1e15] along both axes. *)

val bounds : Picture.t -> Box2.t option
(** [bounds p] is [Picture.bounds p], or {!everywhere} if the bounds lie beyond
    the range of floats, so that a picture that far is taken to show everywhere.
*)

val uncut : Picture.t -> Picture.t
(** [uncut p] is [p] without its clips and tags. Unlike the bounds of [p], which
    a clip cuts at the scale and pens [p] has, its bounds, scaled by an
    instance's scale and grown by the {!reach} of the pens the instance keeps,
    hold what that instance paints: a stroke that misses a clip at scale [1.]
    may meet it at another, with the pen an instance keeps. *)

val meets : Box2.t -> Box2.t -> Box2.t option
(** [meets a b] is the box of the points in both [a] and [b], or [None] if they
    have none or only touch, to within rounding, each extending past the other
    along an axis: what a leaf whose box touches a clip's box paints is nothing,
    however rounding has placed the two. A box within the other meets it,
    however thin. *)

val exists : (Picture.t -> bool) -> Picture.t -> bool
(** [exists f p] is [true] iff [f] holds of [p] or of a picture within it. *)

val fold_strokes :
  (Affine.t -> Stroke.t -> 'a -> 'a) -> Affine.t -> Picture.t -> 'a -> 'a
(** [fold_strokes f m p acc] folds [f] over the strokes of [p], in order, each
    with [m] composed with the transforms above it within [p]. It skips the
    strokes under a composition that {!Hugin_gg.Affine.invert} gives no
    inverse, which paint nothing. *)

val reach : Affine.t -> float -> Picture.t -> float
(** [reach m k p] is how far the strokes of [p] under [m] reach beyond their
    paths, their lengths multiplied by [k]: the largest
    {!Hugin_gg.Stroke.reach} of the pens of [p], mapped through the
    transforms within [p] and through [m]. Stamps within [p] keep their pens, so
    their scales do not change it. *)

val shows : Box2.t -> reach:float -> P2.t -> float -> Box2.t -> bool
(** [shows cut ~reach at s b] is [true] iff [b] scaled by [s], moved to [at] and
    grown by [reach] meets [cut]: an instance at [at] of scale [s] of a picture
    of box [b], whose pens reach [reach], may paint within [cut]. *)

val positions : Box2.t -> reach:float -> Box2.t -> Box2.t option
(** [positions cut ~reach b] is the box of the positions at which an instance of
    scale [1.] {!shows}, up to rounding: [cut] grown by [reach] and moved back
    by the corners of [b]; or [None] if that box is not finite. *)

val color : Color.t array option -> int -> own:(Color.t -> 'a) -> 'a -> 'a
(** [color cs i ~own inherited] is what paints a leaf of instance [i]: its own
    colour [own cs.(i)], or [inherited] if [cs] is [None]. *)
