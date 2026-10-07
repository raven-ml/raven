(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Astronomy over {!Nx} tensors.

    Ymir gives astronomical meaning to tensors: the units astronomy measures in,
    and the models built on them. Every function is a formula of nx operations,
    so it runs batched, compiled and differentiated under rune's transformations
    with no rule of its own.

    [open Ymir] brings the unit algebra of [ymir.units] ({!Unit}, {!Quantity},
    {!Constant}, {!Codata}, {!Vocabulary}) and astronomy's units ({!Units}) into
    scope. {!Frame} names the celestial frames and {!Direction} holds directions
    in them. *)

(** {1:units Units} *)

module Unit = Ymir_units.Unit
(** Units as exact values. *)

module Quantity = Ymir_units.Quantity
(** Values in a unit. *)

module Constant = Ymir_units.Constant
(** Measured constants. *)

module Codata = Ymir_units.Codata
(** CODATA releases of the measured constants. *)

module Vocabulary = Ymir_units.Vocabulary
(** Names for units. *)

(** Astronomy's units.

    Each is an exact {!Unit.t}, built from the SI's units and the IAU's
    definitions. *)
module Units : sig
  val parsec : Unit.t
  (** [parsec] is 648000/π {!Unit.astronomical_unit}, the distance at which one
      astronomical unit subtends one arcsecond. *)

  val julian_year : Unit.t
  (** [julian_year] is 365.25 {!Unit.day}, 31557600 s. *)
end

(** {1:frames Frames and directions} *)

(** Celestial frames.

    A {e frame} is a celestial coordinate system: an orientation of three axes.
    Its {e orientation} [R_f] is the matrix taking ICRS components to frame
    [f]'s, [v_f = R_f · v_icrs]. A {e fixed} frame is one whose orientation from
    ICRS is constant.

    Each frame type has exactly one value, so two values of one type are one
    frame and a frame mismatch is a type error. A fixed frame's type is an
    instance of {!fixed}: {!matrix} and {!Direction.rotate} take any two fixed
    frames, and no other. The users of a frame write its type, as in
    [Frame.galactic Direction.t].

    Each fixed frame's orientation is its standard's own numbers, rounded once
    to float64. A conversion between two fixed frames is computed from their two
    orientations alone, so it depends only on the two frames, never on a path
    through others. *)
module Frame : sig
  type 'f t
  (** The type for celestial frames. Each frame type has exactly one value. *)

  type !'a fixed
  (** The type of a frame at a constant orientation from ICRS. *)

  type icrs = [ `Icrs ] fixed
  type fk5_j2000 = [ `Fk5_j2000 ] fixed
  type galactic = [ `Galactic ] fixed
  type ecliptic_j2000 = [ `Ecliptic_j2000 ] fixed
  type supergalactic = [ `Supergalactic ] fixed

  val icrs : icrs t
  (** [icrs] is the International Celestial Reference System. *)

  val fk5_j2000 : fk5_j2000 t
  (** [fk5_j2000] is FK5, mean equator and equinox of J2000.0, oriented at epoch
      J2000.0: the rotation from ICRS by the vector (−19.9, −9.1, +22.9) mas of
      Mignard and Froeschlé (2000), 31.7 mas in all. *)

  val galactic : galactic t
  (** [galactic] is Galactic coordinates as the Hipparcos Catalogue defines them
      on ICRS: the north Galactic pole at α = 192.85948°, δ = 27.12825°, and the
      north celestial pole at l = 122.93192°. *)

  val ecliptic_j2000 : ecliptic_j2000 t
  (** [ecliptic_j2000] is the IAU 2006 mean ecliptic and equinox of J2000.0, on
      ICRS: the obliquity ε₀ = 84381.406″ applied to the IAU 2006 frame bias at
      J2000.0. *)

  val supergalactic : supergalactic t
  (** [supergalactic] is supergalactic coordinates, defined on {!galactic}: the
      north supergalactic pole at Galactic (47.37°, +6.32°), the origin at l =
      137.37°. *)

  val matrix : 'a fixed t -> 'b fixed t -> (float, Nx.float64_elt) Nx.t
  (** [matrix a b] is [M], [[3; 3]] on the host, with [v_b = M · v_a]: for
      quantities that are not directions, such as a velocity ([M v]) or a
      covariance ([M Σ Mᵀ]).

      [M] is [R_b · R_aᵀ]. [matrix a a] is the identity, [matrix icrs b] is
      [R_b] and [matrix a icrs] is [R_aᵀ]; another pair's entries are each
      computed with one {!Float.fma} chain, so [matrix b a] is the exact
      transpose of [matrix a b]. *)

  val name : 'f t -> string
  (** [name f] is [f]'s canonical name: ["icrs"], ["fk5_j2000"], ["galactic"],
      ["ecliptic_j2000"], ["supergalactic"]. It is the frame's identity in keys,
      text and files. *)

  val pp : Format.formatter -> 'f t -> unit
  (** [pp] prints {!name}. *)

  val walk : ('a, 'b) Nx.Ptree.Walk.cursor -> 'f t -> 'f t
  (** [walk c f] reports {!name} as one case at [c]'s path and is [f]. A frame
      holds no tensors. *)
end

(** Directions in a frame.

    A direction is a frame and float64 vectors [[...; 3]], each read as the ray
    it spans. A unit norm cannot be a property of the type: [jvp] tangents,
    [grad] cotangents and {!Nx.Ptree.map} build directions of any norm. So the
    contract has two halves.

    - {e Constructors return unit vectors.} {!lonlat} and {!of_xyz} give
      [|xyz| = 1] within 3 ulp, and {!rotate} preserves the norm within 3 ulp.
    - {e Readers read the ray.} {!lon}, {!lat}, {!separation} and
      {!position_angle} read every finite row as the direction it points in. A
      positive multiple of a row reads the same up to the rounding of the
      multiple, and bit for bit when the multiple is a power of two that keeps
      the components normal. A NaN component gives NaN, and an infinite one
      raises [Invalid_argument]. {!rotate} and {!xyz} are linear.

    So a sum of two directions read as a direction is their chordal midpoint.

    {b Derivatives.} Every function has its analytic derivative wherever it is
    differentiable. Where it has none it gives a stated value with derivative 0,
    never NaN:
    - {!lon} at a pole ([x = y = 0]): 0.
    - {!lat} at a pole: ±π/2; at a zero row: 0.
    - {!separation} where [a × (b − a) = 0], that is coincident or antipodal
      rows or a zero row: π if [a · b < 0], else 0.
    - {!position_angle} at the same points: 0. With respect to [a] at a pole,
      its derivative is 0.

    At [lat]'s poles and [separation]'s 0 and π the function behaves like [|x|],
    and 0 is a subgradient; the other points are discontinuities. Near a pole
    the derivative of [lon] grows as 1/ρ, ρ the distance from the axis.

    Every angle in or out is a float64 quantity. Indexing and reshaping go
    through {!ptree}:

    {[
    let near =
      Nx.Ptree.map (Direction.ptree ())
        (fun _ x -> Nx.take ~axis:0 ~indices x)
        stars
    ]} *)
module Direction : sig
  type 'f t
  (** The type for directions in frame ['f]: float64 vectors [[...; 3]], each
      read as the ray it spans. *)

  val lonlat :
    'f Frame.t ->
    lon:(float, Nx.float64_elt) Nx.t Quantity.t ->
    lat:(float, Nx.float64_elt) Nx.t Quantity.t ->
    'f t
  (** [lonlat f ~lon ~lat] is the unit vectors
      [(cos b cos l, cos b sin l, sin b)] at longitude [l] and latitude [b],
      their batch axes broadcast. Every finite angle names a direction: latitude
      100° at longitude 0° is latitude 80° at longitude 180°. A NaN angle gives
      a NaN row.

      Raises [Invalid_argument] if [lon] or [lat] is not an angle, or where an
      angle is infinite, as in ["Direction.lonlat: lat at [3] is infinite"]. *)

  val of_xyz : 'f Frame.t -> (float, Nx.float64_elt) Nx.t -> 'f t
  (** [of_xyz f v] is [v / |v|] along [v]'s last axis. A NaN row stays NaN. Its
      derivative is the projection onto the tangent plane.

      Raises [Invalid_argument] if the last axis does not have 3 elements, or
      where a vector is infinite or zero, as in
      ["Direction.of_xyz: the vector at [17] is zero and names no direction"].
  *)

  val frame : 'f t -> 'f Frame.t
  (** [frame d] is [d]'s frame. *)

  val xyz : 'f t -> (float, Nx.float64_elt) Nx.t
  (** [xyz d] is [d]'s vectors, [[...; 3]]: unit vectors for values the
      constructors build. *)

  val lon : 'f t -> (float, Nx.float64_elt) Nx.t Quantity.t
  (** [lon d] is the longitude in \[+0, 2π) radians, [atan2 y x]. It is 0 at a
      pole, so a function that needs a meridian at a pole takes the meridian of
      longitude 0. *)

  val lat : 'f t -> (float, Nx.float64_elt) Nx.t Quantity.t
  (** [lat d] is the latitude in \[−π/2, π/2\] radians, [atan2 z ρ] with
      [ρ = √(x² + y²)]. *)

  val rotate : ('b Frame.fixed as 'g) Frame.t -> 'a Frame.fixed t -> 'g t
  (** [rotate g d] is [d] in frame [g]: [Frame.matrix (frame d) g] times each
      vector. It is linear in [d], within [1.5 · 2⁻⁵² · |v|] of the exact
      product per component, and rounds the same eagerly and compiled. *)

  val separation : 'f t -> 'f t -> (float, Nx.float64_elt) Nx.t Quantity.t
  (** [separation a b] is the angle between the rays, in \[0, π\] radians.
      Leading axes broadcast. It is [atan2 |a × d| (a · b)] with [d = b − a],
      which keeps relative accuracy at any separation: 3e-16 at 1 mas. *)

  val position_angle : 'f t -> 'f t -> (float, Nx.float64_elt) Nx.t Quantity.t
  (** [position_angle a b] is [b]'s bearing from [a], east of north, in \[+0,
      2π) radians. Leading axes broadcast. At a pole, north is the limit along
      longitude 0: at the north pole it points toward longitude 180° and east
      toward 90°. *)

  val ptree : unit -> 'f t Nx.Ptree.t
  (** [ptree ()] is the structure of directions: the frame as {!Frame.walk}
      reports it, then the vectors as a tensor of a fixed type, at the root
      path. {!Nx.Ptree.cast} keeps them float64, and {!Nx.Ptree.map}, placement
      and rune's transformations treat them as any tensor. *)
end
