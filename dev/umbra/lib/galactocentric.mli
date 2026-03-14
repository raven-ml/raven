(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Galactocentric Cartesian coordinates.

    Converts celestial positions with distances to a right-handed Cartesian
    frame centered on the Galactic center. The x-axis points from the Sun toward
    the Galactic center (l=0, b=0), y in the direction of Galactic rotation, z
    toward the North Galactic Pole.

    Coordinates go through the Galactic frame (ICRS {e ->} Galactic {e ->}
    heliocentric Cartesian {e ->} Galactocentric). The Galactic center position
    is defined by the IAU Galactic coordinate system (l=0, b=0).

    Default parameters follow
    {{:https://ui.adsabs.harvard.edu/abs/2018A%26A...615L..15G}GRAVITY
     Collaboration (2018)} for the Galactic center distance.

    {[
      let star =
        Coord.of_radec
          ~ra:(Unit.Angle.of_deg (Nx.create f64 [| 1 |] [| 266.0 |]))
          ~dec:(Unit.Angle.of_deg (Nx.create f64 [| 1 |] [| -29.0 |]))
      in
      let gc =
        Galactocentric.of_coord
          ~distance:(Unit.Length.of_kpc (Nx.create f64 [| 1 |] [| 8.0 |]))
          star
      in
      let x_kpc = Nx.item [ 0 ] (Unit.Length.in_kpc (Galactocentric.x gc))
    ]} *)

(** {1:coords Coordinates} *)

type t
(** The type for Galactocentric Cartesian positions. *)

val x : t -> Unit.length Unit.t
(** [x t] is the x coordinate (toward the Galactic center). *)

val y : t -> Unit.length Unit.t
(** [y t] is the y coordinate (direction of Galactic rotation). *)

val z : t -> Unit.length Unit.t
(** [z t] is the z coordinate (toward the North Galactic Pole). *)

(** {1:converting Converting} *)

val of_coord :
  ?galcen_distance:Unit.length Unit.t ->
  ?z_sun:Unit.length Unit.t ->
  distance:Unit.length Unit.t ->
  Coord.t ->
  t
(** [of_coord ~distance c] converts celestial coordinates [c] with [distance] to
    Galactocentric Cartesian. Not differentiable (scalar-level trigonometry).

    [galcen_distance] is the Sun-GC distance (defaults to 8.122 kpc, GRAVITY
    Collaboration 2018). [z_sun] is the Sun's height above the Galactic midplane
    (defaults to 0.0208 kpc). *)

val to_coord :
  ?galcen_distance:Unit.length Unit.t ->
  ?z_sun:Unit.length Unit.t ->
  t ->
  Coord.t * Unit.length Unit.t
(** [to_coord t] converts Galactocentric Cartesian coordinates [t] back to ICRS
    celestial coordinates and a distance. Not differentiable (scalar-level
    trigonometry).

    [galcen_distance] defaults to 8.122 kpc. [z_sun] defaults to 0.0208 kpc. *)
