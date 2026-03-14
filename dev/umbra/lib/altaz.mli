(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Altitude-azimuth (horizontal) coordinates.

    Converts celestial coordinates to local horizon coordinates for a given
    observer location and time. Uses IAU 2006 precession (Capitaine et al. 2003)
    and the Earth Rotation Angle.

    {b Warning.} Nutation and polar motion are omitted. Atmospheric refraction
    can be applied via {!refraction} or the [~refraction] parameter of
    {!of_coord}. Accuracy is ~1 arcminute for dates within a few centuries of
    J2000.0.

    {[
      let obs = Altaz.make_observer ~lat:(Unit.Angle.deg 28.7624) ~lon:(Unit.Angle.deg (-17.8792)) () in
      let t = Time.of_iso "2024-06-21T22:00:00" in
      let vega =
        Coord.of_radec
          ~ra:(Unit.Angle.deg 279.2347)
          ~dec:(Unit.Angle.deg 38.7837)
      in
      let hz = Altaz.of_coord ~obstime:t ~observer:obs vega in
      let alt_deg = Nx.item [] (Unit.Angle.in_deg (Altaz.alt hz))
    ]} *)

(** {1:observer Observer} *)

type observer
(** The type for a ground-based observer location. *)

val make_observer :
  lat:Unit.angle Unit.t ->
  lon:Unit.angle Unit.t ->
  ?height:Unit.length Unit.t ->
  unit ->
  observer
(** [make_observer ~lat ~lon ?height ()] is an observer at geodetic latitude
    [lat], longitude [lon], and elevation [height] above the reference
    ellipsoid. [lon] is positive East. [height] defaults to sea level.

    [height] is stored for forward compatibility but does not yet affect
    coordinate transforms. *)

val observer_height : observer -> Unit.length Unit.t
(** [observer_height obs] is the observer's elevation above the reference
    ellipsoid. *)

(** {1:coords Horizontal coordinates} *)

type t
(** The type for altitude-azimuth coordinates. Azimuth is measured from North
    through East. *)

val alt : t -> Unit.angle Unit.t
(** [alt t] is the altitude (elevation above the horizon). *)

val az : t -> Unit.angle Unit.t
(** [az t] is the azimuth (North = 0, East = 90 deg). *)

(** {1:derived Derived quantities} *)

val airmass : t -> Nx.float64_t
(** [airmass hz] is the airmass at the altitude of [hz] using the Pickering
    (2002) formula. Well-behaved from zenith to horizon. Not differentiable
    (operates on float-level altitude values). *)

(** {1:refraction Atmospheric refraction} *)

val refraction : t -> Unit.angle Unit.t
(** [refraction hz] is the atmospheric refraction correction at the geometric
    altitude of [hz], using the Bennett (1982) formula. The correction is
    positive (objects appear higher than their geometric position). Returns zero
    for altitudes below -1°.

    Not differentiable (scalar-level trigonometry). *)

(** {1:converting Converting} *)

val of_coord :
  ?refraction:bool ->
  obstime:Time.utc Time.t ->
  observer:observer ->
  Coord.t ->
  t
(** [of_coord ~obstime ~observer c] converts celestial coordinates [c] to
    altitude-azimuth for [observer] at [obstime]. Applies IAU 2006 precession to
    move from ICRS to the mean equator of date.

    When [refraction] is [true], the Bennett (1982) atmospheric refraction
    correction is applied to the computed altitude. [refraction] defaults to
    [false].

    Not differentiable (scalar-level rotation matrices). *)

val to_coord : obstime:Time.utc Time.t -> observer:observer -> t -> Coord.t
(** [to_coord ~obstime ~observer t] converts altitude-azimuth coordinates [t]
    back to ICRS celestial coordinates for [observer] at [obstime]. Not
    differentiable (scalar-level rotation matrices). *)
