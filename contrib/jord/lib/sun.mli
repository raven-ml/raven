(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The Sun's direction from a point on the Earth, and its distance.

    Seen from a point, the Sun lies at a {e zenith} angle from the local
    vertical, the normal to WGS 84's ellipsoid, and an {e azimuth}, measured
    clockwise from north in the horizontal plane: 90° is east. These are the
    angles a product states with its acquisition: STAC's view extension
    states [view:sun_azimuth] as this azimuth ("Measured clockwise in degrees
    (0-360)") and [view:sun_elevation] as 90° less the zenith; Sentinel-2's
    [s2:mean_solar_zenith] and [s2:mean_solar_azimuth] are the means of the
    angles its granule metadata ([MTD_TL.xml]) states at nodes 5 km apart.

    {b Method.} The Sun's direction is the U.S. Naval Observatory's
    {{:https://aa.usno.navy.mil/faq/sun_approx}Approximate Solar Coordinates}:
    with [D] the days since 2000-01-01T12:00, its mean anomaly
    [g = 357.529 + 0.98560028 D], its mean longitude
    [q = 280.459 + 0.98564736 D], its ecliptic longitude
    [L = q + 1.915 sin g + 0.020 sin 2g] (latitude 0) and the obliquity
    [e = 23.439 - 0.00000036 D], in degrees, give the direction
    [(cos L, cos e sin L, sin e sin L)] in the equator and equinox of date.
    The Earth turns that frame by Greenwich mean sidereal time, the form in
    [D] alone of USNO's
    {{:https://aa.usno.navy.mil/faq/GAST}Computing Approximate Sidereal Time},
    [18.697375 + 24.065709824279 D] hours, adapted from USNO Circular 179
    (2005), §2.6.2. At a point the direction is taken into the local frame
    of IOGP's Guidance Note 7-2 (2019), §4.1.2: "The U-axis is locally east,
    the V-axis is locally north and the W-axis is up", the W axis along the
    normal at the point's latitude. The angles are geometric and include no
    atmospheric refraction.

    {b Accuracy.} Against JPL's DE441 ephemeris, as its Horizons system
    states the Sun's apparent place seen from a point, the angles' error is
    bounded by the sum of three terms.
    - The Sun's direction in the Earth's frame: the error of USNO's formula,
      of mean sidereal time in place of apparent, and of taking the instant,
      UTC, as UT1, the time of the Earth's rotation, which UTC keeps within
      0.9 s of ({!Time}). It depends on the instant alone.
    - Parallax: the direction is taken from the Earth's centre, which moves
      it by at most the Earth's radius over the Sun's distance, 8.9″.
    - Polar motion: the Earth's pole has stayed within 0.6″ of the frame's
      since 1962 (IERS EOP 20 C04). It tilts the vertical by as much and
      turns north by up to [0.6″ / cos φ] at latitude φ.

    The module's bounds follow from them. The zenith is within 0.014°, and
    the azimuth's error times the zenith's sine, an arc on the sky, within
    [0.014° + 0.6″ / cos φ], which is 0.015° at 80° and 0.019° at 88°. The
    direction alone is within 0.86′ of DE441's apparent place from 1950 to
    2100 (USNO states "about 1 arcminute within two centuries of 2000"), and
    the distance within 1.1e-4 au. Each second of UT1 - UTC past 0.9 s adds
    0.004°, as the CGPM allows from 2035 (2022, Resolution 4). Beyond two
    centuries of 2000 USNO's formula "degrades gradually".

    Every rule is an nx formula, differentiable in the instants and the
    points except where the Sun is at the zenith or the nadir, whose azimuth
    is undefined. Instants and points are float64, since a float32 instant
    is spaced by minutes ({!Time.section-precision}). *)

val angles :
  'd Nx.float64_t -> 'd Nx.float64_t -> 'd Nx.float64_t * 'd Nx.float64_t
(** [angles t p] is the Sun's zenith, in \[0, 180\], and azimuth, in \[0,
    360\], in degrees, at the instants [t] ({!Time}) seen from the points [p]
    of {!Crs.lonlat}, [[...; 2]], computed where they lie. [t] broadcasts with
    [p]'s batch shape (its shape without the last axis), so one instant
    serves a tile and a column of instants its rows; both results have the
    broadcast shape. A point with a NaN coordinate has NaN angles. At a pole,
    north is along the point's own meridian, as Guidance Note 7-2's axes give
    it.

    Raises [Invalid_argument] if [p]'s last axis is not 2 or [t] does not
    broadcast with its batch shape, and, naming the first point, where a
    latitude is outside \[-90, 90\] or a longitude is not finite, checked by
    {!Nx.check}. *)

val distance : 'd Nx.float64_t -> 'd Nx.float64_t
(** [distance t] is the distance from the Earth to the Sun, in astronomical
    units, at the instants [t]: USNO's [1.00014 - 0.01671 cos g - 0.00014 cos
    2g]. A reflectance at the top of the atmosphere divides by the
    irradiance, which falls as the square of this distance. *)
