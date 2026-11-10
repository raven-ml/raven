(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Astronomy over {!Nx} tensors.

    Ymir gives astronomical meaning to tensors: the units astronomy measures in,
    and the models built on them. Every function is a formula of nx operations,
    {!Healpix}'s Legendre sums one [Rune.scan], so it runs batched, compiled
    and differentiated under rune's transformations with no rule of its own.

    [open Ymir] brings the unit algebra of [ymir.units] ({!Unit}, {!Quantity},
    {!Constant}, {!Codata}, {!Vocabulary}) and astronomy's units ({!Units}) into
    scope. {!Time} holds instants on astronomical time scales, and {!Earth}
    the Earth's orientation and motion at them. {!Frame} names the celestial
    frames and {!Direction} holds directions in them. {!Healpix} divides a
    frame's sphere into equal-area pixels and transforms maps on them to
    spherical harmonics. {!Transform} maps pixels to the sky, {!Grid} is a
    shape of cells seen through a transform, {!Region} is a shape placed on
    a grid's world, and {!Observation} holds data on a grid and integrates
    them over a region. {!Cosmology} is the background of a homogeneous
    expanding universe: its expansion rate, distances, volumes and times.
    {!Wcs} reads and writes a FITS header's world coordinates.

    {b Device sets.} A type that holds tensors names their device set, ['d]. A
    function's results, and the constants it makes, lie where its operands
    lie, as {!Nx.beside} places them, and a cut operand gives a result cut
    alike. What a function makes from nothing lies on the host: FITS reads,
    the cosmology presets, {!Time.of_date}, {!Constant.quantity},
    {!Frame.matrix} and {!Healpix.area}; their walks move them
    ({!Nx.Ptree.place}). {!Wcs.read} and {!Grid.pixels} take the placement of
    what they make, since a {!Transform} cannot move between sets. *)

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

(** {1:times Time scales} *)

(** Instants on astronomical time scales.

    A {e time scale} is a way of numbering instants with days and seconds.
    {e TAI}, International Atomic Time, counts SI seconds on the geoid.
    {e TT}, Terrestrial Time, is TAI + 32.184 s, the time of geocentric
    ephemerides. {e TDB}, Barycentric Dynamical Time, is the time of the
    solar system's barycentre, scaled to keep close to TT: TDB − TT is
    quasi-periodic, under 2 ms. {e UTC}, Coordinated Universal Time, is TAI less
    a whole number of seconds since 1972, changed by a {e leap second} at
    the end of a day to keep near the Earth's rotation, so that a UTC day
    lasts 86399, 86400 or 86401 SI seconds.

    An {e instant} on a scale is a {e day}, the modified Julian date
    (MJD, the Julian date less 2400000.5), and the seconds since that day's
    0h, in \[0, the day's length). On TAI, TT and TDB, the {e uniform}
    scales, every day lasts 86400 s. A UTC day lasts 86400 s plus its leap
    second, so 23:59:60.5 on 2016-12-31 is second 86400.5 of MJD 57753.
    The scale is in the instant's type, so mixing scales is a type error,
    and the functions that read a time as a count of seconds, {!Time.diff},
    {!Time.add} and {!Time.jd}, take uniform scales only.

    {b Precision.} The seconds are float64, whose ulp is at most 2⁻³⁶ s,
    1.5e-11 s, below 2¹⁷ s; a day is an int64, within 2⁵³ days of MJD 0,
    about 2.5e13 years, so that float64 holds it. A rounding of the seconds
    is within half an ulp, 7.3e-12 s, and each function states its bound.

    {b UTC before 1972.} From 1961 to 1972, TAI − UTC was a linear
    expression in the UTC date, [a + (MJD − m) r] seconds, replaced every
    few months by one that jumped by a fraction of a second. A UTC second
    then lasted [1 + r / 86400] SI seconds, and a jump lengthened or
    shortened the day before it as a leap second does: 1961-07-31 lasted
    86399.95 UTC seconds. The IAU SOFA library's UTC functions hold the same
    convention, under which a day that ends in a jump ends, in TAI,
    [jump × r / 86400] s after or before the next day's 0h, under 3.3e-9 s:
    the day's last instants overlap the next day's first, or a gap of TAI
    lies between them, which reads as the next day's 0h. SOFA's UTC begins on
    1960-01-01, with 1961's expression less the 5 ms step the Explanatory
    Supplement to the Astronomical Almanac (1992) gives; ymir's begins on
    1961-01-01, where the IERS's history of TAI − UTC, its source, begins. An
    earlier UTC instant raises [Invalid_argument].

    {b Leap seconds.} TAI − UTC is the IERS's table of it, from its history
    of UTC and Bulletin C, as of this release. A UTC instant from
    {!Time.leap_seconds_known_until} on reads with the table's last value,
    so a conversion past it is off by whole seconds if a leap second is
    announced later.

    {b TDB.} TDB − TT depends on where an event is, through the term
    v⊕ · r / c² of the Earth's barycentric velocity v⊕ and the event's
    geocentric position r, about 2 µs on the Earth's surface. {!Time}
    converts the instants of events at the geocentre: its TDB − TT is the
    series of Fairhead and Bretagnon (1990) in its full form, 787 terms, as
    the IAU SOFA library's [iauDtdb] gives it. SOFA states it within 3 ns of
    numerical integrations of the solar system (JPL's DE405) from 1950 to
    2050; outside those years its error is not stated.

    Times are tensors: a time holds a batch of instants on one scale, its
    days and seconds broadcast together, and each function works on each
    instant alone. The seconds carry their derivative through
    {!Time.convert}, {!Time.diff} and {!Time.add}. A NaN second is a NaN
    instant, which they keep NaN.

    The TDB of a UTC timestamp:

    {[
    let t = Time.of_date Time.utc (2006, 1, 15) (21, 24, 37.5)
    let tdb = Time.convert Time.tdb t
    ]} *)
module Time : sig
  (** {1:scales Scales} *)

  type 's scale
  (** The type for time scales. Each scale type has exactly one value. *)

  type !'a uniform
  (** The type of a scale whose every day lasts 86400 of its seconds. *)

  type utc
  type tai = [ `Tai ] uniform
  type tt = [ `Tt ] uniform
  type tdb = [ `Tdb ] uniform

  val utc : utc scale
  (** [utc] is Coordinated Universal Time. *)

  val tai : tai scale
  (** [tai] is International Atomic Time. *)

  val tt : tt scale
  (** [tt] is Terrestrial Time, TAI + 32.184 s. *)

  val tdb : tdb scale
  (** [tdb] is Barycentric Dynamical Time at the geocentre. *)

  val name : 's scale -> string
  (** [name s] is [s]'s canonical name: ["utc"], ["tai"], ["tt"], ["tdb"]. *)

  (** {1:instants Instants} *)

  type ('s, 'd) t
  (** The type for times: batches of instants on the scale ['s]. *)

  val v :
    's scale ->
    day:(int64, Nx.int64_elt, 'd) Nx.t ->
    second:(float, Nx.float64_elt, 'd) Nx.t Quantity.t ->
    ('s, 'd) t
  (** [v s ~day ~second] is the instants [second] after 0h of the modified
      Julian dates [day] on [s], their shapes broadcast.

      Raises [Invalid_argument] if [second] is not a time, or where it is
      infinite or outside \[0, the day's length), as in
      ["Time.v: second 86400.5 at [2] is outside \[0, 86400) of utc day
       57000"], or where a day is past 2⁵³ days from MJD 0 or a UTC day
      precedes 1961-01-01. *)

  val of_jd :
    ('a uniform as 's) scale ->
    (float, Nx.float64_elt, 'd) Nx.t ->
    (float, Nx.float64_elt, 'd) Nx.t ->
    ('s, 'd) t
  (** [of_jd s j1 j2] is the instants of Julian date [j1 + j2] on [s], their
      shapes broadcast. The sum may be split any way, as in
      [(2400000.5, mjd)]: each part is read as whole days and a fraction, and
      the instant is within 2e-11 s of the exact sum.

      Raises [Invalid_argument] where a part is not finite or the day is past
      2⁵³ days from MJD 0. *)

  val of_date :
    's scale -> int * int * int -> int * int * float -> ('s, Nx.host) t
  (** [of_date s (y, m, d) (h, mi, sec)] is the instant at [h:mi:sec] of the
      date [(y, m, d)] of the proleptic Gregorian calendar on [s], one instant
      on the host. Years are astronomical: year 0 is 1 BC. [sec] is in \[0,
      60), and in a UTC day's last minute up to the day's end: \[0, 61) on a
      day with a leap second.

      A time whose seconds since 0h round to the day's end is the next day's
      0h.

      Raises [Invalid_argument] if the date is not one, the time is not one
      of that day, the date is past 2⁵³ days from MJD 0, or a UTC date
      precedes 1961-01-01. *)

  val scale : ('s, 'd) t -> 's scale
  (** [scale t] is [t]'s scale. *)

  val day : ('s, 'd) t -> (int64, Nx.int64_elt, 'd) Nx.t
  (** [day t] is each instant's modified Julian date. *)

  val second : ('s, 'd) t -> (float, Nx.float64_elt, 'd) Nx.t Quantity.t
  (** [second t] is each instant's seconds since its day's 0h, in \[0, the
      day's length). *)

  val jd :
    ('a uniform, 'd) t ->
    (float, Nx.float64_elt, 'd) Nx.t * (float, Nx.float64_elt, 'd) Nx.t
  (** [jd t] is [(j1, j2)], each instant's Julian date [j1 + j2]: [j1] the
      Julian date of its day's 0h, exact, and [j2] the fraction of the day,
      in \[0, 1\], rounded once. *)

  val date : ('s, Nx.host) t -> (int * int * int) * (int * int * float)
  (** [date t] is [((y, m, d), (h, mi, sec))], the date and time of day of
      [t]'s one instant, as {!of_date} reads them: [of_date s (date t)] is
      [t]. A leap second is 23:59:60 and up.

      Raises [Invalid_argument] if [t] holds other than one instant, or its
      instant is NaN. *)

  val leap_seconds_known_until : (utc, Nx.host) t
  (** [leap_seconds_known_until] is 0h UTC of the day IERS Bulletin C says
      the table of TAI − UTC that ymir carries expires. *)

  (** {1:converting Converting} *)

  val convert : 'b scale -> ('a, 'd) t -> ('b, 'd) t
  (** [convert s t] is [t]'s instants on [s], converted along UTC, TAI, TT,
      TDB one step at a time:
      - UTC to TAI adds TAI − UTC, and TAI to UTC subtracts it. From 1972
        that is whole seconds: each direction rounds at most once, a round
        trip is within 7.3e-12 s, and seconds that are multiples of 2⁻³⁶ s
        convert exactly. Before 1972 a round trip is within 4e-11 s, but
        the last instants of a day that a jump lengthens come back as the
        next day's first, up to 3.3e-9 s later in UTC. A TAI just before
        such a day's next 0h, as a round trip through TT or TDB can give,
        reads as that day's last instants, up to 3.3e-9 s earlier.
      - TAI to TT adds 32.184 s, and TT to TAI subtracts it, each within
        7.3e-12 s.
      - TT to TDB adds TDB − TT at the TDB it gives, the fixed point of
        [tdb = tt + f(tdb)], which two steps reach; TDB to TT subtracts
        [f(tdb)]. A round trip is within 1.5e-11 s.

      A TAI within 4e-11 s before UTC's first instant, as a round trip from
      it can give, is that instant.

      Raises [Invalid_argument] where a UTC instant would precede
      1961-01-01. *)

  (** {1:arithmetic Arithmetic} *)

  val diff :
    ('a uniform as 's, 'd) t ->
    ('s, 'd) t ->
    (float, Nx.float64_elt, 'd) Nx.t Quantity.t
  (** [diff a b] is [a − b] in seconds of their scale, their shapes
      broadcast, within half an ulp of the result and 7.3e-12 s. *)

  val add :
    ('a uniform as 's, 'd) t ->
    (float, Nx.float64_elt, 'd) Nx.t Quantity.t ->
    ('s, 'd) t
  (** [add t span] is [t] moved by [span], a time, their shapes broadcast.
      The instant is within 7.3e-12 s of [span] after [t].

      Raises [Invalid_argument] if [span] is not a time, or where it is not
      finite or the day would pass 2⁵³ days from MJD 0. *)

  (** {1:structure Structure} *)

  val walk : ('a, 'b, 'd, 'e) Nx.Ptree.Walk.cursor -> ('s, 'd) t -> ('s, 'e) t
  (** [walk c t] walks [t]: the scale's {!name} as one case at [c]'s path, then
      the days at field ["day"] and the seconds at field ["second"], as
      tensors of a fixed type. {!Nx.Ptree.cast} keeps them, {!Nx.Ptree.place}
      moves them, and {!Nx.Ptree.map} and rune's transformations treat them
      as any tensor. *)

  val ptree : unit -> (('s, 'd) t, 'd) Nx.Ptree.t
  (** [ptree ()] is [Nx.Ptree.v walk], the structure of times. *)
end

(** {1:earth The Earth} *)

(** The Earth's orientation and motion.

    The Earth's {e orientation} at an instant is how its axes stand among
    the stars: the {e precession-nutation} of its axis, which moves the
    celestial pole and the equator, and its rotation about that axis. Ymir
    holds the IAU 2006/2000A model the IERS Conventions (2010) give in
    chapter 5: the IAU 2006 precession (Capitaine et al. 2003; Hilton et al.
    2006) in the four Fukushima-Williams angles, and the IAU 2000A nutation
    (Mathews et al. 2002), 1365 terms, with its IAU 2006 adjustments. The
    IERS states the model at the 0.2 mas level. It leaves out the free core
    nutation and the celestial pole offsets the IERS observes, together a
    few tenths of a mas, and polar motion: the pole is the {e CIP}, the
    Celestial Intermediate Pole, as the model places it.

    The rotation is the {e Earth rotation angle} (ERA), the angle about the
    CIP from the {e CIO}, the Celestial Intermediate Origin, to the
    terrestrial origin, a linear function of {e UT1}, the time of the
    Earth's rotation. UT1 is measured: the IERS publishes UT1 − UTC, within
    0.9 s, in its bulletins, and ΔT = TT − UT1 for past instants. An error
    of 1 s in UT1 turns the Earth by 15″, so an orientation takes one of
    the two from its caller ({!Earth.of_utc}, {!Earth.of_tt}).

    A {{!Earth.type-t}[t]} holds a batch of instants as TT and UT1. Its
    functions work on each instant alone and carry their derivative in the
    seconds and the offset. {!Frame.matrix_at} and {!Direction.convert} take
    an orientation to place the geocentric frames.

    The Earth's {e ephemeris}, its position and velocity about the Sun and
    the barycentre of the solar system, is the IAU SOFA library's analytic
    model iauEpv00, a simplified VSOP2000 (Moisson and Bretagnon 2001) of
    1951 terms. SOFA states it against JPL's DE405 from 1900 to 2100: the
    barycentric position within 4.6 km rms and 13.4 km at most, and the
    velocity within 1.4 mm/s rms and 4.9 mm/s at most, which bounds the
    aberration it gives at 5 µas. By 1800 and 2200 the position errors
    double, by 1500 and 2500 they grow tenfold. Positions and velocities
    are [[...; 3]] on ICRS's axes, in au and au/d of TDB.

    A bound a function states against the IAU SOFA library is against its
    functions as ERFA 2.0.1 gives them, at instants from 1600 to 2400.

    The Greenwich hour angle of a star at a UTC timestamp, with UT1 − UTC
    from IERS Bulletin A, is 2π less its longitude in TIRS:

    {[
    let e =
      Earth.of_utc (Time.of_date Time.utc (2025, 3, 20) (21, 0, 0.))
        ~dut1:(Quantity.v Unit.second (Nx.scalar Nx.float64 0.0428))

    let gha =
      Quantity.map (Nx.rsub_s (2. *. Float.pi))
        (Direction.lon (Direction.convert e Frame.tirs star))
    ]} *)
module Earth : sig
  type 'd t
  (** The type for the Earth's orientation at a batch of instants. *)

  val of_utc :
    (Time.utc, 'd) Time.t ->
    dut1:(float, Nx.float64_elt, 'd) Nx.t Quantity.t ->
    'd t
  (** [of_utc t ~dut1] is the Earth's orientation at the UTC instants [t],
      with UT1 − UTC [dut1]: UT1 is [t]'s day and seconds since 0h moved by
      [dut1] over days of 86400 s, so a leap second reads as SOFA's
      iauUtcut1 reads it. Before 1972, when UTC's seconds were not SI
      seconds, [dut1] is read against UTC's clock. iauUtcut1 adds it to the
      SI seconds since UTC's 0h instead, up to 1.3 ms apart by the day's end.
      The shapes of [t] and [dut1] broadcast.

      Raises [Invalid_argument] if [dut1] is not a time or where it is
      infinite, as in ["Earth.of_utc: dut1 at [2] is infinite"]. *)

  val of_tt :
    (Time.tt, 'd) Time.t ->
    delta_t:(float, Nx.float64_elt, 'd) Nx.t Quantity.t ->
    'd t
  (** [of_tt t ~delta_t] is the Earth's orientation at the TT instants [t],
      with ΔT = TT − UT1 [delta_t], as {!of_utc} reads its offset. It raises
      as {!of_utc} does. *)

  val tt : 'd t -> (Time.tt, 'd) Time.t
  (** [tt e] is [e]'s instants on TT. *)

  val era : 'd t -> (float, Nx.float64_elt, 'd) Nx.t Quantity.t
  (** [era e] is the Earth rotation angle in \[0, 2π) radians,
      [2π (0.7790572732640 + 1.00273781191135448 T_u)], [T_u] the UT1 Julian
      date less 2451545.0 (IERS Conventions 2010, 5.14), within 4e-15
      radians of the exact value. *)

  val gmst : 'd t -> (float, Nx.float64_elt, 'd) Nx.t Quantity.t
  (** [gmst e] is the IAU 2006 Greenwich mean sidereal time in \[0, 2π)
      radians: {!era} plus a polynomial of TT (IERS Conventions 2010, 5.32),
      within 4e-15 radians of the exact value. *)

  val gst : 'd t -> (float, Nx.float64_elt, 'd) Nx.t Quantity.t
  (** [gst e] is the Greenwich apparent sidereal time in \[0, 2π) radians,
      the angle from the true equinox to the terrestrial origin: {!era} less
      the equation of the origins, the CIO's right ascension of the equinox
      (IERS Conventions 2010, 5.30). [era e − gst e] is iauEo06a's within
      1e-15 radians. *)

  val heliocentric :
    (Time.tdb, 'd) Time.t ->
    (float, Nx.float64_elt, 'd) Nx.t Quantity.t
    * (float, Nx.float64_elt, 'd) Nx.t Quantity.t
  (** [heliocentric t] is [(p, v)], the Earth's position and velocity from
      the Sun's centre at TDB [t], each [[...; 3]] in au and au/d, within
      5e-15 au and 1e-16 au/d of iauEpv00's. *)

  val barycentric :
    (Time.tdb, 'd) Time.t ->
    (float, Nx.float64_elt, 'd) Nx.t Quantity.t
    * (float, Nx.float64_elt, 'd) Nx.t Quantity.t
  (** [barycentric t] is [(p, v)] as {!heliocentric} but from the
      barycentre of the solar system. *)

  val walk : ('a, 'b, 'd, 'e) Nx.Ptree.Walk.cursor -> 'd t -> 'e t
  (** [walk c e] walks [e]: its TT instants at field ["tt"] as {!Time.walk}
      walks them, then UT1's days at ["ut1_day"] and seconds at
      ["ut1_second"], as tensors of a fixed type. *)

  val ptree : unit -> ('d t, 'd) Nx.Ptree.t
  (** [ptree ()] is [Nx.Ptree.v walk], the structure of orientations. *)
end

(** {1:frames Frames and directions} *)

(** Celestial frames.

    A {e frame} is a celestial coordinate system: an orientation of three axes.
    Its {e orientation} [R_f] is the matrix taking ICRS components to frame
    [f]'s, [v_f = R_f · v_icrs]. Frames come in two families. A {e fixed}
    frame is one whose orientation from ICRS is constant: the catalogue
    frames, whose directions are as seen from the barycentre of the solar
    system. A {e geocentric} frame's directions are as seen from the
    Earth's centre: the GCRS, whose axes are ICRS's, and the frames of date,
    whose axes follow the Earth's orientation ({!Earth}) and so depend on
    the instant. Light that reaches the geocentre has been deflected by the
    Sun and arrives from a direction aberrated by the Earth's motion, so a
    direction to a source changes between the families by more than a
    rotation: {!Direction.convert}.

    The frames of date are the {e intermediate} frames of the IERS
    Conventions (2010): each has the CIP as its pole. CIRS, the Celestial
    Intermediate Reference System, takes the CIO as its origin; TIRS, the
    Terrestrial Intermediate Reference System, takes the terrestrial origin,
    so it turns with the Earth; the true equator and equinox of date takes
    the equinox.

    Each frame type has exactly one value, so two values of one type are one
    frame and a frame mismatch is a type error. A fixed frame's type is an
    instance of {!Frame.fixed} and a geocentric frame's of
    {!Frame.geocentric}: {!Frame.matrix} and {!Direction.rotate} take any
    two fixed frames, and no other. The users of a frame write its type, as in
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

  type !'a geocentric
  (** The type of a frame whose directions are as seen from the Earth's
      centre. *)

  type gcrs = [ `Gcrs ] geocentric
  type cirs = [ `Cirs ] geocentric
  type true_of_date = [ `True_of_date ] geocentric
  type tirs = [ `Tirs ] geocentric

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
  (** [supergalactic] is supergalactic coordinates, defined on {!val-galactic}:
      the north supergalactic pole at Galactic (47.37°, +6.32°), the origin at l
      = 137.37°. *)

  val gcrs : gcrs t
  (** [gcrs] is the Geocentric Celestial Reference System: ICRS's axes at the
      Earth's centre. *)

  val cirs : cirs t
  (** [cirs] is the Celestial Intermediate Reference System of date: from GCRS
      by the IAU 2006/2000A precession-nutation, which takes the pole to the
      CIP, and the CIO locator [s], which places the CIO (IERS Conventions
      2010, 5.4.4). *)

  val true_of_date : true_of_date t
  (** [true_of_date] is the true equator and equinox of date: from GCRS by
      the IAU 2006/2000A bias, precession and nutation, the classical,
      equinox based form of {!val-cirs}'s pole (IERS Conventions 2010,
      5.4.4). It is CIRS turned about the CIP by the equation of the
      origins. *)

  val tirs : tirs t
  (** [tirs] is the Terrestrial Intermediate Reference System: CIRS turned
      about the CIP by the Earth rotation angle (IERS Conventions 2010,
      5.4.2), the Earth's frame less polar motion. A direction's longitude
      in it is minus its Greenwich hour angle. *)

  val matrix : 'a fixed t -> 'b fixed t -> (float, Nx.float64_elt, Nx.host) Nx.t
  (** [matrix a b] is [M], [[3; 3]] on the host, with [v_b = M · v_a]: for
      quantities that are not directions, such as a velocity ([M v]) or a
      covariance ([M Σ Mᵀ]).

      [M] is [R_b · R_aᵀ]. [matrix a a] is the identity, [matrix icrs b] is
      [R_b] and [matrix a icrs] is [R_aᵀ]; another pair's entries are each
      computed with one [Float.fma] chain, so [matrix b a] is the exact
      transpose of [matrix a b]. *)

  val matrix_at : 'd Earth.t -> 'a t -> 'b t -> (float, Nx.float64_elt, 'd) Nx.t
  (** [matrix_at e a b] is [M], [[...; 3; 3]] beside and across [e]'s
      instants, with [v_b = M · v_a] at each instant, for any two frames, as
      {!matrix} is for fixed ones: [R_b · R_aᵀ], with GCRS's axes ICRS's. A
      direction to a source converts by it only within a family
      ({!Direction.convert}).

      For two fixed frames it is {!matrix}'s entries. Between the frames of
      date it is the rotation about the CIP by the difference of their
      origins' angles, which the series of the precession-nutation give
      only for the equinox. Its entries from GCRS are within 4e-16 of
      iauPnm06a's to the true equator and equinox, and within 5e-16 of
      iauC2i06a's to CIRS. Each matrix is a rotation within 1e-15 per entry,
      and [matrix_at e b a] is [matrix_at e a b] transposed within 2e-16. *)

  val name : 'f t -> string
  (** [name f] is [f]'s canonical name: ["icrs"], ["fk5_j2000"], ["galactic"],
      ["ecliptic_j2000"], ["supergalactic"], ["gcrs"], ["cirs"],
      ["true_of_date"], ["tirs"]. It is the frame's identity in keys,
      text and files. *)

  val pp : Format.formatter -> 'f t -> unit
  (** [pp] prints {!name}. *)

  val walk : ('a, 'b, 'd, 'e) Nx.Ptree.Walk.cursor -> 'f t -> 'f t
  (** [walk c f] reports {!name} as one case at [c]'s path and is [f]. A frame
      holds no tensors. *)
end

(** Directions in a frame.

    A direction is a frame and float64 vectors [[...; 3]], each read as the ray
    it spans. A unit norm cannot be a property of the type: [jvp] tangents,
    [grad] cotangents and {!Nx.Ptree.map} build directions of any norm. So the
    contract has two halves.

    - {e Constructors return unit vectors.} {!Direction.lonlat} and
      {!Direction.of_xyz} give [|xyz| = 1] within 3 ulp, and {!Direction.rotate}
      preserves the norm within 3 ulp.
    - {e Readers read the ray.} {!Direction.lon}, {!Direction.lat},
      {!Direction.separation} and {!Direction.position_angle} read every finite
      row as the direction it points in. A positive multiple of a row reads the
      same up to the rounding of the multiple, and bit for bit when the multiple
      is a power of two that rounds no component, subnormal ones included. A NaN
      component gives NaN, and an infinite one raises [Invalid_argument].
      {!Direction.rotate} and {!Direction.xyz} are linear.

    So a sum of two directions read as a direction is their chordal midpoint.

    {b Derivatives.} Every function has its analytic derivative wherever it is
    differentiable. Where it has none it gives a stated value with derivative 0,
    never NaN:
    - {!Direction.lon} at a pole ([x = y = 0]): 0.
    - {!Direction.lat} at a pole: ±π/2; at a zero row: 0.
    - {!Direction.separation} where [a × b = 0], that is coincident or
      antipodal rows or a zero row: π if [a · b < 0], else 0.
    - {!Direction.position_angle} at the same points: 0. With respect to [a] at
      a pole, its derivative is 0.

    At [lat]'s poles and [separation]'s 0 and π the function behaves like [|x|],
    and 0 is a subgradient; the other points are discontinuities. Near a pole
    the derivative of [lon] grows as 1/ρ, ρ the distance from the axis.

    Every angle in or out is a float64 quantity. Indexing and reshaping go
    through {!Direction.ptree}:

    {[
    let near =
      Nx.Ptree.map (Direction.ptree ())
        (fun _ x -> Nx.take ~axis:0 indices x)
        stars
    ]} *)
module Direction : sig
  type ('f, 'd) t
  (** The type for directions in frame ['f]: float64 vectors [[...; 3]], each
      read as the ray it spans. *)

  val lonlat :
    'f Frame.t ->
    lon:(float, Nx.float64_elt, 'd) Nx.t Quantity.t ->
    lat:(float, Nx.float64_elt, 'd) Nx.t Quantity.t ->
    ('f, 'd) t
  (** [lonlat f ~lon ~lat] is the unit vectors
      [(cos b cos l, cos b sin l, sin b)] at longitude [l] and latitude [b],
      their batch axes broadcast. Every finite angle names a direction: latitude
      100° at longitude 0° is latitude 80° at longitude 180°. A NaN angle gives
      a NaN row.

      Raises [Invalid_argument] if [lon] or [lat] is not an angle, or where an
      angle is infinite, as in ["Direction.lonlat: lat at [3] is infinite"]. *)

  val of_xyz : 'f Frame.t -> (float, Nx.float64_elt, 'd) Nx.t -> ('f, 'd) t
  (** [of_xyz f v] is [v / |v|] along [v]'s last axis. A NaN row stays NaN. Its
      derivative is the projection onto the tangent plane.

      Raises [Invalid_argument] if the last axis does not have 3 elements, or
      where a vector is infinite or zero, as in
      ["Direction.of_xyz: the vector at [17] is zero and names no direction"].
  *)

  val frame : ('f, 'd) t -> 'f Frame.t
  (** [frame d] is [d]'s frame. *)

  val xyz : ('f, 'd) t -> (float, Nx.float64_elt, 'd) Nx.t
  (** [xyz d] is [d]'s vectors, [[...; 3]]: unit vectors for values the
      constructors build. *)

  val lon : ('f, 'd) t -> (float, Nx.float64_elt, 'd) Nx.t Quantity.t
  (** [lon d] is the longitude in \[+0, 2π) radians, [atan2 y x]. It is 0 at a
      pole, so a function that needs a meridian at a pole takes the meridian of
      longitude 0. *)

  val lat : ('f, 'd) t -> (float, Nx.float64_elt, 'd) Nx.t Quantity.t
  (** [lat d] is the latitude in \[−π/2, π/2\] radians, [atan2 z ρ] with
      [ρ = √(x² + y²)]. *)

  val rotate : ('b Frame.fixed as 'g) Frame.t -> ('a Frame.fixed, 'd) t -> ('g, 'd) t
  (** [rotate g d] is [d] in frame [g]: [Frame.matrix (frame d) g] times each
      vector. It is linear in [d], within [1.5 · 2⁻⁵² · |v|] of the exact
      product per component, and rounds the same eagerly and compiled. *)

  val convert : 'd Earth.t -> 'g Frame.t -> ('f, 'd) t -> ('g, 'd) t
  (** [convert e g d] is [d] in frame [g] at [e]'s instants, their batch
      axes broadcast. Within a family it is {!Frame.matrix_at} times each
      vector. From a fixed frame to a geocentric one it reads [d] as the
      barycentric direction to a source at infinite distance and gives the
      geocentric one as iauAtciq does without proper motion or parallax:
      deflected by the Sun (iauLdsun), aberrated (iauAb), then turned onto
      [g]'s axes. Within
      about 5′ of the Sun's centre the deflection falls linearly to 0 at
      the centre; the planets' is left out. The other way it inverts that
      model: [convert e f (convert e g d)] is [d] within 1e-15.

      Between the families it reads rows as rays and returns unit vectors,
      within 2e-15 of SOFA's; a zero row gives NaN.

      Raises [Invalid_argument] where a row has an infinite component, as in
      ["Direction.convert: the vector at [3] is infinite"]. *)

  val separation : ('f, 'd) t -> ('f, 'd) t -> (float, Nx.float64_elt, 'd) Nx.t Quantity.t
  (** [separation a b] is the angle between the rays, in \[0, π\] radians.
      Leading axes broadcast. It is [atan2 |a × b| (a · b)], each component of
      [a × b] within 2 ulp through a fused multiply-add and lifted by a power
      of two where its square would underflow, so it is within 1.3e-15
      relative at any separation above float64's least normal, whatever the
      rows' norms. *)

  val position_angle : ('f, 'd) t -> ('f, 'd) t -> (float, Nx.float64_elt, 'd) Nx.t Quantity.t
  (** [position_angle a b] is [b]'s bearing from [a], east of north, in \[+0,
      2π) radians. Leading axes broadcast. At a pole, north is the limit along
      longitude 0: at the north pole it points toward longitude 180° and east
      toward 90°. It reads [a × b] as {!separation} does, east [|a| (a × b)_z]
      and north [((a × b) × a)_z], so it is within 2.1e-15 radians at any
      separation above float64's least normal, whatever the rows' norms. *)

  val walk : ('a, 'b, 'd, 'e) Nx.Ptree.Walk.cursor -> ('f, 'd) t -> ('f, 'e) t
  (** [walk c d] walks [d]: the frame as {!Frame.walk} reports it, then the
      vectors as a tensor of a fixed type, at [c]'s path. {!Nx.Ptree.cast}
      keeps them float64, {!Nx.Ptree.place} moves them, and {!Nx.Ptree.map}
      and rune's transformations treat them as any tensor. *)

  val ptree : unit -> (('f, 'd) t, 'd) Nx.Ptree.t
  (** [ptree ()] is [Nx.Ptree.v walk], the structure of directions. *)
end

(** {1:healpix HEALPix} *)

(** HEALPix pixels and spherical harmonics.

    HEALPix, the Hierarchical Equal Area isoLatitude Pixelization (Górski et
    al. 2005, ApJ 622, 759), divides a frame's sphere into 12 base pixels and
    each of those into [nside × nside] pixels, [nside] a power of two from 1
    to 2²⁹. Its [N = 12 nside²] pixels have one area, π / (3 nside²), and
    centres on [4 nside - 1] rings of constant latitude, equally spaced along
    each. With [z = cos θ = sin(lat)], the rings of the polar caps, |z| >
    2/3, hold 4, 8, … pixels from the pole, and the others [4 nside].

    Two {e schemes} number the pixels from 0 to [N - 1]. {!Healpix.Ring}
    counts eastward along each ring, north to south. {!Healpix.Nested} gives
    base pixel [f] the indices from [f nside²], each quarter of it a quarter
    of those, and so on, so an index divided by 4 is that of the pixel
    holding it at [nside / 2]. An index is an [int64]; -1 names no pixel,
    and each function gives -1 or a NaN direction where it reads -1. A map's
    last axis holds one value per pixel in one scheme's order.

    {b Locating directions.} A direction farther than 2e-14 radians from
    every edge gets the pixel holding it, at any [nside]; a nearer one gets
    one of the pixels within 2e-14 radians of it. Both schemes name the same
    pixel, bit for bit, as {!Healpix.renumber} maps them.

    {b Spherical harmonics.} A map [f] band-limited at [lmax] is
    [f(θ, φ) = Σ_(l ≤ lmax) Σ_(|m| ≤ l) a_lm Y_lm(θ, φ)] with the
    orthonormal [Y_lm(θ, φ) = λ_lm(cos θ) e^(imφ)], [λ_lm] with the
    Condon–Shortley phase (-1)^m. A real map has
    [a_l(-m) = (-1)^m conj a_lm], so its coefficients are those of [m ≥ 0],
    a complex128 tensor [[...; lmax + 1; lmax + 1]] with [a_lm] at [[l; m]]
    and 0 at [m > l]. The λ_lm come from the recurrence in [l] that is stable
    upward, with an exponent of its own that keeps λ_mm from underflowing.

    {!Healpix.synthesis} sums at pixel centres. {!Healpix.analysis} takes
    the quadrature [a_lm ≈ (4π / N) Σ_p f(p) conj Y_lm(p)], then [iterations]
    Jacobi steps [a ← a + analysis (f - synthesis a)], each a synthesis and an
    analysis more.

    {b Accuracy.} The recurrence rounds to about float64's precision times
    [lmax] (Reinecke & Seljebotn 2013, "Libsharp", §5.1.2). The quadrature
    is inexact. In the map's L2 norm, [Σ_l (|a_l0|² + 2 Σ_(m > 0)
    |a_lm|²)], an analysis of a map band-limited at [lmax] errs by at most
    ρ^(k+1) of it after [k] iterations, above that rounding, with ρ

    {v
    nside       4       8       16      32      64
    lmax=nside  0.0526  0.0398  0.0339  0.0310  0.0296
    2 nside     0.174   0.147   0.134   0.127   0.124
    v}

    {b Cost.} A transform takes one FFT per ring and one loop of [lmax + 2]
    steps for the Legendre sums, [O(nside lmax²)] terms, holding [O(nside
    lmax)] values per map. A compiled program holds an FFT per ring length,
    [nside] of them, and the loop once, whatever [lmax]. At large sizes a
    transform takes about 100 times as long as the HEALPix C++ library's,
    and compiled it runs no faster than eagerly.

    {b Derivatives.} Indices carry none, and centres and corners are
    constants of them. Both transforms are linear; rune differentiates them
    exactly.

    {[
    let sky = Healpix.v Frame.galactic ~nside:256 Healpix.Ring
    let pixels = Healpix.pixel sky (Direction.rotate Frame.galactic stars)
    let alm = Healpix.analysis sky ~lmax:511 ~iterations:3 map
    ]} *)
module Healpix : sig
  (** The type for pixel numberings. *)
  type scheme =
    | Ring  (** Along the rings, north to south. *)
    | Nested  (** By base pixel, then quarter by quarter. *)

  type 'f t
  (** The type for HEALPix pixelisations of frame ['f]'s sphere, an [nside] and
      a scheme. *)

  val v : 'f Frame.t -> nside:int -> scheme -> 'f t
  (** [v f ~nside s] is the pixelisation of [f]'s sphere at [nside], numbered
      by [s].

      Raises [Invalid_argument] unless [nside] is a power of two from 1 to
      2²⁹, the largest whose indices [int64] holds. *)

  val frame : 'f t -> 'f Frame.t
  (** [frame h] is [h]'s frame. *)

  val nside : 'f t -> int
  (** [nside h] is [h]'s number of divisions of a base pixel's side. *)

  val scheme : 'f t -> scheme
  (** [scheme h] is [h]'s numbering. *)

  val pixels : 'f t -> int
  (** [pixels h] is [12 nside²], [h]'s number of pixels. *)

  val area : 'f t -> (float, Nx.float64_elt, Nx.host) Nx.t Quantity.t
  (** [area h] is each pixel's solid angle, π / (3 nside²) steradians, rounded
      once, on the host. *)

  (** {1:pixels Pixels} *)

  val pixel : 'f t -> ('f, 'd) Direction.t -> 'd Nx.int64_t
  (** [pixel h d] is the index of the pixel holding each direction of [d], of
      [d]'s batch shape, within the bound above, and -1 for a NaN or zero
      vector. It reads each row as the ray it spans.

      Raises [Invalid_argument] where a vector is infinite. *)

  val centre : 'f t -> 'd Nx.int64_t -> ('f, 'd) Direction.t
  (** [centre h p] is the centre of each pixel of [p], unit vectors within
      2e-15 radians of the exact centres, and a NaN row for -1.

      Raises [Invalid_argument] where an index is outside \[-1,
      [pixels h]). *)

  val corners : 'f t -> 'd Nx.int64_t -> ('f, 'd) Direction.t
  (** [corners h p] is each pixel's four corners, [p]'s shape \@ [[4; 3]],
      north, west, south and east, counter-clockwise seen from outside the
      sphere and within 2e-15 radians as {!centre}; NaN rows for -1. The
      edges between them are the curves [z = a + b φ] between the caps and [z
      = a + b / φ²] in them.

      Raises [Invalid_argument] as {!centre} does. *)

  val neighbours : 'f t -> 'd Nx.int64_t -> 'd Nx.int64_t
  (** [neighbours h p] is each pixel's eight neighbours, [p]'s shape \@ [[8]]:
      south-west, west, north-west, north, north-east, east, south-east and
      south, as the HEALPix library orders them. Where three base pixels
      meet, at the eight points at |z| = 2/3, a pixel with a corner there has
      no neighbour across it, and its entry is -1, where {!Nx.take} reads
      zero. Every entry of -1 is -1.

      Raises [Invalid_argument] as {!centre} does. *)

  val renumber : 'f t -> scheme -> 'd Nx.int64_t -> 'd Nx.int64_t
  (** [renumber h s p] is the index in scheme [s] of each pixel whose index in
      [h]'s scheme is in [p], -1 for -1. It is exact, and [p] itself when [s]
      is [h]'s. A map reads in the other scheme through it:

      {[
      let nested = Healpix.v Frame.galactic ~nside:64 Healpix.Nested
      let p = Nx.arange Nx.int64 0 (Healpix.pixels nested) 1
      let nested_map = Nx.take (Healpix.renumber nested Healpix.Ring p) ring_map
      ]}

      Raises [Invalid_argument] as {!centre} does. *)

  (** {1:harmonics Spherical harmonics} *)

  val analysis :
    'f t ->
    lmax:int ->
    iterations:int ->
    (float, Nx.float64_elt, 'd) Nx.t ->
    (Complex.t, Nx.complex64_elt, 'd) Nx.t
  (** [analysis h ~lmax ~iterations f] is the coefficients
      [[...; lmax + 1; lmax + 1]] of each map of [f], [[...; pixels h]] in
      [h]'s scheme, by the quadrature over pixel centres refined [iterations]
      times. Its imaginary parts at [m = 0] are 0.

      Raises [Invalid_argument] if [f]'s last axis is not [pixels h], or
      [lmax] or [iterations] is negative. *)

  val synthesis :
    'f t ->
    (Complex.t, Nx.complex64_elt, 'd) Nx.t ->
    (float, Nx.float64_elt, 'd) Nx.t
  (** [synthesis h a] is the map [[...; pixels h]], in [h]'s scheme, of the
      coefficients [a], [[...; lmax + 1; lmax + 1]], summed at each pixel's
      centre. It ignores the entries at [m > l] and the imaginary parts at
      [m = 0], which no real map has.

      Raises [Invalid_argument] unless [a]'s last two axes are of one
      non-zero size. *)
end

(** {1:grids Transforms, grids and observations} *)

(** Maps between planes and the sky.

    A transform is a list of {e stages}, each one family's parameters and a
    sense, forward or inverse. Its endpoint types say what it maps:
    {!Transform.plane} points, [Frame.icrs Direction.t] directions. Composing
    appends stages and inverting reverses them; no stage is ever fused, so what
    a file stated is what a writer prints.

    A FITS celestial image stored [[NAXIS2; NAXIS1]] reads as

    {[
    axes [| 1; 0 |] ~origin:1
    >> shift crpix >> sip (a, b) >> linear cd
    >> tpv pv >> celestial Tan Frame.icrs ~pv ~native ~crval ~lonpole ~latpole
    ]}

    with {!Transform.sip} and {!Transform.tpv} present when the header has them.
    {!Transform.axes} holds raven's pixel convention (0-based centres in the
    data tensor's axis order) and FITS's (1-based, [NAXIS1] first) in one stage.

    {b Units.} Each planar family reads its input in the unit it states:
    {!Transform.axes} with an origin, {!Transform.sip}, {!Transform.linear} and
    {!Transform.scale} in {!Unit.one}, {!Transform.shift} in its offset's unit,
    {!Transform.tpv} and {!Transform.celestial} in an angle. A list in the wrong
    order raises [Invalid_argument] where a stage reads its input.

    {b Domains.} A stage is a bijection on its domain, the connected region
    about its reference point where its map does not fold. TAN's projection,
    from directions to the plane, needs native θ > 0; ARC's deprojection needs
    a distance from the reference point of at most 180°; ZPN's ends where its
    polynomial first turns. A distortion's domain is where its Jacobian's
    determinant is positive and, inverted, where Newton's method converges.
    {!Transform.apply} at a finite point outside a domain raises
    [Invalid_argument] through {!Nx.check}, naming the stage, the point and its
    distance or residual, as in
    ["Transform.apply: point 12 is 93.1 deg from the TAN reference (110.8375,
     -73.4537) deg, outside the projection's domain; Transform.covers gives the
     mask"]. NaN maps to NaN. {!Transform.covers} gives the mask for callers
    that expect points outside.

    {b Precision.} Planes are float64: every stage holds its parameters and
    computes in float64, so a header's numbers are held as the file wrote them.
    Data at another dtype sit on float64 geometry ({!Grid}, {!Region}).

    Batch axes of points and parameters broadcast. *)
module Transform : sig
  type ('a, 'b, 'd) t
  (** The type for maps from points of type ['a] to points of type ['b]. *)

  type 'd plane = (float, Nx.float64_elt, 'd) Nx.t Quantity.t
  (** The type for points [[...; n]] in a plane. Pixel coordinates are in
      {!Unit.one}, intermediate and tangent-plane coordinates in an angle. *)

  (** The type for projections, by their FITS codes. Zenithal projections put
      the native pole at the plane's origin, cylindrical ones the native point
      (0°, 0°). Each code's parameters [pv] are listed with the FITS keywords
      [PVi_m] of the latitude axis [i] that state them, and their defaults;
      angles are in degrees. *)
  type code =
    | Azp
        (** Zenithal perspective from distance μ, onto a plane tilted by γ:
            [[|μ; γ|]], [PVi_1], [PVi_2], defaults 0, 0. *)
    | Szp
        (** Slant zenithal perspective from distance μ toward (φc, θc):
            [[|μ; φc; θc|]], [PVi_1] to [PVi_3], defaults 0, 0, 90. *)
    | Tan  (** Gnomonic: great circles map to lines. *)
    | Stg  (** Stereographic: circles map to circles. *)
    | Sin
        (** Slant orthographic, along (ξ, η): [[|ξ; η|]], [PVi_1], [PVi_2],
            defaults 0, 0. *)
    | Arc
        (** Zenithal equidistant: distance from the reference is preserved. *)
    | Zpn
        (** Zenithal polynomial [R = Σ Pₘ (90° − θ)ᵐ], θ in radians: 30
            coefficients [P₀] to [P₂₉], [PVi_0] to [PVi_29], default 0. *)
    | Zea  (** Zenithal equal area. *)
    | Air
        (** Airy's minimum-error zenithal, for a boundary at θb: [[|θb|]],
            [PVi_1], default 90. *)
    | Cyp
        (** Cylindrical perspective from distance μ onto a cylinder of radius
            λ: [[|μ; λ|]], [PVi_1], [PVi_2], defaults 1, 1. *)
    | Cea
        (** Cylindrical equal area, λ the square of the cosine of the
            standard parallel: [[|λ|]], [PVi_1], default 1. *)
    | Car  (** Plate carrée: x is longitude and y latitude. *)
    | Mer  (** Mercator's: conformal. *)

  (** {1:constructors Constructors} *)

  val id : ('a, 'a, 'd) t
  (** [id] maps every point to itself. *)

  val axes : int array -> origin:int -> ('d plane, 'd plane, 'd) t
  (** [axes p ~origin] maps [x] to [x.(p.(k)) + origin] in component [k]. With
      [origin = 0] it permutes plane points of any unit; otherwise they are in
      {!Unit.one}.

      Raises [Invalid_argument] if [p] is not a permutation of [0], ...,
      [n - 1]. *)

  val shift : 'd plane -> ('d plane, 'd plane, 'd) t
  (** [shift r] maps [x] to [x - r]: the plane whose origin is [r]. Points are
      read in [r]'s unit.

      Raises [Invalid_argument] if [r] is a scalar. *)

  val linear : 'd plane -> ('d plane, 'd plane, 'd) t
  (** [linear m] maps [x], in {!Unit.one}, to [m · x] in [m]'s unit: CD, or PC
      in {!Unit.one}. [m] is [[...; n; n]].

      Raises [Invalid_argument] if [m] is not square on its last two axes. *)

  val scale : 'd plane -> ('d plane, 'd plane, 'd) t
  (** [scale d] maps [x], in {!Unit.one}, to [x.(k) · d.(k)] in component [k],
      in [d]'s unit: CDELT.

      Raises [Invalid_argument] if [d] is a scalar. *)

  val sip :
    ?seed:(float, Nx.float64_elt, 'd) Nx.t * (float, Nx.float64_elt, 'd) Nx.t ->
    (float, Nx.float64_elt, 'd) Nx.t * (float, Nx.float64_elt, 'd) Nx.t ->
    ('d plane, 'd plane, 'd) t
  (** [sip ~seed (a, b)] maps pixel offsets [(u, v)] from CRPIX, in
      {!Unit.one}, to [(u + f, v + g)] with SIP's polynomials
      [f = Σ a.(p).(q) uᵖ v^q] and [g = Σ b.(p).(q) uᵖ v^q]; [a] and [b] are
      [[...; n; n]], [n] the order plus one. Its inverse solves for each point
      with Newton's method, starting from [seed], the file's AP and BP of the
      same form, applied as [(u + ap, v + bp)], or from the point itself.

      Raises [Invalid_argument] if a matrix is not square on its last two
      axes. *)

  val tpv : ?stated:int array -> (float, Nx.float64_elt, 'd) Nx.t -> ('d plane, 'd plane, 'd) t
  (** [tpv ~stated pv] maps intermediate coordinates [(ξ, η)], in an angle, to
      [(Σ pv.(0).(k) t_k (ξ, η), Σ pv.(1).(k) t_k (η, ξ))] in degrees, over
      TPV's forty terms [t_k]: [1], [x], [y], [r], then the monomials of
      each degree from 2 to 7 in decreasing powers of [x], with [r³], [r⁵]
      and [r⁷] after degrees 3, 5 and 7, [r = √(x² + y²)]. [pv] is
      [[...; 2; 40]], the longitude axis's [PVi_k] then the latitude axis's,
      with the terms the file left out filled: 1 for [PV1_1] and [PV2_1], 0
      otherwise. [stated] lists the indices of [pv] flattened on its last
      two axes that the file gave, all by default. Its inverse solves for
      each point with Newton's method from the point itself.

      Raises [Invalid_argument] if [pv] is not [[...; 2; 40]] or [stated] not
      ascending indices below 80. *)

  val celestial :
    ?stated:int array ->
    code ->
    'f Frame.t ->
    pv:(float, Nx.float64_elt, 'd) Nx.t ->
    native:'d plane ->
    crval:'d plane ->
    lonpole:'d plane ->
    latpole:'d plane ->
    ('d plane, ('f, 'd) Direction.t, 'd) t
  (** [celestial code f ~pv ~native ~crval ~lonpole ~latpole] is the projection
      [code] with parameters [pv], [[...; m]], followed by the rotation taking
      the native point [native] (φ₀, θ₀) to [crval] (lon, lat) in [f], with the
      celestial pole at native longitude [lonpole]; [latpole] picks between two
      poles where FITS defines two. Angles are [[...; 2]] for [native] and
      [crval] and scalars for the poles, in any angle unit, as the file gives
      them. Plane points are angles. [stated] lists the indices of [pv] the file
      gave, all by default.

      The plane's origin is the projection's own reference point, (0°, 90°)
      for a zenithal code and (0°, 0°) for a cylindrical one, whatever
      [native] is. The celestial pole is solved as FITS WCS Paper II states,
      from CRVAL, (φ₀, θ₀) and LONPOLE; where two poles solve, the one nearer
      [latpole] is taken.

      Where [pv] does not define the projection or no pole solves, the stage
      raises [Invalid_argument] through {!Nx.check} where it is applied,
      naming the parameters.

      Raises [Invalid_argument] if [pv]'s last axis is not [m], if [stated] is
      not ascending indices below [m], or if an angle is not [[...; 2]] or not
      in an angle unit. *)

  val rotation :
    ('a Frame.fixed as 'f) Frame.t ->
    ('b Frame.fixed as 'g) Frame.t ->
    (('f, 'd) Direction.t, ('g, 'd) Direction.t, 'd) t
  (** [rotation f g] maps directions in [f] to the same directions in [g], as
      {!Direction.rotate} does. Its inverse is [rotation g f]. *)

  val about : ('f, 'd) Direction.t -> (('f, 'd) Direction.t, 'd plane, 'd) t
  (** [about c] maps directions to angular offsets about [c], in radians, x east
      and y north: the inverse of [celestial Arc] at [c] with LONPOLE 180°. The
      offset's norm is {!Direction.separation} from [c], and its bearing from +y
      toward +x is {!Direction.position_angle}. A disc of radius ρ about the
      origin is exactly the cap of angular radius ρ about [c]. At a pole the
      meridian is longitude 0, as {!Direction.lon} takes it. *)

  val gnomonic : ('f, 'd) Direction.t -> (('f, 'd) Direction.t, 'd plane, 'd) t
  (** [gnomonic c] is the same with [Tan]: great circles map to lines. *)

  (** {1:ops Operations} *)

  val ( >> ) : ('a, 'b, 'd) t -> ('b, 'c, 'd) t -> ('a, 'c, 'd) t
  (** [t >> u] applies [t], then [u]. *)

  val inverse : ('a, 'b, 'd) t -> ('b, 'a, 'd) t
  (** [inverse t] reverses [t]'s stages and flips each one's sense.
      [inverse (inverse t)] has [t]'s stages. *)

  val apply : ('a, 'b, 'd) t -> 'a -> 'b
  (** [apply t x] maps [x] through each stage. Directions are read as the rays
      they span and returned as unit vectors.

      Raises [Invalid_argument] where a point is outside a stage's domain, or
      where a stage reads points of the wrong unit or size. *)

  val covers : ('a, 'b, 'd) t -> 'a -> 'd Nx.bool_t option
  (** [covers t x] is [Some m], [m] where [apply t x] is defined: [false]
      outside a stage's domain and at NaN, of [x]'s batch shape. It is [None]
      for a transform with no stage, which is defined at every point. It does
      not raise at any point. *)

  val pp : Format.formatter -> ('a, 'b, 'd) t -> unit
  (** [pp] formats a transform's stages. *)

  val ptree : unit -> (('a, 'b, 'd) t, 'd) Nx.Ptree.t
  (** [ptree ()] is the structure of transforms: the number of stages, then each
      stage at its index, reporting its family, its sense and its static data
      (permutation, origin, code, frame, stated terms) and walking its
      parameters. No float is static. *)
end

(** Cells seen through a transform.

    A {e cell} is one sample's footprint, a pixel's square. A pixel grid is a
    shape and a transform from its pixel coordinates to its {e world}, ['w]:
    directions on the sky or points in a plane. Cell [(i, j)] has its centre at
    pixel coordinates [(i, j)], 0-based in the data tensor's axis order, and its
    corners at [(i ± ½, j ± ½)].

    A {e window} is a block of a grid's cells of static shape at a traced start,
    [[...; 2]], possibly batched: the grid of a stamp, or of a million stamps. A
    window keeps its {e base}, the whole image's shape, and its transform over
    the whole image's pixel coordinates, so its cells are the base's cells.
    Cells beyond the base exist in a window, mapped through the same transform.
    The {e border} of a window is its outer ring of cells.

    A cell's {e measure} is its solid angle on the sky, or its area in a plane:
    the measure of the quadrilateral of its corners mapped to the world the
    grid was built with, with great-circle edges on the sky. {!Grid.map_world}
    changes the world and keeps the measures. Measures are computed in
    float64. *)
module Grid : sig
  type ('w, 'e, 'd) t
  (** The type for grids of cells whose world is ['w], with measures at dtype
      ['e]. *)

  val cell : Unit.t
  (** [cell] is the unscoped symbol counting cells. Data per cell carry
      [cell⁻¹]: electrons per second in a cell are
      [Unit.(symbol "electron" / second / Grid.cell)]. It never converts to an
      angle; the grid's measure does that. A file's per-pixel unit, such as
      FITS's [pix⁻¹], is [cell⁻¹] when the grid is that file's; the program
      states it. *)

  val pixels :
    at:'d Nx.Placement.t ->
    shape:int array ->
    (float, 'e) Nx.dtype ->
    ('d Transform.plane, 'w, 'd) Transform.t ->
    ('w, 'e, 'd) t
  (** [pixels ~at ~shape dtype t] is the image of [shape] cells,
      [[|rows; columns|]], seen through [t]: the whole image, its own base, with
      measures at [dtype] and its window's start at [at].

      Raises [Invalid_argument] unless [shape] is two non-negative sizes. *)

  (** The type for a grid's kind. *)
  type ('w, 'e, 'd) kind =
    | Pixels : {
        base : int array;  (** The whole image's shape. *)
        shape : int array;  (** The window's shape. *)
        start : 'd Nx.int64_t;  (** The window's first cell, [[...; 2]]. *)
        transform : ('d Transform.plane, 'w, 'd) Transform.t;
            (** From the whole image's pixel coordinates. *)
      }
        -> ('w, 'e, 'd) kind

  val kind : ('w, 'e, 'd) t -> ('w, 'e, 'd) kind
  (** [kind g] is [g]'s kind and parts. *)

  val shape : ('w, 'e, 'd) t -> int array
  (** [shape g] is the shape of [g]'s cells: the data's last axes. *)

  val centres : ('w, 'e, 'd) t -> 'w
  (** [centres g] is each cell's centre in the world, [batch @ shape g] points.
      A window's centres are its base's at the same cells, bit for bit.

      Raises [Invalid_argument] as {!Transform.apply} does. *)

  val corners : ('w, 'e, 'd) t -> 'w
  (** [corners g] is each cell's four corners in the world,
      [batch @ shape g @ [4]] points, counter-clockwise in the pixel plane. *)

  val measure : ('w, 'e, 'd) t -> (float, 'e, 'd) Nx.t Quantity.t
  (** [measure g] is each cell's measure, [batch @ shape g]: in steradians on
      the sky, by Van Oosterom and Strackee's formula on corner differences,
      within about 1e-9 of a 0.031″ TAN pixel's solid angle; in the plane's unit
      squared in a plane, by the shoelace formula. *)

  val map_world : ('w, 'v, 'd) Transform.t -> ('w, 'e, 'd) t -> ('v, 'e, 'd) t
  (** [map_world t g] is [g]'s cells seen in the world [t] maps [g]'s world
      to: its transform followed by [t]. Its cells and their measures are
      [g]'s, taken in the world [g] was built with. *)

  val agree : ('w, 'e, 'd) t -> ('w, 'e, 'd) t -> 'd Nx.bool_t
  (** [agree a b] is where [a] and [b] are the same cells in the same world:
      [false] if their kinds, shapes, stages, codes, frames or units differ,
      and otherwise whether every parameter holds the same numbers, NaN equal
      to NaN, one boolean per batch element. It does not raise. Agreement is
      exact: two headers that differ in their last digit give grids that do
      not agree. *)

  val window : start:'d Nx.int64_t -> shape:int array -> ('w, 'e, 'd) t -> ('w, 'e, 'd) t
  (** [window ~start ~shape g] is the block of [shape] cells of [g] from
      [start], [[...; 2]] along [g]'s own axes. Its batch axes are [start]'s.

      Raises [Invalid_argument] unless [shape] is two non-negative sizes and
      [start]'s last axis is 2. *)

  val around : 'w -> shape:int array -> ('w, 'e, 'd) t -> ('w, 'e, 'd) t
  (** [around x ~shape g] is the [shape] block of [g]'s base centred on the cell
      holding each point [x], located through the inverse transform, with the
      extra cell of an even size on the high side. Its batch axes are [x]'s. It
      carries no derivative and does not raise at any point: a point the inverse
      does not cover gets a window wholly beyond the base, and every start is
      clamped to \[[-shape], [base]\].

      Raises [Invalid_argument] unless [shape] is two non-negative sizes. *)

  val ptree : unit -> (('w, 'e, 'd) t, 'd) Nx.Ptree.t
  (** [ptree ()] is the structure of grids: the kind, the base and window shapes
      and the dtype as static data, then the start and the transform. *)
end

(** Planar shapes placed on a grid's world.

    A region is a planar shape centred on the origin of a plane, and a
    {e placement}: the transform from a world into that plane. On the sky,
    [Transform.about c] is the plane of angular offsets about [c], where a disc
    of radius ρ is exactly the cap of angular radius ρ; [about c >> shift v] is
    a cap to |v|²/6 relative. In pixels, [Transform.shift p] places a shape at
    [p]. The shape's centre is a parameter of the placement, so a fitted
    offset's gradient flows through the placement into the overlap.

    A cell's {e weight} is the fraction of its area inside the shape, both taken
    in the shape's plane: the quadrilateral of the cell's corners mapped there,
    edges straight. The overlap is exact, computed with no data-dependent branch
    and differentiable in the shape's centre and size, and continuous in a
    polygon's vertices. A cell weighs exactly 1 when its corners are all inside
    a disc, and exactly 0 when its nearest point is at least the radius from
    the centre and it does not hold the centre. An annulus weighs the outer
    disc's weight minus the inner's, with the disc's predicates on each; an
    ellipse is a disc after the affine map that takes it there, so its
    weights' rounding grows as the square of its axis ratio, to 1e-12 of its
    area at a ratio of 1000. A cell
    weighs exactly 1 in a polygon when its corners are all inside and no edge
    crosses it, and exactly 0 when no corner is inside, no edge crosses it and
    no vertex is in it.

    A zero size weighs 0 with zero gradient. Regions have no set algebra: the
    fraction of a cell inside a union is no function of the two fractions. *)
module Region : sig
  type ('w, 'e, 'd) t
  (** The type for regions placed from world ['w], sized at dtype ['e]. *)

  val circle :
    ('w, 'd Transform.plane, 'd) Transform.t ->
    radius:(float, 'e, 'd) Nx.t Quantity.t ->
    ('w, 'e, 'd) t
  (** [circle p ~radius] is the disc of [radius] about the origin of [p]'s
      plane, in a unit of that plane. *)

  val annulus :
    ('w, 'd Transform.plane, 'd) Transform.t ->
    inner:(float, 'e, 'd) Nx.t Quantity.t ->
    outer:(float, 'e, 'd) Nx.t Quantity.t ->
    ('w, 'e, 'd) t
  (** [annulus p ~inner ~outer] is the ring between the discs of radii [inner]
      and [outer] about the origin of [p]'s plane. *)

  val ellipse :
    ('w, 'd Transform.plane, 'd) Transform.t ->
    a:(float, 'e, 'd) Nx.t Quantity.t ->
    b:(float, 'e, 'd) Nx.t Quantity.t ->
    angle:(float, 'e, 'd) Nx.t Quantity.t ->
    ('w, 'e, 'd) t
  (** [ellipse p ~a ~b ~angle] is the ellipse about the origin of [p]'s plane
      with semi-axis [a] along the direction at [angle] from the plane's +y
      toward +x, the position angle east of north on the sky, and semi-axis [b]
      across it. [a] and [b] are in a unit of the plane.

      Raises [Invalid_argument] if [angle] is not an angle. *)

  val polygon :
    (float, 'e) Nx.dtype -> ('w, 'd Transform.plane, 'd) Transform.t -> 'w -> ('w, 'e, 'd) t
  (** [polygon dtype p vertices] is the polygon whose vertices, on axis −2 of
      [vertices] in either orientation, are mapped by [p] into its plane, with
      edges straight there. In [about c], edges are straight in angular offsets
      about [c]; in [gnomonic c] they are great circles. The weights are at
      [dtype].

      Raises [Invalid_argument] if [vertices] has fewer than three points on
      axis −2. *)

  val weights : ('w, 'e, 'd) t -> ('w, 'e, 'd) Grid.t -> (float, 'e, 'd) Nx.t
  (** [weights r g] is each cell's weight in \[0, 1\], [batch @ Grid.shape g],
      the region's and the grid's batch axes broadcast. Corners map in float64
      and their offsets in the shape's plane are cast to ['e], or to float32
      for a narrower float, whose weights are then rounded once to ['e].

      Raises [Invalid_argument] through {!Nx.check} where a size is negative or
      NaN, an annulus's [inner] is above its [outer], two edges of a polygon
      cross, or a cell maps to a quadrilateral that is not convex or not
      oriented as its window's first cell. *)

  val ptree : unit -> (('w, 'e, 'd) t, 'd) Nx.Ptree.t
  (** [ptree ()] is the structure of regions: the shape's kind, its sizes and
      its placement. *)
end

(** Measured samples on a grid.

    An observation is data of shape [batch @ Grid.shape g] on a grid [g], with
    optional variance, validity and area.

    {b Validity.} [valid] is true where a sample was measured. Data and variance
    under an invalid sample are replaced by 0 wherever one arises, so a NaN or a
    sentinel there reaches no result under any derivative. Validity only
    narrows: {!Observation.restrict} conjoins, and only {!Observation.v} sets a
    mask anew. Windows always have a mask, false beyond the data. Masks carry no
    gradient.

    {b Area.} A pipeline that defined its surface brightness against a nominal
    area, such as JWST's [PIXAR_SR], or a per-pixel area map, states it as
    [area]. The grid's {!Grid.measure} stays each cell's own.

    {b The unit rule.} Data whose unit has {!Grid.cell}⁻¹ are values per cell,
    and a cell counts as one cell. Every other unit is a field over the cells,
    and a cell counts by the observation's [area] where it has one, its measure
    otherwise. *)
module Observation : sig
  type ('w, 'e, 'd) t
  (** The type for observations on a grid whose world is ['w], at dtype ['e]. *)

  val v :
    ?variance:(float, 'e, 'd) Nx.t Quantity.t ->
    ?valid:'d Nx.bit_t ->
    ?area:(float, 'e, 'd) Nx.t Quantity.t ->
    ('w, 'e, 'd) Grid.t ->
    (float, 'e, 'd) Nx.t Quantity.t ->
    ('w, 'e, 'd) t
  (** [v ?variance ?valid ?area g data] is [data], [batch @ Grid.shape g], on
      [g]. [variance] is in the data's unit squared. [valid] is true where a
      sample was measured. [area] is the measure each cell's data are per, when
      it is not the cell's own: one value per cell, its last axes [g]'s shape,
      or one for all cells.

      Raises [Invalid_argument] if [data] does not end with [g]'s shape, the
      variance's unit is not the data's squared, or [area] is not a measure of
      [g]'s cells or has another shape. *)

  val grid : ('w, 'e, 'd) t -> ('w, 'e, 'd) Grid.t
  (** [grid o] is [o]'s grid. *)

  val data : ('w, 'e, 'd) t -> (float, 'e, 'd) Nx.t Quantity.t
  (** [data o] is [o]'s data, 0 where invalid. *)

  val variance : ('w, 'e, 'd) t -> (float, 'e, 'd) Nx.t Quantity.t option
  (** [variance o] is [o]'s variance, 0 where invalid. *)

  val valid : ('w, 'e, 'd) t -> 'd Nx.bit_t option
  (** [valid o] is where [o]'s samples were measured, if known. *)

  val area : ('w, 'e, 'd) t -> (float, 'e, 'd) Nx.t Quantity.t option
  (** [area o] is the measure [o]'s data are per, when not the cells' own. *)

  val restrict : 'd Nx.bit_t -> ('w, 'e, 'd) t -> ('w, 'e, 'd) t
  (** [restrict m o] keeps a sample valid where it was valid and [m] holds. *)

  val map_world : ('w, 'v, 'd) Transform.t -> ('w, 'e, 'd) t -> ('v, 'e, 'd) t
  (** [map_world t o] is [o] on [Grid.map_world t (grid o)]: its data,
      variance, validity and area are [o]'s. *)

  (** {1:arithmetic Arithmetic}

      Sums, differences and scalings, whose variances combine exactly for
      independent samples. Products, ratios and other functions of data are
      the program's, on {!data}, with variances from [Rune.jvp]. *)

  val add : ('w, 'e, 'd) t -> ('w, 'e, 'd) t -> ('w, 'e, 'd) t
  (** [add a b] is [a]'s data plus [b]'s, in [a]'s unit, on [a]'s grid. The
      variance is the sum of both when both have one; a sample is valid where
      it is valid in both.

      Raises [Invalid_argument] if the grids do not agree ({!Grid.agree}) or
      the areas differ, naming the first difference, as in
      ["Observation.add: the grids do not agree at transform.3.matrix: element
       (0, 1) is -8.6e-06 in the first and -8.7e-06 in the second"]; numbers
      that differ raise through {!Nx.check}. Raises if [b]'s unit does not
      convert to [a]'s. *)

  val sub : ('w, 'e, 'd) t -> ('w, 'e, 'd) t -> ('w, 'e, 'd) t
  (** [sub a b] is [a]'s data less [b]'s, as {!add} combines them. *)

  val scale : (float, 'e, 'd) Nx.t Quantity.t -> ('w, 'e, 'd) t -> ('w, 'e, 'd) t
  (** [scale k o] is [o]'s data times [k], in their units' product, with the
      variance times [k²]. *)

  val window : start:'d Nx.int64_t -> shape:int array -> ('w, 'e, 'd) t -> ('w, 'e, 'd) t
  (** [window ~start ~shape o] is [o] on [Grid.window ~start ~shape (grid o)]:
      data, variance, validity and a per-cell area sliced with the grid, invalid
      beyond [o]'s cells. *)

  val around : 'w -> shape:int array -> ('w, 'e, 'd) t -> ('w, 'e, 'd) t
  (** [around x ~shape o] is [o] on [Grid.around x ~shape (grid o)], sliced as
      {!window} slices. *)

  type ('e, 'd) integral = {
    value : (float, 'e, 'd) Nx.t Quantity.t;
        (** Σ data × w × a over valid cells, [a] a cell's area for a field and
            one cell for values per cell. *)
    variance : (float, 'e, 'd) Nx.t Quantity.t option;
        (** Σ variance × w² × a², the variance of [value] for independent
            samples, when the observation has a variance. *)
    area : (float, 'e, 'd) Nx.t Quantity.t;
        (** Σ w × a over valid cells: [value / area] is the mean. *)
    coverage : (float, 'e, 'd) Nx.t;
        (** The valid cells' overlap with the region as a share of the region's
            own area, both in the region's plane: below 1 where the region
            reaches beyond the image's edge or over invalid samples, through a
            window or the whole image alike, and 0 for a region of zero size. *)
  }
  (** The type for integrals of an observation over a region, each of the
      region's and window's batch axes broadcast. *)

  val integrate : ('w, 'e, 'd) Region.t -> ('w, 'e, 'd) t -> ('e, 'd) integral
  (** [integrate r o] weighs [o]'s cells with {!Region.weights} and sums each
      term once in double-word arithmetic at ['e], with [w] the weight.

      Raises [Invalid_argument] through {!Nx.check} where a cell of the window's
      border is inside the base, the base continues beyond it, and the region is
      not exactly outside it: the window clipped the region, as in
      ["Observation.integrate: the region reaches the border of its 104x104
       window at sample (0, 51); take a larger ~shape"]. Raises as
      {!Region.weights} does. *)

  val ptree : unit -> (('w, 'e, 'd) t, 'd) Nx.Ptree.t
  (** [ptree ()] is the structure of observations: the grid, then the data,
      variance, validity and area, each present or not. *)
end

(** {1:cosmology Cosmology} *)

(** The background of a homogeneous expanding universe.

    A cosmology is one record of tensors: flat and curved ΛCDM, wCDM and
    w0waCDM, with or without radiation and massive neutrinos, are values of it.
    Every parameter is a leaf, so one compiled program and one batch serve every
    model, and a derivative or a sampler moves Ω{_ k} across 0 with no branch.

    {[
    let planck = Cosmology.planck2018 ~codata:Codata.v2022 Nx.float64
    let z = Nx.create Nx.float64 [| 4 |] [| 0.; 1.; 3.; 1100. |]
    let gyr = Unit.giga Units.julian_year
    let ages = Cosmology.age planck z |> Quantity.value gyr
    (* 13.7872 5.85157 2.14385 0.000366537 *)
    ]}

    {b Terms.} The {e scale factor} is a = 1/(1 + z). The {e Hubble rate} H(a)
    is the expansion rate, and E(a) = H(a)/H{_ 0}. A {e component} is a part of
    the energy budget: cold matter (baryons and cold dark matter), photons,
    neutrinos, dark energy and curvature; its {e density parameter} Ω{_ i}(a) is
    its density over the critical density at a. The {e Hubble distance} is
    D{_ H} = c/H{_ 0}, and χ is the line-of-sight comoving distance. A
    cosmology's {e lanes} are the broadcast shape of its leaves and of a call's
    redshifts.

    {b The expansion rate.} With Ω{_ r}(a) the radiation's part,

    {v
    a^4 E^2(a) = Ω_cb a + Ω_k a^2 + Ω_de a^4 exp(-3(1 + w0 + wa) ln a - 3 wa (1 - a))
                 + Ω_r(a)
    Ω_r(a)     = Ω_γ (1 + 7/8 (4/11)^(4/3) N_ur) + Σ_i Ω_γ 7/8 T_ncdm^4 F(y_i a)
    Ω_γ        = 32 π G σ T_CMB^4 / (3 c^3 H0^2)
    y_i        = m_i c^2 / (k_B T_ncdm T_CMB),   T_ncdm = 0.71611
    N_ur       = N_eff - k (T_ncdm / (4/11)^(1/3))^4
    F(y)       = (120 / 7 π^4) ∫_0^∞ x^2 sqrt(x^2 + y^2) / (e^x + 1) dx
    Ω_de       = 1 - Ω_cb - Ω_k - Ω_r(1)
    v}

    for k massive species. Dark energy closes the budget. The massive species
    sit at T{_ ncdm} T{_ CMB}, and each counts as (T{_ ncdm}/(4/11){^ 1/3}){^ 4}
    massless species at zero mass, so every result is continuous in each mass at
    0. G is the {!Codata} release's, rounded once to the payload's dtype. ΛCDM
    is w0 = -1, wa = 0, where the dark energy's factor is 1 exactly; flat is
    Ω{_ k} = 0, exactly; T{_ CMB} = 0 removes the photons and every neutrino,
    massive ones included.

    {b Evaluation.} Every distance and time is one Gauss–Legendre sum of
    a{^ 4}E{^ 2}, in a variable where its integrand is smooth from today to the
    big bang, over nodes fixed per dtype: 64 for distances and 56 for ages in
    float64, 24 and 20 in float32. The age is integrated from the big bang. Each
    function is a formula of nx operations: batched, compiled and differentiable
    in every parameter and in z, its derivative the sum's.
    [Rune.grad (Nx.Ptree.instantiate (module Cosmology.Structure))] returns a cosmology
    whose fields are the derivatives, each per the unit its field holds.

    {b Accuracy.} Against the exact values of the same model, each distance,
    volume and time, {!Cosmology.hubble} and {!Cosmology.critical_density} has a
    relative error below 2{^ -46} in float64 and 2{^ -20} in float32, and each
    density parameter an error below the same bound as a fraction of the budget,
    on the {e validation box}: Ω{_ cb} ∈ \[0.01, 3\], Ω{_ k} ∈ \[-4.5, 2\], w0 ∈
    \[-3, 1\], wa ∈ \[-3, 2\] with w0 + wa < 0, h ∈ \[0.4, 1\], T{_ CMB} ∈ \[0,
    3\] K, N{_ eff} ∈ \[0, 5\], masses ∈ \[0, 1\] eV, redshifts in \[0, 1100\]
    for distances, volumes and lookback times and \[0, 10{^ 6}\] for ages, and
    a{^ 4}E{^ 2} at least 0.3 of the sum of its terms' magnitudes on the range
    each function integrates. Outside the box the same rule runs with no
    promise: where the terms of a{^ 4}E{^ 2} nearly cancel, as in a universe
    whose expansion stalls and resumes, the integrand has a near-singularity no
    fixed rule resolves.

    {b Batching.} A cosmology is an operand of an elementwise function. Its
    leaves, [m_nu]'s without its last axis, broadcast with the redshifts from
    the right, and every result has the lanes' shape: leaves of shape
    [[chains; 1]] against redshifts [[n]] give [[chains; n]]. Leaves of shape
    [[k; 1]] give what [vmap] over k lanes of scalar leaves gives, row for row.

    {b The domain.} A distance, volume or time is NaN, with a zero derivative,
    where z ≤ -1, H{_ 0} ≤ 0 or a{^ 4}E{^ 2} ≤ 0 at a node of its rule;
    {!Cosmology.hubble}, {!Cosmology.density_parameter} and
    {!Cosmology.critical_density} are NaN where z ≤ -1, H{_ 0} ≤ 0 or E{^ 2}(z)
    ≤ 0. {!Cosmology.transverse} is NaN where H{_ 0} ≤ 0. A NaN parameter gives
    NaN with a NaN derivative. Every other lane is untouched, and nothing raises
    on a tensor's values.

    {b Errors.} Each function raises [Invalid_argument], eagerly or at trace, on
    static data: a dtype other than float32 and float64, as in
    ["Cosmology.hubble: float16 has no rule; the cosmology computes in float32
     and float64"]; a unit that does not convert, as in
    ["Cosmology.age: h0 is in 1e3 m s^-1, which does not convert to s^-1: their
     quotient keeps m"]; a scalar [m_nu], as in
    ["Cosmology.comoving_distance: m_nu is a scalar; its last axis lists the
     massive species ([1] for one, [0] for none)"]; and shapes that do not
    broadcast, as in
    ["Cosmology.distance_modulus: the cosmology's lanes [4] and z [1590] do not
     broadcast; for every redshift in each of 4 lanes, give leaves of shape [4;
     1] or map with Rune.vmap"]. *)
module Cosmology : sig
  type 'p t = {
    codata : Codata.t;  (** The CODATA release G is read from. *)
    h0 : 'p Quantity.t;  (** H{_ 0}, in a unit of rate. *)
    omega_cb : 'p;  (** Ω{_ cb} today: baryons and cold dark matter. *)
    omega_k : 'p;  (** Ω{_ k} today; 0 is flat. *)
    w0 : 'p;  (** Dark energy's w today. *)
    wa : 'p;  (** w(a) = w0 + wa (1 - a). *)
    t_cmb : 'p Quantity.t;  (** The photons' temperature today. *)
    n_eff : 'p;
        (** The effective number of neutrino species, massive ones included. *)
    m_nu : 'p Quantity.t;
        (** The rest energies m c{^ 2} of the massive species, on the last axis.
        *)
  }
  (** The type for cosmologies with payloads ['p]. The record holds no derived
      value: Ω{_ de}, Ω{_ γ} and Ω{_ ν} are functions of its fields. Its static
      data are the release's year, the three quantities' units and the leaves'
      shapes, the number of massive species included; everything else is a leaf.
  *)

  val walk : ('a, 'b, 'd, 'e) Nx.Ptree.Walk.cursor -> 'a t -> 'b t
  (** [walk c x] reports [codata]'s year with [Walk.int], then walks the other
      fields in order, quantities with {!Quantity.walk}. *)

  module Structure : Nx.Ptree.S with type ('a, 'd) t = 'a t
  (** [Structure] is {!walk} as a structure whose arrays are all positions:
      [Nx.Ptree.instantiate (module Cosmology.Structure)] describes a
      cosmology of tensors. *)

  val pp : Format.formatter -> (float, 'b, 'd) Nx.t t -> unit
  (** [pp] formats a cosmology's fields. *)

  (** {1:realisations Realisations}

      A realisation is a paper's flat ΛCDM fit, a starting point for record
      update: [{ planck with omega_k; w0 }]. It transcribes the paper's decimals
      and rounds each field once to the dtype: Ω{_ cb} is (ω{_ b} +
      ω{_ c})/h{^ 2} from the physical densities the paper samples, computed
      exactly. Its Ω{_ k}, w0 and wa are 0, -1 and 0. The Planck realisations
      hold T{_ CMB} = 2.7255 K (Fixsen 2009), N{_ eff} = 3.046 and one massive
      species of 0.06 eV; the WMAP ones T{_ CMB} = 2.725 K, N{_ eff} = 3.04 and
      none. Each raises [Invalid_argument] for a dtype other than float32 and
      float64. *)

  val planck2018 : codata:Codata.t -> (float, 'b) Nx.dtype -> (float, 'b, Nx.host) Nx.t t
  (** Planck 2018 VI (Planck Collaboration 2020, A&A 641, A6), Table 2,
      TT,TE,EE+lowE+lensing+BAO: H{_ 0} = 67.66, ω{_ b} = 0.02242, ω{_ c} =
      0.11933. *)

  val planck2015 : codata:Codata.t -> (float, 'b) Nx.dtype -> (float, 'b, Nx.host) Nx.t t
  (** Planck 2015 XIII (Planck Collaboration 2016, A&A 594, A13), Table 4,
      TT,TE,EE+lowP+lensing+ext: H{_ 0} = 67.74, ω{_ b} = 0.02230, ω{_ c} =
      0.1188. *)

  val planck2013 : codata:Codata.t -> (float, 'b) Nx.dtype -> (float, 'b, Nx.host) Nx.t t
  (** Planck 2013 XVI (Planck Collaboration 2014, A&A 571, A16), Table 5,
      Planck+WP+highL+BAO, best fit: H{_ 0} = 67.77, ω{_ b} = 0.022161, ω{_ c} =
      0.11889. *)

  val wmap9 : codata:Codata.t -> (float, 'b) Nx.dtype -> (float, 'b, Nx.host) Nx.t t
  (** WMAP nine-year (Hinshaw et al. 2013, ApJS 208, 19), Table 4,
      WMAP+eCMB+BAO+H{_ 0}: H{_ 0} = 69.32, ω{_ b} = 0.02223, ω{_ c} = 0.1153.
  *)

  val wmap7 : codata:Codata.t -> (float, 'b) Nx.dtype -> (float, 'b, Nx.host) Nx.t t
  (** WMAP seven-year (Komatsu et al. 2011, ApJS 192, 18), Table 1,
      WMAP+BAO+H{_ 0}, maximum likelihood: H{_ 0} = 70.4, ω{_ b} = 0.02253,
      ω{_ c} = 0.1122. *)

  val wmap5 : codata:Codata.t -> (float, 'b) Nx.dtype -> (float, 'b, Nx.host) Nx.t t
  (** WMAP five-year (Komatsu et al. 2009, ApJS 180, 330), Table 1, WMAP+BAO+SN,
      maximum likelihood: H{_ 0} = 70.2, ω{_ b} = 0.02262, ω{_ c} = 0.1138. *)

  val wmap3 : codata:Codata.t -> (float, 'b) Nx.dtype -> (float, 'b, Nx.host) Nx.t t
  (** WMAP three-year (Spergel et al. 2007, ApJS 170, 377), Table 6, WMAP+SN
      Gold: h = 0.701 and ω{_ m} = 0.1349, which the paper prints in place of
      ω{_ c}. *)

  val wmap1 : codata:Codata.t -> (float, 'b) Nx.dtype -> (float, 'b, Nx.host) Nx.t t
  (** WMAP first-year (Spergel et al. 2003, ApJS 148, 175), Table 7,
      WMAP+CBI+ACBAR+2dFGRS+Lyα: h = 0.72 and ω{_ m} = 0.133, which the paper
      prints in place of ω{_ c}. *)

  (** {1:expansion Expansion}

      Each function takes a cosmology and redshifts and returns the lanes'
      shape. *)

  val hubble :
    (float, 'b, 'd) Nx.t t -> (float, 'b, 'd) Nx.t -> (float, 'b, 'd) Nx.t Quantity.t
  (** [hubble c z] is H(z), in [h0]'s unit. *)

  (** The type for components. *)
  type component =
    | Cold_matter  (** [omega_cb]'s component. *)
    | Photons
    | Neutrinos  (** Massless and massive neutrinos, at every redshift. *)
    | Dark_energy
    | Curvature

  val density_parameter :
    (float, 'b, 'd) Nx.t t -> component -> (float, 'b, 'd) Nx.t -> (float, 'b, 'd) Nx.t
  (** [density_parameter c i z] is Ω{_ i}(z). The five sum to 1. *)

  val critical_density :
    (float, 'b, 'd) Nx.t t -> (float, 'b, 'd) Nx.t -> (float, 'b, 'd) Nx.t Quantity.t
  (** [critical_density c z] is 3H(z){^ 2}/8πG, in 3/(8π) u{^ 2} over G's unit
      for [h0] in u. *)

  (** {1:distances Distances}

      A distance is in c/u for [h0] in u. For H{_ 0} in km s{^ -1} Mpc{^ -1}
      that unit is exactly 299792.458 Mpc, and [Quantity.value mpc] applies it
      as one rounded factor. *)

  val comoving_distance :
    (float, 'b, 'd) Nx.t t -> (float, 'b, 'd) Nx.t -> (float, 'b, 'd) Nx.t Quantity.t
  (** [comoving_distance c z] is χ, along the line of sight. *)

  val transverse :
    (float, 'b, 'd) Nx.t t ->
    (float, 'b, 'd) Nx.t Quantity.t ->
    (float, 'b, 'd) Nx.t Quantity.t
  (** [transverse c d] is the transverse comoving distance across a radial
      comoving separation [d], in [d]'s unit: d S(Ω{_ k} (d/D{_ H}){^ 2}), with
      S(x) = Σ{_ n} x{^ n}/(2n+1)!, which is sinh √x/√x for x > 0 and sin
      √-x/√-x for x < 0. S is entire, so [transverse] is analytic in Ω{_ k}
      across 0, where its derivative in Ω{_ k} is d{^ 3}/(6 D{_ H}{^ 2}) and its
      value [d] exactly. The transverse comoving distance D{_ M} is
      [transverse c (comoving_distance c z)]; between two redshifts,
      [transverse c (Quantity.sub chi_s chi_l)].

      Raises [Invalid_argument] if [d] is not a length. *)

  val angular_diameter_distance :
    (float, 'b, 'd) Nx.t t -> (float, 'b, 'd) Nx.t -> (float, 'b, 'd) Nx.t Quantity.t
  (** [angular_diameter_distance c z] is D{_ M}(z)/(1 + z). *)

  val luminosity_distance :
    (float, 'b, 'd) Nx.t t ->
    ?observed:(float, 'b, 'd) Nx.t ->
    (float, 'b, 'd) Nx.t ->
    (float, 'b, 'd) Nx.t Quantity.t
  (** [luminosity_distance c ~observed z] is (1 + observed) D{_ M}(z): [z]
      places the source, [observed] is the redshift the observer measures, [z]
      by default. A supernova fit passes its heliocentric redshift as
      [observed]. *)

  val distance_modulus :
    (float, 'b, 'd) Nx.t t ->
    ?observed:(float, 'b, 'd) Nx.t ->
    (float, 'b, 'd) Nx.t ->
    (float, 'b, 'd) Nx.t
  (** [distance_modulus c ~observed z] is 5 log{_ 10}(D{_ L}/10 pc), a
      magnitude, with D{_ L} as {!luminosity_distance} gives it. It follows
      [log]'s domain: NaN where D{_ M} < 0, past a closed universe's antipode,
      and -∞ where D{_ L} = 0. *)

  (** {1:volumes Volumes}

      A volume is in (c/u){^ 3} sr{^ -1} for [h0] in u. *)

  val comoving_volume :
    (float, 'b, 'd) Nx.t t -> (float, 'b, 'd) Nx.t -> (float, 'b, 'd) Nx.t Quantity.t
  (** [comoving_volume c z] is the comoving volume within [z] per steradian,
      D{_ H}{^ 3} χ̂{^ 3} W(Ω{_ k} χ̂{^ 2}) with χ̂ = χ/D{_ H} and W(x) = (S(4x) -
      1)/(2x). *)

  val comoving_volume_element :
    (float, 'b, 'd) Nx.t t -> (float, 'b, 'd) Nx.t -> (float, 'b, 'd) Nx.t Quantity.t
  (** [comoving_volume_element c z] is dV{_ C}/dz per steradian, c
      D{_ M}{^ 2}/H. *)

  (** {1:times Times}

      A time is in 1/u for [h0] in u. *)

  val lookback_time :
    (float, 'b, 'd) Nx.t t -> (float, 'b, 'd) Nx.t -> (float, 'b, 'd) Nx.t Quantity.t
  (** [lookback_time c z] is the time light from [z] has travelled. *)

  val age :
    (float, 'b, 'd) Nx.t t -> (float, 'b, 'd) Nx.t -> (float, 'b, 'd) Nx.t Quantity.t
  (** [age c z] is the time since the big bang at [z]. *)
end

(** {1:wcs FITS world coordinates} *)

(** FITS world coordinates.

    A header's celestial description reads as the stages

    {[
    axes [| 1; 0 |] ~origin:1 (* tensor order, 0-based -> FITS, 1-based *)
    >> shift crpix
    >> sip (a, b) (* a SIP header *)
    >> linear cd (* a CD header *)
    (* or: linear pc >> scale cdelt, a PC, CROTA2 or CDELT header *)
    >> axes [| 1; 0 |] ~origin:0 (* when latitude is axis 1 *)
    >> tpv pv (* a TPV header *)
    >> celestial code frame ~pv ~native ~crval ~lonpole ~latpole
    ]}

    CRPIX, CD, PC, CDELT, SIP's coefficients and the PV terms are held as
    the header writes them, CRVAL in CUNIT (degrees when absent), and
    LONPOLE, LATPOLE and the native reference point (φ₀, θ₀) =
    ([PVi_1], [PVi_2]) on the longitude axis [i] in degrees. A keyword FITS
    leaves out takes its default: CRPIX and CRVAL 0, CDELT 1, PC the
    identity, a missing CD element or SIP coefficient 0, a projection's
    parameters those {!Transform.code} lists, (φ₀, θ₀) = (0°, 90°) for a
    zenithal projection and (0°, 0°) for a cylindrical one, LATPOLE 90°,
    and LONPOLE 180° + φ₀ when CRVAL's latitude is below θ₀, φ₀ otherwise.
    Where θ₀ = 0, CRVAL's latitude is 0 and LONPOLE is 90° from φ₀, LATPOLE
    alone fixes the celestial pole's latitude and has no default: a header
    without it is an [Error], which astropy (WCSLIB) reads with the pole at
    90°.
    CROTA2 reads as the PC matrix it implies. [PVi_3] and [PVi_4] on the
    longitude axis are LONPOLE's and LATPOLE's other spellings outside TPV; a
    header giving both spellings of one with different values is an [Error].

    {b Distortions.} SIP reads from [A_ORDER], [B_ORDER] and [A_p_q],
    [B_p_q], with [AP_ORDER], [BP_ORDER], [AP_p_q] and [BP_p_q] as the
    inverse's seed, when the CTYPEs end in [-SIP] or the primary description
    has [A_ORDER]. TPV reads from [PVi_0] to [PVi_39] on both axes when the
    CTYPEs say [TPV], or say [TAN] and the latitude axis has PV terms, as
    SCAMP writes; [PV1_1] and [PV2_1] default to 1, the other terms to 0.

    {b Frames.} The CTYPE prefix chooses the system and [RADESYS] (or
    [RADECSYS]) and [EQUINOX] (or [EPOCH]) qualify an equatorial or ecliptic
    one: absent both, ICRS; [EQUINOX] alone, FK4 below 1984.0 and FK5 from it;
    [EQUINOX] 2000.0 under FK5 when absent; [EQUINOX] ignored under ICRS.

    - [RA]/[DEC] under ICRS is {!Frame.val-icrs}, under FK5 at 2000.0
      {!Frame.val-fk5_j2000}.
    - [GLON]/[GLAT] is {!Frame.val-galactic}, [SLON]/[SLAT]
      {!Frame.val-supergalactic}.
    - [ELON]/[ELAT] under ICRS at 2000.0 is {!Frame.val-ecliptic_j2000}.

    Other systems and equinoxes are an [Error] naming what they need.

    {b Scope.} The thirteen projections of {!Transform.code}, SIP and TPV are
    read. Other projections, distortion lookup tables, the fiducial offset
    [PVi_0], [PVi_m] on the longitude axis beyond [m = 1, ..., 4] outside
    TPV, and a third WCS axis are an [Error] naming the keyword. *)
module Wcs : sig
  type keywords = {
    float : string -> (float option, string) result;
    int : string -> (int option, string) result;
    text : string -> (string option, string) result;
  }
  (** The type for a header's keywords, as decoded values. [kw.float k] is
      the real under keyword [k]: [Ok None] when [k] is absent, an [Error]
      when [k] does not decode as a real or its cards disagree. [int] is the
      same for integers, and [text] for strings, without their quotes. These
      errors pass through {!read} unchanged. With [ymir.fits]:

      {[
      let keywords h =
        {
          Wcs.float = (fun k -> Fits.Header.find Fits.Value.float k h);
          int = (fun k -> Fits.Header.find Fits.Value.int k h);
          text = (fun k -> Fits.Header.find Fits.Value.string k h);
        }
      ]} *)

  (** The type for keyword edits. With [ymir.fits]:

      {[
      let edit h = function
        | Wcs.Float (k, x) -> Fits.Header.set Fits.Value.float k x h
        | Int (k, n) -> Fits.Header.set Fits.Value.int k n h
        | Text (k, s) -> Fits.Header.set Fits.Value.string k s h
        | Remove k -> Fits.Header.remove k h
      ]} *)
  type edit =
    | Float of string * float  (** Set a keyword to a real. *)
    | Int of string * int  (** Set a keyword to an integer. *)
    | Text of string * string  (** Set a keyword to a string. *)
    | Remove of string  (** Remove a keyword. *)

  val read :
    ?alt:char ->
    at:'d Nx.Placement.t ->
    'f Frame.t ->
    keywords ->
    (('d Transform.plane, ('f, 'd) Direction.t, 'd) Transform.t, string) result
  (** [read ~alt ~at f kw] is the map from 0-based pixel indices, in tensor
      axis order, to directions in [f], from [kw]'s primary description or its
      alternate [alt] (['A'] to ['Z']), its parameters at [at]. A celestial CUNIT is [deg], [arcmin],
      [arcsec], [mas] or [rad]. It is an [Error] if [kw]'s frame is not [f],
      as in
      ["SCI: FK5 at equinox 2000.0; the caller expects icrs. Read with
       Frame.fk5_j2000."], if a keyword's value is outside what WCS allows,
      if LATPOLE is absent where it alone fixes the pole (see above), or as
      [kw] is. Its own errors start with the header's [EXTNAME].

      Raises [Invalid_argument] if [alt] is not [' '] or ['A'] to ['Z']. *)

  val write :
    ?alt:char ->
    ?window:(int * int) array ->
    ('d Transform.plane, ('f, 'd) Direction.t, 'd) Transform.t ->
    keywords ->
    (edit list, string) result
  (** [write ~alt ~window t kw] is the edits, in order, that make [kw]'s
      description [alt] spell [t], with [window], [(start, stop)] per tensor
      axis, taking each CRPIX less its axis's start. A keyword whose value
      equals [t]'s, bit for bit, gets no edit, nor does one [kw] lacks whose
      value is its default and, for a PV term, that the stage does not list
      as stated. Keywords [t] no longer spells (CD for a PC header, CROTA,
      SIP) are removed. A pole keeps the spelling [kw] gives it.
      [write (read kw) kw] is [Ok []] for every header {!read} reads, except
      a CROTA2 header, whose edits spell the PC matrix it reads as, and
      SCAMP's TAN with PV terms, whose edits spell TPV; their stages are
      equal.

      It is an [Error] naming the stage if [t] is not a list {!read} builds,
      up to an absent PC or a scale stage on its own, or if [t]'s leaves are
      batched or not finite.

      Raises [Invalid_argument] if [alt] is not [' '] or ['A'] to ['Z'], or
      [window] does not have two ranges. *)
end
