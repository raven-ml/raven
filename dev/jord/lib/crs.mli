(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Coordinate reference systems on WGS 84, and the transforms between them.

    A {e coordinate reference system} (CRS) says what a pair of numbers means on
    the Earth. jord knows the systems Earth-observation data on WGS 84 use.
    [Lonlat] is geographic longitude and latitude in degrees, in that order:
    OGC's CRS84, which is EPSG:4326 with its axes swapped. [Utm] is WGS 84 /
    UTM, easting and northing in metres. It is the transverse Mercator
    projection about the {e central meridian} [6 zone - 183] degrees east, with
    scale 0.9996 on it, false easting 500000 m, and false northing 0 in the
    northern hemisphere or 10000000 m in the southern one. These are EPSG:326zz
    and EPSG:327zz for zone [zz]. No other system or datum can be stated, so a
    reader refuses data in another system rather than treat it as one of these.

    A {e point} is the last axis of a float64 tensor, [[...; 2]], in its
    system's axis order: (longitude, latitude) or (easting, northing). It is the
    order of GeoTIFF's model space and of GeoJSON's positions.

    {b Accuracy.} Transverse Mercator is Krüger's series to order n⁶ as Karney,
    "Transverse Mercator with an accuracy of a few nanometers" (J. Geodesy 85,
    2011), eqs. (7)-(22), (35) and (36), gives it, summed by Clenshaw's
    recurrence. The inverse takes Newton's step of eq. (21) twice. The forward
    map and the inverse are within 5 nm of the exact projection for points
    within 3900 km of the central meridian, the bound Karney states (§6), and
    within 6 nm across the transforms' domain ({!section-transforms}). That
    domain is 40° of longitude from the central meridian, the range IOGP's
    Guidance Note 7-2 (§3.2.3.1) states for Krüger's formulas, and reaches 4450
    km on the equator. Karney's published test set bounds both. Float32 would
    err by metres, so the transforms take float64 only. On a device set without
    float64, such as Metal's, the caller places the points on the host. *)

(** The type for coordinate reference systems. [zone] is in [1..60]. *)
type t = private Lonlat | Utm of { zone : int; north : bool }

val lonlat : t
(** [lonlat] is [Lonlat]. *)

val utm : zone:int -> north:bool -> t
(** [utm ~zone ~north] is WGS 84 / UTM zone [zone], northern hemisphere where
    [north].

    Raises [Invalid_argument] unless [1 <= zone <= 60]. *)

val utm_at : lon:float -> lat:float -> t
(** [utm_at ~lon ~lat] is the UTM zone holding the point, by Guidance Note 7-2's
    formula (§3.2.3.2): zone [⌊(lon + 180) / 6⌋ mod 60 + 1], northern where
    [lat >= 0]. It applies no grid-zone exception (32V, 31X to 37X); a caller
    that wants them applies them.

    Raises [Invalid_argument] if [lon] is not finite or [lat] is not in \[-90,
    90\]. *)

val epsg : t -> int
(** [epsg c] is [c]'s EPSG code: 4326, 326zz or 327zz. *)

val of_epsg : int -> (t, string) result
(** [of_epsg code] is the system of [code], [lonlat] for 4326. [Error] names the
    code for any code other than 4326, 32601 to 32660 and 32701 to 32760. *)

val of_wkt : string -> (t, string) result
(** [of_wkt s] is the system that the WKT 1 text [s] (a shapefile's [.prj])
    states. [s] is parsed by the grammar of OGC 18-010r7's Annex C. It names a
    system only when its whole tree equals one of two producers' WKT for
    EPSG:4326, 326zz or 327zz: Esri's, as its Projection Engine's documentation
    publishes it ([GEOGCS["GCS_WGS_1984", ...]],
    [PROJCS["WGS_1984_UTM_Zone_33N", ...]]), which GDAL writes to [.prj] files;
    or OGC's, as PROJ writes it with EPSG's names, axes and identifiers
    ([GEOGCS["WGS 84", ...]], [PROJCS["WGS 84 / UTM zone 33N", ...]]). Keywords
    and quoted names compare without case, and numbers as numbers, so [15]
    equals [15.0].

    Every other tree is an [Error] naming its [PROJCS] or [GEOGCS], since the
    meaning of a WKT 1 projection lies in its method and parameter names, which
    only the producer's vocabulary defines (Annex C.3.4). An [AUTHORITY] never
    decides alone: where an identifier and the text disagree, the text prevails
    (§7.3.3). [Error] also names the position of a syntax error. *)

val equal : t -> t -> bool
(** [equal a b] is [true] iff [a] and [b] are the same system. *)

val pp : Format.formatter -> t -> unit
(** [pp] formats a system by its EPSG name, as [WGS 84 / UTM zone 33N]. *)

(** {1:transforms Transforms}

    A transform maps points from one system to another, each point alone.
    Between two UTM zones it passes through [Lonlat]. A longitude out of
    [Lonlat] lies in \[-180, 180\].

    A point with a NaN coordinate maps to NaN. Every other point must lie in the
    domain of each projection it goes through: a latitude φ in \[-90, 90\] and a
    longitude λ within 40° of the zone's central meridian λ₀, across the
    antimeridian included, or past it by at most 12 nm along the parallel: (|λ -
    λ₀| - 40°) · a · cos φ ≤ 12 nm, [a] being WGS 84's equatorial radius and the
    angle in radians. 12 nm is twice the transforms' error, the most a round
    trip moves a point, so a point of the boundary passes both ways, and so does
    a pole, where every longitude names one point. A UTM point must be the image
    of such a point. The inverse applies the rule to the point it finds, and
    refuses a northing more than half a meridian from the equator, which is no
    point's image. *)

val transform : src:t -> dst:t -> 'd Nx.float64_t -> 'd Nx.float64_t
(** [transform ~src ~dst p] is the points [p] of [src] in [dst], computed where
    [p] lies. It is [p], unchecked, when [equal src dst].

    Raises [Invalid_argument] if [p]'s last axis is not 2, and through
    {!Nx.check}, naming the first point, where a point is outside the domain. *)
