(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Coordinate reference systems, and the transforms between them.

    A {e coordinate reference system} (CRS), which jord calls a {e system},
    says what a pair of numbers means on the Earth. It is a {e datum}, which
    places longitude and latitude on the Earth over an ellipsoid, and, for a
    {e projected} system, a {e map projection} of them onto a plane, an EPSG
    method with its parameters, as ISO 19111 composes a projected system
    from a geographic one and a conversion. A {e geographic} system has no
    projection. jord knows the datums Earth-observation data use, each a
    fact with its own relation to WGS 84, so no other datum can be stated; a
    method's parameters are free numbers.

    {!lonlat} is geographic longitude and latitude on WGS 84 in degrees, in
    that order, which is OGC's CRS84, EPSG:4326 with its axes swapped.
    {!utm} is WGS 84 / UTM, easting and northing in metres of the transverse
    Mercator projection about the {e central meridian} [6 zone - 183]
    degrees east, with scale 0.9996 on it, false easting 500000 m, and false
    northing 0 in the northern hemisphere or 10000000 m in the southern one.
    These are EPSG:326zz and EPSG:327zz for zone [zz], and NAD83 / UTM is the
    same projection on NAD83 (269zz for zones 1 to 23 north, and 9712, 3372
    and 3373 for 24, 59 and 60 north).

    A {e point} is the last axis of a float64 tensor, [[...; 2]], in its
    system's axis order: (longitude, latitude) or (easting, northing), whatever
    order a registry gives the system. It is the order of GeoTIFF's model
    space and of GeoJSON's positions. *)

(** {1:datums Datums}

    WGS 84 is the ensemble of its realisations, which agree to 2 m (EPSG 4326's
    ensemble accuracy), and a WGS 84 coordinate means a point to within that.
    Each other datum takes one step to WGS 84 that changes no coordinate, a
    {e null} step ({!section-steps}). A transform between two systems on one
    datum takes no step, and one between two datums takes both of theirs. A
    reader refuses data on any other datum, so that no data is read as one of
    these. *)

(** The type for datums. Each but WGS 84 is on GRS 1980 (EPSG 7019), with
    semi-major axis 6378137 m and inverse flattening 298.257222101, and
    each constructor's doc gives its step ({!section-steps}). *)
type datum =
  | Wgs84
      (** WGS 84, EPSG's datum ensemble 6326, on its ellipsoid (EPSG 7030):
          semi-major axis 6378137 m and inverse flattening 298.257223563. *)
  | Nad83
      (** NAD83, EPSG datum 6269. Its step is EPSG's transformation 1188,
          4 m, for extent 1325, Canada and the United States' conterminous
          states and mainland Alaska (box 23.81°N to 86.46°N, 172.54°W to
          47.74°W). In the Aleutians and Hawaii EPSG states other
          operations, 1251 and 1252, from which the null step differs by
          less than their own accuracy. NAD83's better operations, NADCON
          and NTv2, are grid files, which jord does not read. *)
  | Euref_fin
      (** ETRS89-FIN \[EUREF-FIN\], ETRS89's realisation in Finland and the
          datum of TM35FIN (3067), EPSG datum 1391. Its step is
          transformation 10701, 1 m, for extent 1095, Finland (box 58.84°N
          to 70.09°N, 19.08°E to 31.59°E). *)
  | Nzgd2000
      (** NZGD2000, EPSG datum 6167. Its step is transformation 1565, 1 m,
          for extent 1175, New Zealand (box 55.95°S to 25.88°S, 160.6°E to
          171.2°W, across the antimeridian). *)
  | Gda94
      (** GDA94, EPSG datum 6283. Its step is transformation 1150, 3 m, for
          extent 4177, Australia (box 60.55°S to 8.47°S, 93.41°E to
          173.34°E). *)

(** {2:steps Steps between datums}

    A datum's step to WGS 84 is EPSG's transformation for it, which changes no
    coordinate, and so holds everywhere. EPSG states the step's accuracy for an
    area, whose bounding box it publishes, and states none outside it. A
    transform between two datums takes the source's step to WGS 84 and the
    destination's from it, and its accuracy is the sum of theirs. *)

(** {1:accuracy Accuracy}

    Each method states an error ε, and each transform's result is within ε
    of the exact image of a point within ε of its input, in metres in the
    plane for a projected point and on the ground for a geographic one. So a
    result lies within ε (1 + s) of the exact image of the input itself, s
    the largest scale of the map there, or for an inverse the largest of its
    own, which [Rune.jacfwd] of [transform] gives. Each ε holds for float64
    points as nx's kernels compute them; float32 would err by metres, so the
    transforms take float64 only. On a device set without float64, such as
    Metal's, the caller places the points on the host. *)

(** {2:accuracy_tm Transverse Mercator}

    Transverse Mercator is Krüger's series to order n⁶ as Karney,
    "Transverse Mercator with an accuracy of a few nanometers" (J. Geodesy
    85, 2011), eqs. (7)-(22), (29), (35) and (36), gives it. The forward map
    and the inverse are within 5 nm of the exact projection for points
    within 3900 km of the central meridian, the bound Karney states (§6),
    and ε is 6 nm for every point the transforms accept. They accept points
    within 40° of longitude of the central meridian ({!section-domains}),
    the range IOGP's Guidance Note 7-2 (§3.2.3.1) states for Krüger's
    formulas, which reaches 4450 km on the equator. GRS80's flattening is
    WGS 84's to 5 parts in 10⁹, so the bounds hold on it too. *)

(** {2:accuracy_albers Albers' conic}

    Albers' cone is Snyder's (USGS Professional Paper 1395, ch. 14) as
    Karney's GeographicLib evaluates it, the two parallels becoming the one
    of least scale, φ₀, with the scale k₀ there. The inverse takes the
    authalic latitude ξ, then φ by Karney's series of "On auxiliary
    latitudes" (Survey Review, 2024), eq. (A20), to order n⁶. ε is 144 u D,
    with u = 2⁻⁵³ and D the largest distance between the image of
    (lon0, φ₀) and a point of the image, 21046 km for 5070, so 336 nm, and
    beyond 10⁵ km for parallels near opposite poles, whose image grows
    without bound ({!section-accuracy_albers_limits}). *)

(** {3:accuracy_albers_limits Where Albers' bound holds}

    ε holds where the C library's [atanh], which computes the cone's
    constants, is as accurate as glibc's, since C states no bound for it.
    Near a pole the inverse's ground error grows as 1/h, h the scale along
    the meridian, which tends to 0 there. *)

(** {2:accuracy_closed Pseudo-Mercator and the sinusoid}

    Pseudo-Mercator and Sinusoidal are closed forms, with ε 10 nm and 8 nm.
    Pseudo-Mercator's forward error in the plane grows as its scale, sec φ,
    29 at 88°. *)

(** {1:projections Projections}

    Each function below makes a projection of its parameters, with [lon0]
    180 taken as -180 and each [-0.] as [0.], so that one system has one
    value. Each raises [Invalid_argument] if a parameter is not finite or
    [lon0] is not in \[-180, 180\]. *)

(** The type for map projections, an EPSG method and its parameters, angles in
    degrees and lengths in metres. A value is made by its function below,
    which checks and normalises it. *)
type projection = private
  | Transverse_mercator of {
      lat0 : float;  (** The latitude of natural origin (EPSG 8801). *)
      lon0 : float;
          (** The longitude of natural origin, the central meridian (8802). *)
      k0 : float;  (** The scale factor on the central meridian (8805). *)
      false_easting : float;  (** (8806) *)
      false_northing : float;  (** (8807) *)
    }
      (** EPSG's method 9807. The point (lon0, lat0) maps to (false_easting,
          false_northing). *)
  | Albers of {
      lat0 : float;  (** The latitude of false origin (EPSG 8821). *)
      lon0 : float;  (** The longitude of false origin (8822). *)
      lat1 : float;
          (** The northern standard parallel, whichever of EPSG's 8823 and
              8824 it is. *)
      lat2 : float;  (** The southern standard parallel. *)
      false_easting : float;  (** The easting at false origin (8826). *)
      false_northing : float;  (** The northing at false origin (8827). *)
    }
      (** EPSG's method 9822, Albers' equal-area conic. The point (lon0,
          lat0) maps to (false_easting, false_northing), and the scale is 1
          along both standard parallels. *)
  | Pseudo_mercator of {
      lon0 : float;  (** The longitude of natural origin (EPSG 8802). *)
      false_easting : float;  (** (8806) *)
      false_northing : float;  (** (8807) *)
    }
      (** EPSG's method 1024, Popular Visualisation Pseudo Mercator, the
          sphere's Mercator of radius the datum's semi-major axis a applied to
          the datum's longitude and latitude, E = FE + a (λ - λ₀) and
          N = FN + a asinh (tan φ) (IOGP Guidance Note 7-2, §3.2.1.2), which is
          therefore not conformal. Its latitude of natural origin (8801) must
          be 0. *)
  | Sinusoidal of {
      lon0 : float;  (** The central meridian. *)
      radius : float;  (** The sphere's radius R, in metres. *)
      false_easting : float;
      false_northing : float;
    }
      (** The sinusoidal projection of a sphere of radius R, applied to the
          datum's longitude and latitude, x = FE + R (λ - λ₀) cos φ and y =
          FN + R φ (Snyder, eqs. (30-1) and (30-2)). EPSG lists no such
          method. *)

val transverse_mercator :
  lat0:float ->
  lon0:float ->
  k0:float ->
  false_easting:float ->
  false_northing:float ->
  projection
(** [transverse_mercator ~lat0 ~lon0 ~k0 ~false_easting ~false_northing] is
    the transverse Mercator projection of those parameters.

    Raises [Invalid_argument] as every projection does, and if [lat0] is not
    in \[-90, 90\] or [k0 <= 0]. *)

val albers :
  lat0:float ->
  lon0:float ->
  lat1:float ->
  lat2:float ->
  false_easting:float ->
  false_northing:float ->
  projection
(** [albers ~lat0 ~lon0 ~lat1 ~lat2 ~false_easting ~false_northing] is
    Albers' equal-area conic of those parameters, its standard parallels in
    either order, the northern one stored as [lat1], since the map is the
    same. Equal parallels make the cone tangent there, a pole as a parallel
    puts that pole at the cone's apex, and opposite parallels make the cone's
    limit, the equal-area cylinder.

    Raises [Invalid_argument] as every projection does, and if a latitude is
    not in \[-90, 90\]. *)

val pseudo_mercator :
  lon0:float -> false_easting:float -> false_northing:float -> projection
(** [pseudo_mercator ~lon0 ~false_easting ~false_northing] is method 1024 of
    those parameters.

    Raises [Invalid_argument] as every projection does. *)

val sinusoidal :
  lon0:float ->
  radius:float ->
  false_easting:float ->
  false_northing:float ->
  projection
(** [sinusoidal ~lon0 ~radius ~false_easting ~false_northing] is the
    sinusoidal projection of those parameters.

    Raises [Invalid_argument] as every projection does, and if
    [radius <= 0]. *)

(** {1:systems Systems}

    A system is a datum and, for a projected system, a projection. This
    section's values are WGS 84's longitude and latitude, MODIS's sinusoid,
    the UTM zones and the zone that holds a point, and a system's equality
    and EPSG code. jord has no polar system, such as UPS's polar
    stereographic, so a grid around a pole is geographic, of the sinusoid
    or of Albers' conic, and a zone of UTM reaches a pole only within 40°
    of longitude of its meridian ({!section-domains_tm}). *)

type t = { datum : datum; projection : projection option }
(** The type for systems, longitude and latitude on [datum] where [projection]
    is [None], else easting and northing of [projection] on [datum]'s
    ellipsoid. *)

val lonlat : t
(** [lonlat] is WGS 84's longitude and latitude. *)

val modis : t
(** [modis] is the system of NASA's MODIS land tiles, WGS 84's longitude and
    latitude under the sinusoidal projection of a sphere of radius
    6371007.181 m, the radius MODLAND's GCTP parameters state, with [lon0] and
    false easting and northing 0. It has no EPSG code. *)

val utm : zone:int -> north:bool -> t
(** [utm ~zone ~north] is WGS 84 / UTM zone [zone], northern hemisphere where
    [north].

    Raises [Invalid_argument] unless [1 <= zone <= 60]. *)

val utm_at : lon:float -> lat:float -> t
(** [utm_at ~lon ~lat] is the UTM zone holding the point, by Guidance Note 7-2's
    formula (§3.2.3.2): zone [⌊(lon + 180) / 6⌋ mod 60 + 1], northern where
    [lat >= 0]. It applies none of {!utm_grid_at}'s zone exceptions (32V,
    31X to 37X).

    Raises [Invalid_argument] if [lon] is not finite or [lat] is not in \[-90,
    90\]. *)

val utm_grid_at : lon:float -> lat:float -> t
(** [utm_grid_at ~lon ~lat] is the zone DMA TM 8358.2 ("The Universal
    Grids: UTM and UPS", 1989) gives the point, which is {!utm_at}'s but for
    the {e zone exceptions} in the latitude bands V and X. Between 56°N and
    64°N, zone 32 covers 3°E to 12°E; between 72°N and 84°N, zones 31, 33,
    35 and 37 cover 0°E to 9°E, 9°E to 21°E, 21°E to 33°E and 33°E to 42°E,
    longitudes taken into \[-180, 180). Each bound lies in the band or zone
    north or east of it.

    Raises [Invalid_argument] as {!utm_at} does. *)

val equal : t -> t -> bool
(** [equal a b] is [true] iff [a] and [b] have the same datum and the same
    projection, each parameter equal as a number. Parameters compare
    exactly, so a system whose parameters differ from another's, however
    little, is another system, and a transform between the two moves
    points. *)

val epsg : t -> int option
(** [epsg c] is the EPSG code of the system [c] is, one {!of_epsg} reads, and
    [None] for any other system. No two codes name equal systems. *)

(** {1:codes Codes and text}

    {!of_epsg} reads the EPSG codes of jord's datums and of the systems on
    them that jord's products and formats use ({!section-code_list}), and
    {!epsg} gives them back. *)

val of_epsg : int -> (t, string) result
(** [of_epsg code] is the system of [code], one of
    {{!section-code_list}the codes it reads}. A geographic system's points are
    (longitude, latitude) though EPSG orders its axes latitude first.

    The result is [Error], naming the code, for any other code. ETRS89 / UTM
    (25828 to 25837) is an [Error], since jord knows no ETRS89 datum but
    EUREF-FIN. *)

val pp : Format.formatter -> t -> unit
(** [pp ppf c] formats [c] on [ppf]. A system that has an EPSG code prints
    as its EPSG name, as [WGS 84 / UTM zone 33N] or
    [ETRS89-FIN [EUREF-FIN] / TM35FIN(E,N)], the names of EPSG's dataset
    12.029, and any other as its datum, its method and each parameter to 17
    significant digits, as [WGS 84 / Transverse Mercator (lat0 0, lon0 15,
    k0 0.99990000000000001, false_easting 500000, false_northing 0)]. Two
    systems print alike only when they are {!equal}. *)

(** {2:code_list The codes}

    The codes {!of_epsg} reads and {!epsg} gives are
    - 4326, {!lonlat}; 32601 to 32660 and 32701 to 32760, WGS 84 / UTM;
    - 4269, NAD83, and 26901 to 26923, 9712, 3372 and 3373, NAD83 / UTM
      zones 1N to 23N, 24N, 59N and 60N;
    - 10690, ETRS89-FIN \[EUREF-FIN\], and 3067, its TM35FIN(E,N), UTM zone
      35N's projection;
    - 4167, NZGD2000, and 2193, its New Zealand Transverse Mercator 2000:
      λ₀ 173°E, scale 0.9996, false easting 1600000 m and false northing
      10000000 m;
    - 5070, NAD83 / Conus Albers, Albers' cone of parallels 29.5°N and
      45.5°N and false origin (96°W, 23°N);
    - 4283, GDA94, and 3577, GDA94 / Australian Albers, the cone of 18°S and
      36°S and (132°E, 0°);
    - 3857, WGS 84 / Pseudo-Mercator, method 1024 about the Greenwich
      meridian.
    Each Albers and Pseudo-Mercator system has false easting and northing
    0. *)

(** {2:wkt Reading WKT}

    A WKT text is WKT 1 (a shapefile's [.prj]) or WKT2, told apart by the
    top keyword, and is parsed by the grammar both share (OGC 18-010r11,
    §6). Keywords and quoted names compare without case, and numbers as
    numbers, so [15] equals [15.0]. A WKT 1 text names a system only by the
    whole tree of one of three producers ({!section-wkt1}), and a WKT2 text
    by EPSG identifiers ({!section-wkt2}). *)

val of_wkt : string -> (t, string) result
(** [of_wkt s] is the system that the WKT text [s] states, read by the WKT 1
    rules ({!section-wkt1}) or the WKT2 rules ({!section-wkt2}).

    The result is [Error], naming the top keyword and name and why, or the
    position of a syntax error, for
    - a text that breaks the grammar;
    - a WKT 1 tree other than Esri's for WGS 84, NAD83 and the UTM zones
      {!of_epsg} reads on them, OGC's for WGS 84 and its UTM zones, and
      GDAL's for MODIS's sinusoid, each with exactly its [AUTHORITY]s,
      none in Esri's;
    - a WKT2 text whose datum, method or parameters lack the EPSG
      identifiers the WKT2 rules read, other than PROJ's MODIS sinusoid, or
      whose ellipsoid, prime meridian or units those rules refuse;
    - a [BOUNDCRS], a [COMPOUNDCRS], an epoch or a 3D system. *)

(** {3:wkt1 WKT 1}

    WKT 1 ([GEOGCS], [PROJCS]) names a system only when its whole tree
    equals one of three producers' trees, since only the producer's
    vocabulary defines a WKT 1 projection's method and parameter names
    (Annex C.3.4). An [AUTHORITY] is part of the tree, so one added to a
    producer's tree, or one that disagrees with its text, makes it another
    tree. The three are Esri's for WGS 84, NAD83 and the UTM zones
    {!of_epsg} reads on them, OGC's for WGS 84 and its UTM zones, and
    GDAL's for MODIS's sinusoid ({!section-wkt1_trees}). *)

(** {4:wkt1_trees The three trees}

    The trees WKT 1 is read by are
    - Esri's for EPSG:4326, 326zz and 327zz, and for NAD83's 4269 and the
      NAD83 / UTM codes {!of_epsg} reads, as its Projection Engine's
      documentation publishes them ([GEOGCS["GCS_WGS_1984", ...]],
      [PROJCS["NAD_1983_UTM_Zone_10N", ...]]), which GDAL writes to [.prj]
      files. Its NAD83 zone 58N, which EPSG does not number, is refused;
    - OGC's for EPSG:4326, 326zz and 327zz, as PROJ writes it with EPSG's
      names, axes and identifiers ([GEOGCS["WGS 84", ...]],
      [PROJCS["WGS 84 / UTM zone 33N", ...]]);
    - GDAL's for MODIS's sinusoid ([PROJCS["unnamed", GEOGCS["Unknown datum
      based upon the custom spheroid", ...]]] with
      [PROJECTION["Sinusoidal"]]), which reads as {!modis} with its three
      parameters' values. *)

(** {3:wkt2 WKT2}

    WKT2 ([GEOGCRS] or [GEODCRS] with an ellipsoidal 2D coordinate system,
    [PROJCRS] with a Cartesian 2D one) is read by the EPSG identifiers of
    its datum ({!section-wkt2_datum}) and of its method and parameters
    ({!section-wkt2_method}), except MODIS's sinusoid
    ({!section-wkt2_modis}). Its axes' units must be the metre or the
    degree; their order is not read, since each format jord reads fixes its
    own. An [ID] of a projected system or of its conversion is not read
    (§7.3.3), so a text that prints a registry's parameters lossily, as
    EPSG's export of 5070 does ([29.5000000000003]), reads as the system it
    states, which has no code. *)

(** {4:wkt2_datum Datum}

    The datum is read by the [ID] of its [DATUM] or [ENSEMBLE], else of the
    geographic system that holds it, and the two must agree: 6326 or 4326
    (WGS 84), 6269 or 4269 (NAD83), 1391 or 10690 (EUREF-FIN), 6167 or 4167
    (NZGD2000), 6283 or 4283 (GDA94). The ellipsoid's numbers must equal
    the datum's, and the prime meridian must be Greenwich. An [ANCHOR] or
    [ANCHOREPOCH] is not read. *)

(** {4:wkt2_method Method and parameters}

    The method is read by its [ID], 9807, 9822 or 1024, and its parameters
    by theirs, exactly those {!type-projection} holds, with 8801 equal to 0
    for 1024. A parameter's name is not read. An angle whose unit is the
    degree (identifier 9102 or 9122, or the factor 0.0174532925199433 as
    printed) is taken as exactly that many degrees; another unit converts
    by its factor, and an omitted one is the degree, the metre or unity
    (§9.3.4). *)

(** {4:wkt2_modis MODIS's sinusoid}

    A [PROJCRS] on an unidentified datum whose ellipsoid is the sphere of
    radius 6371007.181 m, with a [METHOD] named ["Sinusoidal"] and
    parameters 8802, 8806 and 8807, is {!modis} with those values, as PROJ
    writes it. It is the one reading that needs no identifier. *)

(** {1:transforms Transforms}

    A transform maps points from one system to another, each point alone,
    through the longitude and latitude of each system's datum, and the datum
    steps between them. A longitude out of a geographic system lies in \[-180,
    180\].

    A point with a NaN coordinate maps to NaN. Every other point must lie in
    the domain of each projection it goes through ({!section-domains}), and a
    projected point must be the image of such a point. *)

val transform : src:t -> dst:t -> 'd Nx.float64_t -> 'd Nx.float64_t
(** [transform ~src ~dst p] is the points [p] of [src] in [dst], computed where
    [p] lies. It is [p], unchecked, when [equal src dst].

    Raises [Invalid_argument] if [p]'s last axis is not 2, and, naming the
    first point, where a point is outside the domain, checked by
    {!Nx.check}. *)

val in_domain : src:t -> dst:t -> 'd Nx.float64_t -> 'd Nx.bool_t
(** [in_domain ~src ~dst p] is whether each point [p] of [src], [[...; 2]],
    lies in the domain of the transforms from [src] to [dst], by the rules of
    {!section-transforms} and {!section-domains}, computed where [p] lies. A
    point of a geographic [src] must also have a latitude in \[-90, 90\] and
    a finite longitude, as where [src] and [dst] are equal. A point with a
    NaN coordinate is not in it. Where [src] and [dst] are not equal,
    [transform ~src ~dst] refuses exactly the points without a NaN
    coordinate that are not in it.

    Raises [Invalid_argument] if [p]'s last axis is not 2. *)

(** {2:domains Domains}

    Each projection's domain admits points past its boundary by twice the
    transforms' error, the most a round trip moves a point, so a point of
    the boundary passes both ways. Its inverse refuses a point that is not
    the image of a point of the domain, to that margin. Transverse Mercator
    ({!section-domains_tm}) reaches 40° of longitude from its central
    meridian, Albers' conic ({!section-domains_albers}) and the sinusoid
    ({!section-domains_closed}) every point, and Pseudo-Mercator 88° of
    latitude from the equator. *)

(** {3:domains_tm Transverse Mercator}

    Transverse Mercator's domain is a latitude φ in \[-90, 90\] and a
    longitude λ within 40° of the projection's central meridian λ₀, across
    the antimeridian included, or past it by at most 12 nm along the
    parallel, (|λ - λ₀| - 40°) · a · cos φ ≤ 12 nm with [a] the equatorial
    radius, 6378137 m on both ellipsoids, and the angle in radians. 12 nm is
    twice the transforms' error, so a pole passes too, where every
    longitude names one point. The inverse applies the rule to the point it
    finds, and refuses a point more than half a meridian north or south of
    the equator's image, beyond the poles' images. *)

(** {3:domains_albers Albers' conic}

    Albers' domain is every point, the poles included, whose images are arcs
    about the cone's apex, and every longitude, taken into the 360° of [lon0]
    ± 180°. Its inverse refuses a point farther than 2ε from the image,
    beyond a pole's arc or past the image of the meridian [lon0] + 180°
    along the parallel. A point past that meridian by less takes its
    longitude. *)

(** {3:domains_closed Pseudo-Mercator and the sinusoid}

    Pseudo-Mercator's domain is every longitude and a latitude within 88° of
    the equator, GN 7-2's limit, or past it by at most 2ε along the meridian.
    Its inverse refuses a point whose latitude lies past that, or whose
    easting lies more than 2ε beyond π a of the false easting, the image of
    [lon0] ± 180°. Sinusoidal's domain is every point; its inverse refuses a
    point more than 2ε outside the sinusoid, beyond R π/2 of the false
    northing or beyond π R cos φ of the false easting along the parallel. A
    point on a parallel whose image is shorter than 2ε is a pole, and takes
    [lon0]. *)
