(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Features made of polygons, in a coordinate reference system.

    A {e ring} is a sequence of points closed by the edge from its last point to
    its first, so a ring that repeats its first point at its end has a
    zero-length closing edge. A {e feature} is a sequence of rings, such as the
    shells and holes of a polygon or of a multipolygon. A feature's {e region}
    is the set of points from which a ray crosses the feature's edges an odd
    number of times (the even-odd rule), so the orientation of its rings does
    not matter.

    For polygons valid by OGC Simple Features 1.2.1 (§6.1.11.1: holes inside
    their shell, rings that do not cross), the region is the polygon, whichever
    way the rings turn, clockwise shells in a Shapefile and counterclockwise
    ones in GeoJSON. For a multipolygon whose polygons' interiors are disjoint
    (§6.1.14) it is their union, which is why a feature need not say which
    rings form which polygon.

    The layout is GeoArrow's for an array of multipolygons without its
    polygon level, one points tensor and two arrays of offsets. *)

type t = private {
  crs : Crs.t;
  points : Nx.host Nx.float64_t;  (** All rings' points, [[n; 2]]. *)
  rings : int array;
      (** Ring [k] is points [rings.(k)] to [rings.(k + 1) - 1]. *)
  features : int array;
      (** Feature [i] is rings [features.(i)] to [features.(i + 1) - 1]. *)
}
(** The type for features. *)

val v :
  Crs.t -> Nx.host Nx.float64_t -> rings:int array -> features:int array -> t
(** [v crs points ~rings ~features] is the features of those fields. A feature
    with no ring, as a Shapefile's null shape, has an empty region.

    Raises [Invalid_argument] unless [points] is [[n; 2]] of finite numbers, and
    each offset array starts at 0, never decreases and ends at the count it
    indexes, [n] for [rings] and the number of rings for [features]. *)

val count : t -> int
(** [count p] is the number of [p]'s features. *)

val to_crs : Crs.t -> t -> t
(** [to_crs crs p] is [p] with its points moved to [crs] by {!Crs.transform}.
    Edges stay straight in [crs] and gain no points, so an edge straight in
    [p.crs] is a chord of its image.

    Raises as {!Crs.transform} does. *)

(** {1:areas Areas on the Earth}

    An area in a system's plane equals the area it covers on the Earth only
    where the projection is equal-area ({!section-areal_scale}), so a true
    area takes the features to an equal-area system first, as
    [area (to_crs albers p)]. {!val-area} refuses a geographic system
    ({!section-geographic}). *)

val area : t -> Nx.host Nx.float64_t
(** [area p] is the area of each feature's region in [p.crs]'s plane,
    [[count p]], in the square of its coordinates' unit, square metres for
    each projected system jord knows, within a bound of the exact area of
    the region of the feature's points ({!section-area_bound}). It is the
    area of the feature's region, rings that cross included, and a feature
    with no ring has 0.

    Raises [Invalid_argument] if [p.crs] is geographic
    ({!section-geographic}). *)

(** {2:area_bound The bound on an area}

    To first order {!val-area} is within

    {[
      (94 + K) u D W + (4E + 36X) u D²
    ]}

    of the exact area of the region of the feature's points, where u =
    2{^-53}; D is the largest distance of its points from its first point
    along either axis; W the sum of the widths of its {e runs}, the spans of
    abscissae over which one edge bounds the region from one side, at most 2D
    each; K their number; E its edges; and X the crossings of two of its
    edges. For a feature of E edges that do not cross, it is about
    2E(96 + E) u D². *)

(** {2:areal_scale Areal scale}

    A projection's areal scale factor is h k sin θ' (Snyder,
    {e Map Projections: A Working Manual}, USGS Professional Paper 1395, 1987,
    ch. 4, eq. 4-15, {{:https://pubs.usgs.gov/pp/1395/report.pdf}}). It is 1
    on an equal-area map such as Albers' conic ({!Crs.albers}), and k² on a
    conformal one, whose h equals k. So an area in a UTM zone is k² times the
    area on the Earth, 0.9992 of it on the central meridian, where
    k = 0.9996, and more toward the zone's edges. Pseudo-Mercator
    ({!Crs.pseudo_mercator}) applies the sphere's Mercator, h = k = sec φ
    (ch. 7, eq. 7-3), to the ellipsoid's coordinates, so its areas are about
    sec² φ times those on the Earth, 2.4 times at 50°. *)

(** {2:geographic Geographic systems}

    A geographic system's coordinates are degrees of longitude and latitude.
    Read as a plane they are the equidistant cylindrical projection,
    "neither equal-area nor conformal" (Snyder, ch. 12), whose areal scale is
    h k = sec φ (eqs. 12-3, 12-4, the standard parallel at the equator), so a
    square degree covers cos φ times the Earth it covers at the equator, and
    {!val-area} refuses such features. *)
