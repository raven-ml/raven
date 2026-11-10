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
    way the rings turn: clockwise shells in a Shapefile, counterclockwise ones
    in GeoJSON. For a multipolygon whose polygons' interiors are disjoint
    (§6.1.14) it is their union, which is why a feature need not say which rings
    form which polygon.

    The layout is GeoArrow's for an array of multipolygons, without its polygon
    level: one points tensor, and offsets. *)

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
    indexes: [n] for [rings], the number of rings for [features]. *)

val count : t -> int
(** [count p] is the number of [p]'s features. *)

val to_crs : Crs.t -> t -> t
(** [to_crs crs p] is [p] with its points moved to [crs] by {!Crs.transform}.
    Edges stay straight in [crs] and gain no points, so an edge straight in
    [p.crs] is a chord of its image.

    Raises as {!Crs.transform} does. *)
