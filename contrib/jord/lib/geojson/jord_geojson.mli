(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** GeoJSON's polygons and points, by RFC 7946.

    GeoJSON's positions are longitude and latitude in degrees on WGS 84 (§4), so
    its features are in {!Jord.Crs.lonlat}. GeoJSON 2008 let a [crs] member
    name another system, which RFC 7946 dropped (Appendix B), and files still
    carry it. A [crs] member, on any object, must name OGC's CRS84, as
    [urn:ogc:def:crs:OGC::CRS84] (RFC 7946's spelling) or
    [urn:ogc:def:crs:OGC:1.3:CRS84]. Any other is refused, since reading its
    positions as longitude and latitude would place them wrongly.

    Both readers take a GeoJSON text decoded by the caller
    ([Jsont_bytesrw.decode_string Jsont.json]). A FeatureCollection gives its
    features in order and a Feature itself, each with its [properties]
    ([Jsont.Json.null] when absent). A position's numbers after the second
    (an altitude) are dropped. Both readers {e refuse} these texts, with an
    [Error] that names the feature at fault, or the text:
    - a text that is not a GeoJSON object, or a FeatureCollection without
      [features];
    - an element of a collection's [features] that is not a Feature (§3.3);
    - a geometry without [type] or [coordinates];
    - a position with fewer than two numbers, or a coordinate that is not a
      finite number;
    - a [crs] member that does not name CRS84. *)

val polygons :
  Jsont.json -> (Jord.Polygons.t * Jsont.json array, string) result
(** [polygons json] is the features of [json] and their properties. A
    Polygon or MultiPolygon geometry alone is one feature without
    properties. A Polygon is one feature of its rings, a MultiPolygon one
    feature of all its polygons' rings, and a null geometry a feature with
    no ring. Each ring keeps its positions as written, its closing one
    included.

    The result is [Error] for what both readers refuse, and, naming the
    feature, for any other geometry type. *)

val points :
  Jsont.json -> (Nx.host Nx.float64_t * Jsont.json array, string) result
(** [points json] is the positions of [json]'s points, [[n; 2]] longitude and
    latitude in {!Jord.Crs.lonlat}, one per feature in order, and their
    properties. A Point geometry alone is one point without properties. A
    null geometry, a feature RFC 7946 calls unlocated (§3.2), has NaN for
    both, which {!Jord.Crs.transform} keeps and the samplers of {!Jord.Warp}
    read as no data, and so has a Point with empty coordinates, which §3.1
    lets a reader take as null.

    The result is [Error] for what both readers refuse, and, naming the
    feature, for any geometry but a Point, a MultiPoint included, whose several
    positions would share one feature's properties. *)
