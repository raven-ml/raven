(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** GeoJSON's polygons, by RFC 7946.

    GeoJSON's positions are longitude and latitude in degrees on WGS 84 (§4), so
    its features are in {!Crs.lonlat}. GeoJSON 2008 let a [crs] member name
    another system; RFC 7946 dropped it (Appendix B), but files still carry it.
    A [crs] member, on any object, must name OGC's CRS84, as
    [urn:ogc:def:crs:OGC::CRS84] (RFC 7946's spelling) or
    [urn:ogc:def:crs:OGC:1.3:CRS84]. Any other is refused, since reading its
    positions as longitude and latitude would place them wrongly. *)

val polygons : Jsont.json -> (Polygons.t * Jsont.json array, string) result
(** [polygons json] is the features of the GeoJSON text [json], decoded by the
    caller ([Jsont_bytesrw.decode_string Jsont.json]), and each feature's
    [properties] ([Jsont.Json.null] when absent). A FeatureCollection gives its
    features in order, a Feature itself, and a Polygon or MultiPolygon geometry
    one feature without properties.

    A Polygon is one feature of its rings, a MultiPolygon one feature of all its
    polygons' rings, and a null geometry a feature with no ring. Each ring keeps
    its positions as written, its closing one included. A position's numbers
    after the second (an altitude) are dropped.

    [Error] names the feature for an element of a collection's [features] that
    is not a Feature (§3.3), any other geometry type, a position with fewer than
    two numbers, a coordinate that is not a finite number, or a [crs] member
    that does not name CRS84. *)
