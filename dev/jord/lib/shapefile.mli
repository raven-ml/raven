(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** ESRI Shapefiles of polygons and their dBASE attributes.

    A shapefile is a main file ([.shp]) of shapes and a dBASE table ([.dbf]) of
    attributes, one record per shape in the same order, as the ESRI Shapefile
    Technical Description (1998) defines them. jord reads Polygon, PolygonZ and
    PolygonM shapes (it drops z and m) and Null shapes. The [.prj] file beside
    the main file states the system, in WKT ({!Crs.of_wkt}). *)

val read :
  Crs.t ->
  shp:string ->
  dbf:string ->
  (Polygons.t * (string * string array) list, string) result
(** [read crs ~shp ~dbf] is the features of the main file's bytes [shp] in
    [crs], one per shape in order, a Null shape a feature with no ring, and the
    table's fields from the bytes [dbf], each named and holding one text per
    feature. A field's text is the bytes dBASE stores, with trailing blanks and
    NULs dropped, in the encoding the [.cpg] file names. A record the table
    marks deleted is dropped with its shape.

    [Error] names the shape or the file at fault for a main file that breaks the
    description (its header, a record's length, a part's start, a shape of
    another type than the file's), for a file of shapes other than polygons, for
    a table that breaks dBASE's layout, and for files whose records do not
    number alike. *)
