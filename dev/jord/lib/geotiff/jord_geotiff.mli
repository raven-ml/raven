(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The TIFF tags that place an image on the Earth: OGC GeoTIFF 1.1 (19-008r4)
    and GDAL's nodata tag.

    A TIFF image carries {e tags}, each a number and a value
    ({!Nx_io.Tiff.value}), which {!Nx_io.load_tiff} reads and {!Nx_io.save_tiff}
    writes. GeoTIFF's tags are independent of the raster data's, so this module
    reads and makes tags and does no I/O. A landmask's grid:

    {[
      let ifd = List.hd (Nx_io.load_tiff "grid_0.15_52.25.tiff") in
      let grid = Jord_geotiff.grid (Nx_io.Tiff.tags ifd)
    ]}

    {b Reading a grid.} The size is ImageWidth (256) columns and ImageLength
    (257) rows. The map from raster space is ModelPixelScale (33550) with one
    ModelTiepoint (33922), or ModelTransformation (34264), whose rows for x and
    y are taken and whose last row must be [0 0 0 1] (Requirement 1.2). Any
    other combination is refused, ground control points (tiepoints without a
    scale) among them. Its raster space has (I, J) = (column, row). With
    GTRasterTypeGeoKey PixelIsArea (1), the default when the key is absent,
    raster point (I, J) is the grid's pixel point [(J, I)]; with PixelIsPoint
    (2), the raster point (0, 0) is the first cell's centre (Annex B.2.2), so it
    is pixel point [(J + 0.5, I + 0.5)].

    The system is GTModelTypeGeoKey's: projected (1), named by
    ProjectedCRSGeoKey (3072), or geographic (2), named by GeodeticCRSGeoKey
    (2048), as {!Jord.Crs.of_epsg} reads them. A unit key that is present must
    be the metre (9001) for ProjLinearUnitsGeoKey (3076) and
    GeogLinearUnitsGeoKey (2052), or the degree (9102 or 9122) for
    GeogAngularUnitsGeoKey (2054). These keys are read from the GeoKeyDirectory
    (34735), and no other key is, since a registered system is its code (§7.4):
    a projected system's GeodeticCRSGeoKey is not read. *)

val grid : (int * Nx_io.Tiff.value) list -> (Jord.Grid.t, string) result
(** [grid tags] is the grid that [tags] state. An integer tag may be BYTE, SHORT
    or LONG, as TIFF 6.0 asks readers to accept, and a floating-point one FLOAT
    or DOUBLE. BigTIFF's LONG8 holds only offsets and byte counts, so it is
    refused here as another type.

    [Error] names the tag or key for a tag given twice, a tag or key that is
    missing, of another type or count, or of a value GeoTIFF reserves, makes
    user-defined or does not allow, a system {!Jord.Crs.of_epsg} refuses,
    another unit, several tiepoints (ground control points), or a map that is
    not invertible. *)

val nodata : (int * Nx_io.Tiff.value) list -> (float option, string) result
(** [nodata tags] is the GDAL_NODATA tag's (42113) number, the value that marks
    the cells with no data in every band, or [None] without the tag. Its text is
    a decimal number, as C's [strtod] reads one, or [nan], [inf] or [infinity]
    in any case, each with an optional sign. [Error] names the tag for any other
    text. *)

val tags : ?nodata:float -> Jord.Grid.t -> (int * Nx_io.Tiff.value) list
(** [tags ?nodata g] is the tags that state [g], in increasing order:
    PixelIsArea; ModelPixelScale and the tiepoint at raster point (0, 0) where
    [g] is north-up ([row_step = (0, -d)], [col_step = (d', 0)], [d, d' > 0]),
    ModelTransformation otherwise; GTModelTypeGeoKey and the system's key; and
    GDAL_NODATA where [nodata] is given, as the shortest decimal text that reads
    back as [nodata]. The writer adds ImageWidth and ImageLength, which it knows
    from the image; with ImageWidth [g.cols] and ImageLength [g.rows], [grid]
    gives back [g] exactly. *)
