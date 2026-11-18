(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The TIFF tags of OGC GeoTIFF 1.1 (19-008r4) that place an image on the
    Earth, with the grids of a Cloud Optimized GeoTIFF's overviews and GDAL's
    nodata tag.

    A TIFF image carries {e tags}, each a number and a value
    ({!Nx_io.Tiff.value}), which {!Nx_io.load_tiff} reads and {!Nx_io.save_tiff}
    writes. GeoTIFF's tags are independent of the raster data's, so this module
    reads and makes tags and does no I/O. A landmask's grid:

    {[
      let ifd = List.hd (Nx_io.load_tiff "grid_0.15_52.25.tiff") in
      let grid = Jord_geotiff.grid (Nx_io.Tiff.tags ifd)
    ]} *)

(** {1:grids Grids}

    The size is ImageWidth (256) columns and ImageLength (257) rows. The map
    from raster space is ModelPixelScale (33550) with one ModelTiepoint
    (33922), or ModelTransformation (34264), whose rows for x and y are taken
    and whose last row must be [0 0 0 1] (Requirement 1.2). Any other
    combination is refused. Its raster space has (I, J) = (column, row). With
    GTRasterTypeGeoKey PixelIsArea (1), the default when the key is absent,
    raster point (I, J) is the grid's pixel point [(J, I)]; with PixelIsPoint
    (2), the raster point (0, 0) is the first cell's centre (Annex B.2.2), so
    it is pixel point [(J + 0.5, I + 0.5)]. The grid's system comes from the
    tags' GeoKeys ({!section-systems}). *)

val grid : (int * Nx_io.Tiff.value) list -> (Jord.Grid.t, string) result
(** [grid tags] is the grid that [tags] state. An integer tag may be BYTE, SHORT
    or LONG, as TIFF 6.0 asks readers to accept, and a floating-point one FLOAT
    or DOUBLE. BigTIFF's LONG8 holds only offsets and byte counts, so it is
    refused here as another type.

    The result is [Error], naming the tag or key, for a tag given twice, a
    tag or key that is missing, of another type or count, or of a value
    GeoTIFF reserves or does not allow, a system {!Jord.Crs.of_epsg} refuses,
    a user-defined system other than MODIS's sinusoid, another unit, several
    tiepoints (ground control points), or a map that is not invertible. *)

val tags : ?nodata:float -> Jord.Grid.t -> (int * Nx_io.Tiff.value) list
(** [tags ?nodata g] is the tags that state [g], in increasing order:
    PixelIsArea; ModelPixelScale and the tiepoint at raster point (0, 0) where
    [g] is north-up ([row_step = (0, -d)], [col_step = (d', 0)], [d, d' > 0]),
    ModelTransformation otherwise; GTModelTypeGeoKey and the system's key; and
    GDAL_NODATA where [nodata] is given, as the shortest decimal text that reads
    back as [nodata]. The writer adds ImageWidth and ImageLength, which it knows
    from the image; with ImageWidth [g.cols] and ImageLength [g.rows], [grid]
    gives back [g] exactly.

    Raises [Invalid_argument] if [g]'s system has no EPSG code
    ({!Jord.Crs.epsg}). *)

(** {2:systems Systems}

    GTModelTypeGeoKey states the kind of system, projected (1), named by
    ProjectedCRSGeoKey (3072), or geographic (2), named by GeodeticCRSGeoKey
    (2048), as {!Jord.Crs.of_epsg} reads them. A unit key that is present
    must be the metre (9001) for ProjLinearUnitsGeoKey (3076) and
    GeogLinearUnitsGeoKey (2052), or the degree (9102 or 9122) for
    GeogAngularUnitsGeoKey (2054). For a registered system these keys are
    read from the GeoKeyDirectory (34735), and no other key is, since a
    registered system is its code (§7.4), so a projected system's
    GeodeticCRSGeoKey is not read. *)

(** {2:modis MODIS's sinusoid}

    One user-defined system is read, MODIS's sinusoid ({!Jord.Crs.modis}), a
    projected system whose ProjectedCRSGeoKey, ProjectionGeoKey (3074),
    GeodeticCRSGeoKey, GeodeticDatumGeoKey (2050) and EllipsoidGeoKey (2056)
    are all user-defined (32767), with ProjMethodGeoKey (3075) 24, the
    sinusoid of Annex C's Table C.1, on the sphere of radius 6371007.181 m.
    That sphere is GeogSemiMajorAxisGeoKey (2057) of that radius and
    GeogSemiMinorAxisGeoKey (2058) equal to it, or GeogInvFlatteningGeoKey
    (2059) 0. Its central meridian is ProjCenterLongGeoKey (3088) or
    ProjNatOriginLongGeoKey (3080), which must agree where both are given,
    since GeoTIFF does not say which a method takes (Annex C.2), and its false
    easting and northing are 3082 and 3083. These keys are DOUBLEs of
    GeoDoubleParams (34736). A prime meridian (2061) must be Greenwich's. *)

(** {1:overviews Overviews}

    A Cloud Optimized GeoTIFF (OGC 21-026, COG 1.0) may hold {e overviews}
    of an image, reduced-resolution subfiles. They are IFDs whose
    NewSubfileType (254) has bit 0 set, which follow the image's
    full-resolution IFD, each smaller than the one before it (Requirement 3).
    Bit 2 marks a mask, and a mask's overviews have its image's grids. An
    overview states no georeference, and has its image's top left corner and
    its pixel size scaled by the ratio of the sizes ({!section-scale}). In a
    file of one image and no mask, {!overview} gives each overview's grid:

    {[
      let ifds = List.map Nx_io.Tiff.tags (Nx_io.load_tiff "map.tif") in
      let full = Result.get_ok (Jord_geotiff.grid (List.hd ifds)) in
      List.map (Jord_geotiff.overview full) (List.tl ifds)
    ]} *)

val overview :
  Jord.Grid.t -> (int * Nx_io.Tiff.value) list -> (Jord.Grid.t, string) result
(** [overview g tags] is the grid of an overview of an image on grid [g]
    ({!grid} of its full-resolution IFD's tags), where [tags] are the
    overview's. It has [g]'s system and origin, ImageLength rows and ImageWidth
    columns. Each component of [g.row_step] and [g.col_step] is multiplied by
    [g]'s size along its axis and then divided by the overview's, as the
    formula of {{!section-scale}an overview's scale} is written, so where the
    product is exact the step is the exact ratio rounded once.

    The result is [Error], naming the tag, for tags whose NewSubfileType is
    missing or does not set bit 0, that hold ModelPixelScale, ModelTiepoint,
    ModelTransformation or GeoKeyDirectory, or that lack ImageWidth or
    ImageLength. It is [Error], naming both sizes, for a size that is 0 or
    larger than [g]'s along an axis, or equal to [g]'s along both. *)

(** {2:scale An overview's scale}

    "COG readers SHALL consider that all reduced-resolution subfiles share
    the same point of origin for the top left corner of the image as the
    full resolution subfile" (Requirement 6), whose pixel size is multiplied
    by the ratio of the sizes, per axis:

    {[
      IFDn_PixelScaleX = IFD0_PixelScaleX * IFD0_ImageWidth / IFDn_ImageWidth
    ]}

    The ratio need not be 2, as when GDAL halves 75 columns to 37.
    Requirement 3 asks for both sides smaller, but one may equal the
    image's, as GDAL keeps a side of 1. *)

(** {1:nodata Nodata} *)

val nodata : (int * Nx_io.Tiff.value) list -> (float option, string) result
(** [nodata tags] is the GDAL_NODATA tag's (42113) number, the value that marks
    the cells with no data in every band, or [None] without the tag. Its text is
    a decimal number, as C's [strtod] reads one, or [nan], [inf] or [infinity]
    in any case, each with an optional sign.

    The result is [Error], naming the tag, for any other text. *)
