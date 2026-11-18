(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Earth-observation data placed on the Earth.

    jord says where each value of an array lies on the Earth and what a product
    measured there. A {e system} ({!Crs.t}) says what a pair of numbers means
    on the Earth, and a {e point} is the last axis of a float64 tensor,
    [[...; 2]], in its system's axis order. A {e grid} ({!Grid.t}) places an
    array's first two axes in a system. {e Features} ({!Polygons.t}) are
    polygons in a system, and {!Rasterize} gives the cells of a grid that they
    cover. {!Warp} moves an array from one grid to another. A {e sampler} reads
    it at a point of each cell, and an {e area rule} over each cell's area. A
    {e band} ({!Band.t}) says what an array's stored numbers, its {e DNs},
    measure. A {e product module}, such as {!Sentinel2}, states what one
    producer states about its products' {e layers}, the arrays their files
    hold. A layer is a band or a {e quality layer}, whose DNs hold flags or
    classes. Each module defines its terms in full.

    Unless a value's doc says otherwise:
    - it reads and writes no file, and takes the text and values a program has
      read;
    - a value whose result is a [result] is [Error] for input it cannot
      read, with a message for people that names what it refused;
    - a {e misuse}, a call whose arguments break its doc's conditions, data
      outside a value's domain included, raises [Invalid_argument]. An
      array lies on a {e device set} ({!Nx.section-devices}), such as the
      host's, {!Nx.Host}. A condition on the values of an array that can lie
      off the host is checked by {!Nx.check}, which raises at the call on
      the host. On another device set it raises at a later wait on that
      set, or never if no wait follows ({!Nx.section-waiting}), and the
      result is unspecified at the points it refuses;
    - its result arrays lie where its input arrays lie, or where its [~at]
      places them.

    Each file or document format is a library of its own, [jord.<format>]
    with the module [Jord_<format>], which jord does not depend on.

    A program that puts a product's band on a grid of its own goes through
    these steps:
    + It reads, from bytes it fetches, a catalogue's item ([Jord_stac.item]),
      the product's metadata ([Jord_safe.l2a]) and the images they name, or
      windows of them ([Nx_io.Tiff.window]).
    + It takes each file's grid from its tags ([Jord_geotiff.grid]) or the
      item, and each band from the product module, such as
      {!Sentinel2.val-reflectance}.
    + It decodes the band's DNs ({!Band.decode}) into values and a mask.
    + It warps the values onto its own grid ({!Grid.v}) with a sampler of
      {!Warp}, which carries the mask, or, onto a coarser grid, an area
      rule, which gives each cell's {e fraction}, its share with data.
    + It reads the product's quality layer, as {!Sentinel2.Scl.mem} reads
      the scene classification, and joins its mask with a sampler's. For an
      area rule it must join the mask, taken onto the band's cells by
      {!Warp.nearest}, with the band's [valid] before the rule, so that each
      cell's fraction is its clear share. {!Warp.mode} of the classification
      does not give that share, since it keeps each cell's largest class and
      so reads clear where another class covers more of a cell than any
      class of cloud.

    The core of those steps reads as follows, for a Sentinel-2 product's
    [metadata], its band B04's DNs [dn] on grid [src], its scene
    classification [scl] on grid [src'] of the same system, and a grid
    [dst]:

    {[
      let c = Warp.centres ~at:Nx.Host.on src.crs dst in
      let band = Result.get_ok (Sentinel2.reflectance metadata B04) in
      let red, valid = Band.decode band Nx.float32 dn in
      let red, ok = Warp.bilinear ~valid src (Warp.pixel src c) red in
      let scl, seen = Warp.nearest src' (Warp.pixel src' c) scl in
      let masked =
        Sentinel2.Scl.(
          mem
            [ No_data; Saturated_or_defective; Cloud_medium_probability;
              Cloud_high_probability ]
            scl)
      in
      (red, Nx.(logical_and (logical_and ok seen) (logical_not masked)))
    ]}

    It gives the red reflectance on [dst] and a mask that holds where the band
    holds data, the scene classification has a cell, and its class is none of
    the four masked, which hold no data, defective data, or cloud of medium
    and high probability. [dn] and
    [scl] must have one rank, both [[h; w]] or both images of one channel,
    [[h; w; 1]], since masks of different ranks can join into another shape
    without an error ({!Warp.section-samplers}). *)

(** {1:places Places} *)

module Crs = Crs
(** Coordinate reference systems, and the transforms between them. *)

module Grid = Grid
(** A grid places the cells of a raster in a coordinate reference system. *)

module Polygons = Polygons
(** Features made of polygons, in a coordinate reference system. *)

module Rasterize = Rasterize
(** The cells of a grid that features cover, and how much of each. *)

module Warp = Warp
(** Arrays moved from one grid to another. *)

(** {1:measurements Measurements} *)

module Band = Band
(** The bands of a raster, what its stored numbers measure, after STAC's
    Band Object. *)

(** {1:products Products}

    Each quality layer is a submodule named after the layer, such as
    {!Landsat.module-Qa_pixel} ({!Band}). A layer's named one-bit flags are its
    type [flag], whose positions [bit] gives and which [any] reads, and a
    classified layer's classes ({!Band}) are its type [t], whose codes [code]
    gives and which [mem] reads. A field of several bits ({!Band.section-bits})
    has a reader named after it, such as {!Landsat.Qa_pixel.val-confidence} or
    {!Modis.Qc.val-quality}, and {!Landsat.module-Qa_radsat}, whose bits are
    per band, has readers of its own. A layer's fill is its DNs where nothing
    was recorded ({!Band}), and a layer whose fill a program must mask has
    [fill], which reads it. {!Landsat.valid_range} and {!Modis.valid_range}
    give the DNs a band's guide calls valid ({!Band}). *)

module Sentinel2 = Sentinel2
(** The facts ESA states about Sentinel-2's Level-1C and Level-2A
    reflectance products. *)

module Sentinel1 = Sentinel1
(** Sentinel-1's radar backscatter, as radiometrically terrain-corrected γ⁰
    in linear power. *)

module Landsat = Landsat
(** The facts the USGS states about Landsat Collection 2 Level-2 surface
    reflectance and temperature products. *)

module Hls = Hls
(** The facts NASA states about the surface reflectance products of
    Harmonized Landsat and Sentinel-2 (HLS) version 2.0. *)

module Modis = Modis
(** The facts NASA states about MODIS Collection 6.1's vegetation indices and
    surface reflectance, MOD13Q1 and MOD09A1. *)

(** {1:time Time and the Sun} *)

module Time = Time
(** Instants of UTC. *)

module Sun = Sun
(** The Sun's direction from a point on the Earth, and its distance. *)
