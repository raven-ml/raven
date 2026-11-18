(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Embeddings of pixels, and the files GeoTessera reads.

    An embedding array is float32 [[pixels; dims]], a row per pixel of a
    {!Pixels.t}. A pixel with no observation in either sensor has no embedding,
    and its row is NaN. A tile's embedding is [[h; w; dims]], the embedding of
    {!Pixels.of_tile}'s pixels reshaped to the tile's grid [(h, w)]. A
    student's Matryoshka prefix of [k] values ({!Model.prefixes}) is the slice
    of the last axis's first [k], as it is, and every function here takes it as
    it takes a whole embedding.

    The int8 form of an embedding is GeoTessera's. Per pixel, it is a scale
    [s = max (max |e|, 1e-12) / 127] and codes [round (e / s)] clipped to
    [[−127, 127]], each step in float32, halves rounded to even. A pixel with no
    embedding has scale NaN and codes [0], which GeoTessera reads as no data. *)

val encode :
  shape:Batch.shape ->
  'd Nx.Placement.t ->
  ('d Batch.t -> 'd Nx.float32_t) ->
  Nx.host Pixels.t ->
  Nx.host Nx.float32_t
(** [encode ~shape at f px] is the embedding of [px]'s pixels by [f], a model's
    {!Model.apply} or its compiled form. It packs [px] into batches of [shape]
    with {!Batch.v2_counts}, places each at [at] and applies [f]. The host packs
    the next batch while [at]'s devices compute one, and waits for each result
    in turn.

    Raises what {!Batch.pack} raises. *)

val quantize : 'd Nx.float32_t -> 'd Nx.int8_t * 'd Nx.float32_t
(** [quantize e] is the int8 form of [e], [[…; dims]]: the codes, of [e]'s
    shape, and the scales, of [e]'s shape without its last axis. *)

val save : string -> Nx.host Nx.float32_t -> unit
(** [save name e] writes [e], a tile's embedding [[h; w; dims]], to [name.npy].

    Raises [Invalid_argument] if [e] is not of rank 3. *)

val save_int8 : string -> Nx.host Nx.float32_t -> unit
(** [save_int8 name e] writes the int8 form of [e], a tile's embedding
    [[h; w; dims]]: the codes to [name.npy], int8 [[h; w; dims]], and the scales
    to [name_scales.npy], float32 [[h; w]].

    Raises [Invalid_argument] if [e] is not of rank 3. *)

val load : string -> Nx.host Nx.float32_t
(** [load name] is the tile's embedding [[h; w; dims]] in [name.npy]: float32,
    as {!save} writes it, or int8 codes with a scale per pixel in
    [name_scales.npy], float32 [[h; w]], as {!save_int8} and GeoTessera write
    them. A pixel's values are its codes times its scale, in float32, as
    GeoTessera's [dequantize_embedding] reads them, and NaN where its scale is
    not finite.

    After [save_int8 name e], each value of [load name] is within
    [(1/2 + 2⁻¹⁵) s] of [e]'s, [s] its pixel's scale: half a step of the code,
    and the float32 roundings of the code's quotient and of its product, each
    at most [2⁻²⁴ · 127 s]. A pixel without embedding loads as NaN.

    Raises [Failure] naming the file if [name.npy] is neither float32 nor
    int8 of rank 3, or if [name_scales.npy] is not float32 [[h; w]]. A
    scale per value, [[h; w; dims]], which GeoTessera's [dequantize_embedding]
    also reads, is refused: neither {!save_int8} nor GeoTessera's tiles write
    one. Raises what {!Nx_io.load_npy} raises. *)

val save_geotiff : string -> grid:Jord.Grid.t -> Nx.host Nx.float32_t -> unit
(** [save_geotiff name ~grid e] writes [e], a tile's embedding [[h; w; dims]]
    on [grid], to [name.tiff] as GeoTessera's GeoTIFF of a tile writes it: the
    file carries [grid]'s tags ({!Jord_geotiff.tags}), and value [k] of each
    pixel is float32 band [k + 1], described as [Tessera_Band_k] in GDAL's
    metadata (tag 42112). The nodata value is NaN, so a pixel with no embedding
    has no data in every band. The raster is cut into tiles of 256 × 256
    pixels, each compressed by Deflate.

    The file leaves out the metadata of GeoTessera's dataset, such as the
    tile's year and centre, which GeoTessera then reads from a file name of
    the form [grid_<lon>_<lat>_<year>.tiff]. A tile's grid is its place's
    ({!Tile.place}); a tile of TESSERA's own files has none, and lies on the
    grid of its landmask's place ({!Tile.of_landmask}):

    {[
      let place = Tile.load_landmask ~source:"mpc" landmask in
      Embedding.save_geotiff name ~grid:place.grid e
    ]}

    GeoTessera's own GeoTIFF of the tile carries the landmask's transform,
    so it lies up to a cell east and south of this one.

    Raises [Invalid_argument] if [e] is not of rank 3 or not of [grid.rows]
    rows and [grid.cols] columns, and as {!Nx_io.save_tiff} does, such as for
    an embedding of no values. *)
