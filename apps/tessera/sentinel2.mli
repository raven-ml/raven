(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** TESSERA's Sentinel-2 recipe: which Level-2A products of a day it keeps, and
    the frame it makes of them on a tile's cells.

    {!Jord.Sentinel2} states what ESA's products hold; this module states what
    TESSERA does with them, as its preprocessing does
    (github.com/ucam-eo/tessera, [tessera_preprocessing/s2_fast_processor.py]):
    - a cell is {e valid} in a product where its scene class is none of
      {!invalid}, read by the nearest cell;
    - a day's cell takes its bands from the first product, in {!distinct}'s
      order, valid there, and is observed where some product is;
    - a day is kept when its products observe at least {!min_coverage} percent
      of the region of interest;
    - each band is read bilinearly, as GDAL reads it: the mean of the four cells
      around the point, with data, rounded half up to a digital number
      ({!Jord.Warp.bilinear}), then harmonised by the product's offset, so the
      frame holds surface reflectance times 10⁴ ({!Jord.Sentinel2}), negatives
      kept.

    Where TESSERA's code departs from that, this module does not follow it: it
    decides a product's offset by the product's metadata, never by the date; a
    cell whose chosen product lacks a band's data there is not observed; an
    unobserved cell's bands are 0; of two processings of one acquisition it
    keeps one. *)

(** {1:rules Rules} *)

val bands : Jord.Sentinel2.band list
(** [bands] is the ten bands of a frame, in TESSERA's order: [B04], [B02],
    [B03], [B08], [B8A], [B05], [B06], [B07], [B11], [B12]. *)

val invalid : Jord.Sentinel2.Scl.t list
(** [invalid] is the scene classes TESSERA treats as no observation: [No_data],
    [Saturated_or_defective], [Cast_shadows], [Cloud_shadows],
    [Cloud_medium_probability] and [Cloud_high_probability] (codes 0, 1, 2, 3, 8
    and 9). Water, thin cirrus and snow are observations. *)

val min_coverage : float
(** [min_coverage] is [0.01], in percent of the region of interest's cells. *)

val distinct : Jord_stac.item list -> Jord_stac.item list
(** [distinct items] is [items] with one product of each acquisition
    ({!Jord.Sentinel2.key}, from [s2:mgrs_tile] and [s2:datastrip_id]): the one
    of the highest [s2:processing_baseline], compared as numbers
    ({!Jord.Sentinel2.baseline}), and of the greatest id among those,
    as ESA's Collection-1 reprocessing supersedes earlier processings with one
    uniform baseline. They are sorted by time, then id.

    Raises [Failure] naming the item if one lacks a time or those properties, or
    {!Jord.Sentinel2.key} or {!Jord.Sentinel2.baseline} refuses them. *)

val first_valid :
  roi:Nx.host Nx.bool_t -> Nx.host Nx.uint8_t list -> Nx.host Nx.int32_t
(** [first_valid ~roi scls] is, for each cell of [roi], [[h; w]], the index in
    [scls] of the first scene classification, [[h; w]] each, valid at the cell,
    or [-1] where none is or [roi] is false. *)

val covered : roi:Nx.host Nx.bool_t -> Nx.host Nx.int32_t -> bool
(** [covered ~roi choice] is [true] when [choice] ({!first_valid}) gives a
    product to at least {!min_coverage} percent of [roi]'s true cells, computed
    as TESSERA does: [100 · chosen / cells] in float64 against [0.01]. It is
    [false] for an empty region. *)

(** {1:frames Frames} *)

type frame = {
  bands : Nx.host Nx.int16_t;  (** [[h; w; 10]], {!bands} in order. *)
  mask : Nx.host Nx.uint8_t;  (** [[h; w]], [1] where observed. *)
}
(** The type for a day's frame. Bands are 0 where the mask is. *)

val mosaic :
  Nx.host Nx.int32_t ->
  (Nx.host Nx.int16_t * Nx.host Nx.bool_t) option list ->
  frame
(** [mosaic choice products] is the frame whose cell takes, where [choice]
    ({!first_valid}) is [k], the bands [[h; w; 10]] of product [k] of
    [products], observed where its mask [[h; w]] holds. A cell whose product is
    [None], or that no product takes, is not observed. *)

val day :
  Stac.session ->
  roi:Nx.host Nx.bool_t ->
  centres:(Jord.Crs.t -> Nx.host Nx.float64_t) ->
  Jord_stac.item list ->
  frame option
(** [day s ~roi ~centres items] is the frame of MPC's Level-2A products [items],
    of one day, on the cells of a grid whose centres in a system are
    [centres crs] ({!Jord.Warp.centres}), and [roi] its region of interest:
    [None] if they observe less than {!min_coverage} percent of it. It reads
    each product's scene classification, and the bands and metadata
    ([MTD_MSIL2A.xml], {!Jord_safe.l2a}) only of the products some cell takes:
    by {!Stac.window}, so it fetches the tiles of their rasters that the cells
    cover. [items] are taken in order, as {!distinct} sorts them.

    Raises [Failure] naming the item if an item lacks an asset or its grid, its
    metadata does not decode or states no offset of a band
    ({!Jord.Sentinel2.offset}) or a quantification value other than 10⁴, a
    harmonised value in its window lies outside int16, and as {!Stac.window}
    does. ESA states DNs in \[1, 2¹⁵ − 1\], and MPC's band files store 15
    bits, so harmonised values lie in \[−999, 32767\] and the int16 check
    fails only on a product that breaks its specification, which ends the
    day rather than wrapping its value. *)

val days :
  Jord_stac.item list ->
  (Stac.date * Jord_stac.item list) list * Jord_stac.item list
(** [days items] is the {!distinct} items of [items] grouped by the UTC date of
    their time ({!Stac.time}), the days in order, and the items left out for
    lacking an asset {!day} reads: [product-metadata], [SCL] and each of
    {!bands}. Only complete items are deduplicated, so a processing lacking an
    asset leaves an earlier one of its acquisition in its place.

    Raises as {!distinct} does. *)
