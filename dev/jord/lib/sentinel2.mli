(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Sentinel-2 Level-2A: the facts ESA states about its surface reflectance
    products.

    Sentinel-2's MultiSpectral Instrument images the Earth in 13 spectral
    bands. ESA's {e Level-2A} product holds the surface (bottom-of-atmosphere)
    reflectance of 12 of them on one tile of a UTM grid, stored as 16-bit
    digital numbers (DN), and a {e scene classification} (SCL) of each 20 m
    cell. This module states what ESA states about them: the bands and their
    resolutions, the scene classes, the rule that turns a DN into
    reflectance, and the fields that identify the acquisition a product holds
    (ESA SentiWiki, "S2 Products" and "S2 Processing",
    {{:https://sentiwiki.copernicus.eu/web/s2-products}}). Which classes a
    model treats as invalid, and which of two processings of one acquisition
    it keeps, are the model's choices.

    {b Reflectance.} A product states a {e quantification value} Q and, from
    processing baseline 04.00 (25 January 2022), an offset for each band,
    BOA_ADD_OFFSET, which shifts the stored range so that the negative
    reflectances of dark surfaces are kept:

    {[
      reflectance = (DN + BOA_ADD_OFFSET) / Q
    ]}

    Before 04.00 the range is not shifted, so the offset is 0. DN 0 is the
    product's NODATA value and 65535 its SATURATED value. The product's
    metadata, [MTD_MSIL2A.xml], states these values ([Jord_safe.l2a] reads
    them); a catalogue's item may not: Microsoft's Planetary Computer, for
    one, states no offset in its items. *)

(** {1:bands Bands} *)

type band =
  | B01
  | B02
  | B03
  | B04
  | B05
  | B06
  | B07
  | B08
  | B8A
  | B09
  | B11
  | B12
(** The type for Level-2A's reflectance bands, by ESA's names. The
    instrument's 13th band, B10 (cirrus), is omitted from Level-2A products,
    as it holds no surface information. *)

val bands : band list
(** [bands] is every band, in ESA's order: [B01] to [B08], [B8A], [B09],
    [B11], [B12]. *)

val name : band -> string
(** [name b] is [b]'s name in the product's file names and in STAC items, as
    ["B04"] or ["B8A"]. *)

val resolution : band -> int
(** [resolution b] is [b]'s native resolution in metres: 10 for [B02],
    [B03], [B04] and [B08]; 20 for [B05], [B06], [B07], [B8A], [B11] and
    [B12]; 60 for [B01] and [B09]. A product also holds each band but [B08]
    resampled to the coarser resolutions, and [B01] at 20 m; it holds [B08]
    at 10 m only. *)

(** {1:reflectance Reflectance} *)

type metadata = {
  baseline : string;  (** The processing baseline, as ["05.10"]. *)
  quantification : float;  (** Q, BOA_QUANTIFICATION_VALUE. *)
  offsets : (band * int) list;
      (** BOA_ADD_OFFSET of each band the product states one for. *)
  nodata : int;  (** NODATA. *)
  saturated : int;  (** SATURATED. *)
}
(** The type for the values of a product's metadata that decode its DNs. *)

val reflectance : metadata -> band -> (Band.t, string) result
(** [reflectance m b] is band [b] of the product [m] describes: scale 1/Q,
    offset BOA_ADD_OFFSET/Q, nodata and saturated values [m]'s. Its
    {!Band.decode} computes the product and sum in float64, which can differ
    from ESA's quotient in float64's last place; rounded to float32 it is
    ESA's quotient rounded to float32 for every 16-bit DN, at Q 10000 and an
    offset of 0 or -1000. Its [name] is [name b].

    The result is [Error] if [m.baseline] is not two digits, a dot and two
    digits, or if it is 04.00 or later and [m.offsets] has no offset for
    [b]: the reflectance is then unknown, by an offset ESA set to -1000 at
    04.00.

    Raises [Invalid_argument] if [m.quantification] or its inverse is not
    positive and finite. *)

(** {1:scl Scene classification} *)

(** Scene classes. ESA's processor classifies each 20 m cell of a product
    into one of 12 classes, stored as a byte, and resamples the
    classification to 60 m by the most frequent class ("S2 Processing",
    Table 3, {{:https://sentiwiki.copernicus.eu/web/s2-processing}}). *)
module Scl : sig
  type t =
    | No_data  (** 0, NO_DATA. *)
    | Saturated_or_defective  (** 1, SATURATED_OR_DEFECTIVE. *)
    | Cast_shadows
        (** 2, CAST_SHADOWS, named DARK_FEATURES before baseline 05.11. *)
    | Cloud_shadows  (** 3, CLOUD_SHADOWS. *)
    | Vegetation  (** 4, VEGETATION. *)
    | Not_vegetated  (** 5, NOT_VEGETATED. *)
    | Water  (** 6, WATER. *)
    | Unclassified  (** 7, UNCLASSIFIED. *)
    | Cloud_medium_probability  (** 8, CLOUD_MEDIUM_PROBABILITY. *)
    | Cloud_high_probability  (** 9, CLOUD_HIGH_PROBABILITY. *)
    | Thin_cirrus  (** 10, THIN_CIRRUS. *)
    | Snow_or_ice  (** 11, SNOW or ICE. *)
  (** The type for scene classes, each with its code and ESA's name. *)

  val code : t -> int
  (** [code c] is the byte that stands for [c]. *)

  val mem : t list -> ('v, 's, 'd) Nx.t -> 'd Nx.bool_t
  (** [mem cs x] is where the codes [x] stand for one of the classes [cs].
      A code that stands for no class is in none.

      Raises [Invalid_argument] if [x] holds booleans or complex numbers. *)
end

(** {1:identity Identity} *)

type key = {
  platform : string;  (** The satellite, as ["S2A"]. *)
  tile : string;  (** The tile, as ["33UUU"]. *)
  sensing_start : string;
      (** When the datastrip's sensing started, as ["20240629T102556"]. *)
}
(** The type for the acquisition a product holds. One continuous acquisition,
    a {e datatake}, reaches the ground in {e datastrips}, each processed on
    its own, and a product is one tile of one datastrip: where two datastrips
    meet, a tile has a product from each. A product processed again, at
    another baseline, keeps its platform, tile and datastrip sensing start,
    so products with equal keys hold the same acquisition. *)

val key : tile:string -> string -> (key, string) result
(** [key ~tile id] is the key of a product of [tile] whose datastrip
    identifier is [id] (STAC's [s2:datastrip_id]), as
    ["S2A_OPER_MSI_L2A_DS_MSFT_20240629T161811_S20240629T102556_N05.10"]: the
    mission, file class, file type, processing centre and processing time,
    then fields marked by their first letter, [S] the sensing start and [N]
    the baseline. The positions of these fields are those of the
    identifiers catalogues give, which all name a processing centre; ESA's
    naming text in its Products Specification Document is not followed
    here. An identifier of another form is an [Error], never another key.

    The result is [Error] if [tile] is not two digits and three capital
    letters, or [id] is not of that form. *)
