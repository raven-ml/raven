(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The facts ESA states about Sentinel-2's Level-1C and Level-2A
    reflectance products.

    Sentinel-2's MultiSpectral Instrument images the Earth in 13 spectral
    bands. ESA's {e Level-1C} product holds the top-of-atmosphere
    reflectance of all 13 on one tile, a square in a UTM zone, and its
    {e Level-2A} product the surface (bottom-of-atmosphere) reflectance of
    12 of them and a {e scene classification} (SCL) of each 20 m cell, each
    stored as 16-bit digital numbers (DN). This module states what ESA
    states about them: the bands and their resolutions, the scene classes,
    the rule that turns a DN into reflectance, and the fields that identify
    the acquisition a product holds (ESA SentiWiki, "S2 Products" and "S2
    Processing",
    {{:https://sentiwiki.copernicus.eu/web/s2-products}}). A Level-2A
    product also holds aerosol optical thickness (AOT) and water vapour
    (WVP) layers, whose quantification values its metadata states beside
    the reflectance's. This module's bands are the 13 spectral bands, so a
    program decodes those layers itself. *)

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
  | B10
  | B11
  | B12
(** The type for the instrument's 13 bands, by ESA's names. A Level-2A
    product holds an image of each but [B10] (cirrus), which it omits "as it
    does not contain surface information"; its metadata still states [B10]'s
    offset and view angles. *)

val bands : band list
(** [bands] is every band, in ESA's order, the order of {!index}: [B01] to
    [B08], [B8A], [B09] to [B12]. *)

val index : band -> int
(** [index b] is the number a product's metadata gives [b], the [bandId]
    its [Spectral_Information] list pairs with [b]'s [physicalBand], and the
    number by which it names [b]'s offset ([band_id]) and view angles
    ([bandId]). It is 0 for [B01] to 7 for [B08], 8 for [B8A], then 9 to 12
    for [B09] to [B12]. The schemas of ESA's Products Specification Document
    type the number as [A_BAND_NUMBER], ["0"] to ["12"], and the name as
    [A_PHYSICAL_BAND_NAME] ([image.xsd] of PSD 15, as SNAP ships it,
    {{:https://github.com/senbox-org/optical-toolbox/tree/master/opttbx-s2msi-reader/src/main/resources/schemas/PSD150}}). *)

val name : band -> string
(** [name b] is [b]'s name in the product's file names and in STAC items, as
    ["B04"] or ["B8A"]. *)

val resolution : band -> int
(** [resolution b] is [b]'s native resolution in metres: 10 for [B02],
    [B03], [B04] and [B08]; 20 for [B05], [B06], [B07], [B8A], [B11] and
    [B12]; 60 for [B01], [B09] and [B10]. A Level-2A product also holds each
    band but [B08] resampled to the coarser resolutions, and [B01] at 20 m;
    it holds [B08] at 10 m only. *)

type wavelength = {
  central : float;
      (** CENTRAL, which ESA calls the central wavelength (Products
          Specification Document 15.0, [Spectral_Information]). *)
  min : float;  (** MIN. *)
  max : float;  (** MAX. *)
}
(** The type for a band's wavelengths in nanometres, as a product's
    [Spectral_Information] states them, those of its satellite's instrument.
    [MIN] and [MAX] are the first and last wavelengths of the band's
    tabulated response, so [MAX] - [MIN] is the response's extent. *)

(** {1:reflectance Reflectance}

    A product states a {e quantification value} Q and, from processing
    baseline 04.00 (25 January 2022), an offset for each band, which shifts
    the stored range so that the negative reflectances of dark surfaces are
    kept:

    {[
      reflectance = (DN + offset) / Q
    ]}

    Level-1C names them QUANTIFICATION_VALUE and RADIO_ADD_OFFSET, Level-2A
    BOA_QUANTIFICATION_VALUE and BOA_ADD_OFFSET, and the rule is the same.
    Before 04.00 the range is not shifted, so the offset is 0. DN 0 is the
    product's NODATA value and 65535 its SATURATED value. The product's
    metadata, [MTD_MSIL1C.xml] or [MTD_MSIL2A.xml], states these values
    ([Jord_safe.l1c] and [Jord_safe.l2a] read them); a catalogue's item may
    state none of them. The Sun near the horizon leaves valid DNs where no
    reflectance was computed ({!section-horizon}). *)

type metadata = {
  baseline : string;  (** The processing baseline, as ["05.10"]. *)
  quantification : float;
      (** Q: QUANTIFICATION_VALUE, or BOA_QUANTIFICATION_VALUE at Level-2A. *)
  offsets : (band * int) list;
      (** The offset, RADIO_ADD_OFFSET or BOA_ADD_OFFSET, of each band the
          product states one for. *)
  nodata : int;  (** NODATA. *)
  saturated : int;  (** SATURATED. *)
  wavelengths : (band * wavelength) list;
      (** The wavelengths of each band the product states them for. *)
}
(** The type for the values of a product's metadata that decode its DNs and
    state its bands' wavelengths. *)

val baseline : string -> (int * int, string) result
(** [baseline s] is the processing baseline [s], as ["05.10"], ESA's form of
    two digits, a dot and two digits, as its major and minor numbers:
    [(5, 10)], which order baselines as ESA numbers them.

    The result is [Error] if [s] is not of that form. *)

val offset : metadata -> band -> (int, string) result
(** [offset m b] is the offset of band [b] in the product [m]
    describes, the one [m] states, or 0 before baseline 04.00, whose products
    shift no range.

    The result is [Error] if [m.baseline] is not of {!val-baseline}'s form,
    or if it is 04.00 or later and [m.offsets] has no offset for [b], whose
    offset is then unknown, ESA having set it to -1000 at 04.00. A reader of
    a product's metadata may leave out an offset the product states as "Not
    applicable", as [Jord_safe]'s readers do, and from 04.00 such a band's
    offset is then an [Error] here. *)

val reflectance : metadata -> band -> (Band.t, string) result
(** [reflectance m b] is band [b] of the product [m] describes, with scale 1/Q,
    offset {!offset}/Q, and the nodata and saturated values [m] states. Its
    {!Band.decode} computes the product and sum in float64, which can differ
    from ESA's quotient in float64's last place; rounded to float32 it is ESA's
    quotient rounded to float32 for every 16-bit DN, at Q 10000 and an offset
    of 0 or -1000. Its [name] is [name b]. Its [center_wavelength] is [m]'s
    [CENTRAL] for [b], and its [full_width_half_max] [MAX] - [MIN], in
    micrometres, ESA's statement for the product's satellite; both are [None]
    where [m] states no wavelengths for [b].

    The result is [Error] where {!offset} is.

    Raises [Invalid_argument] if [m.quantification] or its inverse is not
    positive and finite. *)

(** {2:horizon The Sun near the horizon}

    ESA's "S2 Products" ("Very high Solar Zenith Angle") states what a
    product holds where the Sun is low. A Level-1C product computes
    reflectance only where the Sun's zenith is below 89.9°, and sets it to
    0 elsewhere, which is DN 1000 at offset -1000, a valid value. At offset
    0, before 04.00, reflectance 0 is DN 0, NODATA, and ESA does not say
    what such a product stored. A Level-2A product is not made for a tile
    whose mean Sun zenith is over 90°. In one that is made, a pixel whose
    Sun zenith is over 89.9° holds DN 1, at Q 10000 and offset -1000 a
    valid reflectance of -0.0999. *)

(** {1:scl Scene classification} *)

(** Scene classes. ESA's processor classifies each 20 m cell of a
    Level-2A product into one of 12 classes, stored as a byte, and resamples the
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
  (** [mem cs x] is where the codes [x] stand for one of the classes [cs],
      compared as {!Band} compares codes. It is false where a code stands for
      none of the scene classes.

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
    its own, and a product is one tile of one datastrip. Where two
    datastrips meet, a tile has a product from each. A product processed
    again, at another baseline or to the other level, keeps its platform,
    tile and datastrip sensing start, so products with equal keys hold the
    same acquisition. A model chooses which of them it keeps. *)

val key : tile:string -> string -> (key, string) result
(** [key ~tile id] is the key of a product of [tile] whose datastrip identifier
    is [id] (STAC's [s2:datastrip_id]), as
    ["S2A_OPER_MSI_L2A_DS_MSFT_20240629T161811_S20240629T102556_N05.10"], whose
    fields are the mission, file class, file type, processing centre and
    processing time, then fields marked by their first letter, [S] the sensing
    start and [N] the baseline. The fields sit at the positions of the
    identifiers catalogues give, which all name a processing centre.

    The result is [Error] if [tile] is not two digits and three capital
    letters, or [id] is not of that form. *)
