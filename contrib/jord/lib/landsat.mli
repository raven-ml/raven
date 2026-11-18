(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The facts the USGS states about Landsat Collection 2 Level-2 surface
    reflectance and temperature products.

    Landsat satellites image the Earth with one sensor each, or a pair. The
    USGS's {e Collection 2 Level-2 Science Products} hold, for one scene of
    the Worldwide Reference System, the surface reflectance of the reflective
    bands and the surface temperature of the thermal band, on a 30 m grid,
    stored as integer digital numbers (DN), with bands that state each
    pixel's quality bit by bit. This module states what the USGS's product
    guides state about them: the bands of each sensor, the rule that turns a
    DN into its quantity, each band's valid range, and the bits of the
    quality bands (Landsat 8-9 Collection 2 Level 2 Science Product Guide,
    LSDS-1619 version 6.0, Section 6,
    {{:https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/media/files/LSDS-1619_Landsat8-9-Collection2-Level2-Science-Product-Guide-v6.pdf}};
    Landsat 4-7, LSDS-1618 version 4.0, Section 5,
    {{:https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/media/files/LSDS-1618_Landsat-4-7_C2-L2-ScienceProductGuide-v4.pdf}}).

    {b Quantities.} Every reflectance band has scale 2.75e-5 and offset -0.2,
    and the thermal band scale 0.00341802 and offset 149, in kelvin:

    {[
      reflectance = 2.75e-5 · DN - 0.2
      temperature = 0.00341802 · DN + 149
    ]}

    so DNs 1 to 65535 span reflectances -0.2 to 1.6 and temperatures 149 K to
    373 K. DN 0 is fill. Neither guide states a saturated value. LSDS-1619
    (Section 6.1.2) says a saturated OLI pixel "can show as" DN 65535 or roll
    over to the low end of the valid range, which it calls oversaturation,
    and that "The L8-L9 QA_RADSAT band will flag only the saturation cases";
    {!module-Qa_radsat} has no bit for TIRS's bands.

    {b Bits.} The quality bands are [Bitfield] bands, and their functions read
    them as bits ({!Band.section-bits}). *)

(** {1:satellites Satellites and sensors} *)

type satellite = Landsat_4 | Landsat_5 | Landsat_7 | Landsat_8 | Landsat_9
(** The type for the satellites of Collection 2 Level-2 products. Each
    satellite's instrument is its own, and two satellites' units of one
    sensor differ in their spectral responses and so in their bands'
    wavelengths ({!section-wavelengths}). Landsat 9's red band is centred at
    0.6543 µm and 36.9 nm wide, and Landsat 8's at 0.6546 µm and 37.5 nm
    wide. *)

type sensor =
  | Tm  (** Landsat 4 and 5's Thematic Mapper. *)
  | Etm_plus  (** Landsat 7's Enhanced Thematic Mapper Plus. *)
  | Oli_tirs
      (** Landsat 8 and 9's Operational Land Imager and Thermal Infrared
          Sensor, OLI-2 and TIRS-2 on Landsat 9. *)
(** The type for the sensors of Collection 2 Level-2 products. The guides
    give TM and ETM+ one set of tables (LSDS-1618), OLI and TIRS another
    (LSDS-1619). *)

val sensor : satellite -> sensor
(** [sensor s] is satellite [s]'s sensor. *)

val satellite : string -> (satellite, string) result
(** [satellite id] is the satellite of the product whose identifier is
    [id], as ["LC09_L2SP_193024_20240629_20240630_02_T1"], read from its
    first field, [LXSS]: [L], the sensor, and the satellite's number. The
    sensor is [T]
    (TM) on Landsat 4 and 5, [E] (ETM+) on Landsat 7, and [C] (OLI and TIRS)
    or [O] (OLI) on Landsat 8 and 9 (LSDS-1618 Section 4, LSDS-1619 Section
    5). TIRS-only data, [LT08] and [LT09], have no Level-2 product:
    "Landsat 8-9 TIRS-only data (LT08/LT09) cannot be processed to TOA
    Brightness Temperature or Surface Temperature" (LSDS-1619 Section 3.2),
    and they hold no reflective bands.

    The result is [Error] if [id]'s first field is not of that form. *)

(** {1:bands Bands} *)

type band =
  | Coastal  (** Coastal aerosol reflectance, OLI only. *)
  | Blue  (** Blue reflectance. *)
  | Green  (** Green reflectance. *)
  | Red  (** Red reflectance. *)
  | Nir  (** Near-infrared reflectance. *)
  | Swir16  (** Shortwave-infrared reflectance near 1.6 µm. *)
  | Swir22  (** Shortwave-infrared reflectance near 2.2 µm. *)
  | Lwir  (** Surface temperature, from the long-wave infrared band. *)
  | St_qa  (** The surface temperature's uncertainty. *)
  | St_trad  (** Thermal radiance. *)
  | St_urad  (** Upwelled radiance. *)
  | St_drad  (** Downwelled radiance. *)
  | St_atran  (** Atmospheric transmittance. *)
  | St_emis
      (** Emissivity, estimated from ASTER's Global Emissivity Dataset. *)
  | St_emsd  (** The emissivity's standard deviation. *)
  | St_cdist  (** Distance to the nearest cloud. *)
  | Sr_atmos_opacity  (** Atmospheric opacity, TM and ETM+ only. *)
  | Qa_pixel  (** Pixel quality, bits ({!module-Qa_pixel}). *)
  | Qa_radsat  (** Radiometric saturation, bits ({!module-Qa_radsat}). *)
  | Sr_qa_aerosol
      (** Aerosol quality, bits ({!module-Sr_qa_aerosol}), OLI only. *)
  | Sr_cloud_qa
      (** Cloud quality, bits ({!module-Sr_cloud_qa}), TM and ETM+ only. *)
(** The type for the bands of a product. It names the reflective and thermal
    bands by the quantity they hold, as their sensors number them
    differently, and the others by their name in the product. *)

val bands : satellite -> band list
(** [bands s] is every band of satellite [s]'s products in its guide's band
    table, in the table's order. A product whose identifier's second field
    is [L2SR] rather than [L2SP], as surface temperature "could not be
    generated" for it, holds none of the temperature bands, [Lwir] and the
    [St_] bands. *)

val name : satellite -> band -> string
(** [name s b] is band [b]'s name in a product of satellite [s], the suffix of
    its file's name, as ["SR_B4"]. The reflective bands are [SR_B1] to
    [SR_B7] and the thermal band [ST_B10] on OLI and TIRS; on TM and ETM+
    [SR_B1] to [SR_B5] and [SR_B7] and the thermal band [ST_B6]. [St_qa] is
    ["ST_QA"], and so on.

    Raises [Invalid_argument] if [s]'s products have no band [b]. *)

val band : satellite -> band -> Band.t
(** [band s b] is band [b] of a product of satellite [s], as its guide's band
    table states it (LSDS-1619 Table 6-1, LSDS-1618 Table 5-1): its [name];
    for a reflective or thermal band its [common_name] in STAC's electro-optical
    extension, ["nir"] for TM and ETM+'s band 4 and ["nir08"] for OLI's band
    5, ["lwir"] for band 6 and ["lwir11"] for band 10; its scale and offset
    (1 and 0 where the table states none); and its fill value as [nodata].
    Units are UDUNITS': ["kelvin"] for [Lwir] and [St_qa],
    ["watt/steradian/square_meter/micrometer"] for the radiances,
    ["kilometer"] for [St_cdist], none for reflectance and the other ratios.
    The quality bands are [Bitfield] bands. No band states a saturated value.

    A reflective or thermal band states its [center_wavelength] and
    [full_width_half_max] in micrometres, those of satellite [s]'s own
    instrument ({!section-wavelengths}).

    Raises [Invalid_argument] if [s]'s products have no band [b]. *)

val valid_range : satellite -> band -> int * int
(** [valid_range s b] is the inclusive range of the DNs its guide calls
    valid for band [b] of a product of satellite [s], the Valid Range column
    of its band table (LSDS-1619 Table 6-1, LSDS-1618 Table 5-1), which
    gives one for every band. It is 7273 to 43636 for reflectance,
    reflectances 0 to 1, and 293 to 61440 for temperature, 150 K to 359 K.
    These bands hold DNs outside it, beside the fill value, which decode to
    reflectances below 0 or above 1 and temperatures below 150 K or above
    359 K. [Sr_cloud_qa]'s, 1 to 63, leaves out DN 0, its fill
    ({!module-Sr_cloud_qa}).

    Raises [Invalid_argument] if [s]'s products have no band [b]. *)

(** {2:wavelengths Wavelengths}

    The products state no wavelength. This module gives each band's centre
    and width at half maximum, STAC eo's definition at that threshold, as
    the midpoint and the distance of the first and the last wavelength
    where the band-average relative spectral response is half its peak. The
    USGS gives the responses (Spectral Characteristics Viewer,
    {{:https://landsat.usgs.gov/spectral-characteristics-viewer}}), and the
    Landsat 7 Science Data Users Handbook's for TM and ETM+'s band 6
    ([HANDBOOK_LandsatThermalRSRs.spectral.dat]). The files of Landsat 9's
    OLI-2 and TIRS-2 and Landsat 8's TIRS state these "50% point"
    wavelengths. For TM, ETM+ and Landsat 8's OLI each is interpolated
    linearly between the samples of the response. The values are written
    to 0.1 nm, finer than some sources resolve. *)

(** {1:quality Quality bits} *)

(** Pixel quality, QA_PIXEL: CFMask's cloud, cloud shadow and snow flags and
    confidences, with high-confidence clouds dilated (LSDS-1619 Section
    6.1.1, Table 6-2; LSDS-1618 Section 5.2, Table 5-5). Its fill value is 1,
    the [Fill] bit alone. Its functions read DNs of an integer dtype bit by
    bit, and a program must mask fill first, with
    {!module-Qa_pixel.val-fill} ({!Band.section-bits}). *)
module Qa_pixel : sig
  type flag =
    | Fill  (** Bit 0: fill rather than image data. *)
    | Dilated_cloud  (** Bit 1: a high-confidence cloud's dilation. *)
    | Cirrus
        (** Bit 2: high-confidence cirrus. OLI only: TM and ETM+ leave it
            unused. *)
    | Cloud  (** Bit 3: high-confidence cloud. *)
    | Cloud_shadow  (** Bit 4: high-confidence cloud shadow. *)
    | Snow  (** Bit 5: high-confidence snow or ice. *)
    | Clear
        (** Bit 6, which the guides define as neither [Cloud] nor
            [Dilated_cloud] set; on every sensor their values contradict
            that. It is set with [Dilated_cloud] (LSDS-1619 Table 6-3,
            21826; LSDS-1618 Table 5-6, 5442), and unset without either over
            water (21888, 23952; 5504) and, on TM and ETM+, under cloud
            shadow (7440).
            LSDS-1618 (Section 5.2.1) advises reading [Dilated_cloud] and
            [Cloud] both unset instead, the complement of
            [any [Cloud; Dilated_cloud]]. *)
    | Water  (** Bit 7: water. *)
  (** The type for the one-bit flags. *)

  val bit : flag -> int
  (** [bit f] is the position of [f]'s bit. *)

  val any : flag list -> ('v, 's, 'd) Nx.t -> 'd Nx.bool_t
  (** [any fs x] is where the DNs [x] set one of the flags [fs]. *)

  val fill : ('v, 's, 'd) Nx.t -> 'd Nx.bool_t
  (** [fill x] is where the DNs [x] set bit 0, [Fill], which the guides
      define as fill rather than image data (LSDS-1619 Table 6-2, LSDS-1618
      Table 5-5). The band's fill value, 1, is that bit alone. *)

  type confidence =
    | Cloud_confidence  (** Bits 8 and 9. *)
    | Cloud_shadow_confidence  (** Bits 10 and 11. *)
    | Snow_ice_confidence  (** Bits 12 and 13. *)
    | Cirrus_confidence  (** Bits 14 and 15, OLI only, as [Cirrus]. *)
  (** The type for the two-bit confidence fields. *)

  type level =
    | Unset  (** 0, no confidence level set. *)
    | Low  (** 1. *)
    | Medium  (** 2, reserved in every field but [Cloud_confidence]. *)
    | High  (** 3. *)
  (** The type for confidence levels, each with its value. [High] cloud
      shadow, snow and cirrus confidences match the [Cloud_shadow], [Snow]
      and [Cirrus] flags (LSDS-1619 Section 6.1.1); the [Cloud] flag follows
      no one level, as Table 6-3 sets it with [Medium] cloud confidence in
      one value (24088) and not in another (22080). LSDS-1618 (Table 5-6)
      advises reading either the flags or the confidences, not both. *)

  val confidence : confidence -> level list -> ('v, 's, 'd) Nx.t -> 'd Nx.bool_t
  (** [confidence c ls x] is where the DNs [x] hold one of the levels [ls] in
      field [c]. *)
end

(** Radiometric saturation, QA_RADSAT: which of the sensor's bands saturated
    where a pixel was recorded, data the guides call unusable (LSDS-1619
    Section 6.1.2, Table 6-4; LSDS-1618 Section 5.1.1, Table 5-7). Each
    sensor numbers its bits by its bands. Bit [n - 1] is band [n] for bands
    1 to 7, except ETM+'s bit 5, its band 6's low gain (6L); OLI's bit 8 is
    its band 9 and ETM+'s bit 8 its band 6's high gain (6H). Two of the
    guides' worked examples disagree with their tables. LSDS-1619's DN 512
    for band 9 is bit 9, where its table puts band 9 at bit 8, and
    LSDS-1618's DN 8 for band 3 is bit 3, which its table gives band 4. This
    module follows the tables. Its band states no fill value ({!val-band}),
    so {!Band.val-nodata} of it is false on every DN, and a program must
    mask its fill with {!module-Qa_pixel.val-fill} of the same product's
    QA_PIXEL, on the same cells. Its functions read DNs of an integer dtype
    bit by bit ({!Band.section-bits}). *)
module Qa_radsat : sig
  val saturated : satellite -> band -> ('v, 's, 'd) Nx.t -> 'd Nx.bool_t
  (** [saturated s b x] is where the DNs [x] state that band [b] of a
      product of satellite [s] is computed from saturated data. ETM+'s
      temperature is computed from band 6's high gain where it is not
      saturated, and else from its low gain (LSDS-1618 Section 1.1.2), so it
      is saturated where both are.

      Raises [Invalid_argument] if [b] is not a reflective or thermal band
      of [s]'s products, or is OLI and TIRS's [Lwir], since the band states
      no saturation of TIRS's bands. *)

  val occluded : ('v, 's, 'd) Nx.t -> 'd Nx.bool_t
  (** [occluded x] is where the DNs [x] set bit 11, terrain occlusion, where
      terrain between the ground and the sensor hides the ground. The bit is
      OLI's; TM and ETM+ leave it unused, 0. *)

  val dropped : ('v, 's, 'd) Nx.t -> 'd Nx.bool_t
  (** [dropped x] is where the DNs [x] set bit 9, a dropped pixel, where a
      detector recorded no value. The bit is TM's and ETM+'s; OLI leaves it
      unused, 0. *)
end

(** Aerosol quality, SR_QA_AEROSOL, of OLI's products, which records the
    aerosol retrieval behind each pixel's surface reflectance (LSDS-1619
    Section 6.1.3, Table 6-5). Its fill value is 1, the [Fill] bit alone. The
    guide recommends against using the pixels of [High] aerosol level. Its
    functions read DNs of an integer dtype bit by bit, and a program must
    mask fill first, with {!module-Sr_qa_aerosol.val-fill}
    ({!Band.section-bits}). *)
module Sr_qa_aerosol : sig
  type flag =
    | Fill  (** Bit 0: fill. *)
    | Valid_retrieval  (** Bit 1: a valid aerosol retrieval. *)
    | Water  (** Bit 2: water. *)
    | Interpolated  (** Bit 5: aerosol interpolated from its neighbours. *)
  (** The type for the one-bit flags. *)

  val bit : flag -> int
  (** [bit f] is the position of [f]'s bit. *)

  val any : flag list -> ('v, 's, 'd) Nx.t -> 'd Nx.bool_t
  (** [any fs x] is where the DNs [x] set one of the flags [fs]. *)

  val fill : ('v, 's, 'd) Nx.t -> 'd Nx.bool_t
  (** [fill x] is where the DNs [x] set bit 0, [Fill], which the guide
      defines as fill (LSDS-1619 Table 6-5). The band's fill value, 1, is
      that bit alone. *)

  type level =
    | Climatology  (** 0, no aerosol correction applied. *)
    | Low  (** 1. *)
    | Medium  (** 2. *)
    | High  (** 3. *)
  (** The type for the aerosol levels of bits 6 and 7, each with its
      value. *)

  val level : level list -> ('v, 's, 'd) Nx.t -> 'd Nx.bool_t
  (** [level ls x] is where the DNs [x] hold one of the levels [ls]. *)
end

(** Cloud quality, SR_CLOUD_QA, of TM and ETM+'s products: LEDAPS's flags
    (LSDS-1618 Tables 5-2 and 5-4). Its fill value is 0: Table 5-4 reads DN 0
    as fill, and Table 5-1 gives the layer's valid range as 1 to 63
    ({!valid_range}). The guide finds {!module-Qa_pixel} likely the more
    accurate. Its functions read DNs of an integer dtype bit by bit, and a
    program must mask fill first, with {!module-Sr_cloud_qa.val-fill}
    ({!Band.section-bits}). *)
module Sr_cloud_qa : sig
  type flag =
    | Ddv  (** Bit 0: dark dense vegetation. *)
    | Cloud  (** Bit 1. *)
    | Cloud_shadow  (** Bit 2. *)
    | Adjacent_cloud  (** Bit 3: adjacent to cloud. *)
    | Snow  (** Bit 4. *)
    | Water  (** Bit 5. *)
  (** The type for the flags. *)

  val bit : flag -> int
  (** [bit f] is the position of [f]'s bit. *)

  val any : flag list -> ('v, 's, 'd) Nx.t -> 'd Nx.bool_t
  (** [any fs x] is where the DNs [x] set one of the flags [fs]. *)

  val fill : ('v, 's, 'd) Nx.t -> 'd Nx.bool_t
  (** [fill x] is where the DNs [x] are 0, which LSDS-1618 Table 5-4 reads
      as fill. *)
end
