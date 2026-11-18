(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The facts NASA states about MODIS Collection 6.1's vegetation indices and
    surface reflectance, MOD13Q1 and MOD09A1.

    The Moderate Resolution Imaging Spectroradiometer (MODIS) flies on NASA's
    Terra and Aqua. Its land products are composites, each pixel the best
    observation of a period, stored as integer digital numbers (DN) on one
    tile of MODLAND's sinusoidal grid. {e MOD13Q1} holds 16 days of
    vegetation indices at 250 m, with the reflectances they were computed
    from (MODIS Vegetation Index User's Guide, MOD13 series, version 3.10,
    Collection 6.1, Section 5,
    {{:https://lpdaac.usgs.gov/documents/621/MOD13_User_Guide_V61.pdf}});
    {e MOD09A1} holds 8 days of surface reflectance in bands 1 to 7 at 500 m
    (MODIS Collection 6.1 Land Surface Reflectance Product User Guide, Section
    3.1.5,
    {{:https://lpdaac.usgs.gov/documents/925/MOD09_User_Guide_V61.pdf}}).
    This module states what the guides state about them: the layers of each
    product, the rule that turns a DN into its quantity, each layer's valid
    range, the bits of the quality layers, and the grid of each tile. Aqua's
    products, MYD13Q1 and MYD09A1, are the same: "The 'MOD' prefix should be
    taken as referring to the datasets in general" (MOD09 guide, Section
    4.5).

    {b Quantities.} Every reflectance layer and both vegetation indices have
    scale 0.0001, the angle layers 0.01 in degrees, and no layer an offset
    (MOD13 Table 1, MOD09 Table 6):

    {[
      reflectance = 0.0001 · DN
      NDVI = 0.0001 · DN
      angle = 0.01 · DN
    ]}

    MOD13Q1's fill is -1000 in its reflectance layers, -3000 in its indices
    and -10000 in its zenith angles; MOD09A1's is -28672 in its reflectance
    layers. Two fills lie inside their layer's valid range: MOD13Q1's relative
    azimuth's -4000, an azimuth of -40°, and MOD09A1's 0 in its three angle
    layers, a zenith or a relative azimuth of 0°. The guides state these DNs
    as fill, and {!Band.decode} masks them, so it also drops the
    observations that lie exactly there. A caller that keeps them decodes
    the layer without its fill and takes validity from another layer. No
    layer states a saturated value.

    The reflectance bands are MODIS's bands 1 to 7, named by the ranges of
    MOD09 Table 6: 620-670, 841-876, 459-479, 545-565, 1230-1250, 1628-1652 and
    2105-2155 nm. Neither guide says at what fraction of a band's response
    those limits lie, and Terra's and Aqua's MODIS are two instruments, so a
    layer states no centre or width, as {!Hls}'s do not.

    {b Tiles.} Each product is one tile of MODLAND's sinusoidal {e tiling} on
    {!Crs.modis}: 36 columns of tiles, numbered [h] from 0 at the west, and 18
    rows, numbered [v] from 0 at the north, "(0,0) ... in the upper left
    corner ... The tile in the bottom right corner is (35,17)" (MODLAND,
    "Sinusoidal Tile Grid",
    {{:https://modis-land.gsfc.nasa.gov/MODLAND_grid.html}}). The tiling's
    upper left corner is at x = -20015109.354 m, y = 10007554.677 m on the
    sphere of radius R = 6371007.181 m (MODLAND's tile calculator,
    [tilemap3.c], release 4.0,
    {{:https://landweb.modaps.eosdis.nasa.gov/data/filePost/tilemap3_r4_0.tar}}),
    so each tile is T = 20015109.354 / 18 m on a side, 1111950.519667 m to
    the micrometre. The corner's x is πR, 20015109.3557974 m, truncated to
    the millimetre, 1.8 mm short, and its y is half of it. So a tile row
    spans 10° of latitude and a tile column 10° of longitude at the equator,
    to under 2e-8°. The calculator's 1 km cell, 926.62543305 m, is T / 1200 to
    6 nm, and the products' corners follow T. *)

(** {1:products Products} *)

type product =
  | Mod13q1  (** Vegetation indices, 16 days at 250 m (MOD13 Section 5). *)
  | Mod09a1  (** Surface reflectance, 8 days at 500 m (MOD09 Section 3.1.5). *)
(** The type for the products, of Terra or of Aqua. *)

(** {1:bands Bands} *)

type band =
  | Ndvi  (** The normalized difference vegetation index, MOD13Q1 only. *)
  | Evi  (** The enhanced vegetation index, MOD13Q1 only. *)
  | Vi_quality
      (** The indices' quality, bits ({!module-Vi_quality}), MOD13Q1 only. *)
  | Red  (** Band 1's reflectance, 620-670 nm. *)
  | Nir  (** Band 2's reflectance, 841-876 nm. *)
  | Blue  (** Band 3's reflectance, 459-479 nm. *)
  | Green  (** Band 4's reflectance, 545-565 nm, MOD09A1 only. *)
  | Swir12  (** Band 5's reflectance, 1230-1250 nm, MOD09A1 only. *)
  | Swir16  (** Band 6's reflectance, 1628-1652 nm, MOD09A1 only. *)
  | Swir22
      (** Band 7's reflectance, 2105-2155 nm, which the MOD13 guide calls
          MIR. *)
  | Qc  (** The reflectance's quality, bits ({!module-Qc}), MOD09A1 only. *)
  | State  (** The pixel's state, bits ({!module-State}), MOD09A1 only. *)
  | View_zenith  (** The view zenith angle. *)
  | Sun_zenith  (** The Sun's zenith angle. *)
  | Relative_azimuth  (** The azimuth of the view relative to the Sun's. *)
  | Day_of_year  (** The day of the year of the pixel's observation. *)
  | Reliability
      (** The pixel's reliability, a classified band
          ({!module-Reliability}), MOD13Q1 only. *)
(** The type for the layers of a product, its science data sets. It names
    the reflective ones by their quantity, as {!Landsat}'s and {!Hls}'s are,
    and the others by what they hold. *)

val bands : product -> band list
(** [bands p] is every layer of product [p], in the order of its table (MOD13
    Table 1, MOD09 Table 6). *)

val name : product -> band -> string
(** [name p b] is layer [b]'s name in a product [p], its science data set's
    name, as the data's providers list it (LP DAAC, "MOD13Q1 v061" and
    "MOD09A1 v061", Variables,
    {{:https://www.earthdata.nasa.gov/data/catalog/lpcloud-mod13q1-061}},
    {{:https://www.earthdata.nasa.gov/data/catalog/lpcloud-mod09a1-061}}):
    ["250m 16 days NDVI"] in MOD13Q1, which MOD13 Table 1 writes as
    ["250/500m 16 days NDVI"] for MOD13Q1 and MOD13A1 together, and
    ["sur_refl_b01"] in MOD09A1. Microsoft's Planetary Computer names an
    item's asset by it, with an underscore for each space, as
    ["250m_16_days_NDVI"].

    Raises [Invalid_argument] if [p] has no layer [b]. *)

val band : product -> band -> Band.t
(** [band p b] is layer [b] of a product [p], as MOD13 Table 1 and MOD09
    Table 6 state it: its [name]; for a reflective layer its [common_name] in
    STAC's electro-optical extension, whose table lists MODIS's bands 1, 2, 3,
    4, 6 and 7 as ["red"], ["nir"], ["blue"], ["green"], ["swir16"] and
    ["swir22"] and band 5 as none (version 2.0.0, "Common Band Names",
    {{:https://github.com/stac-extensions/eo/blob/v2.0.0/README.md}}); its
    scale (1 where the table states none),
    offset 0 and fill value as [nodata]. Units are UDUNITS': ["degree"] for
    the angles, none for reflectance, the indices and the day of the year. The
    quality layers are [Bitfield] bands; [Reliability] is a [Classes] band of
    Table 4's ranks, -1 its fill. No layer states a saturated value, a centre
    wavelength or a width.

    Raises [Invalid_argument] if [p] has no layer [b]. *)

val valid_range : product -> band -> int * int
(** [valid_range p b] is the inclusive range of the DNs its guide calls valid
    for layer [b] of a product [p] (MOD13 Table 1, MOD09 Table 6): -2000 to
    10000 for the indices; 0 to 10000 for MOD13Q1's reflectance and -100 to
    16000 for MOD09A1's; 0 to 18000 for the zenith angles and -18000 to 18000
    for the relative azimuth; 1 to 366 for the day of the year; 0 to 65534 for
    VI quality, 0 to 57343 for the state flags and 0 to 3 for the
    reliability. MOD09 Table 6 gives the reflectance quality's range as NA,
    and LP DAAC's page of the product states it, 0 to 4294966531, which is
    the range here. A layer holds DNs outside its range beside its fill.

    Raises [Invalid_argument] if [p] has no layer [b]. *)

(** {1:grid Tiles} *)

val tile : product -> h:int -> v:int -> Grid.t
(** [tile p ~h ~v] is the grid of tile [(h, v)] of a product [p] on
    {!Crs.modis}: north-up, n × n cells of side T / n, with n 4800 for MOD13Q1
    and 2400 for MOD09A1, its corner at (-20015109.354 + h T, 10007554.677 -
    v T). Each coordinate of every tile's corner is within an ulp of its
    exact value, and the cell's side within half of one. A tile that holds
    no land has a grid too, though no product holds it.

    Raises [Invalid_argument] unless [0 <= h <= 35] and [0 <= v <= 17]. *)

(** {1:quality Quality}

    The quality layers are [Bitfield] bands, and their functions read them
    as bits ({!Band.section-bits}). A guide's bit combination is written
    from its field's highest bit to its lowest. MOD09 Table 10's [1011],
    "missing input", is the value its text calls "band quality value 11"
    (Section 4.1), where read from the lowest bit it would be 13,
    "correction out of bounds"; and MOD13 says of the usefulness index
    that "the lowest quality is equal to a value of 1100 (i.e., 13 levels)"
    (Section 5.5.2), levels 0 to 12. *)

type modland =
  | Good_quality  (** 00: produced at good, ideal quality. *)
  | Other_quality
      (** 01: produced at less than ideal quality; check the other
          fields. *)
  | Not_produced_cloud
      (** 10: not produced because of cloud, or, in MOD13Q1, produced but
          probably cloudy. *)
  | Not_produced_other  (** 11: not produced for another reason. *)
(** The type for the values of MODLAND's bits, bits 0 and 1 of every MODIS
    land product's quality layer, by the names of the summaries they give a
    tile (MOD13 Tables 5 and 6, MOD09 Table 10). *)

type land_water =
  | Shallow_ocean  (** 0. *)
  | Land  (** 1. *)
  | Shoreline  (** 2: ocean coastlines and lake shorelines. *)
  | Shallow_inland_water  (** 3. *)
  | Ephemeral_water  (** 4. *)
  | Deep_inland_water  (** 5. *)
  | Moderate_ocean  (** 6: moderate or continental ocean. *)
  | Deep_ocean  (** 7. *)
(** The type for the classes of the land/water mask, a field of three bits in
    both quality layers that state it (MOD13 Table 5, MOD09 Table 13). MOD13
    Table 5 writes its first two values with two digits, 00 and 01. *)

(** MOD13Q1's VI quality, the conditions under which the vegetation indices
    were made (MOD13 Section 5.5.2, Table 5). Its fill value 65535 sets every
    bit, which {!module-Vi_quality.fill} finds. Its functions read DNs of an
    integer dtype bit by bit, and a program must mask fill first
    ({!Band.section-bits}). *)
module Vi_quality : sig
  val modland : modland list -> ('v, 's, 'd) Nx.t -> 'd Nx.bool_t
  (** [modland ms x] is where the DNs [x] hold one of [ms] in bits 0 and 1. *)

  val usefulness : int list -> ('v, 's, 'd) Nx.t -> 'd Nx.bool_t
  (** [usefulness us x] is where the DNs [x] hold one of [us] in the
      usefulness index, bits 2 to 5: 0 the highest quality and 12 the lowest,
      a sum of Table 7's scores, then 13 "not useful", 14 "L1B data faulty"
      and 15 "not useful for any other reason/not processed". A number outside
      0 to 15 is held nowhere. *)

  type aerosol =
    | Climatology  (** 0, aerosol from climatology. *)
    | Low  (** 1. *)
    | Intermediate  (** 2. *)
    | High  (** 3. *)
  (** The type for the aerosol quantities of bits 6 and 7. *)

  val aerosol : aerosol list -> ('v, 's, 'd) Nx.t -> 'd Nx.bool_t
  (** [aerosol as_ x] is where the DNs [x] hold one of the quantities
      [as_]. *)

  type flag =
    | Adjacent_cloud  (** Bit 8: adjacent cloud detected. *)
    | Brdf_corrected  (** Bit 9: atmosphere-BRDF coupled correction. *)
    | Mixed_clouds  (** Bit 10: possibly mixed clouds. *)
    | Snow_ice  (** Bit 14: possibly snow or ice. *)
    | Shadow  (** Bit 15: possibly shadow. *)
  (** The type for the one-bit flags. The guide says of bits 8 and 9 that "in
      practice no adjacency correction or BRDF correction is applied and
      hence they are always set to 0". *)

  val bit : flag -> int
  (** [bit f] is the position of [f]'s bit. *)

  val any : flag list -> ('v, 's, 'd) Nx.t -> 'd Nx.bool_t
  (** [any fs x] is where the DNs [x] set one of the flags [fs]. *)

  val land_water : land_water list -> ('v, 's, 'd) Nx.t -> 'd Nx.bool_t
  (** [land_water ls x] is where the DNs [x] hold one of [ls] in bits 11 to
      13. The indices are not computed over water (Section 5.5.2). *)

  val fill : ('v, 's, 'd) Nx.t -> 'd Nx.bool_t
  (** [fill x] is where the DNs [x] set every bit from 0 to 15. *)
end

(** MOD09A1's reflectance quality, sur_refl_qc_500m, 32 bits that hold each
    band's data quality and whether the atmospheric corrections ran (MOD09
    Section 3.2.2, Table 10). Its fill value 4294967295 sets every bit. The
    guide takes cloud from {!module-State}. This layer's MODLAND bits "are not
    to be used for determining whether a pixel is cloudy or not", and "All
    cloud information should be derived from State QA SDSs" (Section 4.1). Its
    functions read DNs of an integer dtype bit by bit, and a program must mask
    fill first ({!Band.section-bits}). *)
module Qc : sig
  val modland : modland list -> ('v, 's, 'd) Nx.t -> 'd Nx.bool_t
  (** [modland ms x] is where the DNs [x] hold one of [ms] in bits 0 and 1.
      11 "overrides a value of (01)", and is set where some band was not
      produced, though others may be. *)

  type quality =
    | Highest  (** 0, highest quality. *)
    | Noisy_detector  (** 7. *)
    | Dead_detector  (** 8: dead detector, data interpolated in L1B. *)
    | Sun_zenith_86  (** 9: the Sun's zenith at 86° or more. *)
    | Sun_zenith_85  (** 10: the Sun's zenith from 85° to under 86°. *)
    | Missing_input  (** 11. *)
    | Internal_constant
        (** 12: an internal constant in place of climatological data for
            an atmospheric constant. *)
    | Out_of_bounds
        (** 13: the correction out of bounds, the reflectance constrained to
            -100 or 16000. *)
    | L1b_faulty  (** 14: L1B data faulty. *)
    | Not_processed  (** 15: not processed, for deep ocean or clouds. *)
  (** The type for a band's data quality, a field of four bits, each value
      Table 10 names. Values 1 to 6 are named by none. *)

  val quality : band -> quality list -> ('v, 's, 'd) Nx.t -> 'd Nx.bool_t
  (** [quality b qs x] is where the DNs [x] hold one of [qs] for band [b]:
      bits [2 + 4 (k - 1)] to [5 + 4 (k - 1)] for MODIS band [k], [Red]'s
      bits 2 to 5 to [Swir22]'s 26 to 29.

      Raises [Invalid_argument] if [b] is not a reflectance layer of
      MOD09A1. *)

  type flag =
    | Atmospheric_correction  (** Bit 30: atmospheric correction performed. *)
    | Adjacency_correction  (** Bit 31: adjacency correction performed. *)
  (** The type for the one-bit flags. *)

  val bit : flag -> int
  (** [bit f] is the position of [f]'s bit. *)

  val any : flag list -> ('v, 's, 'd) Nx.t -> 'd Nx.bool_t
  (** [any fs x] is where the DNs [x] set one of the flags [fs]. *)

  val fill : ('v, 's, 'd) Nx.t -> 'd Nx.bool_t
  (** [fill x] is where the DNs [x] set every bit from 0 to 31. *)
end

(** MOD09A1's state flags, sur_refl_state_500m, which say what the pixel is,
    whatever the band (MOD09 Section 3.3, Table 13). Each pixel holds two cloud
    masks, MOD35's in bits 0 and 1 and the processor's own in bit 10 (Section
    4.1). Its fill value 65535 sets every bit. Its functions read DNs of an
    integer dtype bit by bit, and a program must mask fill first
    ({!Band.section-bits}). *)
module State : sig
  type cloud =
    | Clear  (** 0. *)
    | Cloudy  (** 1. *)
    | Mixed  (** 2. *)
    | Not_set  (** 3: not set, assumed clear. *)
  (** The type for MOD35's cloud states, bits 0 and 1. *)

  val cloud : cloud list -> ('v, 's, 'd) Nx.t -> 'd Nx.bool_t
  (** [cloud cs x] is where the DNs [x] hold one of the states [cs]. *)

  val land_water : land_water list -> ('v, 's, 'd) Nx.t -> 'd Nx.bool_t
  (** [land_water ls x] is where the DNs [x] hold one of [ls] in bits 3 to
      5. *)

  type aerosol =
    | Climatology  (** 0. *)
    | Low  (** 1. *)
    | Average  (** 2. *)
    | High  (** 3. *)
  (** The type for bits 6 and 7, which Collection 6.1 renames "aerosol
      quantity: level of uncertainty in aerosol correction" (MOD09 guide,
      "C61 Revision"). *)

  val aerosol : aerosol list -> ('v, 's, 'd) Nx.t -> 'd Nx.bool_t
  (** [aerosol as_ x] is where the DNs [x] hold one of the levels [as_]. *)

  type cirrus =
    | No_cirrus  (** 0. *)
    | Small_cirrus  (** 1. *)
    | Average_cirrus  (** 2. *)
    | High_cirrus  (** 3. *)
  (** The type for the cirrus detected, bits 8 and 9. *)

  val cirrus : cirrus list -> ('v, 's, 'd) Nx.t -> 'd Nx.bool_t
  (** [cirrus cs x] is where the DNs [x] hold one of the levels [cs]. *)

  type flag =
    | Cloud_shadow  (** Bit 2. *)
    | Internal_cloud  (** Bit 10: the internal cloud algorithm's cloud. *)
    | Fire  (** Bit 11: the internal fire algorithm's fire. *)
    | Snow_ice  (** Bit 12: MOD35's snow or ice. *)
    | Adjacent_cloud  (** Bit 13: adjacent to cloud. *)
    | Salt_pan  (** Bit 14. *)
    | Internal_snow  (** Bit 15: the internal snow mask's snow. *)
  (** The type for the one-bit flags. *)

  val bit : flag -> int
  (** [bit f] is the position of [f]'s bit. *)

  val any : flag list -> ('v, 's, 'd) Nx.t -> 'd Nx.bool_t
  (** [any fs x] is where the DNs [x] set one of the flags [fs]. *)

  val fill : ('v, 's, 'd) Nx.t -> 'd Nx.bool_t
  (** [fill x] is where the DNs [x] set every bit from 0 to 15. *)
end

(** MOD13Q1's pixel reliability, a rank that summarises the VI quality,
    stored as an int8 (MOD13 Section 5.5.2, Table 4). *)
module Reliability : sig
  type t =
    | Fill  (** -1, fill or no data: not processed. *)
    | Good  (** 0, good data: use with confidence. *)
    | Marginal  (** 1, marginal data: useful, but look at other QA. *)
    | Snow_ice  (** 2, snow or ice: the target is covered by it. *)
    | Cloudy  (** 3, cloudy: the target is not visible. *)
  (** The type for the ranks, each with its code. *)

  val code : t -> int
  (** [code r] is the DN that stands for [r]. *)

  val mem : t list -> ('v, 's, 'd) Nx.t -> 'd Nx.bool_t
  (** [mem rs x] is where the DNs [x] stand for one of the ranks [rs],
      compared as {!Band} compares codes. A DN that stands for no rank is in
      none.

      Raises [Invalid_argument] if [x] holds booleans or complex numbers. *)
end
