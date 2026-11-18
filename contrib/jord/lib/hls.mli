(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The facts NASA states about the surface reflectance products of
    Harmonized Landsat and Sentinel-2 (HLS) version 2.0.

    NASA's HLS project makes the surface reflectance of Landsat 8 and 9's
    Operational Land Imager (OLI) and Sentinel-2's MultiSpectral Instrument
    (MSI) comparable. Its data are corrected for the atmosphere, masked for
    cloud, co-registered at 30 m on Sentinel-2's tiles, normalised to a nadir
    view (NBAR) and, for MSI, adjusted to OLI's bandpasses. Its
    two products are {e L30}, made from Landsat Collection 2 Level-1, and
    {e S30}, from Sentinel-2 Level-1C, each tile stored as one Cloud
    Optimized GeoTIFF per layer of digital numbers (DN). This module states
    what the product's user guide states about them: the layers of each
    product, the rule that turns a DN into its quantity, the bits of the
    quality layer and the identifier (HLS Product User Guide, product version
    2.0, last updated April 2026,
    {{:https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf}}).

    {b Quantities.} Every reflectance layer has scale 0.0001, L30's thermal
    layers 0.01 in degrees Celsius, and the angle layers 0.01 in degrees, all
    with offset 0 (Tables 6 to 8):

    {[
      reflectance = 0.0001 · DN
      temperature = 0.01 · DN
      angle = 0.01 · DN
    ]}

    DN -9999 is fill in the reflective and thermal layers, which are int16;
    40000 in the angle layers, uint16; and 255 in the quality layer, a byte.
    No layer states a saturated value. A product's own metadata states its
    scale and fill, as REF_SCALE_FACTOR and FILLVALUE (Table 11). The guide's
    wavelengths are nominal ranges, one per band for every satellite, "as
    they can vary slightly with the satellites" (Table 3), so a layer states
    no centre or width. A catalogue's items may state their own centres and
    widths ({!Band}).

    {b Grid.} A product is one tile of Sentinel-2's tiling, 109,800 m on a
    side in the UTM system of its zone (Section 3.5). "HLS inherits the USGS
    Landsat UTM convention of keeping the y coordinate for the Southern
    Hemisphere negative" (Section 3.5), so a tile south of the equator lies
    in its zone's northern system. Tile 21JVF, whose latitude band [J] lies
    south of the equator, has its grid in WGS 84 / UTM zone 21N (EPSG 32621)
    with its corner at northing -3399960. A tile's system is the one its
    file's or its catalogue's grid states. *)

(** {1:products Products} *)

type product =
  | L30  (** From Landsat 8 and 9's OLI and TIRS. *)
  | S30  (** From Sentinel-2's MSI. *)
(** The type for HLS's two products (Table 2). *)

(** {1:bands Bands} *)

type band =
  | Coastal  (** Coastal aerosol reflectance. *)
  | Blue  (** Blue reflectance. *)
  | Green  (** Green reflectance. *)
  | Red  (** Red reflectance. *)
  | Red_edge1  (** Red-edge 1 reflectance, S30 only. *)
  | Red_edge2  (** Red-edge 2 reflectance, S30 only. *)
  | Red_edge3  (** Red-edge 3 reflectance, S30 only. *)
  | Nir_broad  (** Broad near-infrared reflectance, S30 only. *)
  | Nir_narrow  (** Narrow near-infrared reflectance. *)
  | Swir16  (** Shortwave-infrared reflectance near 1.6 µm, SWIR 1. *)
  | Swir22  (** Shortwave-infrared reflectance near 2.2 µm, SWIR 2. *)
  | Water_vapour
      (** Water vapour, top-of-atmosphere reflectance, S30 only. *)
  | Cirrus  (** Cirrus, top-of-atmosphere reflectance. *)
  | Lwir11
      (** Thermal infrared 1, top-of-atmosphere brightness temperature, L30
          only. *)
  | Lwir12
      (** Thermal infrared 2, top-of-atmosphere brightness temperature, L30
          only. *)
  | Fmask  (** Quality, bits ({!module-Fmask}). *)
  | Sun_zenith  (** The Sun's zenith angle. *)
  | Sun_azimuth  (** The Sun's azimuth. *)
  | View_zenith  (** The view zenith angle. *)
  | View_azimuth  (** The view azimuth. *)
(** The type for the layers of a product. It names the spectral ones by
    Table 3's names, as L30 and S30 number them differently, and the others
    by what they hold. The reflective layers are NBAR surface reflectance,
    except [Water_vapour] and [Cirrus] (Tables 6 and 7). L30 holds Landsat's
    angles, derived for the red band; S30 holds angles interpolated from
    ESA's angles at nodes 5 km apart, its view angles those of its red-edge
    2 band (Section 6.3). *)

val bands : product -> band list
(** [bands p] is every layer of product [p], in the order of its table
    (Table 6 or 7), then the angle layers in Table 8's order. *)

val name : product -> band -> string
(** [name p b] is layer [b]'s name in a product [p], the suffix of its
    file's name: Table 3's code, as ["B05"], which is [Nir_narrow] in L30
    and [Red_edge1] in S30; ["Fmask"]; and ["SZA"], ["SAA"], ["VZA"] and
    ["VAA"] for the angles (Section 6.1).

    Raises [Invalid_argument] if [p] has no layer [b]. *)

val band : product -> band -> Band.t
(** [band p b] is layer [b] of a product [p], as Tables 6 to 8 state it: its
    [name]; for a spectral layer its [common_name] in STAC's electro-optical
    extension, from ["coastal"] to ["lwir12"], with ["rededge071"],
    ["rededge075"] and ["rededge078"] for the red edge, ["nir"] for
    [Nir_broad], ["nir08"] for [Nir_narrow] and ["nir09"] for [Water_vapour];
    its scale, offset 0 and fill value as [nodata]. Units are
    UDUNITS': ["degree_Celsius"] for [Lwir11] and [Lwir12], ["degree"] for
    the angles, none for reflectance. [Fmask] is a [Bitfield] band. No layer
    states a saturated value, a centre wavelength or a width.

    Raises [Invalid_argument] if [p] has no layer [b]. *)

(** {1:quality Quality bits} *)

(** The quality layer, Fmask: Fmask's cloud, cloud shadow, snow and water
    classes, cloud and shadow dilated by five pixels, and the aerosol
    optical thickness level of the atmospheric correction (Section 6.4,
    Table 9). Bit 0 is reserved, unused in version 2.0, and holds no class:
    "cirrus is aggregated into the generic 'cloud' class" (Section 4.2). The
    guide warns that bits 0 to 5 "may not be mutually exclusive due to the
    possible mixture of categorical labels when data are resampled to 30 m".

    Its fill value 255 sets every bit, which {!module-Fmask.fill} finds. Its
    functions read the byte as bits ({!Band.section-bits}). *)
module Fmask : sig
  type flag =
    | Cloud  (** Bit 1. *)
    | Adjacent  (** Bit 2: adjacent to cloud or cloud shadow. *)
    | Cloud_shadow  (** Bit 3. *)
    | Snow_ice  (** Bit 4: snow or ice. *)
    | Water  (** Bit 5. *)
  (** The type for the one-bit flags. *)

  val bit : flag -> int
  (** [bit f] is the position of [f]'s bit. *)

  val any : flag list -> ('v, 's, 'd) Nx.t -> 'd Nx.bool_t
  (** [any fs x] is where the DNs [x] set one of the flags [fs]. *)

  type aerosol =
    | Climatology  (** 0, aerosol from climatology. *)
    | Low  (** 1. *)
    | Medium  (** 2. *)
    | High  (** 3. *)
  (** The type for the aerosol levels of bits 6 and 7, each with its value.
      Table 9 writes each level's two bits from bit 6 to bit 7, the order
      it lists bits in, so it writes [Low] as 10 and [Medium] as 01;
      Appendix A's DN 100, binary 01100100, is "low" aerosol with bit 6
      set. *)

  val aerosol : aerosol list -> ('v, 's, 'd) Nx.t -> 'd Nx.bool_t
  (** [aerosol ls x] is where the DNs [x] hold one of the levels [ls]. *)

  val fill : ('v, 's, 'd) Nx.t -> 'd Nx.bool_t
  (** [fill x] is where the byte of the DNs [x] is 255, where no observation
      was made. *)
end

(** {1:identity Identity} *)

type id = {
  product : product;  (** L30 or S30. *)
  tile : string;  (** The tile, as ["17SLU"]. *)
  stamp : float;
      (** The instant the identifier states, as a {!Time} instant. *)
}
(** The type for what a product's identifier states. Its [stamp] is the
    instant the identifier names, which the guide defines per product. In
    L30 it is "the sensing time at the input Landsat scene center", one of
    two scenes' "chosen by chance" where two cover the tile; in S30 it is
    "the time the sensor begins to sense the sun-lit side of the Earth for
    each orbit" (Section 6.1). The guide's words: "it is intended mainly as
    a unique identifier, not for quantitative analysis". *)

val id : string -> (id, string) result
(** [id s] is what the identifier [s] states, as
    ["HLS.L30.T17SLU.2020209T155956.v2.0"]: [HLS], the product, [T] and the
    tile, the year, its day from 001 and the UTC time [hhmmss], and the
    version, joined by dots (Section 6.1). The tile is read by its form, two
    digits and three capital letters, and not checked against Sentinel-2's
    tiling, so [T01ABC] reads, though Sentinel-2's latitude bands run from C
    to X. The version is ["v2.0"], as the products and catalogues write it,
    or ["V2.0"], as the guide's examples do.

    The result is [Error] if [s] is not of that form, as a layer's file
    name (the identifier, a dot and {!name}) is not, if its day is past its
    year's last or its time past 23:59:59, or if its version is not 2.0. *)
