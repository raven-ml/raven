(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The bands of a raster, what its stored numbers measure, after STAC's
    Band Object.

    A {e band} is one quantity of a raster, an array on a grid, stored as
    {e digital numbers} (DN), often integers to save space. It is STAC's Band
    Object (STAC 1.1, Common Metadata, "Bands" and "Data Values"), with the
    raster extension's scale and offset, which turn a DN into the measured
    value

    {[
      value = scale · DN + offset
    ]}

    and its [unit] (raster extension 2.0, "Scale and Offset"). A
    {e special value} is a DN that marks a cell without a measurement, the
    band's [nodata], where no value was recorded, which product guides call
    {e fill}, or its [saturated] value, where the sensor's range was exceeded
    (Sentinel-2's product metadata states NODATA 0 and SATURATED 65535). Both
    are invalid; {!decode} masks both, and {!val-nodata} and
    {!val-saturated} give each one's mask, on a band of any kind. A DN is
    compared with a special value, or with a class's code, in the DN's own
    dtype, so no DN equals a value its dtype does not hold, as [-9999.] in
    [uint16], and a NaN DN equals a NaN special value.

    A {e bitfield} band, such as Landsat's pixel quality band, packs flags in
    the bits of each DN. Its DNs hold flags, so it is read as bits
    ({!section-bits}) and never decoded. A {e classified} band, such as a
    land cover map, stores in each DN a code that stands for a class. A code
    is a nominal value, compared and never decoded, which {!Warp.nearest}
    and {!Warp.mode} resample.

    A {e product module}, such as {!Sentinel2} or {!Landsat}, states what
    its producer states about its products. A product's {e layers} are the
    arrays its files hold, each a band or a quality layer of bits or
    classes. {!Landsat}, {!Hls} and {!Modis} give each layer as a band, and
    {!Landsat} and {!Modis} its valid range. {!Sentinel2} makes its
    reflectance bands from a product's metadata, and {!Sentinel1} names
    polarisations. [Jord_stac] reads a catalogue's statement of a product's
    bands as the catalogue states it, which can differ from the producer's.
    A model chooses which pixels it masks, from which bits, classes or
    confidences, and which DNs outside a valid range it keeps. *)


type class_ = {
  value : int;  (** The code. *)
  name : string option;
      (** A short name for machines, which classification 2.0 requires. *)
  title : string option;
      (** A name for people, "for use in, e.g., a map legend"
          (classification 2.0). *)
  description : string option;
      (** A description, which classification 1.x requires instead. *)
  nodata : bool;  (** Whether the code marks no data (classification 2.0). *)
}
(** The type for the classes of a classified band, STAC's Class Object
    (classification extension 1.0 to 2.0). *)

(** The kinds of bands. *)
type kind =
  | Measurement  (** Values of a quantity, by {!decode}. *)
  | Bitfield  (** Flags in each value's bits, read as bits. *)
  | Classes of class_ list  (** Codes, each of one of the classes. *)

type t = {
  name : string option;
  (** The producer's name for the band, as ["B04"] or ["SR_B4"]. *)
  common_name : string option;
  (** The name across sensors, as ["red"] or ["nir08"] (STAC's eo
      extension). *)
  center_wavelength : float option;
  (** The band's centre wavelength in micrometres, as its producer or
      catalogue states it (STAC's eo extension 2.0, [eo:center_wavelength]).
      Each source defines it. STAC defines it as "the midpoint between the
      min and max wavelengths", where transmission drops below "some
      threshold, and this threshold is often half of the maximum
      transmission", and ESA as the [central] of a Sentinel-2 product's
      {!Sentinel2.type-wavelength}. *)
  full_width_half_max : float option;
  (** The band's width in micrometres between those min and max
      wavelengths, as its source states it ([eo:full_width_half_max]),
      STAC's "full width at half maximum" where the source's threshold is
      half the maximum. *)
  unit : string option;  (** The unit of the decoded value. *)
  kind : kind;  (** How its DNs are read. *)
  scale : float;  (** 1 when the producer states none. *)
  offset : float;  (** 0 when the producer states none. *)
  nodata : float option;  (** A number, NaN or an infinity. *)
  saturated : float option;
}
(** The type for bands. *)

val decode :
  t ->
  (float, 'k) Nx.dtype ->
  ('v, 's, 'd) Nx.t ->
  (float, 'k, 'd) Nx.t * 'd Nx.bool_t
(** [decode b dtype x] is the values of the DNs [x] of band [b], in [dtype],
    and a mask, of [x]'s shape, of the DNs that are not special. A DN is
    special when it equals [b.nodata] or [b.saturated]. The value is
    [b.scale · DN + b.offset] in float64, then rounded once to [dtype].
    Where the mask is false the value is 0. A NaN DN that is not special is
    data and gives NaN.

    Raises [Invalid_argument] if [b] is a [Bitfield] or [Classes] band, or [x]
    holds booleans or complex numbers. *)

val nodata : t -> ('v, 's, 'd) Nx.t -> 'd Nx.bool_t
(** [nodata b x] is where the DNs [x] equal [b.nodata], false everywhere
    when [b] states none. It masks the fill of a [Bitfield] or [Classes]
    band, which {!decode} refuses, before the program reads its bits or
    codes, where the band's product module has no [fill]
    ({!section-bits}).

    Raises [Invalid_argument] if [x] holds booleans or complex numbers. *)

val saturated : t -> ('v, 's, 'd) Nx.t -> 'd Nx.bool_t
(** [saturated b x] is where the DNs [x] equal [b.saturated], false
    everywhere when [b] states none. Those are the DNs that {!decode} masks
    for saturation; the others it masks equal [b.nodata].

    Raises [Invalid_argument] if [x] holds booleans or complex numbers. *)

(** {1:bits Bitfield bands}

    A bitfield band is read bit by bit, bit 0 the least significant, and a
    {e field} of [length] bits from bit [offset] holds the number
    [DN >> offset & (2{^length} - 1)], in the names of STAC's classification
    extension ("Bit Fields"), where [offset] is a bit's position. A product
    module's functions that read bits take DNs of any integer dtype, read in
    that dtype, and raise [Invalid_argument] for another dtype
    ({!section-bit_dtypes}). A fill value that sets every bit reads as every
    flag set and as every field's highest value, and one that sets no bit or
    its fill bit alone, as SR_CLOUD_QA's 0 and QA_PIXEL's 1 do, reads as every
    other flag unset, so as free of cloud. So a program must mask fill before
    it reads a flag or a field. Where the layer's product module has
    [fill], the program masks by it, since a layer's fill can be more DNs
    than its band's [nodata]. QA_PIXEL's and SR_QA_AEROSOL's [nodata] is 1,
    and {!Landsat.Qa_pixel.val-fill} and {!Landsat.Sr_qa_aerosol.val-fill}
    hold wherever bit 0 is set, on DNs 3 and 65535 too, where {!val-nodata}
    is false. {!val-nodata} masks the fill of a band whose product has no
    module, such as one a catalogue alone states. *)

(** {2:bit_dtypes Dtypes}

    A bit at or past the dtype's width is unset, so a [uint8] DN holds no
    bit 8 and sets no field above it, and the top bit of a signed dtype is
    its sign, so bit 15 of an [int16] DN is set where the DN is
    negative. *)
