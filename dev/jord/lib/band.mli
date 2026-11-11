(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Bands: what a raster's stored numbers measure.

    A {e band} is one quantity of a raster, an array on a grid, stored as
    {e digital numbers} (DN), often integers to save space. It is STAC's Band
    Object (STAC 1.1, Common Metadata, "Bands" and "Data Values"), with the
    raster extension's scale and offset, which turn a DN into the measured
    value

    {[
      value = scale · DN + offset
    ]}

    and its [unit] (raster extension 2.0, "Scale and Offset"). A DN equal to a
    {e special value} measures nothing: the band's [nodata], where no value
    was recorded, and its [saturated] value, where the sensor's range was
    exceeded (Sentinel-2's product metadata states NODATA 0 and SATURATED
    65535). Both are invalid; {!decode} masks both and {!saturated} tells them
    apart.

    A {e bitfield} band, such as Landsat's pixel quality band, packs flags in
    the bits of each DN. Its numbers are not quantities, so it is read as bits,
    with {!Nx}'s bitwise operations, and never decoded. *)

(** The kinds of bands. *)
type kind =
  | Measurement  (** Values of a quantity, by {!decode}. *)
  | Bitfield  (** Flags in each value's bits, read as bits. *)

type t = {
  name : string option;
  (** The producer's name for the band, as ["B04"] or ["OLI_B4"]. *)
  common_name : string option;
  (** The name across sensors, as ["red"] or ["nir08"] (STAC's eo
      extension). *)
  unit : string option;  (** The unit of the decoded value. *)
  kind : kind;
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
    special when its float64 value equals [b.nodata] or [b.saturated], or is
    NaN where that value is NaN; a float64 holds every DN of 32 bits or fewer
    exactly. The value is [b.scale · DN + b.offset] in float64, then rounded
    once to [dtype]. Where the mask is false the value is 0. A NaN DN that is
    not special is data and gives NaN.

    Raises [Invalid_argument] if [b] is a [Bitfield] band, or [x] holds
    booleans or complex numbers. *)

val saturated : t -> ('v, 's, 'd) Nx.t -> 'd Nx.bool_t
(** [saturated b x] is where the DNs [x] equal [b.saturated], compared as
    {!decode} compares them: false everywhere when [b] states none. Those are
    the DNs that {!decode} masks for saturation rather than for no data. *)
