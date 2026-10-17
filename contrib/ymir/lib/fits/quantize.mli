(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Quantized floats (FITS 4.0 §10.2), as the archives' files hold them.

    Those files are all written by cfitsio or code derived from it, whose
    bytes differ from the standard's text in the zero value of
    SUBTRACTIVE_DITHER_2 and in the dither index, which restarts at 10000
    from [(tile + ZDITHER0 - 2) mod 10000] for 1-based tiles. Both
    directions follow cfitsio's arithmetic, roundings included. The dither
    deviates are computed when the library loads, so domains share them
    without a lock. *)

type ints = (int, Bigarray.int_elt, Bigarray.c_layout) Bigarray.Array1.t
(** The type for arrays of quantized values. *)

type floats =
  (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Array1.t
(** The type for arrays of physical values. *)

(** The type for [ZQUANTIZ]'s dithering methods. *)
type dither = No_dither | Subtractive_1 | Subtractive_2

val null_value : int
(** [null_value] is [-2147483647], the quantized value of NaN, which
    [ZBLANK] names. *)

val dequantize :
  dither:dither ->
  row:int ->
  scale:float ->
  zero:float ->
  blank:int option ->
  ints ->
  int ->
  floats ->
  unit
(** [dequantize ~dither ~row ~scale ~zero ~blank i n out] writes the
    physical values of the first [n] values of [i] to [out]:
    [I × ZSCALE + ZZERO], or [((I − R) + 0.5) × ZSCALE + ZZERO] dithered,
    each operation rounded in float64, NaN for [blank] and 0 for
    SUBTRACTIVE_DITHER_2's zero value. [row] is [tile + ZDITHER0] for a
    0-based tile. [i] and [out] hold at least [n] values, which
    [dequantize] does not check. *)

val quantize :
  dither:dither ->
  row:int ->
  q:float ->
  floats ->
  int ->
  int ->
  ints ->
  (float * float) option
(** [quantize ~dither ~row ~q data nx ny out] is cfitsio's
    [fits_quantize_float]: it writes to [out] the integers of the tile
    [data] of [ny] rows of [nx] pixels, in steps of its noise divided by
    [q], NaN as {!null_value}, and is [Some (zscale, zzero)]. It is [None],
    and writes nothing, for a tile of at most one pixel, with no noise, or
    with a range past int32, which then stays lossless. [data] and [out]
    hold at least [nx × ny] values, which [quantize] does not check. *)
