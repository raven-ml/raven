(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The valid observations of pixels, as Sentinel-2 and Sentinel-1 recorded
    them.

    A pixel's year is two irregular series of {e observations}, one per sensor,
    each with its day of year. Row [i] of {!field-s2} and row [i] of {!field-s1}
    are the same pixel's, as {!Nx_ragged} rows. A row holds only the valid
    observations, raw, in the tile's acquisition order:
    - a Sentinel-2 observation is valid where the tile's mask is nonzero;
    - a Sentinel-1 observation is valid where its VV or VH is nonzero, read
      before any scaling. A row holds the ascending passes, then the descending
      ones.

    Values are as the preprocessing stored them: Sentinel-2 bands as L2A digital
    numbers, Sentinel-1 bands as its scaled decibels. No statistic of a model is
    applied here, so one value serves every model. *)

type 'd t = {
  s2 : (int, Nx.int16_unsigned_elt, 'd) Nx_ragged.t;
      (** Cells [[11]]: the bands B04 B02 B03 B08 B8A B05 B06 B07 B11 B12, then
          the day of year. *)
  s1 : (int, Nx.int16_signed_elt, 'd) Nx_ragged.t;
      (** Cells [[4]]: VV, VH, the day of year, and the orbit, [0] for an
          ascending pass and [1] for a descending one. *)
}

val s2_bands : int
(** [s2_bands] is [10], the Sentinel-2 bands of a cell. *)

val s1_bands : int
(** [s1_bands] is [2], the Sentinel-1 bands of a cell. *)

val length : 'd t -> int
(** [length px] is the number of pixels of [px]. *)

val of_tile : string -> Nx.host t * (int * int)
(** [of_tile dir] is the pixels of the tile [dir] in raster order, row
    [y · w + x] the pixel at [(y, x)], with the tile's grid [(h, w)]. [dir]
    holds the files of TESSERA's preprocessing:
    - [bands.npy], uint16 [[t; h; w; 10]];
    - [masks.npy], uint8 [[t; h; w]], and [doys.npy], uint16 [[t]];
    - [sar_ascending.npy] and [sar_descending.npy], int16 [[t'; h; w; 2]], with
      [sar_ascending_doy.npy] and [sar_descending_doy.npy], int16 [[t']].

    A missing mask gives no Sentinel-2 observation, a missing [doys.npy] days
    [0], and a missing orbit no observation of it. Cost: one pass over the
    files' elements.

    Raises [Invalid_argument] naming the file if [bands.npy] is missing, or if a
    file's dtype or shape disagrees with the above. *)
