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

(** {1:corpora Corpora}

    A {e corpus} is pixels on disk, read by position: a directory holding, per
    sensor [s] of [s2] and [s1], [s.npy] with its rows' cells one after the
    other and [s_offsets.npy], int64 [[n + 1]], each row's first cell from [0].
    A run reads a corpus written shuffled ({!shuffle}) in order, so a stretch of
    positions is contiguous on disk. *)

val sub : 'd t -> pos:int -> len:int -> 'd t
(** [sub px ~pos ~len] is pixels [pos] to [pos + len − 1] of [px], over [px]'s
    values.

    Raises [Invalid_argument] if [pos] or [len] is negative or
    [pos + len > length px]. *)

val save : string -> Nx.host t -> unit
(** [save dir px] writes [px] as a corpus to [dir], made if missing. *)

val load : string -> Nx.host t
(** [load dir] is the corpus [dir], its arrays mapping the files.

    Raises [Invalid_argument] naming the file if a file is missing, or of
    another dtype or cells than the corpus's, and if the sensors have different
    numbers of pixels. *)

val shuffle :
  key:Nx.host Nx.Rng.key -> bytes:int -> string -> string list -> int
(** [shuffle ~key ~bytes dir parts] writes to [dir] the corpus of the pixels of
    the corpora [parts], in order, pixel [j] of their concatenation at position
    [Nx.Rng.permute ~key ~n j], and is their number [n]. It writes the positions
    in runs of at most [bytes] bytes of observations, or one pixel, and holds
    about three runs at a time, beside a few integers per pixel.

    Raises [Invalid_argument] if [bytes] is not positive, and as {!load} does.
*)
