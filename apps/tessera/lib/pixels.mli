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

    Values are as the tile holds them ({!Tile}): Sentinel-2 bands as surface
    reflectance times 10⁴, negatives kept, Sentinel-1 bands as TESSERA's code
    of γ⁰ in decibels; a tile read from TESSERA's own files holds TESSERA's
    values. No statistic or encoding of a model is applied here, so one value
    serves every model: TESSERA's model raises a negative band by 1000
    ({!Model}). *)

type 'd t = {
  s2 : (int, Nx.int16_signed_elt, 'd) Nx_ragged.t;
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

val of_tile : Tile.t -> Nx.host t * (int * int)
(** [of_tile tile] is the pixels of [tile] in raster order, row [y · w + x]
    the pixel at [(y, x)], with the tile's cells [(h, w)]: its Sentinel-2
    frames where the mask is nonzero, and its Sentinel-1 frames where VV or
    VH is nonzero, the ascending passes first, each orbit in its frames'
    order. Cost: one pass over the tile's elements. *)

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

val take : indices:'d Nx.int64_t -> 'd t -> 'd t
(** [take ~indices px] is the pixels [indices] of [px], in memory of their
    own: its pixel [j] is pixel [indices.{j}] of [px], each sensor's row by
    [Nx_ragged.take]. An index outside \[[0], [length px]) gives a pixel of
    no observation, as [Nx_ragged.take] reads an empty row.

    Raises [Invalid_argument] if [indices] is not 1-D. *)

val save : string -> Nx.host t -> unit
(** [save dir px] writes [px] as a corpus to [dir], made if missing. *)

val load : string -> Nx.host t
(** [load dir] is the corpus [dir], its arrays mapping the files.

    Raises [Failure] naming the file if a file is missing, or of another
    dtype or cells than the corpus's, and if the sensors have different
    numbers of pixels. *)

val read : string -> Nx.host Nx.int64_t -> Nx.host t
(** [read dir indices] is the pixels [indices] of the corpus [dir], as
    {!take} of its {!load} is, in memory of its own. For each sensor in turn
    it reads the pixels' offsets, then their observations, by their ranges of
    [dir]'s files ({!Nx_io.load_npy_rows}): four calls, each made once the
    one before it returns. So it reads from
    storage only the pages that hold them, whether or not the system holds
    the files in memory. A run reads random pixels with it, and a stretch of
    positions with {!take} of the {!load}, whose pages the system reads
    ahead in order.

    Raises [Invalid_argument] if [indices] is not 1-D or holds an index
    outside \[[0], [n]) of a corpus of [n] pixels. Raises [Failure] naming
    the file as {!Nx_io.load_npy_rows} does for a file it cannot read, and
    for one that holds what a corpus cannot: offsets that decrease or reach
    past the values, or values of another dtype or cells than the corpus's.
    {!load} checks these of the whole corpus; [read] of the rows it reads, as
    the files may have changed since. *)

val stream :
  key:Nx.host Nx.Rng.key -> n:int -> first:int -> int -> Nx.host Nx.int64_t
(** [stream ~key ~n ~first len] is the corpus index of each stream position
    [first] to [first + len - 1] of a run over a corpus of [n] pixels. A run
    reads its corpus as a stream with no end: position [p], of epoch
    [e = p / n], is pixel [π_e (p mod n)], where [π_0] is the identity, since
    a corpus is written shuffled ({!shuffle}) and so read in order in its
    first epoch, and [π_e] for [e > 0] is [Nx.Rng.permute] keyed by
    [Nx.Rng.fold_in key e]. Each index is a function of [key], [n] and its
    position alone, so a stream read in windows of any lengths is the same
    stream.

    Raises [Invalid_argument] if [n] is not positive, or [first] or [len] is
    negative. *)

val window : string -> Nx.host t -> first:int -> Nx.host Nx.int64_t -> Nx.host t
(** [window dir corpus ~first indices] is the pixels [indices] of [corpus], the
    {!load} of [dir], that a run takes at the stream positions [first] to
    [first + k - 1], [k] the length of [indices], as {!stream} gives them.
    Where those positions lie in the first epoch, below [length corpus], it is
    {!take} of [corpus], through the files' mappings, which read such a
    stretch in order. Otherwise, the window that straddles the first epoch's
    end included, it is [read dir indices]. *)

val shuffle :
  key:Nx.host Nx.Rng.key -> bytes:int -> string -> string list -> int
(** [shuffle ~key ~bytes dir parts] writes to [dir] the corpus of the pixels of
    the corpora [parts], in order, pixel [j] of their concatenation at position
    [Nx.Rng.permute ~key ~n j], and is their number [n]. It writes the positions
    in runs of at most [bytes] bytes of observations, or one pixel, and holds
    about three runs at a time, beside a few integers per pixel.

    Raises [Invalid_argument] if [bytes] is not positive, and [Failure] as
    {!load} does. *)

(** {1:views Views} *)

val draw : keys:Nx.host Nx.Rng.key -> length:int -> Nx.host t -> Nx.host t
(** [draw ~keys ~length px] is a {e view} of each pixel of [px]: [length]
    observations of each sensor, each drawn uniformly with replacement from the
    pixel's, in draw order. Row [i]'s draws are a function of row [i] of [keys]
    ([[n; 2]]) alone, its Sentinel-2 draws [Nx.Rng.randint] below the row's
    count from [Nx.Rng.fold_in] of the key and [0], its Sentinel-1 draws from
    that of [1]. TESSERA v1.0 trains on two views of [40] observations of each
    pixel.

    Raises [Invalid_argument] if [length] is not positive, if [keys] is not
    [[n; 2]], or if a pixel has no observation of a sensor. *)
