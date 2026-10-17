(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Tile-compressed images (FITS 4.0 §10): a [BINTABLE] with [ZIMAGE = T]
    whose rows are tiles, each coded in the heap.

    Reading raises {!Err.Fail} at the place that fails, the tile's when a
    tile does not decode. However its descriptors overlap, a read fetches
    at most the heap's size in tile bytes, and inflates a gzip tile to at
    most 8 bytes per pixel. *)

(** {1:description Description} *)

(** The type for tile codecs. *)
type codec =
  | Rice of { block : int; bytepix : int }
  | Gzip_1
  | Gzip_2
  | Nocompress

val codec_name : codec -> string
(** [codec_name c] is [c]'s [ZCMPTYPE], as ["RICE_1"]. *)

(** The type for a per-tile parameter: a column of the table, a keyword, or
    absent. *)
type 'a param = Column of Bintable.column | Keyword of 'a | Absent

type t = {
  table : Bintable.t;
  zbitpix : int;
  axes : int array;  (** ZNAXISn, in file order *)
  tile : int array;  (** ZTILEn, in file order *)
  counts : int array;  (** tiles along each axis *)
  codec : codec;
  quantized : bool;
  dither : Quantize.dither;
  dither_name : string;
  zdither0 : int;
  zscale : float param;
  zzero : float param;
  zblank : int param;
  compressed : Bintable.column;
  gzip : Bintable.column option;
  uncompressed : Bintable.column option;
  null_mask : Bintable.column option;
}
(** The type for a tiled image's description. *)

val describe : Header.t -> Hdu.store -> t
(** [describe h store] is the tiled image that the header [h] gives the data
    unit [store]. A dithered image without [ZDITHER0] has seed 1, as
    cfitsio reads the files written before the keyword existed. Raises
    {!Err.Fail} if a keyword does not read or is out of range, a codec is
    unknown or not read (HCOMPRESS_1, PLIO_1), or the table does not hold
    the tiles. *)

val read : t -> (int * int) array -> Nx.packed
(** [read t bounds] is the window [bounds], given in tensor axis order, in
    [ZBITPIX]'s format: the tiles it meets decoded, each once. [bounds] lie
    within the image, which [read] does not check. *)

(** {1:writing Writing} *)

(** The type for how a tiled image is written: losslessly by a codec, or
    quantized and Rice-coded in steps of each tile's noise divided by
    [q]. *)
type plan = Lossless of codec | Quantized of { q : float }

type table = {
  ntiles : int;
  row_bytes : int;
  wide : bool;  (** Q descriptors *)
  columns : (string * string) list;  (** TTYPEn, TFORMn *)
  keys : Structure.entry list;  (** the tile-compression keywords *)
  pcount : int;
}
(** The type for the table a tiled image is written as. Everything but the
    heap's sizes is known before a tile is coded, so a writer can print a
    header of the final one's records first and stream the tiles after
    it. *)

val table :
  bitpix:int ->
  axes:int array ->
  tile:int array ->
  plan ->
  zdither0:int ->
  maxes:int * int ->
  heap:int ->
  table
(** [table ~bitpix ~axes ~tile plan ~zdither0 ~maxes ~heap] describes the
    table of an image coded by [plan] whose heap holds [heap] bytes, its
    heap columns' longest arrays [maxes]. [axes] and [tile] are in file
    order. The descriptors are Q when the heap could pass 2{^ 31} − 1
    bytes. A coded tile is at most twice its pixels' bytes and 64 more, so
    the choice is made before coding. *)

val zdither0 : int -> int
(** [zdither0 sum] is [ZDITHER0] for the ones' complement sum [sum] of the
    stored pixels: 1 plus [sum] modulo 10000, so equal images compress to
    equal bytes. *)

val code :
  bitpix:int ->
  axes:int array ->
  tile:int array ->
  plan ->
  zdither0:int ->
  slab:(int -> int -> ((int, 'a, 'b) Bigarray.Array1.t -> unit) -> unit) ->
  heap:(string -> unit) ->
  string * table
(** [code ~bitpix ~axes ~tile plan ~zdither0 ~slab ~heap] codes the image
    tile row by tile row: [slab a b f] calls [f] with the big-endian
    elements, in C order, of the image's indices [a] to [b - 1] on its
    slowest file axis, which are valid during the call, and each coded tile
    goes to [heap]. It is the table's rows and description. Memory beyond
    the image is one slab of whole tile rows, at most 4 MiB unless one tile
    row is larger. *)
