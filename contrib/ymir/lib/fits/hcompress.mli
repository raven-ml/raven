(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** H-compress decoding (FITS 4.0 §10.4.4, White 1992), as cfitsio's
    [fits_hdecompress64] reads a stream.

    A stream holds one tile's H-transform: a magic, the tile's size
    [nx × ny] with [ny] the fastest axis, the scale its coefficients were
    divided by, the first coefficient in full, then each quadrant's bit
    planes, quadtree-coded or written directly, and the signs of the nonzero
    coefficients. Decoding multiplies the coefficients by the scale and
    inverts the transform in integer arithmetic, adjusting them toward
    their neighbours' when asked to smooth.

    A few bytes can code a tile of any size. Decoding a window of it costs
    memory and work in proportion to the stream and the window. *)

val size : Checksum.bigbytes -> int * int
(** [size src] is the stream's [(nx, ny)]. Raises {!Err.Fail} as {!decode}
    does for its header. *)

val decode :
  smooth:bool ->
  Checksum.bigbytes ->
  x0:int ->
  x1:int ->
  y0:int ->
  y1:int ->
  (int, Bigarray.int_elt, Bigarray.c_layout) Bigarray.Array1.t ->
  int
(** [decode ~smooth src ~x0 ~x1 ~y0 ~y1 out] decodes rows [x0] to [x1 - 1]
    and columns [y0] to [y1 - 1] of the stream [src] into [out], row by row,
    and is the stream's scale: 0 or 1 when the tile is lossless. The window
    is nonempty and lies in the stream's size, and [out] holds its values,
    which [decode] does not check. Bytes after the signs are ignored.
    Raises {!Err.Fail} if [src] does not start with the magic, its size is
    not positive, a quadrant has more than 62 bit planes, a bit plane's code
    is not 0 or 15, the planes do not end with the zero nybble, or the
    stream ends early. *)
