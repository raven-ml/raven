(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Raster rendering.

    Pictures are drawn with analytic antialiasing into an RGBA pixel buffer. *)

val render :
  ?background:Color.t -> width:int -> height:int -> Picture.t -> Nx.uint8_t
(** [render ~background ~width ~height p] is [p] drawn on a [width] by [height]
    canvas as an [[|height; width; 4|]] RGBA tensor with straight alpha. Pixel
    [(i, j)] covers the unit square with corner [(j, i)]. [background] defaults
    to {!Color.transparent}.

    Raises [Invalid_argument] if [width] or [height] is not positive. *)
