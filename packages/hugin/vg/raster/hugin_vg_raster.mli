(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Raster rendering of {!Hugin_vg} pictures.

    The picture's unit is one pixel. Edges are antialiased analytically, so the
    coverage of fills and strokes is exact; an image shown smaller than its
    pixels is averaged over up to sixteen samples per pixel. Stamps are drawn
    once and placed on the pixel grid, so their positions round to whole pixels.
*)

val render :
  ?background:Hugin_vg.Color.t ->
  width:int ->
  height:int ->
  Hugin_vg.Picture.t ->
  Nx.uint8_t
(** [render ~background ~width ~height p] is [p] drawn on a canvas of [width] by
    [height] pixels, as an [[|height; width; 4|]] RGBA tensor with straight
    alpha. Pixel [(i, j)] covers the unit square with corner [(j, i)]. The
    canvas starts filled with [background], which defaults to
    {!Hugin_vg.Color.transparent}. Drawing outside the canvas is clipped.

    Raises [Invalid_argument] if [width] or [height] is not positive. *)
