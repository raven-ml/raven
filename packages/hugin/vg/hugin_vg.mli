(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** 2D vector pictures.

    A {!Picture.t} is an immutable description of a drawing: paths filled or
    stroked with a color, text, images, and groups of those clipped, transformed
    or stamped at many points. Coordinates are in a y-down plane whose unit is
    one pixel or one point, depending on the renderer.

    Build geometry with {!Path}, describe how it is drawn with {!Color} and
    {!Stroke}, lay out text with {!Font}, and assemble the drawing with
    {!Picture}. {!Affine} and {!Box} are the transforms and boxes the others
    speak in.

    Rendering lives in separate libraries so that a program links only what it
    uses: [hugin.vg.raster] draws pictures into pixel tensors, [hugin.vg.svg]
    and [hugin.vg.pdf] write them as documents. Every renderer is a single
    [render] function over a picture. *)

module Color = Color
module Affine = Affine
module Box = Box
module Path = Path
module Stroke = Stroke
module Font = Font
module Picture = Picture
