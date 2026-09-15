(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** 2D vector pictures.

    A {!Picture.t} is an immutable description of a drawing in a y-down plane
    whose unit is the device pixel. Renderers live in their own libraries:
    [hugin.vg.raster] draws pictures into pixel tensors, [hugin.vg.svg] and
    [hugin.vg.pdf] write them as documents. *)

module Color = Color
module Affine = Affine
module Path = Path
module Stroke = Stroke
module Font = Font
module Picture = Picture
