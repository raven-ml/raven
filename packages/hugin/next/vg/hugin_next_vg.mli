(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Pictures and renderables.

    {!Picture} describes drawings and {!Renderable} puts one on a page of a
    physical size. Three libraries render renderables, each with a [render]
    function:
    - [hugin.next.vg.raster] draws them as RGBA tensors, and as PNG files, at a
      density in device pixels per point;
    - [hugin.next.vg.svg] writes them as SVG documents;
    - [hugin.next.vg.pdf] writes them as PDF documents.

    The renderers are separate libraries so that a program links only those it
    uses; they depend on [nx.io] for PNG encoding and compression.

    Points, boxes, affine maps, paths, stroke styles and colours come from
    {!Hugin_next_gg}, whose {{!Hugin_next_gg.section-conventions}conventions}
    apply here, and glyph runs from {!Hugin_next_font}. The meaning of a picture
    is stated once, in {!Picture.section-semantics}. Every renderer writes every
    picture to that meaning, within the accuracy it documents, and no renderer
    drops or replaces part of a picture. Raster output is the meaning itself;
    SVG and PDF output show it in viewers that have the features each renderer
    lists. *)

module Picture = Picture
module Renderable = Renderable
