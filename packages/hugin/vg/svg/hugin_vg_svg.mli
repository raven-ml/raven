(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** SVG rendering of {!Hugin_vg} pictures.

    The picture's unit is one CSS pixel. The document is self-contained: text
    stays text, with each font it uses embedded once as a data URI, and images
    are embedded as PNG. *)

val render : width:float -> height:float -> Hugin_vg.Picture.t -> string
(** [render ~width ~height p] is [p] as an SVG 1.1 document of [width] by
    [height] pixels, with a matching [viewBox]. Fills, strokes and clips map to
    their SVG counterparts, transforms to groups with a [matrix], stamps to one
    definition used at every point, and images to [image] elements with
    [image-rendering: pixelated]. *)
