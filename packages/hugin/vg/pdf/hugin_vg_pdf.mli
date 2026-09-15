(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** PDF rendering of {!Hugin_vg} pictures.

    The picture's unit is one point, a 72nd of an inch. Text stays text and
    copies as text: each font the picture uses is embedded once, glyphs are
    positioned with their kerning, and a Unicode map is attached. Images are
    stored losslessly, and every stream is deflated. *)

val render : width:float -> height:float -> Hugin_vg.Picture.t -> string
(** [render ~width ~height p] is [p] as a single-page PDF 1.7 document whose
    page is [width] by [height] points. Alpha becomes graphics states, stamps
    become form objects placed at every point, and RGBA images get a soft mask.
*)
