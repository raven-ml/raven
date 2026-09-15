(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** SVG rendering. *)

val render : width:float -> height:float -> Picture.t -> string
(** [render ~width ~height p] is [p] as an SVG document of the given size in CSS
    pixels. Text stays text: the fonts it uses are embedded in the document, so
    it renders the same everywhere. Images are embedded as PNG and shown with
    crisp pixels. *)
