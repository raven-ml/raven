(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Rendering through [hugin.vg].

    {b Internal module.} Turns a {!Scene.t} into a {!Hugin_vg.Picture.t} and
    measures text with the bundled fonts. *)

val text_measurer : Resolve.text_measurer
(** [text_measurer ~font s] is the ink width and height of [s] in [font]. *)

val picture : Scene.t -> Hugin_vg.Picture.t
(** [picture scene] is [scene] as a picture. *)

val render_png : string -> width:float -> height:float -> Scene.t -> unit
(** [render_png filename ~width ~height scene] writes [scene] as a PNG file. *)

val render_to_buffer : width:float -> height:float -> Scene.t -> string
(** [render_to_buffer ~width ~height scene] is [scene] as PNG bytes. *)

val render_pdf : string -> width:float -> height:float -> Scene.t -> unit
(** [render_pdf filename ~width ~height scene] writes [scene] as a single-page
    PDF file. *)

val render_svg : width:float -> height:float -> Scene.t -> string
(** [render_svg ~width ~height scene] is [scene] as an SVG document. *)
