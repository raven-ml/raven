(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Cairo rendering backend.

    {b Internal module.} Renders {!Scene.t} to PNG, PDF, or an interactive SDL
    window via Cairo. *)

(** {1:measurer Text measurement} *)

val text_measurer : Ucairo.t -> Resolve.text_measurer
(** [text_measurer cr] is a text measurer backed by {!Ucairo.text_extents}. *)

(** {1:rendering Rendering} *)

val render_scene : Ucairo.t -> Scene.t -> unit
(** [render_scene cr scene] draws [scene] onto [cr]. *)

val render_to_png : string -> width:float -> height:float -> Scene.t -> unit
(** [render_to_png filename ~width ~height scene] writes [scene] as a PNG image.
*)

val render_to_pdf : string -> width:float -> height:float -> Scene.t -> unit
(** [render_to_pdf filename ~width ~height scene] writes [scene] as a
    single-page PDF. *)

val render_to_buffer : width:float -> height:float -> Scene.t -> string
(** [render_to_buffer ~width ~height scene] is the PNG-encoded contents of
    [scene] as a string. *)

(** {1:interactive Interactive display} *)

val show_interactive :
  theme:Theme.t -> width:float -> height:float -> Prepared.t -> unit
(** [show_interactive ~theme ~width ~height prepared] opens an SDL window and
    renders [prepared]. Compiles data once; only re-resolves layout on resize.
    Exits on Escape, Q, or window close. *)
