(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** SVG rendering backend.

    {b Internal module.} Renders {!Scene.t} to SVG markup. Pure OCaml, no
    external dependencies beyond Cairo for image encoding. *)

(** {1:measurer Text measurement} *)

val text_measurer : Resolve.text_measurer
(** [text_measurer] estimates text dimensions from character count and font
    size. Heuristic: width is [String.length s * 0.6 * font.size]. *)

(** {1:rendering Rendering} *)

val render : Scene.t -> string
(** [render scene] is [scene] as an SVG document string. *)

val render_to_file : string -> Scene.t -> unit
(** [render_to_file filename scene] writes [scene] as an SVG file. *)
