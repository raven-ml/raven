(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** PDF rendering of {!Hugin_vg} pictures. *)

val render : width:float -> height:float -> Hugin_vg.Picture.t -> string
(** [render ~width ~height p] is [p] as a single-page PDF document; one canvas
    unit is one point. Text stays text, with the fonts it uses embedded, and
    images are stored losslessly. *)
