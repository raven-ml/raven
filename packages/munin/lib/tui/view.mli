(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The screen: the bars, the panes, what is over them, and the screens of
    a store without runs, of a wait and of a terminal too small. *)

val view : Model.t -> Model.msg Mosaic.t
(** [view m] is the screen of [m]. *)
