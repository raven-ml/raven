(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Important info panel component for kaun-console TUI. *)

open Mosaic

(* ───── View ───── *)

let view () =
  box ~padding:(padding 1)
    [ text ~style:(Ansi.Style.make ~bold:true ()) "imp info" ]
