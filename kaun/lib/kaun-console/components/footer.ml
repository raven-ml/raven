(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Footer component for kaun-console TUI. *)

open Mosaic

(* ───── Styles ───── *)

let hint_style = Ansi.Style.make ~fg:(Ansi.Color.grayscale ~level:14) ()

(* ───── View ───── *)

let view () =
  box ~padding:(padding 1)
    [ text ~style:hint_style
        "1–9 open chart • ← → change batch • Esc back • Ctrl-C quit"
    ]
