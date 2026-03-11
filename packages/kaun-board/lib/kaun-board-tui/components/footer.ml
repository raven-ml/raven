(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Mosaic

let hint_style = Ansi.Style.make ~fg:(Ansi.Color.grayscale ~level:14) ()

type mode = Dashboard | Detail of { smooth : int }

let hints = function
  | Dashboard -> "←↑↓→ Select • Enter Open • [] Batch • Esc back • Ctrl-C quit"
  | Detail { smooth } ->
      if smooth = 0 then "S Smooth • Esc back"
      else Printf.sprintf "S Smooth [%d] • Esc back" smooth

let view ~mode =
  box ~padding:(padding_xy 2 0)
    ~background:(Ansi.Color.grayscale ~level:2)
    ~size:{ width = pct 100; height = auto }
    [ text ~style:hint_style (hints mode) ]
