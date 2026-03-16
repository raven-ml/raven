(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Mosaic

let hints = function
  | `Dashboard -> "←↑↓→ Select • Enter Open • <> Batch • [] System • q Quit"
  | `Detail smooth -> (
      match smooth with
      | Theme.Off -> "S Smooth • Esc back"
      | Light | Medium | Heavy ->
          Printf.sprintf "S Smooth [%d] • Esc back"
            (Theme.smooth_display smooth))

let view ~mode =
  box ~padding:(padding_xy 2 0) ~background:Theme.header_bg
    ~size:{ width = pct 100; height = auto }
    [ text ~style:Theme.muted_style (hints mode) ]
