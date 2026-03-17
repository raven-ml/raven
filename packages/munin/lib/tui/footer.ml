(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Mosaic

let hints = function
  | `Dashboard ->
      "\u{2190}\u{2191}\u{2193}\u{2192} Select \u{00B7} Enter Open \u{00B7} <> \
       Batch \u{00B7} [] System \u{00B7} i Info \u{00B7} q Quit"
  | `Detail smooth -> (
      match smooth with
      | Theme.Off -> "S Smooth \u{00B7} Esc back"
      | Light | Medium | Heavy ->
          Printf.sprintf "S Smooth [%d] \u{00B7} Esc back"
            (Theme.smooth_display smooth))
  | `Info -> "q/Esc back"

let view ~mode =
  box ~padding:(padding_xy 2 0) ~background:Theme.header_bg
    ~size:{ width = pct 100; height = auto }
    [ text ~style:Theme.muted_style (hints mode) ]
