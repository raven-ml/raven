(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Mosaic

let hint_style = Ansi.Style.make ~fg:(Ansi.Color.grayscale ~level:14) ()

let view () =
  box ~padding:(padding 1)
    [
      text ~style:hint_style
        "1–9 Chart View • ← → Change Batch • Esc back • Ctrl-C quit";
    ]
