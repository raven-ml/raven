(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Mosaic

let hint_style = Ansi.Style.make ~fg:(Ansi.Color.grayscale ~level:14) ()

let view () =
  box ~padding:(padding_xy 2 0)
    ~background:(Ansi.Color.grayscale ~level:2)
    ~size:{ width = pct 100; height = auto }
    [
      text ~style:hint_style
        "←↑↓→ Select • Enter Open • [] Batch • Esc back • Ctrl-C quit";
    ]
