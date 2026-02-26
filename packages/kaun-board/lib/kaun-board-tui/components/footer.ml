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
        "1\xe2\x80\x939 Chart View \xe2\x80\xa2 \xe2\x86\x90 \xe2\x86\x92 \
         Change Batch \xe2\x80\xa2 Esc back \xe2\x80\xa2 Ctrl-C quit";
    ]
