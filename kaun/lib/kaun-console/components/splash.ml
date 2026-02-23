(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Splash screen component for Kaun Console. *)

open Mosaic

let view () =
  box ~flex_direction:Column ~justify_content:Center ~align_items:Center
    ~size:{ width = pct 100; height = pct 100 }
    ~background:(Ansi.Color.of_rgb 20 20 30)
    [
      box ~flex_direction:Column ~gap:(gap 1) ~align_items:Center
        [
          text
            ~style:
              (Ansi.Style.make ~bold:true ~fg:(Ansi.Color.of_rgb 100 200 255) ())
            "Kaun Console";
          spinner ~frame_set:Spinner.dots
            ~color:(Ansi.Color.grayscale ~level:14) ();
        ];
    ]
