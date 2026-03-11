(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Mosaic

(* Run status *)

type run_status = Live | Stopped | Done

let status_label = function
  | Live -> "LIVE"
  | Stopped -> "Stopped"
  | Done -> "Done"

let badge_color = function
  | Live -> Ansi.Color.green
  | Stopped -> Ansi.Color.grayscale ~level:12
  | Done -> Ansi.Color.of_rgb 80 140 200

(* Styles *)

let header_bg = Ansi.Color.of_rgb 30 80 100
let hint_style = Ansi.Style.make ~fg:(Ansi.Color.grayscale ~level:14) ()
let step_color = Ansi.Color.cyan
let epoch_color = Ansi.Color.cyan

(* View *)

let view ~run_id ~latest_epoch ~status =
  let badge_bg = badge_color status in
  box ~padding:(padding 1) ~background:header_bg
    ~size:{ width = pct 100; height = auto }
    [
      box ~flex_direction:Row ~gap:(gap 2) ~align_items:Center
        ~size:{ width = pct 100; height = auto }
        [
          text ~style:(Ansi.Style.make ~bold:true ()) "\xe2\x96\xb8 Kaun Board";
          text
            ~style:(Ansi.Style.make ~fg:step_color ())
            (Printf.sprintf "Run: %s" run_id);
          (match latest_epoch with
          | None -> text ~style:hint_style "Epoch: -"
          | Some e ->
              text
                ~style:(Ansi.Style.make ~fg:epoch_color ())
                (Printf.sprintf "Epoch: %d" e));
          box ~flex_grow:1.0 ~size:{ width = auto; height = auto } [];
          box ~padding:(padding 1) ~background:badge_bg
            [
              text
                ~style:(Ansi.Style.make ~bold:true ~fg:Ansi.Color.white ())
                (status_label status);
            ];
        ];
    ]
