(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Important info panel component for kaun-console TUI.
    Displays the best (min/max) value for each metric being tracked. *)

open Mosaic

(* ───── Styles ───── *)

let hint_style = Ansi.Style.make ~fg:(Ansi.Color.grayscale ~level:14) ()
let value_style = Ansi.Style.make ~bold:true ~fg:Ansi.Color.green ()
let step_style = Ansi.Style.make ~fg:Ansi.Color.cyan ()
let tag_style = Ansi.Style.make ~fg:(Ansi.Color.grayscale ~level:18) ()

(* ───── Types ───── *)

type best_metric = { step : int; value : float }

(* ───── View ───── *)

let view_metric_row tag (best : best_metric) =
  box ~flex_direction:Column ~gap:(gap 0)
    ~size:{ width = pct 100; height = auto }
    [
      text ~style:tag_style tag;
      box ~flex_direction:Row ~gap:(gap 1) ~align_items:Center
        [
          text ~style:value_style (Printf.sprintf "%.4f" best.value);
          text ~style:step_style (Printf.sprintf "@ step %d" best.step);
        ];
    ]

let view ~(best_metrics : (string * best_metric) list) =
  box ~flex_direction:Column ~padding:(padding 1) ~gap:(gap 1)
    ~size:{ width = pct 100; height = auto }
    [
      text ~style:(Ansi.Style.make ~bold:true ()) "Best Values";
      (if best_metrics = [] then
         text ~style:hint_style "  Waiting for metrics..."
       else
         box ~flex_direction:Column ~gap:(gap 1)
           (List.map (fun (tag, best) -> view_metric_row tag best) best_metrics));
    ]
