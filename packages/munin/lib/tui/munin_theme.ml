(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Mosaic
module Charts = Matrix_charts

(* Run status *)

type run_status = Live | Stopped | Done | Failed | Killed

let status_label = function
  | Live -> "LIVE"
  | Stopped -> "Stopped"
  | Done -> "Done"
  | Failed -> "Failed"
  | Killed -> "Killed"

let status_color = function
  | Live -> Ansi.Color.green
  | Stopped -> Ansi.Color.grayscale ~level:12
  | Done -> Ansi.Color.of_rgb 80 140 200
  | Failed -> Ansi.Color.red
  | Killed -> Ansi.Color.yellow

(* Smooth level *)

type smooth = Off | Light | Medium | Heavy

let smooth_alpha = function
  | Off -> None
  | Light -> Some 0.5
  | Medium -> Some 0.3
  | Heavy -> Some 0.15

let next_smooth = function
  | Off -> Light
  | Light -> Medium
  | Medium -> Heavy
  | Heavy -> Off

let smooth_display = function Off -> 0 | Light -> 1 | Medium -> 2 | Heavy -> 3

(* Shared styles *)

let muted_style = Ansi.Style.make ~fg:(Ansi.Color.grayscale ~level:14) ()

let axis_style =
  Ansi.Style.make ~fg:(Ansi.Color.grayscale ~level:12) ~dim:true ()

let grid_style =
  Ansi.Style.make ~fg:(Ansi.Color.grayscale ~level:6) ~dim:true ()

let header_bg = Ansi.Color.grayscale ~level:2

(* Shared helpers *)

let last_value history =
  let rec go = function
    | [] -> None
    | [ (_, v) ] -> Some v
    | _ :: rest -> go rest
  in
  go history

(* Chart drawing *)

let chart_theme =
  { Charts.Theme.dark with axis = axis_style; grid = grid_style }

let to_points history =
  Array.of_list
    (List.map (fun (step, value) -> (float_of_int step, value)) history)

let metric_line color history =
  Charts.line ~color:(Charts.const color) ~x:fst ~y:snd (to_points history)

(* An axis spaces its ticks by cells: [width / x_ticks] cells apart gives about
   [x_ticks] ticks across the chart. *)
let draw_metric_chart ~compact ?smoothed history grid ~width ~height =
  if history = [] then ()
  else
    let x_ticks, y_ticks, y_format =
      if compact then (4, 2, fun _ v -> Printf.sprintf "%.1f" v)
      else (6, 4, fun _ v -> Printf.sprintf "%.4g" v)
    in
    let lines =
      match smoothed with
      | None -> [ metric_line Ansi.Color.cyan history ]
      | Some smoothed ->
          [
            metric_line (Ansi.Color.grayscale ~level:8) history;
            metric_line Ansi.Color.cyan smoothed;
          ]
    in
    let axes =
      [
        Charts.axis ~grid:`Ticks ~spacing:(width / x_ticks) "x";
        Charts.axis ~grid:`Ticks ~spacing:(height / y_ticks) ~format:y_format
          "y";
      ]
    in
    let fig = Charts.layer (lines @ axes) in
    let layout = Charts.layout ~theme:chart_theme ~width ~height fig in
    Charts.draw layout grid ~x:0 ~y:0
