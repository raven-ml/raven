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

let draw_metric_chart ~compact history grid ~width ~height =
  if history = [] then ()
  else
    let data =
      Array.of_list
        (List.map (fun (step, value) -> (float_of_int step, value)) history)
    in
    let margins, x_ticks, y_ticks, y_format =
      if compact then ((1, 0, 0, 2), 4, 2, fun _ v -> Printf.sprintf "%.1f" v)
      else ((1, 1, 1, 4), 6, 4, fun _ v -> Printf.sprintf "%.4g" v)
    in
    let chart =
      Charts.empty ()
      |> Charts.with_frame (Charts.manual_frame ~margins ())
      |> Charts.with_axes
           ~x:
             (Charts.Axis.default
             |> Charts.Axis.with_ticks x_ticks
             |> Charts.Axis.with_style axis_style)
           ~y:
             (Charts.Axis.default
             |> Charts.Axis.with_ticks y_ticks
             |> Charts.Axis.with_style axis_style
             |> Charts.Axis.with_format y_format)
      |> Charts.with_grid
           (Charts.Gridlines.default
           |> Charts.Gridlines.with_style grid_style
           |> Charts.Gridlines.with_x true
           |> Charts.Gridlines.with_y true)
      |> Charts.line ~id:"metric" ~resolution:`Braille2x4
           ~style:(Ansi.Style.make ~fg:Ansi.Color.cyan ())
           ~x:fst ~y:snd data
    in
    ignore (Charts.draw chart grid ~width ~height)
