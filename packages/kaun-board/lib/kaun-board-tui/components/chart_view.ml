(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Mosaic
module Charts = Matrix_charts

let axis_style =
  Ansi.Style.make ~fg:(Ansi.Color.grayscale ~level:12) ~dim:true ()

let y_axis_style =
  Ansi.Style.make ~fg:(Ansi.Color.grayscale ~level:12) ~dim:true ()

let grid_style =
  Ansi.Style.make ~fg:(Ansi.Color.grayscale ~level:6) ~dim:true ()

(** Exponential moving average: alpha in (0,1]. *)
let ema alpha history =
  match history with
  | [] -> []
  | (s0, v0) :: rest ->
      let rec loop acc prev = function
        | [] -> List.rev acc
        | (s, v) :: xs ->
            let smoothed = (alpha *. v) +. ((1. -. alpha) *. prev) in
            loop ((s, smoothed) :: acc) smoothed xs
      in
      (s0, v0) :: loop [] v0 rest

let draw_metric_chart history grid ~width ~height =
  if history = [] then ()
  else
    let data =
      Array.of_list
        (List.map (fun (step, value) -> (float_of_int step, value)) history)
    in
    let chart =
      Charts.empty ()
      |> Charts.with_frame (Charts.manual_frame ~margins:(1, 1, 1, 4) ())
      |> Charts.with_axes
           ~x:
             (Charts.Axis.default |> Charts.Axis.with_ticks 6
             |> Charts.Axis.with_style axis_style)
           ~y:
             (Charts.Axis.default |> Charts.Axis.with_ticks 4
             |> Charts.Axis.with_style y_axis_style
             |> Charts.Axis.with_format (fun _ v -> Printf.sprintf "%.4g" v))
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

let latest_value history =
  let rec last = function
    | [] -> None
    | [ (_, v) ] -> Some v
    | _ :: rest -> last rest
  in
  last history

let view ~tag ~history_for_tag ~best ~size ~smooth =
  let history = history_for_tag tag in
  let display_history = if smooth then ema 0.2 history else history in
  let title =
    match latest_value history with
    | None -> tag
    | Some v -> Printf.sprintf "%s [%.4f]" tag v
  in
  let title = if smooth then title ^ " (EMA)" else title in
  box ~flex_direction:Column ~gap:(gap 1) ~align_items:Center ~size
    [
      box ~border:true ~title ~padding:(padding 1)
        ~size:{ width = pct 100; height = pct 100 }
        ~flex_grow:1.0
        [
          canvas
            ~size:{ width = pct 100; height = pct 100 }
            (fun c ~delta:_ ->
              draw_metric_chart display_history (Canvas.grid c)
                ~width:(Canvas.width c) ~height:(Canvas.height c));
        ];
      (match best with
      | None -> box [] ~size:{ width = px 0; height = px 0 }
      | Some value ->
          box ~justify_content:Center ~align_items:Center
            ~size:{ width = pct 100; height = auto }
            [ text (Printf.sprintf "Best: %.4f" value) ]);
    ]
