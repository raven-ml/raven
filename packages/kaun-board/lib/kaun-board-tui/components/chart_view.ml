(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Mosaic
module Charts = Matrix_charts

let axis_style =
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

let draw_metric_chart ~hover history grid ~width ~height =
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
             |> Charts.Axis.with_style axis_style
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
    let layout = Charts.draw chart grid ~width ~height in
    match hover with
    | None -> ()
    | Some (px, py) -> (
        if not (Charts.Layout.is_inside_plot layout ~px ~py) then ()
        else
          let radius, policy = (4, `Nearest_x) in
          match Charts.Layout.hit_test layout ~px ~py ~radius ~policy with
          | Some hit -> (
              match hit.payload with
              | Charts.Hit.XY { x; y } ->
                  let lines =
                    [
                      Printf.sprintf "Step: %d" (int_of_float x);
                      Printf.sprintf "Value: %.4g" y;
                    ]
                  in
                  Charts.Overlay.tooltip layout grid ~x ~y lines
              | _ -> ())
          | None -> (
              (* Free cursor: tooltip at interpolated data coords *)
              match Charts.Layout.data_of_px layout ~px ~py with
              | None -> ()
              | Some (x, y) ->
                  let lines =
                    [
                      Printf.sprintf "Step: %.0f" x;
                      Printf.sprintf "Value: %.4g" y;
                    ]
                  in
                  Charts.Overlay.tooltip ~anchor:`Right layout grid ~x ~y lines))

let latest_value history =
  let rec last = function
    | [] -> None
    | [ (_, v) ] -> Some v
    | _ :: rest -> last rest
  in
  last history

let view ~tag ~history_for_tag ~best ~size ~smooth ~hover ~on_mouse =
  let history = history_for_tag tag in
  let display_history =
    match smooth with None -> history | Some alpha -> ema alpha history
  in
  let title =
    match latest_value history with
    | None -> tag
    | Some v -> Printf.sprintf "%s [%.4f]" tag v
  in
  let title = if Option.is_some smooth then title ^ " (EMA)" else title in
  let canvas_content =
    canvas
      ~live:true
      ~on_mouse
      ~size:{ width = pct 100; height = pct 100 }
      (fun c ~delta:_ ->
        Canvas.clear c;
        draw_metric_chart ~hover display_history (Canvas.grid c)
          ~width:(Canvas.width c) ~height:(Canvas.height c))
  in
  box ~flex_direction:Column ~gap:(gap 1) ~align_items:Center ~size
    [
      box ~border:true ~title ~padding:(padding 1)
        ~size:{ width = pct 100; height = pct 100 }
        ~flex_grow:1.0
        [ canvas_content ];
      (match best with
      | None -> box [] ~size:{ width = px 0; height = px 0 }
      | Some value ->
          box ~justify_content:Center ~align_items:Center
            ~size:{ width = pct 100; height = auto }
            [ text (Printf.sprintf "Best: %.4f" value) ]);
    ]
