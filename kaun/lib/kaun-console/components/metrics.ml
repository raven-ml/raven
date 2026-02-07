(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Metrics chart component for kaun-console TUI. *)

open Mosaic
module Charts = Matrix_charts

(* ───── Styles ───── *)

let hint_style = Ansi.Style.make ~fg:(Ansi.Color.grayscale ~level:14) ()

let axis_style =
  Ansi.Style.make ~fg:(Ansi.Color.grayscale ~level:12) ~dim:true ()

let y_axis_style =
  Ansi.Style.make ~fg:(Ansi.Color.grayscale ~level:12) ~dim:true ()

let grid_style =
  Ansi.Style.make ~fg:(Ansi.Color.grayscale ~level:6) ~dim:true ()

(* ───── Constants ───── *)

let graph_height = 14
let header_height = 3
let footer_height = 1
let metrics_padding = 2

(* ───── Helpers ───── *)

(** Calculate how many graphs fit in available height *)
let calculate_graphs_per_batch (screen_height : int) : int =
  let available_height =
    screen_height - header_height - footer_height - (metrics_padding * 2)
  in
  if available_height < graph_height then 1
  else max 1 (available_height / graph_height)

(* ───── Chart Drawing ───── *)

let draw_metric_chart _tag history grid ~width ~height =
  if history = [] then (* No data yet - show placeholder *)
    ()
  else
    (* Convert (step, value) list to array of (x, y) tuples *)
    let data =
      Array.of_list
        (List.map (fun (step, value) -> (float_of_int step, value)) history)
    in
    let chart =
      Charts.empty ()
      |> Charts.with_frame (Charts.manual_frame ~margins:(1, 0, 0, 2) ())
      |> Charts.with_axes
           ~x:
             (Charts.Axis.default
             |> Charts.Axis.with_ticks 4
             |> Charts.Axis.with_style axis_style)
           ~y:
             (Charts.Axis.default
             |> Charts.Axis.with_ticks 2
             |> Charts.Axis.with_style y_axis_style
             |> Charts.Axis.with_format (fun _ v -> Printf.sprintf "%.1f" v))
      |> Charts.with_grid
           (Charts.Gridlines.default
           |> Charts.Gridlines.with_style grid_style
           |> Charts.Gridlines.with_x true
           |> Charts.Gridlines.with_y true)
      |> Charts.line ~resolution:`Braille2x4
           ~style:(Ansi.Style.make ~fg:Ansi.Color.cyan ())
           ~x:fst ~y:snd data
    in
    ignore (Charts.draw chart grid ~width ~height)

let view_metric_chart ~history_for_tag tag =
  let history = history_for_tag tag in
  box ~border:true ~title:tag ~padding:(padding 0)
    ~size:{ width = pct 100; height = px 14 }
    [
      canvas
        ~draw:(fun grid ~width ~height ->
          draw_metric_chart tag history grid ~width ~height)
        ~size:{ width = pct 100; height = pct 100 }
        ();
    ]

(* ───── View ───── *)

(* We use 'a since we don't need the metric value - only the tag for lookup *)
type 'a view_params = {
  latest_metrics : (string * 'a) list;
  history_for_tag : string -> (int * float) list;
  screen_height : int;
  current_batch : int;
}

let view (params : _ view_params) =
  let latest = params.latest_metrics in
  if latest = [] then
    box ~padding:(padding 1)
      [ text ~style:hint_style "  Waiting for metrics..." ]
  else
    let total_metrics = List.length latest in
    let graphs_per_batch = calculate_graphs_per_batch params.screen_height in
    let total_batches =
      if total_metrics = 0 then 1
      else (total_metrics + graphs_per_batch - 1) / graphs_per_batch
    in
    let current_batch = min params.current_batch (max 0 (total_batches - 1)) in
    let start_idx = current_batch * graphs_per_batch in
    let end_idx = min (start_idx + graphs_per_batch) total_metrics in
    let visible_metrics =
      List.mapi (fun i (tag, metric) -> (i, tag, metric)) latest
      |> List.filter (fun (i, _, _) -> i >= start_idx && i < end_idx)
      |> List.map (fun (_, tag, metric) -> (tag, metric))
    in
    box ~flex_direction:Column ~padding:(padding 1) ~gap:(gap 1)
      [
        (if total_batches > 1 then
           box ~flex_direction:Row ~justify_content:Space_between
             ~align_items:Center
             [
               text ~style:(Ansi.Style.make ~bold:true ()) "Metrics:";
               text ~style:hint_style
                 (Printf.sprintf "Batch %d/%d (← →)" (current_batch + 1)
                    total_batches);
             ]
         else
           box ~flex_direction:Row
             [ text ~style:(Ansi.Style.make ~bold:true ()) "Metrics:" ]);
        box ~flex_direction:Column ~gap:(gap 1)
          (List.map
             (fun (tag, _metric) ->
               view_metric_chart ~history_for_tag:params.history_for_tag tag)
             visible_metrics);
      ]
