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
let min_graph_width = 25
let header_height = 3
let footer_height = 1
let metrics_padding = 2
let metrics_width_ratio = 0.66

(* ───── Helpers ───── *)

(** Calculate how many columns fit in available width *)
let calculate_columns (screen_width : int) : int =
  let metrics_width = int_of_float (float_of_int screen_width *. metrics_width_ratio) in
  if metrics_width < min_graph_width * 2 then 1 else 2

(** Calculate how many rows of graphs fit in available height *)
let calculate_rows_per_batch (screen_height : int) : int =
  let available_height =
    screen_height - header_height - footer_height - (metrics_padding * 2)
  in
  if available_height < graph_height then 1
  else max 1 (available_height / graph_height)

(** Calculate how many graphs fit in available space *)
let calculate_graphs_per_batch ~width ~height : int =
  let columns = calculate_columns width in
  let rows = calculate_rows_per_batch height in
  rows * columns

(* ───── Chart Drawing ───── *)

let draw_metric_chart ~hover history grid ~width ~height =
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
      |> Charts.line ~id:"metric" ~resolution:`Braille2x4
           ~style:(Ansi.Style.make ~fg:Ansi.Color.cyan ())
           ~x:fst ~y:snd data
    in
    let layout = Charts.draw chart grid ~width ~height in
    (* Draw tooltip if hovering *)
    match hover with
    | None -> ()
    | Some (px, py) ->
        if Charts.Layout.is_inside_plot layout ~px ~py then
          match
            Charts.Layout.hit_test layout ~px ~py ~radius:4 ~policy:`Nearest_x
          with
          | Some hit -> (
              match hit.payload with
              | Charts.Hit.XY { x; y } ->
                  let lines =
                    [
                      Printf.sprintf "Step: %d" (int_of_float x);
                      Printf.sprintf "Value: %.4f" y;
                    ]
                  in
                  Charts.Overlay.crosshair layout grid ~x ~y;
                  Charts.Overlay.marker layout grid ~x ~y;
                  Charts.Overlay.tooltip layout grid ~x ~y lines
              | _ -> ())
          | None -> ()

let view_metric_chart ~history_for_tag ~columns tag =
  let history = history_for_tag tag in
  let width_pct = if columns = 1 then 100 else 49 in
  (* Show current (latest) value in title at all times. *)
  let rec last_value = function
    | [] -> None
    | [ (_, v) ] -> Some v
    | _ :: rest -> last_value rest
  in
  let title =
    match last_value history with
    | Some value -> Printf.sprintf "%s [%.4f]" tag value
    | None -> tag
  in
  box ~key:tag ~border:true ~title ~padding:(padding 0)
    ~size:{ width = pct width_pct; height = px 14 }
    [
      canvas
        ~draw:(fun grid ~width ~height ->
          draw_metric_chart ~hover:None history grid ~width ~height)
        ~size:{ width = pct 100; height = pct 100 }
        ();
    ]

(* ───── View ───── *)

(* We use 'a since we don't need the metric value - only the tag for lookup *)
type 'a view_params = {
  latest_metrics : (string * 'a) list;
  history_for_tag : string -> (int * float) list;
  screen_width : int;
  screen_height : int;
  current_batch : int;
}

(** Chunk a list into groups of n *)
let rec chunk_by n lst =
  if lst = [] then []
  else
    let rec take k acc = function
      | [] -> (List.rev acc, [])
      | x :: xs -> if k = 0 then (List.rev acc, x :: xs) else take (k - 1) (x :: acc) xs
    in
    let group, rest = take n [] lst in
    group :: chunk_by n rest

let view (params : _ view_params) =
  let latest = params.latest_metrics in
  if latest = [] then
    box ~padding:(padding 1) ~size:{ width = pct 66; height = pct 100 }
      [ text ~style:hint_style "  Waiting for metrics..." ]
  else
    let columns = calculate_columns params.screen_width in
    let total_metrics = List.length latest in
    let graphs_per_batch =
      calculate_graphs_per_batch ~width:params.screen_width
        ~height:params.screen_height
    in
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
      |> List.map (fun (_, tag, _metric) -> tag)
    in
    let rows = chunk_by columns visible_metrics in
    box ~flex_direction:Column ~padding:(padding 1) ~gap:(gap 1)
      ~size:{ width = pct 66; height = pct 100 }
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
          (List.mapi
             (fun row_idx row ->
               box ~key:(Printf.sprintf "row-%d" row_idx) ~flex_direction:Row
                 ~gap:(gap 1)
                 ~size:{ width = pct 100; height = auto }
                 (List.map
                    (fun tag ->
                      view_metric_chart ~history_for_tag:params.history_for_tag
                        ~columns tag)
                    row))
             rows);
      ]
