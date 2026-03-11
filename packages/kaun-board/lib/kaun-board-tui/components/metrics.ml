(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Mosaic
module Charts = Matrix_charts

(* Styles *)

let hint_style = Ansi.Style.make ~fg:(Ansi.Color.grayscale ~level:14) ()

let axis_style =
  Ansi.Style.make ~fg:(Ansi.Color.grayscale ~level:12) ~dim:true ()

let grid_style =
  Ansi.Style.make ~fg:(Ansi.Color.grayscale ~level:6) ~dim:true ()

(* Constants *)

let graph_height = 14
let min_graph_width = 25
let header_height = 3
let footer_height = 1
let metrics_padding = 2
let metrics_width_ratio = 0.66

(* Layout helpers *)

let calculate_columns (screen_width : int) : int =
  let metrics_width =
    int_of_float (float_of_int screen_width *. metrics_width_ratio)
  in
  if metrics_width < min_graph_width * 2 then 1 else 2

let calculate_rows_per_batch (screen_height : int) : int =
  let available_height =
    screen_height - header_height - footer_height - (metrics_padding * 2)
  in
  if available_height < graph_height then 1
  else max 1 (available_height / graph_height)

let calculate_graphs_per_batch ~width ~height : int =
  let columns = calculate_columns width in
  let rows = calculate_rows_per_batch height in
  rows * columns

(* Component state and update *)

type state = {
  screen_width : int;
  screen_height : int;
  current_batch : int;
  selected : int;
}

type msg =
  | Resize of int * int
  | Next_batch
  | Prev_batch
  | Select_left
  | Select_right
  | Select_up
  | Select_down

type batch_window = {
  start_idx : int;
  end_idx : int;
  total_batches : int;
  current_batch : int;
}

let batch_window ~width ~height ~current_batch ~total_metrics =
  if total_metrics = 0 then
    { start_idx = 0; end_idx = 0; total_batches = 1; current_batch = 0 }
  else
    let per_batch = calculate_graphs_per_batch ~width ~height in
    let total_batches = (total_metrics + per_batch - 1) / per_batch in
    let current_batch = min current_batch (max 0 (total_batches - 1)) in
    let start_idx = current_batch * per_batch in
    let end_idx = min (start_idx + per_batch) total_metrics in
    { start_idx; end_idx; total_batches; current_batch }

let initial_state () =
  { screen_width = 80; screen_height = 24; current_batch = 0; selected = 0 }

let visible_count s ~total_metrics =
  let w =
    batch_window ~width:s.screen_width ~height:s.screen_height
      ~current_batch:s.current_batch ~total_metrics
  in
  w.end_idx - w.start_idx

let update (msg : msg) (s : state) ~total_metrics : state =
  match msg with
  | Resize (width, height) ->
      let w =
        batch_window ~width ~height ~current_batch:s.current_batch
          ~total_metrics
      in
      let n = w.end_idx - w.start_idx in
      {
        screen_width = width;
        screen_height = height;
        current_batch = w.current_batch;
        selected = min s.selected (max 0 (n - 1));
      }
  | Next_batch ->
      let w =
        batch_window ~width:s.screen_width ~height:s.screen_height
          ~current_batch:(s.current_batch + 1) ~total_metrics
      in
      if w.current_batch = s.current_batch then s
      else { s with current_batch = w.current_batch; selected = 0 }
  | Prev_batch ->
      let prev = max 0 (s.current_batch - 1) in
      if prev = s.current_batch then s
      else { s with current_batch = prev; selected = 0 }
  | Select_left ->
      let n = visible_count s ~total_metrics in
      if n = 0 || s.selected = 0 then s
      else { s with selected = s.selected - 1 }
  | Select_right ->
      let n = visible_count s ~total_metrics in
      if n = 0 || s.selected >= n - 1 then s
      else { s with selected = s.selected + 1 }
  | Select_up ->
      let cols = calculate_columns s.screen_width in
      if s.selected >= cols then { s with selected = s.selected - cols } else s
  | Select_down ->
      let n = visible_count s ~total_metrics in
      let cols = calculate_columns s.screen_width in
      if s.selected + cols < n then { s with selected = s.selected + cols }
      else s

let visible_chart_tags (s : state) ~total_metrics ~all_tags : string list =
  let w =
    batch_window ~width:s.screen_width ~height:s.screen_height
      ~current_batch:s.current_batch ~total_metrics
  in
  List.mapi (fun i tag -> (i, tag)) all_tags
  |> List.filter (fun (i, _) -> i >= w.start_idx && i < w.end_idx)
  |> List.map snd

(* Chart drawing *)

let draw_metric_chart history grid ~width ~height =
  if history = [] then ()
  else
    let data =
      Array.of_list
        (List.map (fun (step, value) -> (float_of_int step, value)) history)
    in
    let chart =
      Charts.empty ()
      |> Charts.with_frame (Charts.manual_frame ~margins:(1, 0, 0, 2) ())
      |> Charts.with_axes
           ~x:
             (Charts.Axis.default |> Charts.Axis.with_ticks 4
             |> Charts.Axis.with_style axis_style)
           ~y:
             (Charts.Axis.default |> Charts.Axis.with_ticks 2
             |> Charts.Axis.with_style axis_style
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
    ignore (Charts.draw chart grid ~width ~height)

let dim_border = Ansi.Color.grayscale ~level:6
let selected_border = Ansi.Color.cyan

let view_metric_chart ~history_for_tag ~columns ~selected tag =
  let history = history_for_tag tag in
  let width_pct = if columns = 1 then 100 else 49 in
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
  let border_color = if selected then selected_border else dim_border in
  box ~key:tag ~border:true ~border_color ~title ~padding:(padding 0)
    ~size:{ width = pct width_pct; height = px 14 }
    [
      canvas
        ~size:{ width = pct 100; height = pct 100 }
        (fun c ~delta:_ ->
          draw_metric_chart history (Canvas.grid c) ~width:(Canvas.width c)
            ~height:(Canvas.height c));
    ]

(* View *)

type view_params = {
  metric_tags : string list;
  history_for_tag : string -> (int * float) list;
  screen_width : int;
  screen_height : int;
  current_batch : int;
  selected : int;
}

let rec chunk_by n lst =
  if lst = [] then []
  else
    let rec take k acc = function
      | [] -> (List.rev acc, [])
      | x :: xs ->
          if k = 0 then (List.rev acc, x :: xs) else take (k - 1) (x :: acc) xs
    in
    let group, rest = take n [] lst in
    group :: chunk_by n rest

let view (params : view_params) =
  if params.metric_tags = [] then
    box ~padding:(padding 1)
      ~size:{ width = pct 66; height = pct 100 }
      [ text ~style:hint_style "  Waiting for metrics..." ]
  else
    let columns = calculate_columns params.screen_width in
    let total_metrics = List.length params.metric_tags in
    let w =
      batch_window ~width:params.screen_width ~height:params.screen_height
        ~current_batch:params.current_batch ~total_metrics
    in
    let visible_metrics =
      List.mapi (fun i tag -> (i, tag)) params.metric_tags
      |> List.filter (fun (i, _) -> i >= w.start_idx && i < w.end_idx)
      |> List.mapi (fun local_idx (_, tag) -> (local_idx, tag))
    in
    let rows = chunk_by columns visible_metrics in
    box ~flex_direction:Column ~padding:(padding 1) ~gap:(gap 1)
      ~size:{ width = pct 66; height = pct 100 }
      ([
         (if w.total_batches > 1 then
            box ~flex_direction:Row ~justify_content:Flex_end
              ~align_items:Center
              ~size:{ width = pct 100; height = auto }
              [
                text ~style:hint_style
                  (Printf.sprintf "Batch %d/%d" (w.current_batch + 1)
                     w.total_batches);
              ]
          else box ~size:{ width = px 0; height = px 0 } []);
       ]
      @ [
          box ~flex_direction:Column ~gap:(gap 1)
            (List.mapi
               (fun row_idx row ->
                 box
                   ~key:(Printf.sprintf "row-%d" row_idx)
                   ~flex_direction:Row ~gap:(gap 1)
                   ~size:{ width = pct 100; height = auto }
                   (List.map
                      (fun (local_idx, tag) ->
                        view_metric_chart
                          ~history_for_tag:params.history_for_tag ~columns
                          ~selected:(local_idx = params.selected)
                          tag)
                      row))
               rows);
        ])
