(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Mosaic

(* Constants *)

let graph_height = 14
let min_graph_width = 25
let header_height = 3
let footer_height = 1
let metrics_padding = 2
(* Layout helpers *)

let calculate_columns (available_width : int) : int =
  if available_width < min_graph_width * 2 then 1 else 2

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

let selected_tag (s : state) ~total_metrics ~all_tags =
  let w =
    batch_window ~width:s.screen_width ~height:s.screen_height
      ~current_batch:s.current_batch ~total_metrics
  in
  let idx = w.start_idx + s.selected in
  if idx < w.end_idx then List.nth_opt all_tags idx else None

(* Metric grouping *)

let group_prefix tag =
  match String.index_opt tag '/' with
  | Some i -> String.sub tag 0 i
  | None -> ""

let section_style =
  Ansi.Style.make ~fg:(Ansi.Color.grayscale ~level:12) ~dim:true ()

let view_group_header prefix =
  box
    ~size:{ width = pct 100; height = auto }
    [ text ~style:section_style (Printf.sprintf "\u{2500}\u{2500} %s " prefix) ]

let prefix_groups metrics =
  let rec go current_prefix acc group = function
    | [] ->
        let groups =
          if group = [] then acc else (current_prefix, List.rev group) :: acc
        in
        List.rev groups
    | ((_, tag) as item) :: rest ->
        let p = group_prefix tag in
        if String.equal p current_prefix then
          go current_prefix acc (item :: group) rest
        else
          let acc =
            if group = [] then acc else (current_prefix, List.rev group) :: acc
          in
          go p acc [ item ] rest
  in
  match metrics with
  | [] -> []
  | ((_, tag) as item) :: rest -> go (group_prefix tag) [] [ item ] rest

(* Chart rendering *)

let dim_border = Ansi.Color.grayscale ~level:6
let selected_border = Ansi.Color.white

let view_metric_chart ~history_for_tag ~(best_for_tag : string -> float option)
    ~goal_for_tag ~columns ~selected tag =
  let history = history_for_tag tag in
  let best = best_for_tag tag in
  let width_pct = if columns = 1 then 100 else 49 in
  let goal_arrow =
    match goal_for_tag tag with
    | Some `Minimize -> " \u{2193}"
    | Some `Maximize -> " \u{2191}"
    | None -> ""
  in
  let title =
    match (Theme.last_value history, best) with
    | Some value, Some best_val ->
        Printf.sprintf "%s [%.4f] best: %.4f%s" tag value best_val goal_arrow
    | Some value, None -> Printf.sprintf "%s [%.4f]" tag value
    | None, _ -> tag
  in
  let border_color = if selected then selected_border else dim_border in
  box ~key:tag ~border:true ~border_color ~title ~padding:(padding 0)
    ~size:{ width = pct width_pct; height = px 14 }
    [
      canvas
        ~size:{ width = pct 100; height = pct 100 }
        (fun c ~delta:_ ->
          Theme.draw_metric_chart ~compact:true history (Canvas.grid c)
            ~width:(Canvas.width c) ~height:(Canvas.height c));
    ]

(* View *)

let rec chunk_by n = function
  | [] -> []
  | lst ->
      let rec take k acc = function
        | [] -> (List.rev acc, [])
        | x :: xs ->
            if k = 0 then (List.rev acc, x :: xs)
            else take (k - 1) (x :: acc) xs
      in
      let group, rest = take n [] lst in
      group :: chunk_by n rest

let view (s : state) ~metric_tags ~history_for_tag ~best_for_tag ~goal_for_tag =
  if metric_tags = [] then
    box ~padding:(padding 1)
      ~size:{ width = pct 100; height = auto }
      [ text ~style:Theme.muted_style "  Waiting for metrics..." ]
  else
    let columns = calculate_columns s.screen_width in
    let total_metrics = List.length metric_tags in
    let w =
      batch_window ~width:s.screen_width ~height:s.screen_height
        ~current_batch:s.current_batch ~total_metrics
    in
    let visible_metrics =
      List.mapi (fun i tag -> (i, tag)) metric_tags
      |> List.filter (fun (i, _) -> i >= w.start_idx && i < w.end_idx)
      |> List.mapi (fun local_idx (_, tag) -> (local_idx, tag))
    in
    let groups = prefix_groups visible_metrics in
    let batch_header =
      if w.total_batches > 1 then
        [
          box ~flex_direction:Row ~justify_content:Flex_end ~align_items:Center
            ~size:{ width = pct 100; height = auto }
            [
              text ~style:Theme.muted_style
                (Printf.sprintf "Batch %d/%d" (w.current_batch + 1)
                   w.total_batches);
            ];
        ]
      else []
    in
    let charts =
      List.concat_map
        (fun (prefix, group_metrics) ->
          let header =
            if prefix <> "" then [ view_group_header prefix ] else []
          in
          let rows = chunk_by columns group_metrics in
          let chart_rows =
            List.mapi
              (fun row_idx row ->
                box
                  ~key:(Printf.sprintf "row-%s-%d" prefix row_idx)
                  ~flex_direction:Row ~gap:(gap 1)
                  ~size:{ width = pct 100; height = auto }
                  (List.map
                     (fun (local_idx, tag) ->
                       view_metric_chart ~history_for_tag ~best_for_tag
                         ~goal_for_tag ~columns
                         ~selected:(local_idx = s.selected) tag)
                     row))
              rows
          in
          header @ chart_rows)
        groups
    in
    box ~flex_direction:Column ~padding:(padding_lrtb 1 1 1 0) ~gap:(gap 1)
      ~size:{ width = pct 100; height = auto }
      (batch_header @ charts)
