(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Mosaic
open Kaun_runlog
module Charts = Matrix_charts

(* ───── Model ───── *)

type model = {
  run_id : string;
  store : Metric_store.t;
  stream : Run.event_stream;
  screen_height : int;
  current_batch : int;
}

type msg = Tick of float | Quit | Resize of int * int | Next_batch | Prev_batch

(* ───── Constants ───── *)

let header_bg = Ansi.Color.of_rgb 30 80 100
let hint_style = Ansi.Style.make ~fg:(Ansi.Color.grayscale ~level:14) ()
let step_color = Ansi.Color.cyan
let epoch_color = Ansi.Color.cyan

(* Graph dimensions *)
let graph_height = 14
let header_height = 3
let footer_height = 1
let metrics_padding = 2

(* Calculate how many graphs fit in available height *)
let calculate_graphs_per_batch (screen_height : int) : int =
  let available_height =
    screen_height - header_height - footer_height - (metrics_padding * 2)
  in
  if available_height < graph_height then 1
  else max 1 (available_height / graph_height)

(* ───── View Components ───── *)

let view_header ~run_id store =
  box ~padding:(padding 1) ~background:header_bg
    ~size:{ width = pct 100; height = auto }
    [
      box ~flex_direction:Row ~gap:(gap 2) ~align_items:Center
        ~size:{ width = pct 100; height = auto }
        [
          text ~style:(Ansi.Style.make ~bold:true ()) "▸ Kaun Console";
          text
            ~style:(Ansi.Style.make ~fg:step_color ())
            (Printf.sprintf "Run: %s" run_id);
          (match Metric_store.latest_epoch store with
          | None -> text ~style:hint_style "Epoch: -"
          | Some e ->
              text
                ~style:(Ansi.Style.make ~fg:epoch_color ())
                (Printf.sprintf "Epoch: %d" e));
          box ~flex_grow:1.0 ~size:{ width = auto; height = auto } [];
          box ~padding:(padding 1) ~background:Ansi.Color.green
            [
              text
                ~style:(Ansi.Style.make ~bold:true ~fg:Ansi.Color.white ())
                "LIVE";
            ];
        ];
    ]

(* ───── Chart Drawing ───── *)

let axis_style =
  Ansi.Style.make ~fg:(Ansi.Color.grayscale ~level:12) ~dim:true ()

let y_axis_style =
  Ansi.Style.make ~fg:(Ansi.Color.grayscale ~level:12) ~dim:true ()

let grid_style =
  Ansi.Style.make ~fg:(Ansi.Color.grayscale ~level:6) ~dim:true ()

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

let view_metric_chart store tag (_m : Metric_store.metric) =
  let history = Metric_store.history_for_tag store tag in
  box ~border:true ~title:tag ~padding:(padding 0)
    ~size:{ width = pct 100; height = px 14 }
    [
      canvas
        ~draw:(fun grid ~width ~height ->
          draw_metric_chart tag history grid ~width ~height)
        ~size:{ width = pct 100; height = pct 100 }
        ();
    ]

let view_metrics m =
  let latest = Metric_store.latest_metrics m.store in
  if latest = [] then
    box ~padding:(padding 1)
      [ text ~style:hint_style "  Waiting for metrics..." ]
  else
    let total_metrics = List.length latest in
    let graphs_per_batch = calculate_graphs_per_batch m.screen_height in
    let total_batches =
      if total_metrics = 0 then 1
      else (total_metrics + graphs_per_batch - 1) / graphs_per_batch
    in
    let current_batch = min m.current_batch (max 0 (total_batches - 1)) in
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
             (fun (tag, metric) -> view_metric_chart m.store tag metric)
             visible_metrics);
      ]

let view_imp_info () =
  box ~padding:(padding 1)
    [ text ~style:(Ansi.Style.make ~bold:true ()) "imp info" ]

let view_sys_panel () =
  box ~padding:(padding 1)
    [ text ~style:(Ansi.Style.make ~bold:true ()) "sys panel" ]

let view_footer () =
  box ~padding:(padding 1)
    [ text ~style:hint_style "(Press Ctrl-C to quit)" ]

let view m =
  box ~flex_direction:Column
    ~size:{ width = pct 100; height = pct 100 }
    [
      view_header ~run_id:m.run_id m.store;
      box ~flex_direction:Row ~flex_grow:1.0
        ~size:{ width = pct 100; height = pct 100 }
        [
          (* Left column: imp info *)
          scroll_box ~scroll_y:true ~scroll_x:false
            ~size:{ width = pct 33; height = pct 100 }
            [ view_imp_info () ];
          box
            ~size:{ width = px 1; height = pct 100 }
            ~background:(Ansi.Color.grayscale ~level:8)
            [ text " " ];
          (* Middle column: metrics - scroll disabled since we use batch navigation *)
          scroll_box ~scroll_y:false ~scroll_x:false
            ~size:{ width = pct 34; height = pct 100 }
            [ view_metrics m ];
          box
            ~size:{ width = px 1; height = pct 100 }
            ~background:(Ansi.Color.grayscale ~level:8)
            [ text " " ];
          (* Right column: sys panel *)
          scroll_box ~scroll_y:true ~scroll_x:false
            ~size:{ width = pct 33; height = pct 100 }
            [ view_sys_panel () ];
        ];
      view_footer ();
    ]

(* ───── TEA Core ───── *)

let get_initial_terminal_height () : int =
  try
    let ic = Unix.open_process_in "stty size 2>/dev/null" in
    let line = input_line ic in
    ignore (Unix.close_process_in ic);
    match String.split_on_char ' ' line with
    | rows :: _ -> int_of_string rows
    | _ -> 24
  with _ -> 24

let init ~run =
  let run_id = Run.run_id run in
  let stream = Run.open_events run in
  let store = Metric_store.create () in
  (* Load initial events *)
  let initial_events = Run.read_events stream in
  Metric_store.update store initial_events;
  (* Get actual terminal height at startup *)
  let initial_height = get_initial_terminal_height () in
  ({ run_id; store; stream; screen_height = initial_height; current_batch = 0 }, Cmd.none)

let update msg m =
  match msg with
  | Tick _ ->
      let new_events = Run.read_events m.stream in
      Metric_store.update m.store new_events;
      ({ m with store = m.store }, Cmd.none)
  | Resize (_width, height) ->
      (* Recalculate current batch to ensure it's still valid after resize *)
      let latest = Metric_store.latest_metrics m.store in
      let total_metrics = List.length latest in
      let graphs_per_batch = calculate_graphs_per_batch height in
      let total_batches =
        if total_metrics = 0 then 1
        else (total_metrics + graphs_per_batch - 1) / graphs_per_batch
      in
      let max_batch = max 0 (total_batches - 1) in
      let current_batch = min m.current_batch max_batch in
      ({ m with screen_height = height; current_batch }, Cmd.none)
  | Next_batch ->
      let latest = Metric_store.latest_metrics m.store in
      let total_metrics = List.length latest in
      let graphs_per_batch = calculate_graphs_per_batch m.screen_height in
      let total_batches =
        if total_metrics = 0 then 1
        else (total_metrics + graphs_per_batch - 1) / graphs_per_batch
      in
      let max_batch = max 0 (total_batches - 1) in
      ({ m with current_batch = min (m.current_batch + 1) max_batch }, Cmd.none)
  | Prev_batch ->
      ({ m with current_batch = max 0 (m.current_batch - 1) }, Cmd.none)
  | Quit ->
      Run.close_events m.stream;
      (m, Cmd.quit)

let subscriptions _model =
  Sub.batch
    [
      Sub.on_tick (fun ~dt -> Tick dt);
      Sub.on_resize (fun ~width ~height -> Resize (width, height));
      Sub.on_key (fun ev ->
          match (Mosaic_ui.Event.Key.data ev).key with
          | Char c when Uchar.equal c (Uchar.of_char 'q') -> Some Quit
          | Char c when Uchar.equal c (Uchar.of_char 'Q') -> Some Quit
          | Escape -> Some Quit
          | Left -> Some Prev_batch
          | Right -> Some Next_batch
          | _ -> None);
    ]

let run ?(base_dir = "./runs") ?experiment:_ ?tags:_ ?runs () =
  match runs with
  | Some [ run_id ] -> (
      let run_dir = Filename.concat base_dir run_id in
      match Run.load run_dir with
      | Some run ->
          let init () = init ~run in
          Mosaic.run { init; update; view; subscriptions }
      | None -> Printf.printf "kaun-console: run not found: %s\n%!" run_id)
  | _ -> Printf.printf "kaun-console: please specify a single run\n%!"
