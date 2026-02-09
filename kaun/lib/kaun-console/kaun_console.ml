(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Kaun Console - TUI for monitoring training runs. *)

open Mosaic
open Kaun_runlog

(* Import components *)
module Header = Kaun_console_components.Header
module Footer = Kaun_console_components.Footer
module Metrics = Kaun_console_components.Metrics
module Imp_info = Kaun_console_components.Imp_info
module Sys_panel = Kaun_console_components.Sys_panel

(* ───── Model ───── *)

type model = {
  run_id : string;
  store : Metric_store.t;
  stream : Run.event_stream;
  screen_height : int;
  current_batch : int;
  sys_panel : Sys_panel.t;
}

type msg = Tick of float | Quit | Resize of int * int | Next_batch | Prev_batch

(* ───── View ───── *)

let divider () =
  box
    ~size:{ width = px 1; height = pct 100 }
    ~background:(Ansi.Color.grayscale ~level:8)
    [ text " " ]

let view m =
  (* Convert Metric_store.best_value to Imp_info.best_metric *)
  let best_metrics =
    Metric_store.best_metrics m.store
    |> List.map (fun (tag, (bv : Metric_store.best_value)) ->
           (tag, ({ step = bv.step; value = bv.value } : Imp_info.best_metric)))
  in
  box ~flex_direction:Column
    ~size:{ width = pct 100; height = pct 100 }
    [
      Header.view ~run_id:m.run_id
        ~latest_epoch:(Metric_store.latest_epoch m.store);
      box ~flex_direction:Row ~flex_grow:1.0
        ~size:{ width = pct 100; height = pct 100 }
        [
          (* Left column: imp info *)
          scroll_box ~scroll_y:true ~scroll_x:false
            ~size:{ width = pct 33; height = pct 100 }
            [ Imp_info.view ~best_metrics ];
          divider ();
          (* Middle column: metrics *)
          scroll_box ~scroll_y:false ~scroll_x:false
            ~size:{ width = pct 34; height = pct 100 }
            [
              Metrics.view
                {
                  latest_metrics = Metric_store.latest_metrics m.store;
                  history_for_tag = Metric_store.history_for_tag m.store;
                  screen_height = m.screen_height;
                  current_batch = m.current_batch;
                };
            ];
          divider ();
          (* Right column: sys panel *)
          scroll_box ~scroll_y:true ~scroll_x:false
            ~size:{ width = pct 33; height = pct 100 }
            [ Sys_panel.view m.sys_panel ];
        ];
      Footer.view ();
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
  (* Initialize system panel *)
  let sys_panel = Sys_panel.create () in
  ( { run_id; store; stream; screen_height = initial_height; current_batch = 0; sys_panel },
    Cmd.none )

let update msg m =
  match msg with
  | Tick dt ->
      let new_events = Run.read_events m.stream in
      Metric_store.update m.store new_events;
      let sys_panel = Sys_panel.update m.sys_panel ~dt in
      ({ m with store = m.store; sys_panel }, Cmd.none)
  | Resize (_width, height) ->
      (* Recalculate current batch to ensure it's still valid after resize *)
      let latest = Metric_store.latest_metrics m.store in
      let total_metrics = List.length latest in
      let graphs_per_batch = Metrics.calculate_graphs_per_batch height in
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
      let graphs_per_batch = Metrics.calculate_graphs_per_batch m.screen_height in
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

let run ?base_dir ?experiment:_ ?tags:_ ?runs () =
  let base_dir = Option.value base_dir ~default:(Kaun_runlog.base_dir ()) in
  match runs with
  | Some [ run_id ] -> (
      let run_dir = Filename.concat base_dir run_id in
      match Run.load run_dir with
      | Some run ->
          let init () = init ~run in
          Mosaic.run { init; update; view; subscriptions }
      | None -> Printf.printf "kaun-console: run not found: %s\n%!" run_id)
  | _ -> Printf.printf "kaun-console: please specify a single run\n%!"
