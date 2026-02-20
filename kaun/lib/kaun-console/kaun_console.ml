(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Kaun Console - TUI for monitoring training runs. *)

open Mosaic
open Kaun_runlog

(* ───── Model ───── *)

type model = {
  loading : bool;
  loading_time : float;
  run_id : string;
  store : Metric_store.t;
  stream : Run.event_stream;
  metrics_state : Metrics.state;
  sys_panel : Sys_panel.t;
  mode : mode;
}

and mode =
  | Dashboard
  | Detail of string

type msg =
  | Tick of float
  | Quit
  | Metrics_msg of Metrics.msg
  | Open_metric of string
  | Close_metric

(* ───── Helpers ───── *)

let visible_chart_tags (m : model) : string list =
  let latest = Metric_store.latest_metrics m.store in
  let all_tags = List.map fst latest in
  let total_metrics = List.length all_tags in
  Metrics.visible_chart_tags m.metrics_state ~total_metrics ~all_tags

(* ───── View ───── *)

let divider () =
  box
    ~size:{ width = px 1; height = pct 100 }
    ~background:(Ansi.Color.grayscale ~level:8)
    [ text " " ]

let view_dashboard m =
  (* Convert Metric_store.best_value to Imp_info.best_metric *)
  (* let best_metrics =
       Metric_store.best_metrics m.store
       |> List.map (fun (tag, (bv : Metric_store.best_value)) ->
              (tag, ({ step = bv.step; value = bv.value } : Imp_info.best_metric)))
     in *)
  box ~flex_direction:Column
    ~size:{ width = pct 100; height = pct 100 }
    [
      Header.view ~run_id:m.run_id
        ~latest_epoch:(Metric_store.latest_epoch m.store);
      box ~flex_direction:Row ~flex_grow:1.0
        ~size:{ width = pct 100; height = pct 100 }
        [
          (* Left column: metrics (2/3 width) - no scroll_box to allow mouse events *)
          Metrics.view
            {
              latest_metrics = Metric_store.latest_metrics m.store;
              history_for_tag = Metric_store.history_for_tag m.store;
              screen_width = m.metrics_state.screen_width;
              screen_height = m.metrics_state.screen_height;
              current_batch = m.metrics_state.current_batch;
            };
          divider ();
          (* Right column: sys panel (1/3 width) *)
          scroll_box ~scroll_y:true ~scroll_x:false
            ~size:{ width = pct 34; height = pct 100 }
            [ Sys_panel.view m.sys_panel ];
        ];
      Footer.view ();
    ]

let view_detail m tag =
  box ~flex_direction:Column
    ~size:{ width = pct 100; height = pct 100 }
    ~background:(Ansi.Color.of_rgb 20 20 30)
    [
      box ~padding:(padding 1) ~size:{ width = pct 100; height = auto }
        [
          text
            ~style:(Ansi.Style.make ~bold:true ~fg:(Ansi.Color.grayscale ~level:14) ())
            "Chart View  •  [Esc/q] back";
        ];
      box ~flex_grow:1.0 ~justify_content:Center ~align_items:Center
        ~size:{ width = pct 100; height = pct 100 }
        [
          Chart_view.view ~tag
            ~history_for_tag:(Metric_store.history_for_tag m.store)
            ~best:
              (Option.map
                 (fun (b : Metric_store.best_value) -> b.value)
                 (Metric_store.best_for_tag m.store tag))
            ~size:{ width = pct 80; height = pct 80 };
        ];
    ]

let view m =
  if m.loading then Splash.view ()
  else
    match m.mode with
    | Dashboard -> view_dashboard m
    | Detail tag -> view_detail m tag

(* ───── TEA Core ───── *)

let get_initial_terminal_size () : int * int =
  try
    let ic = Unix.open_process_in "stty size 2>/dev/null" in
    let line = input_line ic in
    ignore (Unix.close_process_in ic);
    match String.split_on_char ' ' line with
    | [ rows; cols ] -> (int_of_string cols, int_of_string rows)
    | _ -> (80, 24)
  with _ -> (80, 24)

let init ~run =
  let run_id = Run.run_id run in
  let stream = Run.open_events run in
  let store = Metric_store.create () in
  (* Load initial events *)
  let initial_events = Run.read_events stream in
  Metric_store.update store initial_events;
  (* Get actual terminal size at startup *)
  let initial_width, initial_height = get_initial_terminal_size () in
  let metrics_state =
    { (Metrics.initial_state ()) with
      screen_width = initial_width;
      screen_height = initial_height;
    }
  in
  (* Initialize system panel *)
  let sys_panel = Sys_panel.create () in
  ( {
      loading = true;
      loading_time = 0.0;
      run_id;
      store;
      stream;
      metrics_state;
      sys_panel;
      mode = Dashboard;
    },
    Cmd.none )

let splash_duration = 1.5

let update msg m =
  match msg with
  | Tick dt ->
      (* Handle splash screen transition *)
      let new_loading_time =
        if m.loading then m.loading_time +. dt else m.loading_time
      in
      let should_finish_loading = new_loading_time >= splash_duration in
      let m =
        if should_finish_loading then
          { m with loading = false; loading_time = 0.0 }
        else { m with loading_time = new_loading_time }
      in
      (* Don't update metrics while showing splash screen *)
      if m.loading then (m, Cmd.none)
      else
        let new_events = Run.read_events m.stream in
        Metric_store.update m.store new_events;
        let sys_panel = Sys_panel.update m.sys_panel ~dt in
        ({ m with store = m.store; sys_panel }, Cmd.none)
  | Metrics_msg metrics_msg ->
      let total_metrics =
        List.length (Metric_store.latest_metrics m.store)
      in
      let metrics_state' =
        Metrics.update metrics_msg m.metrics_state ~total_metrics
      in
      ({ m with metrics_state = metrics_state' }, Cmd.none)
  | Open_metric tag -> ({ m with mode = Detail tag }, Cmd.none)
  | Close_metric -> ({ m with mode = Dashboard }, Cmd.none)
  | Quit ->
      Run.close_events m.stream;
      (m, Cmd.quit)

let subscriptions m =
  Sub.batch
    [
      Sub.on_tick (fun ~dt -> Tick dt);
      Sub.on_resize (fun ~width ~height ->
          Metrics_msg (Metrics.Resize (width, height)));
      Sub.on_key (fun ev ->
          let key_data = Mosaic_ui.Event.Key.data ev in
          match key_data.key with
          | Char c when Uchar.equal c (Uchar.of_char 'q') -> (
              match m.mode with Dashboard -> Some Quit | Detail _ -> Some Close_metric)
          | Char c when Uchar.equal c (Uchar.of_char 'Q') -> (
              match m.mode with Dashboard -> Some Quit | Detail _ -> Some Close_metric)
          | Escape -> (
              match m.mode with Dashboard -> Some Quit | Detail _ -> Some Close_metric)
          | Left -> (
              match m.mode with Dashboard -> Some (Metrics_msg Metrics.Prev_batch) | Detail _ -> None)
          | Right -> (
              match m.mode with Dashboard -> Some (Metrics_msg Metrics.Next_batch) | Detail _ -> None)
          | Char c when m.mode = Dashboard ->
              let one = Uchar.of_char '1' and nine = Uchar.of_char '9' in
              let idx =
                if Uchar.compare c one >= 0 && Uchar.compare c nine <= 0 then
                  Uchar.to_int c - Uchar.to_int one
                else
                  -1
              in
              if idx >= 0 then
                let visible = visible_chart_tags m in
                if idx < List.length visible then
                  Some (Open_metric (List.nth visible idx))
                else None
              else None
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
