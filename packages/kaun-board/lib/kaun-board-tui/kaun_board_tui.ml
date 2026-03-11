(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Mosaic
open Kaun_board

(* Model *)

type model = {
  run_id : string;
  total_epochs : int option;
  created_at : float;
  run_status : Header.run_status;
  store : Store.t;
  stream : Run.event_stream;
  metrics_state : Metrics.state;
  sys_panel : Sys_panel.t;
  mode : mode;
  chart_smooth : int; (* 0 = off, 1..3 = smoothness level *)
}

and mode = Dashboard | Detail of string

type msg =
  | Tick of float
  | Quit
  | Metrics_msg of Metrics.msg
  | Open_metric of string
  | Open_selected
  | Close_metric
  | Toggle_smooth

(* Helpers *)

let stopped_threshold_sec = 5.0

let compute_run_status ~stream ~total_epochs ~latest_epoch : Header.run_status =
  match (total_epochs, latest_epoch) with
  | Some total, Some e when e >= total -> Done
  | _ ->
      let age = Unix.gettimeofday () -. Run.last_mtime stream in
      if age > stopped_threshold_sec then Stopped else Live

let latest_step store =
  let metrics = Store.latest_metrics store in
  match metrics with
  | [] -> None
  | _ ->
      Some
        (List.fold_left
           (fun acc (_, (m : Store.metric)) -> max acc m.step)
           0 metrics)

let elapsed_secs m =
  let end_time =
    match m.run_status with
    | Header.Live -> Unix.gettimeofday ()
    | Stopped | Done -> Run.last_mtime m.stream
  in
  end_time -. m.created_at

let visible_chart_tags (m : model) : string list =
  let latest = Store.latest_metrics m.store in
  let all_tags = List.map fst latest in
  let total_metrics = List.length all_tags in
  Metrics.visible_chart_tags m.metrics_state ~total_metrics ~all_tags

(* View *)

let divider () =
  box
    ~size:{ width = px 1; height = pct 100 }
    ~background:(Ansi.Color.grayscale ~level:8)
    [ text " " ]

let view_dashboard m =
  box ~flex_direction:Column
    ~size:{ width = pct 100; height = pct 100 }
    [
      Header.view ~run_id:m.run_id
        ~latest_epoch:(Store.latest_epoch m.store)
        ~total_epochs:m.total_epochs ~latest_step:(latest_step m.store)
        ~elapsed_secs:(elapsed_secs m) ~status:m.run_status;
      box ~flex_direction:Row ~flex_grow:1.0
        ~size:{ width = pct 100; height = pct 100 }
        [
          Metrics.view
            {
              metric_tags = List.map fst (Store.latest_metrics m.store);
              history_for_tag = Store.history_for_tag m.store;
              best_for_tag = Store.best_for_tag m.store;
              screen_width = m.metrics_state.screen_width;
              screen_height = m.metrics_state.screen_height;
              current_batch = m.metrics_state.current_batch;
              selected = m.metrics_state.selected;
            };
          divider ();
          box
            ~size:{ width = pct 34; height = pct 100 }
            [ Sys_panel.view m.sys_panel ];
        ];
      Footer.view ~mode:Dashboard;
    ]

let smooth_alpha = function
  | 1 -> 0.5
  | 2 -> 0.3
  | 3 -> 0.15
  | _ -> assert false

let view_detail m tag =
  let smooth_param =
    if m.chart_smooth = 0 then None else Some (smooth_alpha m.chart_smooth)
  in
  box ~flex_direction:Column
    ~size:{ width = pct 100; height = pct 100 }
    ~background:(Ansi.Color.of_rgb 20 20 30)
    [
      box ~flex_grow:1.0 ~justify_content:Center ~align_items:Center
        ~size:{ width = pct 100; height = pct 100 }
        [
          Chart_view.view ~tag
            ~history_for_tag:(Store.history_for_tag m.store)
            ~best:
              (Option.map
                 (fun (b : Store.best_value) -> b.value)
                 (Store.best_for_tag m.store tag))
            ~smooth:smooth_param
            ~size:{ width = pct 80; height = pct 80 };
        ];
      Footer.view ~mode:(Detail { smooth = m.chart_smooth });
    ]

let view m =
  match m.mode with
  | Dashboard -> view_dashboard m
  | Detail tag -> view_detail m tag

(* TEA core *)

let init ~run =
  let run_id = Run.run_id run in
  let total_epochs = Run.total_epochs run in
  let created_at = Run.created_at run in
  let stream = Run.open_events run in
  let store = Store.create () in
  let initial_events = Run.read_events stream in
  Store.update store initial_events;
  let latest_epoch = Store.latest_epoch store in
  let run_status = compute_run_status ~stream ~total_epochs ~latest_epoch in
  let metrics_state = Metrics.initial_state () in
  let sys_panel = Sys_panel.create () in
  ( {
      run_id;
      total_epochs;
      created_at;
      run_status;
      store;
      stream;
      metrics_state;
      sys_panel;
      mode = Dashboard;
      chart_smooth = 0;
    },
    Cmd.none )

let update msg m =
  match msg with
  | Tick dt ->
      let new_events = Run.read_events m.stream in
      Store.update m.store new_events;
      let sys_panel = Sys_panel.update m.sys_panel ~dt in
      let run_status =
        compute_run_status ~stream:m.stream ~total_epochs:m.total_epochs
          ~latest_epoch:(Store.latest_epoch m.store)
      in
      ({ m with sys_panel; run_status }, Cmd.none)
  | Metrics_msg metrics_msg ->
      let total_metrics = List.length (Store.latest_metrics m.store) in
      let metrics_state' =
        Metrics.update metrics_msg m.metrics_state ~total_metrics
      in
      ({ m with metrics_state = metrics_state' }, Cmd.none)
  | Open_metric tag -> ({ m with mode = Detail tag }, Cmd.none)
  | Open_selected ->
      let visible = visible_chart_tags m in
      let idx = m.metrics_state.selected in
      if idx < List.length visible then
        ({ m with mode = Detail (List.nth visible idx) }, Cmd.none)
      else (m, Cmd.none)
  | Close_metric -> ({ m with mode = Dashboard }, Cmd.none)
  | Toggle_smooth -> (
      match m.mode with
      | Dashboard -> (m, Cmd.none)
      | Detail _ ->
          ({ m with chart_smooth = (m.chart_smooth + 1) mod 4 }, Cmd.none))
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
          let is c ch =
            let lo = Uchar.of_char ch in
            let hi = Uchar.of_char (Char.uppercase_ascii ch) in
            Uchar.equal c lo || Uchar.equal c hi
          in
          let data = Mosaic_ui.Event.Key.data ev in
          match (data.key, m.mode) with
          | Char c, Detail _ when is c 's' -> Some Toggle_smooth
          | Char c, Dashboard when is c 'q' -> Some Quit
          | Char c, Detail _ when is c 'q' -> Some Close_metric
          | Escape, Dashboard -> Some Quit
          | Escape, Detail _ -> Some Close_metric
          | Left, Dashboard -> Some (Metrics_msg Metrics.Select_left)
          | Right, Dashboard -> Some (Metrics_msg Metrics.Select_right)
          | Up, Dashboard -> Some (Metrics_msg Metrics.Select_up)
          | Down, Dashboard -> Some (Metrics_msg Metrics.Select_down)
          | Char c, Dashboard when is c '[' ->
              Some (Metrics_msg Metrics.Prev_batch)
          | Char c, Dashboard when is c ']' ->
              Some (Metrics_msg Metrics.Next_batch)
          | Enter, Dashboard -> Some Open_selected
          | Char c, Dashboard when Uchar.equal c (Uchar.of_char ' ') ->
              Some Open_selected
          | _ -> None);
    ]

let run ?base_dir ?runs () =
  let base_dir = Option.value base_dir ~default:(Kaun_board.Env.base_dir ()) in
  let run =
    match runs with
    | Some [ run_id ] -> Run.load (Filename.concat base_dir run_id)
    | None | Some [] -> Run.latest base_dir
    | Some _ ->
        Printf.printf "kaun-board: please specify a single run\n%!";
        None
  in
  match run with
  | Some run ->
      let init () = init ~run in
      Mosaic.run { init; update; view; subscriptions }
  | None -> (
      match runs with
      | Some [ run_id ] ->
          Printf.printf "kaun-board: run not found: %s\n%!" run_id
      | _ -> Printf.printf "kaun-board: no runs found in %s\n%!" base_dir)
