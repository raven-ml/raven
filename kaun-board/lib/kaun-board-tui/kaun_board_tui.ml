(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Mosaic
open Kaun_board

(* Model *)

type model = {
  run_id : string;
  run_dir : string;
  store : Store.t;
  stream : Run.event_stream;
  metrics_state : Metrics.state;
  sys_panel : Sys_panel.t;
  mode : mode;
}

and mode = Dashboard | Detail of string

type msg =
  | Tick of float
  | Quit
  | Metrics_msg of Metrics.msg
  | Open_metric of string
  | Close_metric

(* Helpers *)

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
        ~run_dir:m.run_dir;
      box ~flex_direction:Row ~flex_grow:1.0
        ~size:{ width = pct 100; height = pct 100 }
        [
          Metrics.view
            {
              latest_metrics = Store.latest_metrics m.store;
              history_for_tag = Store.history_for_tag m.store;
              screen_width = m.metrics_state.screen_width;
              screen_height = m.metrics_state.screen_height;
              current_batch = m.metrics_state.current_batch;
            };
          divider ();
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
      box ~padding:(padding 1)
        ~size:{ width = pct 100; height = auto }
        [
          text
            ~style:
              (Ansi.Style.make ~bold:true
                 ~fg:(Ansi.Color.grayscale ~level:14)
                 ())
            "Chart View  \xe2\x80\xa2  [Esc/q] back";
        ];
      box ~flex_grow:1.0 ~justify_content:Center ~align_items:Center
        ~size:{ width = pct 100; height = pct 100 }
        [
          Chart_view.view ~tag
            ~history_for_tag:(Store.history_for_tag m.store)
            ~best:
              (Option.map
                 (fun (b : Store.best_value) -> b.value)
                 (Store.best_for_tag m.store tag))
            ~size:{ width = pct 80; height = pct 80 };
        ];
    ]

let view m =
  match m.mode with
  | Dashboard -> view_dashboard m
  | Detail tag -> view_detail m tag

(* TEA core *)

let init ~run =
  let run_id = Run.run_id run in
  let run_dir = Run.dir run in
  let stream = Run.open_events run in
  let store = Store.create () in
  let initial_events = Run.read_events stream in
  Store.update store initial_events;
  let metrics_state = Metrics.initial_state () in
  let sys_panel = Sys_panel.create () in
  ( {
      run_id;
      run_dir;
      store;
      stream;
      metrics_state;
      sys_panel;
      mode = Dashboard;
    },
    Cmd.none )

let update msg m =
  match msg with
  | Tick dt ->
      let new_events = Run.read_events m.stream in
      Store.update m.store new_events;
      let sys_panel = Sys_panel.update m.sys_panel ~dt in
      ({ m with sys_panel }, Cmd.none)
  | Metrics_msg metrics_msg ->
      let total_metrics = List.length (Store.latest_metrics m.store) in
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
              match m.mode with
              | Dashboard -> Some Quit
              | Detail _ -> Some Close_metric)
          | Char c when Uchar.equal c (Uchar.of_char 'Q') -> (
              match m.mode with
              | Dashboard -> Some Quit
              | Detail _ -> Some Close_metric)
          | Escape -> (
              match m.mode with
              | Dashboard -> Some Quit
              | Detail _ -> Some Close_metric)
          | Left -> (
              match m.mode with
              | Dashboard -> Some (Metrics_msg Metrics.Prev_batch)
              | Detail _ -> None)
          | Right -> (
              match m.mode with
              | Dashboard -> Some (Metrics_msg Metrics.Next_batch)
              | Detail _ -> None)
          | Char c when m.mode = Dashboard ->
              let one = Uchar.of_char '1' and nine = Uchar.of_char '9' in
              let idx =
                if Uchar.compare c one >= 0 && Uchar.compare c nine <= 0 then
                  Uchar.to_int c - Uchar.to_int one
                else -1
              in
              if idx >= 0 then
                let visible = visible_chart_tags m in
                if idx < List.length visible then
                  Some (Open_metric (List.nth visible idx))
                else None
              else None
          | _ -> None);
    ]

let latest_run base_dir =
  if not (Sys.file_exists base_dir) then None
  else
    let entries = Sys.readdir base_dir in
    Array.sort (fun a b -> String.compare b a) entries;
    let rec find i =
      if i >= Array.length entries then None
      else
        let dir = Filename.concat base_dir entries.(i) in
        match Run.load dir with Some run -> Some run | None -> find (i + 1)
    in
    find 0

let run ?base_dir ?runs () =
  let base_dir = Option.value base_dir ~default:(Kaun_board.Env.base_dir ()) in
  let run =
    match runs with
    | Some [ run_id ] -> Run.load (Filename.concat base_dir run_id)
    | None | Some [] -> latest_run base_dir
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
