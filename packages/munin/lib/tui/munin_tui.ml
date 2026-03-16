(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Mosaic

(* Model *)

type model = {
  run : Munin.Run.t;
  monitor : Munin.Run_monitor.t;
  run_status : Theme.run_status;
  metrics_state : Metrics.state;
  mode : mode;
  smooth : Theme.smooth;
  show_system : bool;
  screen_width : int;
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
  | Toggle_system
  | Terminal_resize of int * int

(* Helpers *)

let run_status_of_live_status :
    Munin.Run_monitor.live_status -> Theme.run_status = function
  | `Live -> Theme.Live
  | `Stopped -> Theme.Stopped
  | `Done `finished -> Theme.Done
  | `Done `failed -> Theme.Failed
  | `Done `killed -> Theme.Killed
  | `Done `running -> Theme.Live

let latest_step monitor =
  let ms = Munin.Run_monitor.metrics monitor in
  match ms with
  | [] -> None
  | _ ->
      Some
        (List.fold_left
           (fun acc (_, (m : Munin.Run.metric)) -> max acc m.step)
           0 ms)

let latest_epoch monitor =
  let ms = Munin.Run_monitor.metrics monitor in
  match List.assoc_opt "epoch" ms with
  | Some (m : Munin.Run.metric) -> Some (int_of_float m.value)
  | None -> None

let total_epochs run =
  match Munin.Run.find_param run "epochs" with
  | Some v -> Munin.Value.to_int v
  | None -> None

let elapsed_secs m =
  let end_time =
    match m.run_status with
    | Theme.Live -> Unix.gettimeofday ()
    | Stopped | Done | Failed | Killed -> (
        match Munin.Run.ended_at m.run with
        | Some t -> t
        | None -> Unix.gettimeofday ())
  in
  end_time -. Munin.Run.started_at m.run

let metrics_width m =
  if m.show_system then int_of_float (float_of_int m.screen_width *. 0.66)
  else m.screen_width

(* View *)

let divider () =
  box
    ~size:{ width = px 1; height = pct 100 }
    ~background:(Ansi.Color.grayscale ~level:8)
    [ text " " ]

let is_sys_metric tag = String.length tag > 4 && String.sub tag 0 4 = "sys/"

let step_metrics monitor =
  let defs = Munin.Run_monitor.metric_defs monitor in
  let from_defs =
    List.filter_map (fun (_, (d : Munin.Run.metric_def)) -> d.step_metric) defs
  in
  if List.mem "epoch" from_defs then from_defs else "epoch" :: from_defs

let user_metric_tags monitor =
  let sms = step_metrics monitor in
  List.filter_map
    (fun (tag, _) ->
      if is_sys_metric tag || List.mem tag sms then None else Some tag)
    (Munin.Run_monitor.metrics monitor)

let view_dashboard m =
  let all_metrics = Munin.Run_monitor.metrics m.monitor in
  let metric_tags = user_metric_tags m.monitor in
  let history_for_tag tag = Munin.Run_monitor.history m.monitor tag in
  let best_for_tag tag =
    Option.map
      (fun (b : Munin.Run.metric) -> b.value)
      (Munin.Run_monitor.best m.monitor tag)
  in
  let goal_for_tag tag =
    match List.assoc_opt tag (Munin.Run_monitor.metric_defs m.monitor) with
    | Some (d : Munin.Run.metric_def) -> d.goal
    | None -> None
  in
  let metrics_pct = if m.show_system then 66 else 100 in
  let sys_latest tag =
    match List.assoc_opt tag all_metrics with
    | Some (m : Munin.Run.metric) -> m.value
    | None -> 0.0
  in
  let sys_values =
    System.
      {
        cpu_user = sys_latest "sys/cpu_user";
        cpu_system = sys_latest "sys/cpu_system";
        mem_pct = sys_latest "sys/mem_used_pct";
        mem_gb = sys_latest "sys/mem_used_gb";
        proc_cpu = sys_latest "sys/proc_cpu_pct";
        proc_mem_mb = sys_latest "sys/proc_mem_mb";
        disk_read_mbs = sys_latest "sys/disk_read_mbs";
        disk_write_mbs = sys_latest "sys/disk_write_mbs";
        disk_util_pct = sys_latest "sys/disk_util_pct";
      }
  in
  let right_panel =
    if m.show_system then
      [
        divider ();
        box ~flex_direction:Column ~padding:(padding_lrtb 1 1 1 0) ~gap:(gap 1)
          ~size:{ width = pct 34; height = auto }
          [
            System.view sys_values ~history_for_tag;
            Overview.view ~run:m.run ~latest_metrics:all_metrics
              ~step_metrics:(step_metrics m.monitor);
          ];
      ]
    else []
  in
  box ~flex_direction:Column
    ~size:{ width = pct 100; height = pct 100 }
    [
      Header.view ~run_id:(Munin.Run.id m.run) ~run_name:(Munin.Run.name m.run)
        ~tags:(Munin.Run.tags m.run) ~latest_epoch:(latest_epoch m.monitor)
        ~total_epochs:(total_epochs m.run) ~latest_step:(latest_step m.monitor)
        ~elapsed_secs:(elapsed_secs m) ~status:m.run_status;
      box ~flex_direction:Row ~flex_grow:1.0 ~flex_shrink:1.0
        ~overflow:{ x = Hidden; y = Hidden }
        ~size:{ width = pct 100; height = auto }
        ([
           box
             ~size:{ width = pct metrics_pct; height = auto }
             [
               Metrics.view m.metrics_state ~metric_tags ~history_for_tag
                 ~best_for_tag ~goal_for_tag;
             ];
         ]
        @ right_panel);
      Footer.view ~mode:`Dashboard;
    ]

let view_detail m tag =
  let smooth_param = Theme.smooth_alpha m.smooth in
  let history_for_tag t = Munin.Run_monitor.history m.monitor t in
  box ~flex_direction:Column
    ~size:{ width = pct 100; height = pct 100 }
    [
      box ~flex_grow:1.0 ~justify_content:Center ~align_items:Center
        ~size:{ width = pct 100; height = pct 100 }
        [
          Detail.view ~tag ~history_for_tag
            ~best:
              (Option.map
                 (fun (b : Munin.Run.metric) -> b.value)
                 (Munin.Run_monitor.best m.monitor tag))
            ~smooth:smooth_param
            ~size:{ width = pct 80; height = pct 80 };
        ];
      Footer.view ~mode:(`Detail m.smooth);
    ]

let view m =
  match m.mode with
  | Dashboard -> view_dashboard m
  | Detail tag -> view_detail m tag

(* TEA core *)

let init ~run =
  let monitor = Munin.Run_monitor.start run in
  Munin.Run_monitor.poll monitor;
  let run_status =
    run_status_of_live_status (Munin.Run_monitor.live_status monitor)
  in
  let metrics_state = Metrics.initial_state () in
  ( {
      run;
      monitor;
      run_status;
      metrics_state;
      mode = Dashboard;
      smooth = Theme.Off;
      show_system = true;
      screen_width = 80;
    },
    Cmd.none )

let update msg m =
  match msg with
  | Tick _dt ->
      Munin.Run_monitor.poll m.monitor;
      let run_status =
        run_status_of_live_status (Munin.Run_monitor.live_status m.monitor)
      in
      ({ m with run_status }, Cmd.none)
  | Terminal_resize (width, height) ->
      let m = { m with screen_width = width } in
      let mw = metrics_width m in
      let metrics_state' =
        Metrics.update
          (Metrics.Resize (mw, height))
          m.metrics_state
          ~total_metrics:(List.length (user_metric_tags m.monitor))
      in
      ({ m with metrics_state = metrics_state' }, Cmd.none)
  | Metrics_msg metrics_msg ->
      let total_metrics = List.length (user_metric_tags m.monitor) in
      let metrics_state' =
        Metrics.update metrics_msg m.metrics_state ~total_metrics
      in
      ({ m with metrics_state = metrics_state' }, Cmd.none)
  | Open_metric tag -> ({ m with mode = Detail tag }, Cmd.none)
  | Open_selected -> (
      let all_tags = user_metric_tags m.monitor in
      let total_metrics = List.length all_tags in
      match Metrics.selected_tag m.metrics_state ~total_metrics ~all_tags with
      | Some tag -> ({ m with mode = Detail tag }, Cmd.none)
      | None -> (m, Cmd.none))
  | Close_metric -> ({ m with mode = Dashboard }, Cmd.none)
  | Toggle_smooth -> (
      match m.mode with
      | Dashboard -> (m, Cmd.none)
      | Detail _ -> ({ m with smooth = Theme.next_smooth m.smooth }, Cmd.none))
  | Toggle_system ->
      let m = { m with show_system = not m.show_system } in
      let mw = metrics_width m in
      let total_metrics = List.length (user_metric_tags m.monitor) in
      let metrics_state' =
        Metrics.update
          (Metrics.Resize (mw, m.metrics_state.screen_height))
          m.metrics_state ~total_metrics
      in
      ({ m with metrics_state = metrics_state' }, Cmd.none)
  | Quit ->
      Munin.Run_monitor.close m.monitor;
      (m, Cmd.quit)

let subscriptions m =
  Sub.batch
    [
      Sub.on_tick (fun ~dt -> Tick dt);
      Sub.on_resize (fun ~width ~height -> Terminal_resize (width, height));
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
          | Char c, Dashboard when is c '[' -> Some Toggle_system
          | Char c, Dashboard when is c ']' -> Some Toggle_system
          | Char c, Dashboard when Uchar.equal c (Uchar.of_char '<') ->
              Some (Metrics_msg Metrics.Prev_batch)
          | Char c, Dashboard when Uchar.equal c (Uchar.of_char '>') ->
              Some (Metrics_msg Metrics.Next_batch)
          | Enter, Dashboard -> Some Open_selected
          | Char c, Dashboard when is c ' ' -> Some Open_selected
          | _ -> None);
    ]

let run ?root ?experiment ?runs () =
  let store = Munin.Store.open_ ?root () in
  let run =
    match runs with
    | Some [ run_id ] -> Munin.Store.find_run store run_id
    | None | Some [] -> Munin.Store.latest_run store ?experiment ()
    | Some _ ->
        Printf.printf "munin: please specify a single run\n%!";
        None
  in
  match run with
  | Some run ->
      let init () = init ~run in
      Mosaic.run { init; update; view; subscriptions }
  | None -> (
      match runs with
      | Some [ run_id ] -> Printf.printf "munin: run not found: %s\n%!" run_id
      | _ ->
          Printf.printf "munin: no runs found in %s\n%!"
            (Munin.Store.root store))
