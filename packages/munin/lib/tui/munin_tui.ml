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
  was_live : bool;
  run_completed : bool;
}

and mode = Dashboard | Detail of string | Info

type msg =
  | Tick of float
  | Quit
  | Metrics_msg of Metrics.msg
  | Open_metric of string
  | Open_selected
  | Close_metric
  | Open_info
  | Close_info
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

let best_value monitor tag =
  Option.map
    (fun (b : Munin.Run.metric) -> b.value)
    (Munin.Run_monitor.best monitor tag)

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

let blocks =
  [|
    "\u{2581}";
    "\u{2582}";
    "\u{2583}";
    "\u{2584}";
    "\u{2585}";
    "\u{2586}";
    "\u{2587}";
    "\u{2588}";
  |]

let mini_sparkline history ~width =
  let values = List.map snd history in
  let n = List.length values in
  if n = 0 then ""
  else
    let arr =
      if n <= width then Array.of_list values
      else
        let a = Array.of_list values in
        Array.sub a (n - width) width
    in
    let len = Array.length arr in
    let lo = Array.fold_left min infinity arr in
    let hi = Array.fold_left max neg_infinity arr in
    let range = hi -. lo in
    let buf = Buffer.create (len * 3) in
    Array.iter
      (fun v ->
        let idx =
          if range = 0. then 3
          else int_of_float ((v -. lo) /. range *. 7.) |> max 0 |> min 7
        in
        Buffer.add_string buf blocks.(idx))
      arr;
    Buffer.contents buf

let spark_style = Ansi.Style.make ~fg:(Ansi.Color.grayscale ~level:10) ()
let bold_white = Ansi.Style.make ~bold:true ~fg:Ansi.Color.white ()

let view_summary_banner m =
  let elapsed = elapsed_secs m in
  let h = int_of_float elapsed / 3600 in
  let mi = int_of_float elapsed mod 3600 / 60 in
  let s = int_of_float elapsed mod 60 in
  let duration = Printf.sprintf "%02d:%02d:%02d" h mi s in
  let status_color = Theme.status_color m.run_status in
  let status_label = Theme.status_label m.run_status in
  let metric_tags = user_metric_tags m.monitor in
  let capped =
    if List.length metric_tags > 8 then
      List.filteri (fun i _ -> i < 8) metric_tags
    else metric_tags
  in
  let metric_entries =
    List.map
      (fun tag ->
        let history = Munin.Run_monitor.history m.monitor tag in
        let spark = mini_sparkline history ~width:8 in
        let value =
          match best_value m.monitor tag with
          | Some v -> Printf.sprintf "%.4g" v
          | None -> (
              match Theme.last_value history with
              | Some v -> Printf.sprintf "%.4g" v
              | None -> "-")
        in
        box ~flex_direction:Row ~gap:(gap 1)
          ~size:{ width = pct 50; height = auto }
          [
            text ~style:Theme.muted_style tag;
            text ~style:spark_style spark;
            text ~style:bold_white value;
          ])
      capped
  in
  let rec pairs = function
    | [] -> []
    | [ x ] -> [ [ x ] ]
    | x :: y :: rest -> [ x; y ] :: pairs rest
  in
  let metric_rows =
    List.map
      (fun row ->
        box ~flex_direction:Row ~size:{ width = pct 100; height = auto } row)
      (pairs metric_entries)
  in
  box ~border:true
    ~border_color:(Ansi.Color.grayscale ~level:8)
    ~title:(Printf.sprintf " Run %s " status_label)
    ~padding:(padding_xy 2 0)
    ~size:{ width = pct 100; height = auto }
    ([
       box ~flex_direction:Row ~gap:(gap 1)
         ~size:{ width = pct 100; height = auto }
         [
           text
             ~style:(Ansi.Style.make ~fg:status_color ())
             (Printf.sprintf "%s in %s" status_label duration);
         ];
     ]
    @ metric_rows)

let view_dashboard m =
  let all_metrics = Munin.Run_monitor.metrics m.monitor in
  let metric_tags = user_metric_tags m.monitor in
  let history_for_tag tag = Munin.Run_monitor.history m.monitor tag in
  let best_for_tag tag = best_value m.monitor tag in
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
              ~step_metrics:(step_metrics m.monitor)
              ~metric_defs:(Munin.Run_monitor.metric_defs m.monitor)
              ~best_for_tag:(best_value m.monitor);
          ];
      ]
    else []
  in
  let banner = if m.run_completed then [ view_summary_banner m ] else [] in
  box ~flex_direction:Column
    ~size:{ width = pct 100; height = pct 100 }
    ([
       Header.view ~run_id:(Munin.Run.id m.run) ~run_name:(Munin.Run.name m.run)
         ~tags:(Munin.Run.tags m.run) ~latest_epoch:(latest_epoch m.monitor)
         ~total_epochs:(total_epochs m.run) ~latest_step:(latest_step m.monitor)
         ~elapsed_secs:(elapsed_secs m) ~status:m.run_status;
     ]
    @ banner
    @ [
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
      ])

let view_detail m tag =
  let smooth_param = Theme.smooth_alpha m.smooth in
  let history_for_tag t = Munin.Run_monitor.history m.monitor t in
  let metric_def =
    List.assoc_opt tag (Munin.Run_monitor.metric_defs m.monitor)
  in
  box ~flex_direction:Column
    ~size:{ width = pct 100; height = pct 100 }
    [
      box ~flex_grow:1.0 ~justify_content:Center ~align_items:Center
        ~size:{ width = pct 100; height = pct 100 }
        [
          Detail.view ~tag ~history_for_tag ~best:(best_value m.monitor tag)
            ~smooth:smooth_param ~metric_def
            ~size:{ width = pct 80; height = pct 80 };
        ];
      Footer.view ~mode:(`Detail m.smooth);
    ]

let view_info m =
  let all_metrics = Munin.Run_monitor.metrics m.monitor in
  box ~flex_direction:Column
    ~size:{ width = pct 100; height = pct 100 }
    [
      Header.view ~run_id:(Munin.Run.id m.run) ~run_name:(Munin.Run.name m.run)
        ~tags:(Munin.Run.tags m.run) ~latest_epoch:(latest_epoch m.monitor)
        ~total_epochs:(total_epochs m.run) ~latest_step:(latest_step m.monitor)
        ~elapsed_secs:(elapsed_secs m) ~status:m.run_status;
      box ~flex_grow:1.0 ~flex_shrink:1.0 ~overflow:{ x = Hidden; y = Hidden }
        ~size:{ width = pct 100; height = auto }
        [
          Info.view ~run:m.run ~status:m.run_status
            ~elapsed_secs:(elapsed_secs m)
            ~metric_defs:(Munin.Run_monitor.metric_defs m.monitor)
            ~latest_metrics:all_metrics ~step_metrics:(step_metrics m.monitor)
            ~best_for_tag:(best_value m.monitor);
        ];
      Footer.view ~mode:`Info;
    ]

let view m =
  match m.mode with
  | Dashboard -> view_dashboard m
  | Detail tag -> view_detail m tag
  | Info -> view_info m

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
      was_live = run_status = Theme.Live;
      run_completed = false;
    },
    Cmd.none )

let update msg m =
  match msg with
  | Tick _dt ->
      Munin.Run_monitor.poll m.monitor;
      let run_status =
        run_status_of_live_status (Munin.Run_monitor.live_status m.monitor)
      in
      let run_completed =
        m.run_completed
        || m.was_live && run_status <> Theme.Live
           && run_status <> Theme.Stopped
      in
      ({ m with run_status; run_completed }, Cmd.none)
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
  | Open_info -> ({ m with mode = Info }, Cmd.none)
  | Close_info -> ({ m with mode = Dashboard }, Cmd.none)
  | Toggle_smooth -> (
      match m.mode with
      | Detail _ -> ({ m with smooth = Theme.next_smooth m.smooth }, Cmd.none)
      | Dashboard | Info -> (m, Cmd.none))
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
          | Char c, Info when is c 'q' -> Some Close_info
          | Char c, Dashboard when is c 'i' -> Some Open_info
          | Escape, Dashboard -> Some Quit
          | Escape, Detail _ -> Some Close_metric
          | Escape, Info -> Some Close_info
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
