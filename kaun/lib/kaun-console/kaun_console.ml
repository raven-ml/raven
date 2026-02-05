(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Mosaic
open Kaun_runlog

(* ───── Model ───── *)

type model = {
  run_id : string;
  store : Metric_store.t;
  stream : Run.event_stream;
}

type msg = Tick of float | Quit

(* ───── Constants ───── *)

let header_bg = Ansi.Color.of_rgb 30 80 100
let hint_style = Ansi.Style.make ~fg:(Ansi.Color.grayscale ~level:14) ()
let step_color = Ansi.Color.cyan
let epoch_color = Ansi.Color.cyan
let metric_value_style = Ansi.Style.make ~fg:(Ansi.Color.grayscale ~level:15) ()

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
          box ~padding:(padding 1) ~background:Ansi.Color.green
            [
              text
                ~style:(Ansi.Style.make ~bold:true ~fg:Ansi.Color.white ())
                "LIVE";
            ];
        ];
    ]

let view_metrics store =
  let latest = Metric_store.latest_metrics store in
  if latest = [] then
    box ~padding:(padding 1)
      [ text ~style:hint_style "  Waiting for metrics..." ]
  else
    box ~flex_direction:Column ~padding:(padding 1) ~gap:(gap 1)
      [
        text ~style:(Ansi.Style.make ~bold:true ()) "Metrics:";
        box ~flex_direction:Column ~gap:(gap 1)
          (List.map
             (fun (tag, (m : Metric_store.metric)) ->
               let epoch_str =
                 match m.epoch with
                 | None -> ""
                 | Some e -> Printf.sprintf ", epoch %d" e
               in
               box ~flex_direction:Row ~gap:(gap 2)
                 [
                   text ~style:hint_style (Printf.sprintf "  %-30s" tag);
                   text ~style:metric_value_style
                     (Printf.sprintf "%8.4f" m.value);
                   text ~style:hint_style
                     (Printf.sprintf "(step %d%s)" m.step epoch_str);
                 ])
             latest);
      ]

let view_footer () =
  box ~padding:(padding 1) [ text ~style:hint_style "(Press Ctrl-C to quit)" ]

let view m =
  box ~flex_direction:Column
    ~size:{ width = pct 100; height = pct 100 }
    [
      view_header ~run_id:m.run_id m.store;
      scroll_box ~scroll_y:true ~scroll_x:false ~flex_grow:1.
        ~size:{ width = pct 100; height = pct 100 }
        [ view_metrics m.store ];
      view_footer ();
    ]

(* ───── TEA Core ───── *)

let init ~run =
  let run_id = Run.run_id run in
  let stream = Run.open_events run in
  let store = Metric_store.create () in
  (* Load initial events *)
  let initial_events = Run.read_events stream in
  Metric_store.update store initial_events;
  ({ run_id; store; stream }, Cmd.none)

let update msg m =
  match msg with
  | Tick _ ->
      let new_events = Run.read_events m.stream in
      Metric_store.update m.store new_events;
      ({ m with store = m.store }, Cmd.none)
  | Quit ->
      Run.close_events m.stream;
      (m, Cmd.quit)

let subscriptions _model =
  Sub.batch
    [
      Sub.on_tick (fun ~dt -> Tick dt);
      Sub.on_key (fun ev ->
          match (Mosaic_ui.Event.Key.data ev).key with
          | Char c when Uchar.equal c (Uchar.of_char 'q') -> Some Quit
          | Char c when Uchar.equal c (Uchar.of_char 'Q') -> Some Quit
          | Escape -> Some Quit
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
