(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Mosaic

(* Run status *)

type run_status = Live | Stopped | Done

let stopped_threshold_sec = 5.0
let events_path run_dir = Filename.concat run_dir "events.jsonl"

let read_total_epochs run_dir : int option =
  let path = Filename.concat run_dir "run.json" in
  if not (Sys.file_exists path) then None
  else
    try
      let ic = open_in path in
      let s =
        Fun.protect
          ~finally:(fun () -> close_in ic)
          (fun () -> really_input_string ic (in_channel_length ic))
      in
      match Jsont_bytesrw.decode_string Jsont.json s with
      | Error _ -> None
      | Ok json -> (
          let find name = function
            | Jsont.Object (mems, _) -> (
                match Jsont.Json.find_mem name mems with
                | Some (_, v) -> v
                | None -> Jsont.Null ((), Jsont.Meta.none))
            | _ -> Jsont.Null ((), Jsont.Meta.none)
          in
          match find "total_epochs" json with
          | Jsont.Number (f, _) -> Some (int_of_float f)
          | _ -> None)
    with _ -> None

let compute_status ~run_dir ~(latest_epoch : int option) : run_status =
  let now = Unix.gettimeofday () in
  let mtime =
    try (Unix.stat (events_path run_dir)).Unix.st_mtime with _ -> 0.0
  in
  match (read_total_epochs run_dir, latest_epoch) with
  | Some total, Some e when e >= total -> Done
  | _ -> if now -. mtime > stopped_threshold_sec then Stopped else Live

let status_label = function
  | Live -> "LIVE"
  | Stopped -> "Stopped"
  | Done -> "Done"

let badge_color = function
  | Live -> Ansi.Color.green
  | Stopped -> Ansi.Color.grayscale ~level:12
  | Done -> Ansi.Color.of_rgb 80 140 200

(* Styles *)

let header_bg = Ansi.Color.of_rgb 30 80 100
let hint_style = Ansi.Style.make ~fg:(Ansi.Color.grayscale ~level:14) ()
let step_color = Ansi.Color.cyan
let epoch_color = Ansi.Color.cyan

(* View *)

let view ~run_id ~latest_epoch ~run_dir =
  let status = compute_status ~run_dir ~latest_epoch in
  let badge_bg = badge_color status in
  box ~padding:(padding 1) ~background:header_bg
    ~size:{ width = pct 100; height = auto }
    [
      box ~flex_direction:Row ~gap:(gap 2) ~align_items:Center
        ~size:{ width = pct 100; height = auto }
        [
          text ~style:(Ansi.Style.make ~bold:true ()) "\xe2\x96\xb8 Kaun Board";
          text
            ~style:(Ansi.Style.make ~fg:step_color ())
            (Printf.sprintf "Run: %s" run_id);
          (match latest_epoch with
          | None -> text ~style:hint_style "Epoch: -"
          | Some e ->
              text
                ~style:(Ansi.Style.make ~fg:epoch_color ())
                (Printf.sprintf "Epoch: %d" e));
          box ~flex_grow:1.0 ~size:{ width = auto; height = auto } [];
          box ~padding:(padding 1) ~background:badge_bg
            [
              text
                ~style:(Ansi.Style.make ~bold:true ~fg:Ansi.Color.white ())
                (status_label status);
            ];
        ];
    ]
