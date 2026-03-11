(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Mosaic

(* Run status *)

type run_status = Live | Stopped | Done

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

(* Helpers *)

let format_elapsed secs =
  let secs = int_of_float secs in
  let h = secs / 3600 in
  let m = secs mod 3600 / 60 in
  let s = secs mod 60 in
  Printf.sprintf "%02d:%02d:%02d" h m s

let format_step step =
  if step < 1000 then string_of_int step
  else
    let s = string_of_int step in
    let len = String.length s in
    let buf = Buffer.create (len + ((len - 1) / 3)) in
    for i = 0 to len - 1 do
      if i > 0 && (len - i) mod 3 = 0 then Buffer.add_char buf ',';
      Buffer.add_char buf s.[i]
    done;
    Buffer.contents buf

(* View *)

let view ~run_id ~latest_epoch ~total_epochs ~latest_step ~created_at ~status =
  let badge_bg = badge_color status in
  let epoch_text =
    match (latest_epoch, total_epochs) with
    | None, _ -> None
    | Some e, Some t -> Some (Printf.sprintf "Epoch %d/%d" e t)
    | Some e, None -> Some (Printf.sprintf "Epoch %d" e)
  in
  let step_text =
    match latest_step with
    | None -> None
    | Some s -> Some (Printf.sprintf "Step %s" (format_step s))
  in
  let elapsed = Unix.gettimeofday () -. created_at in
  let elapsed_text = format_elapsed elapsed in
  let sep () = text ~style:hint_style "\u{2022}" in
  box ~padding:(padding_xy 2 0) ~background:header_bg
    ~size:{ width = pct 100; height = auto }
    [
      box ~flex_direction:Row ~gap:(gap 2) ~align_items:Center
        ~size:{ width = pct 100; height = auto }
        ([
           text ~style:(Ansi.Style.make ~bold:true ()) "\u{25B8} Kaun Board";
           text
             ~style:(Ansi.Style.make ~fg:step_color ())
             (Printf.sprintf "Run: %s" run_id);
         ]
        @ (match epoch_text with
          | None -> []
          | Some s ->
              [ sep (); text ~style:(Ansi.Style.make ~fg:epoch_color ()) s ])
        @ (match step_text with
          | None -> []
          | Some s ->
              [ sep (); text ~style:(Ansi.Style.make ~fg:step_color ()) s ])
        @ [
            sep ();
            text ~style:hint_style elapsed_text;
            box ~flex_grow:1.0 ~size:{ width = auto; height = auto } [];
            box ~padding:(padding_xy 1 0) ~background:badge_bg
              [
                text
                  ~style:(Ansi.Style.make ~bold:true ~fg:Ansi.Color.white ())
                  (status_label status);
              ];
          ]);
    ]
