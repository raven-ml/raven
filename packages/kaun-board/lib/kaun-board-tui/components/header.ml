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

let border_color = Ansi.Color.grayscale ~level:6
let dim_style = Ansi.Style.make ~fg:(Ansi.Color.grayscale ~level:14) ()
let label_style = Ansi.Style.make ~fg:(Ansi.Color.grayscale ~level:16) ()
let value_style = Ansi.Style.make ~bold:true ~fg:Ansi.Color.white ()
let muted_style = Ansi.Style.make ~fg:(Ansi.Color.grayscale ~level:14) ()

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

let view ~run_id ~latest_epoch ~total_epochs ~latest_step ~elapsed_secs ~status
    =
  let status_color = badge_color status in
  let epoch_items =
    match (latest_epoch, total_epochs) with
    | None, _ -> []
    | Some e, Some t ->
        [
          text ~style:label_style "Epoch ";
          text ~style:value_style (Printf.sprintf "%d/%d" e t);
        ]
    | Some e, None ->
        [
          text ~style:label_style "Epoch ";
          text ~style:value_style (string_of_int e);
        ]
  in
  let step_items =
    match latest_step with
    | None -> []
    | Some s ->
        [
          text ~style:label_style "Step ";
          text ~style:value_style (format_step s);
        ]
  in
  let sep () = text ~style:dim_style " \u{00B7} " in
  let stats =
    [ epoch_items; step_items ]
    |> List.filter (fun l -> l <> [])
    |> List.mapi (fun i items -> if i > 0 then sep () :: items else items)
    |> List.flatten
  in
  let stats =
    if stats <> [] then
      stats @ [ sep (); text ~style:muted_style (format_elapsed elapsed_secs) ]
    else [ text ~style:muted_style (format_elapsed elapsed_secs) ]
  in
  box ~flex_direction:Column
    ~size:{ width = pct 100; height = auto }
    [
      box ~padding:(padding_xy 2 0) ~flex_direction:Row ~gap:(gap 2)
        ~align_items:Center
        ~size:{ width = pct 100; height = auto }
        ([
           text ~style:value_style "Kaun Board";
           text ~style:dim_style "\u{2502}";
           text ~style:muted_style run_id;
         ]
        @ stats
        @ [
            box ~flex_grow:1.0 ~size:{ width = auto; height = auto } [];
            text ~style:(Ansi.Style.make ~fg:status_color ()) "\u{25CF}";
            text
              ~style:(Ansi.Style.make ~bold:true ~fg:status_color ())
              (status_label status);
          ]);
      box
        ~size:{ width = pct 100; height = px 1 }
        ~background:border_color
        [ text " " ];
    ]
