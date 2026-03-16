(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Mosaic

(* Styles *)

let label_style = Ansi.Style.make ~fg:(Ansi.Color.grayscale ~level:16) ()
let value_style = Ansi.Style.make ~bold:true ~fg:Ansi.Color.white ()

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

let tag_style =
  Ansi.Style.make
    ~fg:(Ansi.Color.grayscale ~level:18)
    ~bg:(Ansi.Color.grayscale ~level:5)
    ()

let view ~run_id ~run_name ~tags ~latest_epoch ~total_epochs ~latest_step
    ~elapsed_secs ~status =
  let color = Theme.status_color status in
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
  let sep () = text ~style:Theme.muted_style " \u{00B7} " in
  let stats =
    [ epoch_items; step_items ]
    |> List.filter (fun l -> l <> [])
    |> List.mapi (fun i items -> if i > 0 then sep () :: items else items)
    |> List.flatten
  in
  let stats =
    if stats <> [] then
      stats
      @ [ sep (); text ~style:Theme.muted_style (format_elapsed elapsed_secs) ]
    else [ text ~style:Theme.muted_style (format_elapsed elapsed_secs) ]
  in
  box ~padding:(padding_xy 2 0) ~flex_direction:Row ~gap:(gap 2)
    ~align_items:Center ~background:Theme.header_bg
    ~size:{ width = pct 100; height = auto }
    ([
       text ~style:value_style "Munin"; text ~style:Theme.muted_style "\u{2502}";
     ]
    @ (match run_name with
      | Some name ->
          [
            text ~style:value_style name;
            text ~style:Theme.muted_style (Printf.sprintf " (%s)" run_id);
          ]
      | None -> [ text ~style:Theme.muted_style run_id ])
    @ List.map (fun t -> text ~style:tag_style (Printf.sprintf " %s " t)) tags
    @ stats
    @ [
        box ~flex_grow:1.0 ~size:{ width = auto; height = auto } [];
        text ~style:(Ansi.Style.make ~fg:color ()) "\u{25CF}";
        text
          ~style:(Ansi.Style.make ~bold:true ~fg:color ())
          (Theme.status_label status);
      ])
