(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Mosaic

(* Types *)

type values = {
  cpu_user : float option;
  cpu_system : float option;
  mem_pct : float option;
  mem_gb : float option;
  proc_cpu : float option;
  proc_mem_mb : float option;
  disk_read_mbs : float option;
  disk_write_mbs : float option;
  disk_util_pct : float option;
}

(* Styles *)

let label_style = Ansi.Style.make ~fg:Ansi.Color.white ()
let bracket_style = Ansi.Style.make ~fg:(Ansi.Color.grayscale ~level:10) ()
let value_style = Ansi.Style.make ~fg:Ansi.Color.white ()
let user_style = Ansi.Style.make ~fg:Ansi.Color.green ()
let system_style = Ansi.Style.make ~fg:Ansi.Color.cyan ()
let spark_style = Ansi.Style.make ~fg:(Ansi.Color.grayscale ~level:14) ()

let bar_color pct =
  if pct > 80. then Ansi.Color.red
  else if pct > 50. then Ansi.Color.yellow
  else Ansi.Color.green

(* Sparkline rendering *)

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

let sparkline history ~width =
  let n = List.length history in
  if n = 0 then ""
  else
    let values =
      if n <= width then Array.of_list history
      else
        let arr = Array.of_list history in
        Array.sub arr (n - width) width
    in
    let len = Array.length values in
    let lo = Array.fold_left min infinity values in
    let hi = Array.fold_left max neg_infinity values in
    let range = hi -. lo in
    let buf = Buffer.create (len * 3) in
    Array.iter
      (fun v ->
        let idx =
          if range = 0. then 3
          else int_of_float ((v -. lo) /. range *. 7.) |> max 0 |> min 7
        in
        Buffer.add_string buf blocks.(idx))
      values;
    Buffer.contents buf

(* Bar drawing *)

let draw_bar c ~y ~width ~label ~value_text ~percent ~spark =
  let label_len = String.length label in
  let value_len = String.length value_text in
  Canvas.draw_text c ~x:0 ~y ~text:label ~style:label_style;
  let spark_width = 12 in
  let bar_end = width - spark_width - 2 in
  let bar_start = label_len in
  if bar_end - bar_start < 2 then ()
  else begin
    Canvas.draw_text c ~x:bar_start ~y ~text:"[" ~style:bracket_style;
    Canvas.draw_text c ~x:bar_end ~y ~text:"]" ~style:bracket_style;
    let inner = bar_end - bar_start - 1 in
    let fill_space = inner - value_len - 1 in
    let fill_count =
      if fill_space > 0 then
        int_of_float (percent /. 100. *. float_of_int fill_space)
        |> max 0 |> min fill_space
      else 0
    in
    let color = bar_color percent in
    let style = Ansi.Style.make ~fg:color () in
    for i = 0 to fill_count - 1 do
      Canvas.draw_text c ~x:(bar_start + 1 + i) ~y ~text:"|" ~style
    done;
    let vx = bar_end - value_len in
    if vx > bar_start + 1 then
      Canvas.draw_text c ~x:vx ~y ~text:value_text ~style:value_style;
    Canvas.draw_text c ~x:(bar_end + 2) ~y ~text:spark ~style:spark_style
  end

let draw_cpu_bar c ~y ~width ~label ~cpu_user ~cpu_system ~spark =
  let label_len = String.length label in
  let total = cpu_user +. cpu_system in
  let value_text = Printf.sprintf "%.0f%%" total in
  let value_len = String.length value_text in
  Canvas.draw_text c ~x:0 ~y ~text:label ~style:label_style;
  let spark_width = 12 in
  let bar_end = width - spark_width - 2 in
  let bar_start = label_len in
  if bar_end - bar_start < 2 then ()
  else begin
    Canvas.draw_text c ~x:bar_start ~y ~text:"[" ~style:bracket_style;
    Canvas.draw_text c ~x:bar_end ~y ~text:"]" ~style:bracket_style;
    let inner = bar_end - bar_start - 1 in
    let fill_space = inner - value_len - 1 in
    let user_count =
      if fill_space > 0 then
        int_of_float (cpu_user /. 100. *. float_of_int fill_space)
        |> max 0 |> min fill_space
      else 0
    in
    let system_count =
      if fill_space > 0 then
        int_of_float (cpu_system /. 100. *. float_of_int fill_space)
        |> max 0
        |> min (fill_space - user_count)
      else 0
    in
    for i = 0 to user_count - 1 do
      Canvas.draw_text c ~x:(bar_start + 1 + i) ~y ~text:"|" ~style:user_style
    done;
    for i = 0 to system_count - 1 do
      Canvas.draw_text c
        ~x:(bar_start + 1 + user_count + i)
        ~y ~text:"|" ~style:system_style
    done;
    let vx = bar_end - value_len in
    if vx > bar_start + 1 then
      Canvas.draw_text c ~x:vx ~y ~text:value_text ~style:value_style;
    Canvas.draw_text c ~x:(bar_end + 2) ~y ~text:spark ~style:spark_style
  end

let draw_value_line c ~y ~width ~label ~value_text ~spark =
  let label_len = String.length label in
  Canvas.draw_text c ~x:0 ~y ~text:label ~style:label_style;
  let spark_width = 12 in
  let vx = width - spark_width - 2 - String.length value_text in
  if vx > label_len then
    Canvas.draw_text c ~x:vx ~y ~text:value_text ~style:value_style;
  Canvas.draw_text c ~x:(width - spark_width) ~y ~text:spark ~style:spark_style

(* View *)

let extract_values history = List.map snd history

let view (v : values) ~(history_for_tag : string -> (int * float) list) =
  let spark tag = sparkline (extract_values (history_for_tag tag)) ~width:10 in
  let cpu_spark =
    let h1 = extract_values (history_for_tag "sys/cpu_user") in
    let h2 = extract_values (history_for_tag "sys/cpu_system") in
    let rec zip_add a b =
      match (a, b) with
      | x :: xs, y :: ys -> (x +. y) :: zip_add xs ys
      | [], _ | _, [] -> []
    in
    sparkline (zip_add h1 h2) ~width:10
  in
  let format_rate mbs =
    if mbs >= 1024. then Printf.sprintf "%.1f GB/s" (mbs /. 1024.)
    else if mbs >= 1. then Printf.sprintf "%.1f MB/s" mbs
    else Printf.sprintf "%.0f KB/s" (mbs *. 1024.)
  in
  (* Each line draws at the [y] it is given: the lines present stack without
     gaps. *)
  let line value draw = Option.map draw value in
  let both a b =
    match (a, b) with Some a, Some b -> Some (a, b) | _ -> None
  in
  let lines =
    List.filter_map Fun.id
      [
        line (both v.cpu_user v.cpu_system) (fun (cpu_user, cpu_system) ->
            fun c ~y ~width ->
             draw_cpu_bar c ~y ~width ~label:"CPU  " ~cpu_user ~cpu_system
               ~spark:cpu_spark);
        line (both v.mem_pct v.mem_gb) (fun (mem_pct, mem_gb) ->
            fun c ~y ~width ->
             draw_bar c ~y ~width ~label:"Mem  "
               ~value_text:(Printf.sprintf "%.0f%%  %.1fG" mem_pct mem_gb)
               ~percent:mem_pct ~spark:(spark "sys/mem_used_pct"));
        line v.proc_cpu (fun proc_cpu c ~y ~width ->
            draw_bar c ~y ~width ~label:"Proc "
              ~value_text:(Printf.sprintf "%.0f%%" proc_cpu)
              ~percent:proc_cpu ~spark:(spark "sys/proc_cpu_pct"));
        line v.proc_mem_mb (fun proc_mem_mb c ~y ~width ->
            draw_value_line c ~y ~width ~label:"RSS  "
              ~value_text:(Printf.sprintf "%.0fM" proc_mem_mb)
              ~spark:(spark "sys/proc_mem_mb"));
        line v.disk_util_pct (fun disk_util_pct c ~y ~width ->
            draw_bar c ~y ~width ~label:"Disk "
              ~value_text:(Printf.sprintf "%.0f%%" disk_util_pct)
              ~percent:disk_util_pct
              ~spark:(spark "sys/disk_util_pct"));
        line v.disk_read_mbs (fun disk_read_mbs c ~y ~width ->
            draw_value_line c ~y ~width ~label:"  R  "
              ~value_text:(format_rate disk_read_mbs)
              ~spark:(spark "sys/disk_read_mbs"));
        line v.disk_write_mbs (fun disk_write_mbs c ~y ~width ->
            draw_value_line c ~y ~width ~label:"  W  "
              ~value_text:(format_rate disk_write_mbs)
              ~spark:(spark "sys/disk_write_mbs"));
      ]
  in
  canvas
    ~size:{ width = pct 100; height = px (List.length lines) }
    (fun c ~dt:_ ->
      Canvas.clear c;
      let width = Canvas.width c in
      List.iteri (fun y draw -> draw c ~y ~width) lines)
