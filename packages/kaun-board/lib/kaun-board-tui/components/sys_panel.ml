(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Mosaic

(* Model *)

type t = {
  cpu : Sysstat.Cpu.stats;
  cpu_per_core : Sysstat.Cpu.stats array;
  memory : Sysstat.Mem.t;
  cpu_prev : Sysstat.Cpu.t;
  cpu_per_core_prev : Sysstat.Cpu.t array;
  num_cores : int;
  sample_acc : float;
}

(* Helpers *)

let bytes_to_gb b = Int64.to_float b /. (1024. *. 1024. *. 1024.)

let mem_used_percent (m : Sysstat.Mem.t) =
  if m.total > 0L then Int64.to_float m.used /. Int64.to_float m.total *. 100.
  else 0.0

(* Init *)

let create () : t =
  let cpu_prev = Sysstat.Cpu.sample () in
  let cpu_per_core_prev = Sysstat.Cpu.sample_per_core () in
  let num_cores = Array.length cpu_per_core_prev in
  Unix.sleepf 0.05;
  let cpu_next = Sysstat.Cpu.sample () in
  let cpu_per_core_next = Sysstat.Cpu.sample_per_core () in
  let cpu = Sysstat.Cpu.compute ~prev:cpu_prev ~next:cpu_next in
  let cpu_per_core =
    Array.map2
      (fun p n -> Sysstat.Cpu.compute ~prev:p ~next:n)
      cpu_per_core_prev cpu_per_core_next
  in
  let cpu_prev = cpu_next in
  let cpu_per_core_prev = cpu_per_core_next in
  let memory = Sysstat.Mem.sample () in
  {
    cpu;
    cpu_per_core;
    memory;
    cpu_prev;
    cpu_per_core_prev;
    num_cores;
    sample_acc = 0.0;
  }

(* Update *)

let update (t : t) ~(dt : float) : t =
  let sample_acc = t.sample_acc +. dt in
  if sample_acc < 0.2 then { t with sample_acc }
  else
    let cpu_next = Sysstat.Cpu.sample () in
    let cpu_per_core_next = Sysstat.Cpu.sample_per_core () in
    let cpu = Sysstat.Cpu.compute ~prev:t.cpu_prev ~next:cpu_next in
    let cpu_per_core =
      Array.map2
        (fun p n -> Sysstat.Cpu.compute ~prev:p ~next:n)
        t.cpu_per_core_prev cpu_per_core_next
    in
    let memory = Sysstat.Mem.sample () in
    {
      cpu;
      cpu_per_core;
      memory;
      cpu_prev = cpu_next;
      cpu_per_core_prev = cpu_per_core_next;
      num_cores = t.num_cores;
      sample_acc = 0.0;
    }

(* View *)

let label_style = Ansi.Style.make ~fg:Ansi.Color.white ()
let muted = Ansi.Style.make ~fg:(Ansi.Color.grayscale ~level:14) ()
let bracket_style = Ansi.Style.make ~fg:(Ansi.Color.grayscale ~level:10) ()
let value_style = Ansi.Style.make ~fg:Ansi.Color.white ()
let user_style = Ansi.Style.make ~fg:Ansi.Color.green ()
let system_style = Ansi.Style.make ~fg:Ansi.Color.cyan ()

let bar_color pct =
  if pct > 80. then Ansi.Color.red
  else if pct > 50. then Ansi.Color.yellow
  else Ansi.Color.green

(* Draw a simple bar: label[|||||| value] on a single canvas row *)
let draw_bar c ~y ~width ~label ~value_text ~percent =
  let label_len = String.length label in
  let value_len = String.length value_text in
  Canvas.draw_text c ~x:0 ~y ~text:label ~style:label_style;
  let bar_start = label_len in
  let bar_end = width - 1 in
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
      Canvas.draw_text c ~x:vx ~y ~text:value_text ~style:value_style
  end

(* Draw a CPU bar with user (green) + system (cyan) split *)
let draw_cpu_bar c ~y ~width ~label ~(stats : Sysstat.Cpu.stats) =
  let label_len = String.length label in
  let total = stats.user +. stats.system in
  let value_text = Printf.sprintf "%.0f%%" total in
  let value_len = String.length value_text in
  Canvas.draw_text c ~x:0 ~y ~text:label ~style:label_style;
  let bar_start = label_len in
  let bar_end = width - 1 in
  if bar_end - bar_start < 2 then ()
  else begin
    Canvas.draw_text c ~x:bar_start ~y ~text:"[" ~style:bracket_style;
    Canvas.draw_text c ~x:bar_end ~y ~text:"]" ~style:bracket_style;
    let inner = bar_end - bar_start - 1 in
    let fill_space = inner - value_len - 1 in
    let user_count =
      if fill_space > 0 then
        int_of_float (stats.user /. 100. *. float_of_int fill_space)
        |> max 0 |> min fill_space
      else 0
    in
    let system_count =
      if fill_space > 0 then
        int_of_float (stats.system /. 100. *. float_of_int fill_space)
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
      Canvas.draw_text c ~x:vx ~y ~text:value_text ~style:value_style
  end

let view_cpu_mem ~(cpu : Sysstat.Cpu.stats) ~(memory : Sysstat.Mem.t) =
  let mem_pct = mem_used_percent memory in
  let has_swap = memory.swap_total > 0L in
  let rows = if has_swap then 3 else 2 in
  canvas
    ~size:{ width = pct 100; height = px rows }
    (fun c ~delta:_ ->
      Canvas.clear c;
      let w = Canvas.width c in
      draw_cpu_bar c ~y:0 ~width:w ~label:"CPU " ~stats:cpu;
      draw_bar c ~y:1 ~width:w ~label:"Mem "
        ~value_text:
          (Printf.sprintf "%.1f/%.1fG" (bytes_to_gb memory.used)
             (bytes_to_gb memory.total))
        ~percent:mem_pct;
      if has_swap then begin
        let swap_pct =
          Int64.to_float memory.swap_used
          /. Int64.to_float memory.swap_total
          *. 100.
        in
        draw_bar c ~y:2 ~width:w ~label:"Swp "
          ~value_text:
            (Printf.sprintf "%.1f/%.1fG"
               (bytes_to_gb memory.swap_used)
               (bytes_to_gb memory.swap_total))
          ~percent:swap_pct
      end)

let view_cores (cpu_per_core : Sysstat.Cpu.stats array) =
  let num_cores = Array.length cpu_per_core in
  if num_cores = 0 then box ~size:{ width = px 0; height = px 0 } []
  else
    let max_digits = String.length (string_of_int (num_cores - 1)) in
    canvas
      ~size:{ width = pct 100; height = px num_cores }
      (fun c ~delta:_ ->
        Canvas.clear c;
        let w = Canvas.width c in
        for i = 0 to num_cores - 1 do
          let label = Printf.sprintf "%*d " max_digits i in
          draw_cpu_bar c ~y:i ~width:w ~label ~stats:cpu_per_core.(i)
        done)

let view (t : t) =
  box ~flex_direction:Column ~padding:(padding_lrtb 1 1 1 0) ~gap:(gap 1)
    ~size:{ width = pct 100; height = auto }
    [ view_cpu_mem ~cpu:t.cpu ~memory:t.memory; view_cores t.cpu_per_core ]
