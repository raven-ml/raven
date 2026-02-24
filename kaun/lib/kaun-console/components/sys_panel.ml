(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Compact system metrics panel for kaun-console.

    Displays CPU, memory, and process stats in a narrow vertical column. *)

open Mosaic
module Charts = Matrix_charts

(* ───── Model ───── *)

type t = {
  cpu : Sysstat.Cpu.stats;
  cpu_per_core : Sysstat.Cpu.stats array;
  memory : Sysstat.Mem.t;
  process : Sysstat.Proc.Self.stats;
  cpu_prev : Sysstat.Cpu.t;
  cpu_per_core_prev : Sysstat.Cpu.t array;
  proc_prev : Sysstat.Proc.Self.t;
  num_cores : int;
  sparkline_cpu : Charts.Sparkline.t;
  sparkline_mem : Charts.Sparkline.t;
  sample_acc : float;
}

(* ───── Helpers ───── *)

let bytes_to_gb b = Int64.to_float b /. (1024. *. 1024. *. 1024.)
let bytes_to_mb b = Int64.to_float b /. (1024. *. 1024.)

let mem_used_percent (m : Sysstat.Mem.t) =
  if m.total > 0L then Int64.to_float m.used /. Int64.to_float m.total *. 100.
  else 0.0

(* ───── Init ───── *)

let create () : t =
  let sparkline_cpu =
    Charts.Sparkline.create
      ~style:(Ansi.Style.make ~fg:Ansi.Color.cyan ())
      ~auto_max:false ~max_value:100. ~capacity:15 ()
  in
  let sparkline_mem =
    Charts.Sparkline.create
      ~style:(Ansi.Style.make ~fg:Ansi.Color.blue ())
      ~auto_max:false ~max_value:100. ~capacity:15 ()
  in
  (* Initial CPU sample *)
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
  (* Memory *)
  let memory = Sysstat.Mem.sample () in
  (* Process self *)
  let proc_prev = Sysstat.Proc.Self.sample () in
  let process =
    { Sysstat.Proc.Self.cpu_percent = 0.0; rss_bytes = 0L; vsize_bytes = 0L }
  in
  (* Push initial values to sparklines *)
  let total_cpu = cpu.user +. cpu.system in
  Charts.Sparkline.push sparkline_cpu total_cpu;
  Charts.Sparkline.push sparkline_mem (mem_used_percent memory);
  {
    cpu;
    cpu_per_core;
    memory;
    process;
    cpu_prev;
    cpu_per_core_prev;
    proc_prev;
    num_cores;
    sparkline_cpu;
    sparkline_mem;
    sample_acc = 0.0;
  }

(* ───── Update ───── *)

let update (t : t) ~(dt : float) : t =
  let sample_acc = t.sample_acc +. dt in
  (* Sample at ~5Hz *)
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
    let proc_next = Sysstat.Proc.Self.sample () in
    let process =
      Sysstat.Proc.Self.compute ~prev:t.proc_prev ~next:proc_next ~dt:sample_acc
        ~num_cores:(if t.num_cores > 0 then Some t.num_cores else None)
    in
    (* Update sparklines *)
    let total_cpu = cpu.user +. cpu.system in
    Charts.Sparkline.push t.sparkline_cpu total_cpu;
    Charts.Sparkline.push t.sparkline_mem (mem_used_percent memory);
    {
      cpu;
      cpu_per_core;
      memory;
      process;
      cpu_prev = cpu_next;
      cpu_per_core_prev = cpu_per_core_next;
      proc_prev = proc_next;
      num_cores = t.num_cores;
      sparkline_cpu = t.sparkline_cpu;
      sparkline_mem = t.sparkline_mem;
      sample_acc = 0.0;
    }

(* ───── View ───── *)

let muted = Ansi.Style.make ~fg:(Ansi.Color.grayscale ~level:14) ()
let bold_cyan = Ansi.Style.make ~bold:true ~fg:Ansi.Color.cyan ()
let bold_blue = Ansi.Style.make ~bold:true ~fg:Ansi.Color.blue ()
let bold_green = Ansi.Style.make ~bold:true ~fg:Ansi.Color.green ()
let bold_yellow = Ansi.Style.make ~bold:true ~fg:Ansi.Color.yellow ()

let draw_progress_bar c ~value ~max_value ~fill_color =
  let width = Canvas.width c in
  let height = Canvas.height c in
  if width > 0 && height > 0 then (
    (* Background *)
    Canvas.fill_rect c ~x:0 ~y:0 ~width ~height
      ~color:(Ansi.Color.grayscale ~level:3);
    (* Filled portion *)
    let filled = int_of_float (value /. max_value *. float_of_int width) in
    let filled = max 0 (min width filled) in
    if filled > 0 then
      Canvas.fill_rect c ~x:0 ~y:0 ~width:filled ~height ~color:fill_color)

let view_cpu_bar (cpu : Sysstat.Cpu.stats) =
  let total = cpu.user +. cpu.system in
  let color =
    if total > 80. then Ansi.Color.red
    else if total > 50. then Ansi.Color.yellow
    else Ansi.Color.green
  in
  box ~flex_direction:Column ~gap:(gap 0)
    ~size:{ width = pct 100; height = auto }
    [
      box ~flex_direction:Row ~justify_content:Space_between ~align_items:Center
        [
          text ~style:muted "CPU";
          text
            ~style:(Ansi.Style.make ~bold:true ~fg:color ())
            (Printf.sprintf "%.0f%%" total);
        ];
      canvas
        ~size:{ width = pct 100; height = px 1 }
        (fun c ~delta:_ ->
          draw_progress_bar c ~value:total ~max_value:100. ~fill_color:color);
    ]

let view_mem_bar (mem : Sysstat.Mem.t) =
  let mem_pct = mem_used_percent mem in
  let color =
    if mem_pct > 90. then Ansi.Color.red
    else if mem_pct > 70. then Ansi.Color.yellow
    else Ansi.Color.blue
  in
  box ~flex_direction:Column ~gap:(gap 0)
    ~size:{ width = pct 100; height = auto }
    [
      box ~flex_direction:Row ~justify_content:Space_between ~align_items:Center
        [
          text ~style:muted "MEM";
          text
            ~style:(Ansi.Style.make ~bold:true ~fg:color ())
            (Printf.sprintf "%.0f%%" mem_pct);
        ];
      canvas
        ~size:{ width = pct 100; height = px 1 }
        (fun c ~delta:_ ->
          draw_progress_bar c ~value:mem_pct ~max_value:100. ~fill_color:color);
    ]

let view_sparklines ~sparkline_cpu ~sparkline_mem =
  box ~flex_direction:Row ~gap:(gap 1)
    ~size:{ width = pct 100; height = auto }
    [
      (* CPU sparkline *)
      box ~flex_direction:Column ~gap:(gap 0)
        ~size:{ width = pct 50; height = auto }
        [
          text ~style:muted "CPU";
          canvas
            ~size:{ width = pct 100; height = px 3 }
            (fun c ~delta:_ ->
              Charts.Sparkline.draw sparkline_cpu ~kind:`Braille (Canvas.grid c)
                ~width:(Canvas.width c) ~height:(Canvas.height c));
        ];
      (* Memory sparkline *)
      box ~flex_direction:Column ~gap:(gap 0)
        ~size:{ width = pct 50; height = auto }
        [
          text ~style:muted "MEM";
          canvas
            ~size:{ width = pct 100; height = px 3 }
            (fun c ~delta:_ ->
              Charts.Sparkline.draw sparkline_mem ~kind:`Braille (Canvas.grid c)
                ~width:(Canvas.width c) ~height:(Canvas.height c));
        ];
    ]

let view_process (proc : Sysstat.Proc.Self.stats) =
  box ~flex_direction:Column ~gap:(gap 1)
    ~size:{ width = pct 100; height = auto }
    [
      text ~style:(Ansi.Style.make ~bold:true ()) "Process";
      box ~flex_direction:Row ~justify_content:Space_between ~align_items:Center
        [
          text ~style:muted "CPU:";
          text ~style:bold_cyan (Printf.sprintf "%.1f%%" proc.cpu_percent);
        ];
      box ~flex_direction:Row ~justify_content:Space_between ~align_items:Center
        [
          text ~style:muted "RSS:";
          text ~style:bold_blue
            (Printf.sprintf "%.1f MB" (bytes_to_mb proc.rss_bytes));
        ];
    ]

let view_per_core_cpu (cpu_per_core : Sysstat.Cpu.stats array) =
  let num_cores = Array.length cpu_per_core in
  if num_cores = 0 then box ~size:{ width = px 0; height = px 0 } []
  else
    let cores = Array.to_list (Array.mapi (fun i s -> (i, s)) cpu_per_core) in
    let rec chunk_pairs = function
      | [] -> []
      | [ x ] -> [ [ x ] ]
      | x :: y :: rest -> [ x; y ] :: chunk_pairs rest
    in
    let rows = chunk_pairs cores in
    box ~flex_direction:Column ~gap:(gap 0)
      ~size:{ width = pct 100; height = auto }
      [
        text ~style:(Ansi.Style.make ~bold:true ()) "Cores";
        box ~flex_direction:Column ~gap:(gap 0)
          ~size:{ width = pct 100; height = auto }
          (List.mapi
             (fun row_idx row ->
               box
                 ~key:(Printf.sprintf "core-row-%d" row_idx)
                 ~flex_direction:Row ~gap:(gap 1)
                 ~size:{ width = pct 100; height = auto }
                 (List.map
                    (fun (i, (stats : Sysstat.Cpu.stats)) ->
                      let total = stats.user +. stats.system in
                      let color =
                        if total > 80. then Ansi.Color.red
                        else if total > 50. then Ansi.Color.yellow
                        else Ansi.Color.green
                      in
                      box
                        ~key:(Printf.sprintf "core-%d" i)
                        ~flex_direction:Row ~gap:(gap 0) ~align_items:Center
                        ~size:{ width = pct 50; height = auto }
                        [
                          text ~style:muted (Printf.sprintf "%d:" i);
                          canvas
                            ~size:{ width = pct 70; height = px 1 }
                            (fun c ~delta:_ ->
                              draw_progress_bar c ~value:total ~max_value:100.
                                ~fill_color:color);
                          text
                            ~style:(Ansi.Style.make ~fg:color ())
                            (Printf.sprintf "%2.0f" total);
                        ])
                    row))
             rows);
      ]

let view_memory_detail (mem : Sysstat.Mem.t) =
  box ~flex_direction:Column ~gap:(gap 1)
    ~size:{ width = pct 100; height = auto }
    [
      text ~style:(Ansi.Style.make ~bold:true ()) "Memory";
      box ~flex_direction:Row ~justify_content:Space_between ~align_items:Center
        [
          text ~style:muted "Used:";
          text ~style:bold_blue
            (Printf.sprintf "%.1f GB" (bytes_to_gb mem.used));
        ];
      box ~flex_direction:Row ~justify_content:Space_between ~align_items:Center
        [
          text ~style:muted "Total:";
          text ~style:bold_green
            (Printf.sprintf "%.1f GB" (bytes_to_gb mem.total));
        ];
      (if mem.swap_used > 0L then
         box ~flex_direction:Row ~justify_content:Space_between
           ~align_items:Center
           [
             text ~style:muted "Swap:";
             text ~style:bold_yellow
               (Printf.sprintf "%.1f GB" (bytes_to_gb mem.swap_used));
           ]
       else box ~size:{ width = px 0; height = px 0 } []);
    ]

let view (t : t) =
  box ~flex_direction:Column ~padding:(padding 1) ~gap:(gap 2)
    ~size:{ width = pct 100; height = auto }
    [
      text ~style:(Ansi.Style.make ~bold:true ()) "System";
      view_cpu_bar t.cpu;
      view_mem_bar t.memory;
      view_sparklines ~sparkline_cpu:t.sparkline_cpu
        ~sparkline_mem:t.sparkline_mem;
      view_per_core_cpu t.cpu_per_core;
      view_memory_detail t.memory;
      view_process t.process;
    ]
