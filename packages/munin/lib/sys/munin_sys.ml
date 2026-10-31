(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Munin
module Stat = Sysstat

type t = {
  stop : bool Atomic.t;
  wake : Unix.file_descr * Unix.file_descr;
  thread : Thread.t;
}

(* The monitor waits for its next sample by selecting on one end of a socket
   pair, with the time left as timeout; [stop] writes a byte to the other end
   to wake it. A socket, because Windows selects on sockets only. More than
   [interval] left means the wall clock stepped back: the wait ends, rather
   than last until the clock reaches the deadline again. A millisecond of
   slack keeps the subtraction's rounding from reading as a step back. *)
let rec wait fd ~interval deadline =
  let left = deadline -. Unix.gettimeofday () in
  if left > 0. && left <= interval +. 1e-3 then
    match Unix.select [ fd ] [] [] left with
    | [], _, _ -> wait fd ~interval deadline
    | _ -> ()
    | exception Unix.Unix_error (Unix.EINTR, _, _) -> wait fd ~interval deadline

(* Windows has no socketpair. OCaml's binds a socket at a name in the temporary
   directory made of the process id and a count, and removes it once the pair
   is connected: a process killed in between leaves it there, and a later
   process of the same id fails to bind it with EADDRINUSE. A loopback
   connection names no file. Its listener takes the connection from the other
   end and closes any other, from another process of the machine. *)
let tcp_pair () =
  let socket () = Unix.socket ~cloexec:true Unix.PF_INET Unix.SOCK_STREAM 0 in
  let l = socket () in
  Fun.protect ~finally:(fun () -> Unix.close l) @@ fun () ->
  Unix.bind l (Unix.ADDR_INET (Unix.inet_addr_loopback, 0));
  Unix.listen l 1;
  let a = socket () in
  let rec from me =
    let b, peer = Unix.accept ~cloexec:true l in
    if peer = me then b
    else begin
      Unix.close b;
      from me
    end
  in
  match
    Unix.connect a (Unix.getsockname l);
    from (Unix.getsockname a)
  with
  | b -> (a, b)
  | exception e ->
      Unix.close a;
      raise e

let socket_pair () =
  if Sys.win32 then tcp_pair ()
  else Unix.socketpair ~cloexec:true Unix.PF_UNIX Unix.SOCK_STREAM 0

(* [source session sample define] samples now and, if [sample ()] reads the
   system, defines the metrics [keys] of [define first = (keys, values)], the
   first sample. It is then the tick of that source: it samples again and pairs
   each metric with its value from the previous and the new sample, [dt]
   seconds apart, where [values] gives one. A system that cannot read the
   source when the monitor starts, or that it gives no metric, gets none of
   its metrics. A tick whose sample fails logs none of them and keeps the
   previous sample: the next tick's rates span both intervals. A tick after the
   wall clock stepped back ([dt <= 0]) logs none of them either, and its sample
   becomes the previous one. *)
let source session sample define =
  let timed () =
    let s = sample () in
    (s, Unix.gettimeofday ())
  in
  match timed () with
  | exception Sys_error _ -> None
  | (first, _) as sampled -> (
      match define first with
      | [], _ -> None
      | keys, values ->
          let metrics = List.map (fun key -> Session.metric session key) keys in
          let prev = ref sampled in
          let logged (m, v) = Option.map (fun v -> (m, v)) v in
          let tick () =
            match timed () with
            | exception Sys_error _ -> []
            | (next, t) as sampled ->
                let prev_sample, prev_t = !prev in
                prev := sampled;
                let dt = t -. prev_t in
                if dt <= 0. then []
                else
                  List.combine metrics (values ~prev:prev_sample ~next ~dt)
                  |> List.filter_map logged
          in
          Some tick)

(* The [define] of a source whose metrics [keys] each sample gives. *)
let fixed keys values _first =
  (keys, fun ~prev ~next ~dt -> List.map Option.some (values ~prev ~next ~dt))

(* Each GPU's metrics, named by its place in the first sample and kept to its
   id, and each one its first sample gives. A GPU gone from a later sample
   logs nothing then. *)
let gpu_fields : (string * (Sysstat.Gpu.t -> float option)) list =
  let gb bytes = Int64.to_float bytes /. 1_073_741_824. in
  [
    ("util_pct", fun g -> g.busy_percent);
    ("mem_used_gb", fun g -> Option.map gb g.memory_used);
    ( "mem_used_pct",
      fun g ->
        match (g.memory_used, g.memory_total) with
        | Some used, Some total when Int64.compare total 0L > 0 ->
            Some (Int64.to_float used *. 100. /. Int64.to_float total)
        | _ -> None );
    ("temp_c", fun g -> g.temperature);
    ("power_w", fun g -> g.power);
  ]

let gpu_define (first : Sysstat.Gpu.t list) =
  let plan =
    List.concat
      (List.mapi
         (fun i (g : Sysstat.Gpu.t) ->
           List.filter_map
             (fun (name, field) ->
               match field g with
               | None -> None
               | Some _ ->
                   Some (Printf.sprintf "sys/gpu%d_%s" i name, g.id, field))
             gpu_fields)
         first)
  in
  let values ~prev:_ ~(next : Sysstat.Gpu.t list) ~dt:_ =
    List.map
      (fun (_, id, field) ->
        match List.find_opt (fun (g : Sysstat.Gpu.t) -> g.id = id) next with
        | Some g -> field g
        | None -> None)
      plan
  in
  (List.map (fun (key, _, _) -> key) plan, values)

let mb bytes = bytes /. 1_048_576.

let start ?(interval = 15.0) session =
  let cpu =
    source session Sysstat.Cpu.sample
      (fixed [ "sys/cpu_user"; "sys/cpu_system" ] (fun ~prev ~next ~dt:_ ->
           let stats = Sysstat.Cpu.compute ~prev ~next in
           [ stats.user; stats.system ]))
  in
  let mem =
    source session Sysstat.Mem.sample
      (fixed [ "sys/mem_used_pct"; "sys/mem_used_gb" ]
         (fun ~prev:_ ~(next : Sysstat.Mem.t) ~dt:_ ->
           let used = Int64.to_float next.used in
           [ used *. 100. /. Int64.to_float next.total; used /. 1_073_741_824. ]))
  in
  let proc =
    source session Sysstat.Proc.Self.sample
      (fixed [ "sys/proc_cpu_pct"; "sys/proc_mem_mb" ] (fun ~prev ~next ~dt ->
           let stats =
             Sysstat.Proc.Self.compute ~prev ~next ~dt ~num_cores:None
           in
           [ stats.cpu_percent; mb (Int64.to_float stats.rss_bytes) ]))
  in
  let disk =
    source session Sysstat.Disk_io.sample
      (fixed
         [ "sys/disk_read_mbs"; "sys/disk_write_mbs"; "sys/disk_util_pct" ]
         (fun ~prev ~next ~dt ->
           let stats = Sysstat.Disk_io.compute ~prev ~next ~dt in
           [
             mb stats.read_bytes_per_sec;
             mb stats.write_bytes_per_sec;
             stats.utilization_percent;
           ]))
  in
  let gpu = source session Sysstat.Gpu.sample gpu_define in
  let sources = List.filter_map Fun.id [ cpu; mem; proc; disk; gpu ] in
  let stop_flag = Atomic.make false in
  let wake = socket_pair () in
  let prev_time = ref (Unix.gettimeofday ()) in
  let step = ref 0 in
  let thread =
    Thread.create
      (fun () ->
        while not (Atomic.get stop_flag) do
          wait (fst wake) ~interval (!prev_time +. interval);
          if not (Atomic.get stop_flag) then begin
            incr step;
            prev_time := Unix.gettimeofday ();
            match List.concat_map (fun tick -> tick ()) sources with
            | [] -> ()
            | metrics -> Session.log_metrics session ~step:!step metrics
          end
        done)
      ()
  in
  { stop = stop_flag; wake; thread }

(* Both ends close only after the join: had the monitor closed its end when an
   exception ended it, the write would deliver SIGPIPE and end the process. *)
let stop t =
  if Atomic.compare_and_set t.stop false true then begin
    ignore (Unix.write_substring (snd t.wake) "x" 0 1);
    Thread.join t.thread;
    Unix.close (fst t.wake);
    Unix.close (snd t.wake)
  end
  else Thread.join t.thread
