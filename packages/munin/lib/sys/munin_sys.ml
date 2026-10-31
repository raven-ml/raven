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
   system, is the tick of that source, where [define first = (keys, values)]
   for the first sample. It declares the metrics [keys] at once. Each tick
   samples again and logs the [(key, value)] pairs that [values] gives from
   the previous and the new sample, [dt] seconds apart, declaring a key the
   first time it is logged. A system that cannot read the source when the
   monitor starts gets none of its metrics. A tick whose sample fails logs
   none of them and keeps the previous sample: the next tick's rates span
   both intervals. A tick after the wall clock stepped back ([dt <= 0]) logs
   none of them either, and its sample becomes the previous one. *)
let source session sample define =
  let timed () =
    let s = sample () in
    (s, Unix.gettimeofday ())
  in
  match timed () with
  | exception Sys_error _ -> None
  | (first, _) as sampled ->
      let keys, values = define first in
      let metrics = Hashtbl.create 16 in
      let metric key =
        match Hashtbl.find_opt metrics key with
        | Some m -> m
        | None ->
            let m = Session.metric session key in
            Hashtbl.replace metrics key m;
            m
      in
      List.iter (fun key -> ignore (metric key)) keys;
      let prev = ref sampled in
      let tick () =
        match timed () with
        | exception Sys_error _ -> []
        | (next, t) as sampled ->
            let prev_sample, prev_t = !prev in
            prev := sampled;
            let dt = t -. prev_t in
            if dt <= 0. then []
            else
              List.map
                (fun (key, v) -> (metric key, v))
                (values ~prev:prev_sample ~next ~dt)
      in
      Some tick

(* The [define] of a source whose metrics [keys] each sample gives. *)
let fixed keys values _first =
  (keys, fun ~prev ~next ~dt -> List.combine keys (values ~prev ~next ~dt))

(* GPU metrics' keys *)

let gpu_prefix = "sys/gpu"
let gpu_key i name = Printf.sprintf "%s%d_%s" gpu_prefix i name

(* The digits after the prefix, as [gpu_key] prints them, without a leading
   zero, up to the first '_', then a name. *)
let gpu_metric key =
  let p = String.length gpu_prefix and n = String.length key in
  if not (String.starts_with ~prefix:gpu_prefix key) then None
  else
    match String.index_from_opt key p '_' with
    | None -> None
    | Some u ->
        let digits = String.sub key p (u - p) in
        let all_digits =
          digits <> ""
          && String.for_all (function '0' .. '9' -> true | _ -> false) digits
          && (digits = "0" || digits.[0] <> '0')
        in
        if (not all_digits) || u + 1 = n then None
        else
          Option.map
            (fun i -> (i, String.sub key (u + 1) (n - u - 1)))
            (int_of_string_opt digits)

(* A GPU's metrics, each logged when its sample has its value. *)
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

(* GPU [i] is the [i]th id the monitor saw, in the order of the first sample
   and then of the samples a new id appears in, so a GPU keeps its number. The
   metrics of the first sample's values are declared at the start; each other
   is declared the first time a sample has its value, as the load of a GPU
   that was suspended at the start. *)
let gpu_define (first : Sysstat.Gpu.t list) =
  let numbers = Hashtbl.create 4 in
  let number (g : Sysstat.Gpu.t) =
    match Hashtbl.find_opt numbers g.id with
    | Some i -> i
    | None ->
        let i = Hashtbl.length numbers in
        Hashtbl.replace numbers g.id i;
        i
  in
  let values (gpus : Sysstat.Gpu.t list) =
    List.concat_map
      (fun g ->
        let i = number g in
        List.filter_map
          (fun (name, field) ->
            Option.map (fun v -> (gpu_key i name, v)) (field g))
          gpu_fields)
      gpus
  in
  ( List.map fst (values first),
    fun ~prev:_ ~next ~dt:_ -> values next )

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
           let total = Int64.to_float next.total in
           [ used *. 100. /. total; used /. 1_073_741_824. ]))
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
  let gpu =
    source session (Sysstat.Gpu.sample ~window:interval) gpu_define
  in
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
