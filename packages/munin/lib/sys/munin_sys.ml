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

(* [source session keys sample values] defines the metrics [keys] if [sample ()]
   reads the system now, and is then the tick of that source: it samples again
   and pairs each metric with its value from the previous and the new sample,
   [dt] seconds apart. A system that cannot read the source when the monitor
   starts gets none of its metrics. A tick whose sample fails logs none of them
   and keeps the previous sample: the next tick's rates span both intervals. A
   tick after the wall clock stepped back ([dt <= 0]) logs none of them either,
   and its sample becomes the previous one. *)
let source session keys sample values =
  let timed () =
    let s = sample () in
    (s, Unix.gettimeofday ())
  in
  match timed () with
  | exception Sys_error _ -> None
  | first ->
      let metrics = List.map (fun key -> Session.metric session key) keys in
      let prev = ref first in
      let tick () =
        match timed () with
        | exception Sys_error _ -> []
        | (next, t) as sampled ->
            let prev_sample, prev_t = !prev in
            prev := sampled;
            let dt = t -. prev_t in
            if dt <= 0. then []
            else List.combine metrics (values ~prev:prev_sample ~next ~dt)
      in
      Some tick

let mb bytes = bytes /. 1_048_576.

let start ?(interval = 15.0) session =
  let cpu =
    source session [ "sys/cpu_user"; "sys/cpu_system" ] Sysstat.Cpu.sample
      (fun ~prev ~next ~dt:_ ->
        let stats = Sysstat.Cpu.compute ~prev ~next in
        [ stats.user; stats.system ])
  in
  let mem =
    source session [ "sys/mem_used_pct"; "sys/mem_used_gb" ] Sysstat.Mem.sample
      (fun ~prev:_ ~(next : Sysstat.Mem.t) ~dt:_ ->
        let used = Int64.to_float next.used in
        [ used *. 100. /. Int64.to_float next.total; used /. 1_073_741_824. ])
  in
  let proc =
    source session [ "sys/proc_cpu_pct"; "sys/proc_mem_mb" ]
      Sysstat.Proc.Self.sample (fun ~prev ~next ~dt ->
        let stats = Sysstat.Proc.Self.compute ~prev ~next ~dt ~num_cores:None in
        [ stats.cpu_percent; mb (Int64.to_float stats.rss_bytes) ])
  in
  let disk =
    source session
      [ "sys/disk_read_mbs"; "sys/disk_write_mbs"; "sys/disk_util_pct" ]
      Sysstat.Disk_io.sample (fun ~prev ~next ~dt ->
        let stats = Sysstat.Disk_io.compute ~prev ~next ~dt in
        [
          mb stats.read_bytes_per_sec;
          mb stats.write_bytes_per_sec;
          stats.utilization_percent;
        ])
  in
  let sources = List.filter_map Fun.id [ cpu; mem; proc; disk ] in
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
