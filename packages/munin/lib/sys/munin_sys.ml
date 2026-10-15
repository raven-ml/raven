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
   to wake it. A socket, because Windows selects on sockets only. *)
let rec wait fd deadline =
  let left = deadline -. Unix.gettimeofday () in
  if left > 0. then
    match Unix.select [ fd ] [] [] left with
    | [], _, _ -> wait fd deadline
    | _ -> ()
    | exception Unix.Unix_error (Unix.EINTR, _, _) -> wait fd deadline

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

let start ?(interval = 15.0) session =
  let cpu_user = Session.metric session "sys/cpu_user" in
  let cpu_system = Session.metric session "sys/cpu_system" in
  let mem_used_pct = Session.metric session "sys/mem_used_pct" in
  let mem_used_gb = Session.metric session "sys/mem_used_gb" in
  let proc_cpu_pct = Session.metric session "sys/proc_cpu_pct" in
  let proc_mem_mb = Session.metric session "sys/proc_mem_mb" in
  let disk_read_mbs = Session.metric session "sys/disk_read_mbs" in
  let disk_write_mbs = Session.metric session "sys/disk_write_mbs" in
  let disk_util_pct = Session.metric session "sys/disk_util_pct" in
  let stop_flag = Atomic.make false in
  let wake = socket_pair () in
  let prev_cpu = ref (Sysstat.Cpu.sample ()) in
  let prev_proc = ref (Sysstat.Proc.Self.sample ()) in
  let prev_disk = ref (Sysstat.Disk_io.sample ()) in
  let prev_time = ref (Unix.gettimeofday ()) in
  let step = ref 0 in
  let thread =
    Thread.create
      (fun () ->
        while not (Atomic.get stop_flag) do
          wait (fst wake) (!prev_time +. interval);
          if not (Atomic.get stop_flag) then begin
            incr step;
            let now = Unix.gettimeofday () in
            let dt = now -. !prev_time in
            (* System CPU *)
            let cpu = Sysstat.Cpu.sample () in
            let cpu_stats = Sysstat.Cpu.compute ~prev:!prev_cpu ~next:cpu in
            prev_cpu := cpu;
            (* System memory *)
            let mem = Sysstat.Mem.sample () in
            let mem_pct =
              Int64.to_float mem.used *. 100. /. Int64.to_float mem.total
            in
            let mem_gb = Int64.to_float mem.used /. 1_073_741_824. in
            (* Process stats *)
            let proc = Sysstat.Proc.Self.sample () in
            let proc_stats =
              Sysstat.Proc.Self.compute ~prev:!prev_proc ~next:proc ~dt
                ~num_cores:None
            in
            prev_proc := proc;
            (* Disk I/O *)
            let disk = Sysstat.Disk_io.sample () in
            let disk_stats =
              Sysstat.Disk_io.compute ~prev:!prev_disk ~next:disk ~dt
            in
            prev_disk := disk;
            prev_time := now;
            Session.log_metrics session ~step:!step
              [
                (cpu_user, cpu_stats.user);
                (cpu_system, cpu_stats.system);
                (mem_used_pct, mem_pct);
                (mem_used_gb, mem_gb);
                (proc_cpu_pct, proc_stats.cpu_percent);
                (proc_mem_mb, Int64.to_float proc_stats.rss_bytes /. 1_048_576.);
                (disk_read_mbs, disk_stats.read_bytes_per_sec /. 1_048_576.);
                (disk_write_mbs, disk_stats.write_bytes_per_sec /. 1_048_576.);
                (disk_util_pct, disk_stats.utilization_percent);
              ]
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
