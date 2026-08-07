(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Munin
include Sysstat

type t = { stop : bool Atomic.t; thread : Thread.t }

let start session ?(interval = 2.0) () =
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
  let prev_cpu = ref (Sysstat.Cpu.sample ()) in
  let prev_proc = ref (Sysstat.Proc.Self.sample ()) in
  let prev_disk = ref (Sysstat.Disk_io.sample ()) in
  let prev_time = ref (Unix.gettimeofday ()) in
  let step = ref 0 in
  let thread =
    Thread.create
      (fun () ->
        while not (Atomic.get stop_flag) do
          Thread.delay interval;
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
  { stop = stop_flag; thread }

let stop t =
  if not (Atomic.get t.stop) then begin
    Atomic.set t.stop true;
    Thread.join t.thread
  end
