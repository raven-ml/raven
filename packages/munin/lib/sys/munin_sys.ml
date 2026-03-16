include Sysstat

type t = { stop : bool Atomic.t; thread : Thread.t }

let define session key = Session.define_metric session key ~summary:`Last ()

let start session ?(interval = 2.0) () =
  define session "sys/cpu_user";
  define session "sys/cpu_system";
  define session "sys/mem_used_pct";
  define session "sys/mem_used_gb";
  define session "sys/proc_cpu_pct";
  define session "sys/proc_mem_mb";
  define session "sys/disk_read_mbs";
  define session "sys/disk_write_mbs";
  define session "sys/disk_util_pct";
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
                ("sys/cpu_user", cpu_stats.user);
                ("sys/cpu_system", cpu_stats.system);
                ("sys/mem_used_pct", mem_pct);
                ("sys/mem_used_gb", mem_gb);
                ("sys/proc_cpu_pct", proc_stats.cpu_percent);
                ( "sys/proc_mem_mb",
                  Int64.to_float proc_stats.rss_bytes /. 1_048_576. );
                ( "sys/disk_read_mbs",
                  disk_stats.read_bytes_per_sec /. 1_048_576. );
                ( "sys/disk_write_mbs",
                  disk_stats.write_bytes_per_sec /. 1_048_576. );
                ("sys/disk_util_pct", disk_stats.utilization_percent);
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
