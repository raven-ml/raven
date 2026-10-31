(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Munin

(* [Session.start] defaults its provenance to [Provenance.detect ()], which runs
   git twice: in this repository, most of the suite's time. Sessions here record
   [provenance] unless a test passes its own. *)
let provenance =
  {
    Provenance.command = [ "test_sys" ];
    cwd = "/";
    hostname = None;
    pid = 0;
    git_commit = None;
    git_dirty = None;
    env = [];
  }

module Session = struct
  include Session

  let start ?(provenance = provenance) = start ~provenance
end

let rec rm_rf path =
  if Sys.is_directory path then begin
    Array.iter (fun e -> rm_rf (Filename.concat path e)) (Sys.readdir path);
    Sys.rmdir path
  end
  else Sys.remove path

let with_temp_dir f =
  let base = Filename.temp_file "munin" "test" in
  Sys.remove base;
  Unix.mkdir base 0o755;
  Fun.protect ~finally:(fun () -> rm_rf base) (fun () -> f base)

(* Munin reads no disk I/O counters on Windows: there the monitor neither
   defines nor logs the disk metrics. *)
let sys_keys =
  List.sort compare
    ([
       "sys/cpu_user";
       "sys/cpu_system";
       "sys/mem_used_pct";
       "sys/mem_used_gb";
       "sys/proc_cpu_pct";
       "sys/proc_mem_mb";
     ]
    @
    if Sys.win32 then []
    else [ "sys/disk_read_mbs"; "sys/disk_write_mbs"; "sys/disk_util_pct" ])

let sys_only keys =
  List.sort compare (List.filter (String.starts_with ~prefix:"sys/") keys)

let test_system_monitor_logs_metrics () =
  with_temp_dir @@ fun root ->
  let store = Store.open_ ~root () in
  let session = Session.start ~store ~experiment:"exp" () in
  let monitor = Munin_sys.start ~interval:0.1 session in
  let rec sampled () =
    if not (List.mem "sys/cpu_user" (Run.metric_keys (Session.run session)))
    then begin
      Thread.delay 0.01;
      sampled ()
    end
  in
  sampled ();
  Munin_sys.stop monitor;
  Session.finish session;
  let run = Session.run session in
  equal (list string) sys_keys (sys_only (Run.metric_keys run))

let test_system_monitor_defines_metrics () =
  with_temp_dir @@ fun root ->
  let store = Store.open_ ~root () in
  let session = Session.start ~store ~experiment:"exp" () in
  let monitor = Munin_sys.start ~interval:100.0 session in
  Munin_sys.stop monitor;
  Session.finish session;
  let run = Session.run session in
  let last =
    List.filter_map
      (fun (key, (d : Metric.def)) ->
        if d.summary = `Last then Some key else None)
      (Run.metric_defs run)
  in
  equal (list string) sys_keys (sys_only last)

let test_system_monitor_stop_idempotent () =
  with_temp_dir @@ fun root ->
  let store = Store.open_ ~root () in
  let session = Session.start ~store ~experiment:"exp" () in
  let monitor = Munin_sys.start ~interval:100.0 session in
  Munin_sys.stop monitor;
  Munin_sys.stop monitor;
  Session.finish session

(* The monitor sleeps between samples; stop must not wait out the interval. The
   delay lets the monitor thread reach its sleep: if it has not, stop returns at
   once and the test passes without reaching its case, never failing falsely. *)
let test_system_monitor_stop_prompt () =
  with_temp_dir @@ fun root ->
  let store = Store.open_ ~root () in
  let session = Session.start ~store ~experiment:"exp" () in
  let monitor = Munin_sys.start ~interval:10.0 session in
  Thread.delay 0.1;
  let t0 = Unix.gettimeofday () in
  Munin_sys.stop monitor;
  less ~msg:"seconds stop blocked" float_exact ~than:1.0
    (Unix.gettimeofday () -. t0);
  Session.finish session

(* A process's CPU percentage is its CPU time over the interval, whatever the
   interval and however many cores the time came from. *)
let test_proc_cpu_mean =
  let gen = Gen.pair (Gen.float_range 0.02 3600.) (Gen.float_range 0. 64.) in
  prop "process CPU is a mean over the interval" gen @@ fun (dt, cores) ->
  let cpu = cores *. dt in
  let prev =
    {
      Munin_sys.Stat.Proc.Self.utime = 0.;
      stime = 0.;
      rss_bytes = 0L;
      vsize_bytes = 0L;
    }
  in
  let next = { prev with utime = cpu /. 2.; stime = cpu /. 2. } in
  cover "interval past 10 s" (dt > 10.);
  cover "more than 8 cores busy" (cores > 8.);
  let stats =
    Munin_sys.Stat.Proc.Self.compute ~prev ~next ~dt ~num_cores:None
  in
  equal (float_rel ~rel:1e-9 ~abs:1e-9) (cores *. 100.) stats.cpu_percent

(* Every sampler reads the system on Linux and macOS. On Windows the per-core,
   network, disk I/O, process table and load average samplers raise
   [Sys_error]. *)
let reads name read check = test name @@ fun () -> check (read ())

let unread_on_windows name read check =
  test name @@ fun () ->
  if Sys.win32 then raises_match ~msg:name Exn.sys_error read
  else check (read ())

let test_per_core =
  unread_on_windows "Cpu.sample_per_core" Munin_sys.Stat.Cpu.sample_per_core
  @@ fun cores -> at_least ~msg:"cores" int ~than:1 (Array.length cores)

let test_net =
  unread_on_windows "Net.sample" Munin_sys.Stat.Net.sample @@ fun n ->
  at_least ~msg:"bytes_rx" int64 ~than:0L n.bytes_rx;
  at_least ~msg:"packets_rx" int64 ~than:0L n.packets_rx;
  at_least ~msg:"bytes_tx" int64 ~than:0L n.bytes_tx;
  at_least ~msg:"packets_tx" int64 ~than:0L n.packets_tx

let test_disk_io =
  unread_on_windows "Disk_io.sample" Munin_sys.Stat.Disk_io.sample @@ fun d ->
  at_least ~msg:"bytes_read" int64 ~than:0L d.bytes_read;
  at_least ~msg:"bytes_written" int64 ~than:0L d.bytes_written;
  at_least ~msg:"time_ms" int64 ~than:0L d.time_ms;
  at_least ~msg:"num_disks" int64 ~than:0L d.num_disks

let test_fs =
  reads "Fs.sample" (fun () -> Munin_sys.Stat.Fs.sample ()) @@ fun fs ->
  greater ~msg:"total_bytes" int64 ~than:0L fs.total_bytes;
  at_most ~msg:"used_bytes" int64 ~than:fs.total_bytes fs.used_bytes

(* Windows boots from a fixed drive, the partitions it lists. *)
let test_partitions =
  reads "Fs.partitions" Munin_sys.Stat.Fs.partitions @@ fun ps ->
  if Sys.win32 then at_least ~msg:"fixed drives" int ~than:1 (List.length ps);
  List.iter
    (fun (p : Munin_sys.Stat.Fs.partition) ->
      at_most ~msg:p.mount_point int64 ~than:p.total_bytes p.used_bytes)
    ps

(* A process is resident in memory, and its resident pages are mapped. *)
let test_proc_self =
  reads "Proc.Self.sample" Munin_sys.Stat.Proc.Self.sample @@ fun p ->
  greater ~msg:"rss_bytes" int64 ~than:0L p.rss_bytes;
  at_least ~msg:"vsize_bytes" int64 ~than:p.rss_bytes p.vsize_bytes

let test_proc_table =
  unread_on_windows "Proc.Table.sample" Munin_sys.Stat.Proc.Table.sample
  @@ fun ps ->
  mem int (Unix.getpid ())
    (List.map (fun (p : Munin_sys.Stat.Proc.Table.t) -> p.pid) ps)

let test_loadavg =
  unread_on_windows "loadavg" Munin_sys.Stat.loadavg @@ fun (l1, l5, l15) ->
  at_least ~msg:"1 min" float_exact ~than:0. l1;
  at_least ~msg:"5 min" float_exact ~than:0. l5;
  at_least ~msg:"15 min" float_exact ~than:0. l15

let test_uptime =
  reads "uptime" Munin_sys.Stat.uptime @@ fun s ->
  greater ~msg:"seconds" int64 ~than:0L s

let system_monitor_tests =
  [
    test "logs metrics" test_system_monitor_logs_metrics;
    test "defines metrics" test_system_monitor_defines_metrics;
    test "stop idempotent" test_system_monitor_stop_idempotent;
    test "stop does not wait out the interval" test_system_monitor_stop_prompt;
  ]

let () =
  exit
    (run "Munin_sys"
       [
         group "System monitor" system_monitor_tests;
         group "Stat"
           [
             test_proc_cpu_mean;
             test_per_core;
             test_net;
             test_disk_io;
             test_fs;
             test_partitions;
             test_proc_self;
             test_proc_table;
             test_loadavg;
             test_uptime;
           ];
       ])
