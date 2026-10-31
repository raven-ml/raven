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

let sys_keys =
  List.sort compare
    [
      "sys/cpu_user";
      "sys/cpu_system";
      "sys/mem_used_pct";
      "sys/mem_used_gb";
      "sys/proc_cpu_pct";
      "sys/proc_mem_mb";
      "sys/disk_read_mbs";
      "sys/disk_write_mbs";
      "sys/disk_util_pct";
    ]

(* The GPU metrics depend on the machine's GPUs; the GPU tests check them. *)
let sys_only keys =
  List.sort compare
    (List.filter
       (fun k ->
         String.starts_with ~prefix:"sys/" k
         && not (String.starts_with ~prefix:"sys/gpu" k))
       keys)

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

(* Every sampler reads the system on Linux, macOS and Windows, except the load
   average, which Windows keeps none of. *)
let reads name read check = test name @@ fun () -> check (read ())

let unread_on_windows name read check =
  test name @@ fun () ->
  if Sys.win32 then raises_match ~msg:name Exn.sys_error read
  else check (read ())

let test_per_core =
  reads "Cpu.sample_per_core" Munin_sys.Stat.Cpu.sample_per_core
  @@ fun cores -> at_least ~msg:"cores" int ~than:1 (Array.length cores)

let test_net =
  reads "Net.sample" Munin_sys.Stat.Net.sample @@ fun n ->
  at_least ~msg:"bytes_rx" int64 ~than:0L n.bytes_rx;
  at_least ~msg:"packets_rx" int64 ~than:0L n.packets_rx;
  at_least ~msg:"bytes_tx" int64 ~than:0L n.bytes_tx;
  at_least ~msg:"packets_tx" int64 ~than:0L n.packets_tx

let test_disk_io =
  reads "Disk_io.sample" Munin_sys.Stat.Disk_io.sample @@ fun d ->
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

(* This process is in the table, named and resident, whatever user owns the
   others. *)
let test_proc_table =
  reads "Proc.Table.sample" Munin_sys.Stat.Proc.Table.sample @@ fun ps ->
  let me = Unix.getpid () in
  let pid (p : Munin_sys.Stat.Proc.Table.t) = p.pid in
  mem int me (List.map pid ps);
  let p = List.find (fun p -> pid p = me) ps in
  not_equal ~msg:"name" string "" p.name;
  greater ~msg:"resident_size" int64 ~than:0L p.resident_size

let test_loadavg =
  unread_on_windows "loadavg" Munin_sys.Stat.loadavg @@ fun (l1, l5, l15) ->
  at_least ~msg:"1 min" float_exact ~than:0. l1;
  at_least ~msg:"5 min" float_exact ~than:0. l5;
  at_least ~msg:"15 min" float_exact ~than:0. l15

(* GPUs *)

let rec mkdir_p dir =
  if not (Sys.file_exists dir) then begin
    mkdir_p (Filename.dirname dir);
    Unix.mkdir dir 0o755
  end

let write path contents =
  mkdir_p (Filename.dirname path);
  Out_channel.with_open_bin path (fun oc -> output_string oc contents)

(* An amdgpu GPU's files, each present or not, in the units of the kernel's
   amdgpu documentation: bytes, millidegrees Celsius and microwatts. *)
type amdgpu = {
  busy : int option;
  used : int option;
  total : int option;
  millic : int option;
  avg_uw : int option;
  input_uw : int option;
  suspended : bool;
}

let gen_amdgpu =
  let open Gen in
  let some g = option g in
  let+ busy = some (int_range 0 100)
  and+ used = some (int_range 0 (1 lsl 40))
  and+ total = some (int_range 1 (1 lsl 40))
  and+ millic = some (int_range (-40_000) 125_000)
  and+ avg_uw = some (int_range 0 1_000_000_000)
  and+ input_uw = some (int_range 0 1_000_000_000)
  and+ suspended = bool in
  { busy; used; total; millic; avg_uw; input_uw; suspended }

let slot i = Printf.sprintf "0000:%02x:00.0" (i + 3)

(* Card [i] is the amdgpu [g]; the root also holds a connector and a GPU of
   another driver, which munin does not read. *)
let make_card root i g =
  let dev = Filename.concat root (Printf.sprintf "sys/class/drm/card%d/device" i) in
  let file name v =
    Option.iter (fun v -> write (Filename.concat dev name) (Printf.sprintf "%d\n" v)) v
  in
  write (Filename.concat dev "uevent")
    (Printf.sprintf "DRIVER=amdgpu\nPCI_CLASS=30000\nPCI_SLOT_NAME=%s\n" (slot i));
  write (Filename.concat dev "power/runtime_status")
    (if g.suspended then "suspended\n" else "active\n");
  file "gpu_busy_percent" g.busy;
  file "mem_info_vram_used" g.used;
  file "mem_info_vram_total" g.total;
  file "hwmon/hwmon4/temp1_input" g.millic;
  file "hwmon/hwmon4/power1_average" g.avg_uw;
  file "hwmon/hwmon4/power1_input" g.input_uw

let make_others root =
  write (Filename.concat root "sys/class/drm/card0-DP-1/status") "connected\n";
  write
    (Filename.concat root "sys/class/drm/card9/device/uevent")
    "DRIVER=i915\nPCI_SLOT_NAME=0000:00:02.0\n"

let check_gpu i g (got : Munin_sys.Stat.Gpu.t) =
  let f = option (float_rel ~rel:1e-12 ~abs:0.) and i64 = option int64 in
  let awake x = if g.suspended then None else x in
  let scaled s = Option.map (fun v -> float_of_int v /. s) in
  equal ~msg:"id" string (slot i) got.id;
  equal ~msg:"busy_percent" f (awake (scaled 1. g.busy)) got.busy_percent;
  equal ~msg:"memory_used" i64 (Option.map Int64.of_int g.used) got.memory_used;
  equal ~msg:"memory_total" i64 (Option.map Int64.of_int g.total)
    got.memory_total;
  equal ~msg:"temperature" f (awake (scaled 1e3 g.millic)) got.temperature;
  let power = match g.avg_uw with Some _ -> g.avg_uw | None -> g.input_uw in
  equal ~msg:"power" f (awake (scaled 1e6 power)) got.power

(* Every amdgpu GPU reads back what its files hold, in bytes, Celsius and
   watts; the average power before the instantaneous one; a suspended GPU its
   memory alone; and no other DRM entry. *)
let test_gpu_sysfs =
  prop "Gpu.sample reads amdgpu's files"
    (Gen.list ~size:(Gen.int_range 0 3) gen_amdgpu)
  @@ fun gpus ->
  cover "a suspended GPU" (List.exists (fun g -> g.suspended) gpus);
  cover "power from power1_input alone"
    (List.exists (fun g -> g.avg_uw = None && g.input_uw <> None) gpus);
  cover "no GPU" (gpus = []);
  with_temp_dir @@ fun root ->
  List.iteri (make_card root) gpus;
  make_others root;
  let got = Munin_sys.Stat.Gpu.sample_under ~root in
  equal ~msg:"GPUs" int (List.length gpus) (List.length got);
  List.iteri (fun i (g, got) -> check_gpu i g got) (List.combine gpus got)

let test_gpu_no_drm =
  test "Gpu.sample without /sys/class/drm" @@ fun () ->
  with_temp_dir @@ fun root ->
  equal int 0 (List.length (Munin_sys.Stat.Gpu.sample_under ~root))

let test_gpu =
  test "Gpu.sample" @@ fun () ->
  if Sys.win32 then raises_match Exn.sys_error Munin_sys.Stat.Gpu.sample
  else
    List.iter
      (fun (g : Munin_sys.Stat.Gpu.t) ->
        Option.iter
          (fun b -> at_most ~msg:g.id float_exact ~than:100. b)
          g.busy_percent)
      (Munin_sys.Stat.Gpu.sample ())

(* Every NVIDIA GPU of the machine is read from its RM: its memory, used
   within its total, and no temperature or power. *)
let test_gpu_nvidia =
  test "Gpu.sample reads NVIDIA's GPUs" @@ fun () ->
  let buses = Result.value (Rig_nv_nvidia.buses ()) ~default:[] in
  if buses = [] then skip ~reason:"the machine has no NVIDIA GPU" ();
  let gpus = Munin_sys.Stat.Gpu.sample () in
  let read bus =
    let g = List.find (fun (g : Munin_sys.Stat.Gpu.t) -> g.id = bus) gpus in
    equal ~msg:"temperature" (option float_exact) None g.temperature;
    equal ~msg:"power" (option float_exact) None g.power;
    match (g.memory_used, g.memory_total) with
    | Some used, Some total ->
        greater ~msg:"memory_total" int64 ~than:0L total;
        at_most ~msg:"memory_used" int64 ~than:total used
    | _ -> failf "%s: memory not read" bus
  in
  List.iter read buses

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
         group "Gpu"
           [ test_gpu_sysfs; test_gpu_no_drm; test_gpu; test_gpu_nvidia ];
       ])
