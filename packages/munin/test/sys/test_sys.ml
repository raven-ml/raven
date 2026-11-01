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

(* This process is in the table by its ID, named and resident, whatever user
   owns the others. *)
let test_proc_table =
  reads "Proc.Table.sample" Munin_sys.Stat.Proc.Table.sample @@ fun ps ->
  let me = Munin_sys.Stat.Proc.Self.pid () in
  let pid (p : Munin_sys.Stat.Proc.Table.t) = p.pid in
  mem int me (List.map pid ps);
  let p = List.find (fun p -> pid p = me) ps in
  not_equal ~msg:"name" string "" p.name;
  greater ~msg:"resident_size" int64 ~than:0L p.resident_size

(* A run's provenance names this process as the process table does. *)
let test_provenance_pid =
  test "Provenance.detect gives the process ID the table lists" @@ fun () ->
  let p = Provenance.detect ~cwd:(Filename.get_temp_dir_name ()) () in
  equal int (Munin_sys.Stat.Proc.Self.pid ()) p.pid

let test_loadavg =
  unread_on_windows "loadavg" Munin_sys.Stat.loadavg @@ fun (l1, l5, l15) ->
  at_least ~msg:"1 min" float_exact ~than:0. l1;
  at_least ~msg:"5 min" float_exact ~than:0. l5;
  at_least ~msg:"15 min" float_exact ~than:0. l15

(* GPUs *)

let gpus () = Munin_sys.Stat.Gpu.sample ()

(* [id] as a PCI address reads it, [domain:bus:device.function] in
   hexadecimal, or [""]: an id that is one reads back as itself. *)
let as_pci id =
  try
    Scanf.sscanf id "%x:%x:%x.%x%!" (fun domain bus dev fn ->
        Printf.sprintf "%04x:%02x:%02x.%x" domain bus dev fn)
  with Scanf.Scan_failure _ | Failure _ | End_of_file -> ""

(* The machine's amdgpu GPUs, by their DRM devices' uevent files, and its
   NVIDIA GPUs, as rig lists them. *)
let machine_gpus () =
  let drm = "/sys/class/drm" in
  let amd card =
    let path = Filename.concat drm (card ^ "/device/uevent") in
    match In_channel.with_open_bin path In_channel.input_all with
    | exception Sys_error _ -> None
    | text ->
        let lines = String.split_on_char '\n' text in
        if not (List.mem "DRIVER=amdgpu" lines) then None
        else
          List.find_map
            (fun l ->
              if String.starts_with ~prefix:"PCI_SLOT_NAME=" l then
                Some (String.sub l 14 (String.length l - 14))
              else None)
            lines
  in
  let cards =
    match Sys.readdir drm with
    | exception Sys_error _ -> []
    | names ->
        List.filter
          (fun n ->
            String.starts_with ~prefix:"card" n && not (String.contains n '-'))
          (Array.to_list names)
  in
  List.sort compare
    (List.filter_map amd cards
    @ Nvidia.buses ())

(* Every value within its unit's bounds: a percentage, a die's temperature in
   Celsius, a GPU's power in watts, memory within the GPU's. On Linux a GPU is
   named by its PCI address, and the GPUs are the machine's amdgpu and NVIDIA
   ones, in order of name. *)
let test_gpu =
  test "Gpu.sample" @@ fun () ->
  if Sys.win32 then raises_match Exn.sys_error gpus
  else begin
    let gs = gpus () in
    if Sys.file_exists "/proc" then
      equal ~msg:"ids" (list string) (machine_gpus ())
        (List.map (fun (g : Munin_sys.Stat.Gpu.t) -> g.id) gs);
    if gs = [] then skip ~reason:"the machine has no GPU munin reads" ();
    List.iter
      (fun (g : Munin_sys.Stat.Gpu.t) ->
        let within name lo hi =
          Option.iter (fun v ->
              greater ~msg:(g.id ^ " " ^ name) float_exact ~than:lo v;
              less ~msg:(g.id ^ " " ^ name) float_exact ~than:hi v)
        in
        if Sys.file_exists "/proc" then
          equal ~msg:"a PCI address" string g.id (as_pci g.id);
        Option.iter
          (fun b ->
            at_least ~msg:(g.id ^ " busy") float_exact ~than:0. b;
            at_most ~msg:(g.id ^ " busy") float_exact ~than:100. b)
          g.busy_percent;
        within "temperature" (-40.) 150. g.temperature;
        within "power" 0. 2000. g.power;
        match (g.memory_used, g.memory_total) with
        | Some used, Some total ->
            at_most ~msg:(g.id ^ " memory_used") int64 ~than:total used
        | _ -> ())
      gs
  end

(* Every NVIDIA GPU of the machine is read from its RM: its memory, used
   within its total, and no temperature or power; a GPU the RM does not
   answer for, every field [None]. *)
let test_gpu_nvidia =
  test "Gpu.sample reads NVIDIA's GPUs" @@ fun () ->
  let buses = Nvidia.buses () in
  if buses = [] then skip ~reason:"the machine has no NVIDIA GPU" ();
  let gpus = gpus () in
  let read bus =
    let g = List.find (fun (g : Munin_sys.Stat.Gpu.t) -> g.id = bus) gpus in
    let f = option float_exact in
    equal ~msg:(bus ^ " temperature") f None g.temperature;
    equal ~msg:(bus ^ " power") f None g.power;
    match (g.memory_used, g.memory_total) with
    | Some used, Some total ->
        greater ~msg:"memory_total" int64 ~than:0L total;
        at_most ~msg:"memory_used" int64 ~than:total used
    | used, total ->
        (* A GPU the RM does not answer for. *)
        equal ~msg:(bus ^ " busy_percent") f None g.busy_percent;
        equal ~msg:(bus ^ " memory_used") (option int64) None used;
        equal ~msg:(bus ^ " memory_total") (option int64) None total
  in
  List.iter read buses

(* The keys of the values a sample of the GPUs has, GPU [i] the [i]th. *)
let gpu_keys gs =
  List.concat
    (List.mapi
       (fun i (g : Munin_sys.Stat.Gpu.t) ->
         List.filter_map
           (fun (name, present) ->
             if present then Some (Metric.gpu_key i name) else None)
           [
             ("util_pct", g.busy_percent <> None);
             ("mem_used_gb", g.memory_used <> None);
             ( "mem_used_pct",
               g.memory_used <> None && g.memory_total <> None );
             ("temp_c", g.temperature <> None);
             ("power_w", g.power <> None);
           ])
       gs)
  |> List.sort compare

(* The monitor defines a GPU's metric for each value that its samples have:
   every value a sample before and one after the monitor ran both have, and
   none that neither has. Windows has no sample: [Gpu.sample] raises there. *)
let test_gpu_monitor =
  test "the monitor defines the GPU metrics its samples have" @@ fun () ->
  let gpus () = if Sys.win32 then [] else gpus () in
  let before = gpus () in
  if before = [] then skip ~reason:"the machine has no GPU munin reads" ();
  with_temp_dir @@ fun root ->
  let store = Store.open_ ~root () in
  let session = Session.start ~store ~experiment:"exp" () in
  let monitor = Munin_sys.start ~interval:0.1 session in
  Thread.delay 0.5;
  Munin_sys.stop monitor;
  Session.finish session;
  let after = gpus () in
  let defined =
    List.filter
      (String.starts_with ~prefix:"sys/gpu")
      (List.map fst (Run.metric_defs (Session.run session)))
    |> List.sort_uniq compare
  in
  let b = gpu_keys before and a = gpu_keys after in
  List.iter
    (fun k -> if List.mem k a then mem ~msg:"defined" string k defined)
    b;
  List.iter (fun k -> mem ~msg:"in a sample" string k (b @ a)) defined

let test_uptime =
  reads "uptime" Munin_sys.Stat.uptime @@ fun s ->
  greater ~msg:"seconds" int64 ~than:0L s

(* An interval that is not positive raises before the monitor declares any
   metric. *)
let test_system_monitor_bad_interval () =
  with_temp_dir @@ fun root ->
  let store = Store.open_ ~root () in
  let session = Session.start ~store ~experiment:"exp" () in
  List.iter
    (fun interval ->
      raises_match Exn.invalid_arg (fun () ->
          Munin_sys.start ~interval session))
    [ 0.; -1.; Float.nan ];
  (* The interval prints as the shortest decimal that reads back, the same on
     every system: the C library's [%g] wrote 1e+06, 1e+006 under
     mingw-w64. *)
  raises
    (Invalid_argument "Munin_sys.start: interval -1000000 is not positive")
    (fun () -> Munin_sys.start ~interval:(-1e6) session);
  raises
    (Invalid_argument "Munin_sys.start: interval -4.58e-72 is not positive")
    (fun () -> Munin_sys.start ~interval:(-4.58e-72) session);
  Session.finish session;
  let defs = Run.metric_defs (Session.run session) in
  equal (list string) [] (sys_only (List.map fst defs))

let system_monitor_tests =
  [
    test "an interval that is not positive raises"
      test_system_monitor_bad_interval;
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
             test_provenance_pid;
             test_loadavg;
             test_uptime;
           ];
         group "Gpu"
           [
             test_gpu;
             test_gpu_nvidia;
             test_gpu_monitor;
           ];
       ])
