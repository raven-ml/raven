(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

let pairs = list (pair string float_exact)
let keys pairs = List.map fst pairs
let value key pairs = List.assoc key pairs
let linux = Sys.file_exists "/proc/self/status"
let macos = Sys.os_type = "Unix" && not linux

let in_range ~low ~high key pairs =
  let v = value key pairs in
  at_least ~msg:key float_exact ~than:low v;
  at_most ~msg:key float_exact ~than:high v

let non_negative key pairs =
  at_least ~msg:key float_exact ~than:0. (value key pairs)

(* The machine *)

let host_keys =
  [
    "sys/cpu_busy_pct";
    "sys/load_per_core";
    "sys/mem_used_bytes";
    "sys/mem_total_bytes";
    "sys/swap_used_bytes";
    "sys/disk_read_bytes_per_s";
    "sys/disk_write_bytes_per_s";
    "sys/disk_busy_pct";
    "sys/interval_s";
  ]

(* What every supported system reads. Windows keeps no load average; a Linux
   machine or a Windows VM may have no physical disk that gives counters. *)
let host_required =
  [
    "sys/cpu_busy_pct";
    "sys/mem_used_bytes";
    "sys/mem_total_bytes";
    "sys/swap_used_bytes";
    "sys/interval_s";
  ]
  @ (if Sys.win32 then [] else [ "sys/load_per_core" ])
  @
  if macos then [ "sys/disk_read_bytes_per_s"; "sys/disk_write_bytes_per_s" ]
  else []

let window = 0.05

let test_host_first () =
  let read = Munin_sys.host () in
  equal pairs [] (read ())

let test_host_second () =
  let read = Munin_sys.host () in
  ignore (read ());
  Unix.sleepf window;
  let got = read () in
  let ks = keys got in
  equal (list string) (List.sort_uniq compare ks) (List.sort compare ks);
  List.iter (fun k -> mem string k host_keys) ks;
  List.iter (fun k -> mem string k ks) host_required;
  in_range ~low:0. ~high:100. "sys/cpu_busy_pct" got;
  if List.mem_assoc "sys/load_per_core" got then
    non_negative "sys/load_per_core" got;
  in_range ~low:0.
    ~high:(value "sys/mem_total_bytes" got)
    "sys/mem_used_bytes" got;
  non_negative "sys/swap_used_bytes" got;
  let disks = List.mem_assoc "sys/disk_read_bytes_per_s" got in
  equal bool disks (List.mem_assoc "sys/disk_write_bytes_per_s" got);
  equal bool (linux && disks) (List.mem_assoc "sys/disk_busy_pct" got);
  if disks then begin
    non_negative "sys/disk_read_bytes_per_s" got;
    non_negative "sys/disk_write_bytes_per_s" got
  end;
  if List.mem_assoc "sys/disk_busy_pct" got then
    in_range ~low:0. ~high:100. "sys/disk_busy_pct" got;
  at_least float_exact ~than:window (value "sys/interval_s" got)

(* This process *)

let test_process_first () =
  let read = Munin_sys.process () in
  equal pairs [] (read ())

let cpu_seconds () =
  let t = Unix.times () in
  t.Unix.tms_utime +. t.Unix.tms_stime

(* Computes until the process spent [s] seconds of CPU. *)
let burn s =
  let start = cpu_seconds () in
  while cpu_seconds () -. start < s do
    ignore (Sys.opaque_identity (Array.make 64 0))
  done

let block_bytes = 32 * 1024 * 1024

let test_process_second () =
  let read = Munin_sys.process () in
  ignore (read ());
  (* Written, so resident. *)
  let block = Bytes.make block_bytes 'x' in
  burn 0.02;
  let got = read () in
  ignore (Sys.opaque_identity block);
  equal (list string)
    [ "sys/proc_cpu_cores"; "sys/proc_rss_bytes"; "sys/proc_interval_s" ]
    (keys got);
  greater float_exact ~than:0. (value "sys/proc_cpu_cores" got);
  at_least float_exact ~than:(Float.of_int block_bytes)
    (value "sys/proc_rss_bytes" got);
  greater float_exact ~than:0. (value "sys/proc_interval_s" got)

(* CPU busy on Linux *)

(* A /proc/stat text whose aggregate line holds [fields]: user, nice, system,
   idle, iowait, irq, softirq, steal, guest, guest_nice, or a prefix. *)
let proc_stat fields =
  String.concat "\n"
    [
      "cpu  " ^ String.concat " " (List.map string_of_int fields);
      "cpu0 1 2 3 4 5 6 7 8 9 10";
      "intr 12345 0 0";
      "";
    ]

let zero = proc_stat [ 0; 0; 0; 0; 0; 0; 0; 0; 0; 0 ]
let busy = option (float 1e-9)

(* Linux counts guest time in user (guest_nice in nice) as well: a window half
   running a virtual machine and half idle is 50% busy. *)
let busy_cases =
  [
    ("guest once in user", [ 50; 0; 0; 50; 0; 0; 0; 0; 50; 0 ], Some 50.);
    ("guest_nice once in nice", [ 0; 50; 0; 50; 0; 0; 0; 0; 0; 50 ], Some 50.);
    ("iowait is not busy", [ 25; 0; 0; 25; 50; 0; 0; 0; 0; 0 ], Some 25.);
    ("irq, softirq, steal", [ 0; 0; 0; 25; 0; 25; 25; 25 ], Some 75.);
    ("four fields", [ 10; 0; 10; 20 ], Some 50.);
    ("no time", [ 0; 0; 0; 0; 0; 0; 0; 0; 0; 0 ], None);
  ]

let test_busy_cases =
  cases "/proc/stat windows"
    ~id:(fun (n, _, _) -> n)
    busy_cases
    (fun (_, next, expected) ->
      equal busy expected (Munin_sys.proc_stat_busy zero (proc_stat next)))

(* Ten counters with guest <= user and guest_nice <= nice. *)
let counters =
  let open Gen in
  let+ base = list ~size:(constant 8) (int_range 0 1_000_000)
  and+ guest = int_range 0 100
  and+ guest_nice = int_range 0 100 in
  match base with
  | user :: nice :: _ -> base @ [ user * guest / 100; nice * guest_nice / 100 ]
  | _ -> base

let add = List.map2 ( + )

let test_guest_ignored =
  prop "guest time leaves busy unchanged"
    Gen.(pair counters counters)
    (fun (prev, delta) ->
      let f (prev, next) =
        Munin_sys.proc_stat_busy (proc_stat prev) (proc_stat next)
      in
      (* All of the window's user and nice time ran virtual machines. *)
      let all_guest (prev, next) =
        let guest l =
          List.mapi
            (fun i v ->
              match i with 8 -> List.nth l 0 | 9 -> List.nth l 1 | _ -> v)
            l
        in
        (guest prev, guest next)
      in
      Law.ignores
        (pair (list int) (list int))
        busy f all_guest
        (prev, add prev delta))

(* Disk rates *)

let disk name (read_bytes, written_bytes, busy_ms) =
  { Munin_sys.name; read_bytes; written_bytes; busy_ms }

let counter = Gen.int_range 0 (1 lsl 40)
let change = Gen.int_range 0 (1 lsl 30)

(* Disks d0, d1, ... and their counters at the start and end of a window. *)
let window_gen =
  let open Gen in
  let+ dt = float_range 0.001 100.
  and+ linux = bool
  and+ ds =
    list ~size:(int_range 1 4)
      (pair (triple counter counter counter) (triple change change change))
  in
  let busy b = if linux then Some b else None in
  let prev, next =
    List.split
      (List.mapi
         (fun i ((r, w, b), (dr, dw, db)) ->
           let name = "d" ^ string_of_int i in
           (disk name (r, w, busy b), disk name (r + dr, w + dw, busy (b + db))))
         ds)
  in
  (dt, prev, next)

let rate = float_rel ~rel:1e-12 ~abs:0.
let sum f l = List.fold_left (fun acc d -> acc + f d) 0 l

let test_disk_sums =
  prop "disk rates sum each disk's change" window_gen (fun (dt, prev, next) ->
      cover "busy" (List.for_all (fun d -> d.Munin_sys.busy_ms <> None) next);
      cover "several disks" (List.length next > 1);
      (* The order of the disks does not matter. *)
      let got = Munin_sys.disk_rates ~dt ~prev (List.rev next) in
      let per_s f = Float.of_int (sum f next - sum f prev) /. dt in
      equal rate
        (per_s (fun d -> d.read_bytes))
        (value "sys/disk_read_bytes_per_s" got);
      equal rate
        (per_s (fun d -> d.written_bytes))
        (value "sys/disk_write_bytes_per_s" got);
      match (List.hd next).busy_ms with
      | None -> equal bool false (List.mem_assoc "sys/disk_busy_pct" got)
      | Some _ -> in_range ~low:0. ~high:100. "sys/disk_busy_pct" got)

let test_disk_set_changes =
  prop "a disk that appears or leaves gives no disk rates"
    Gen.(pair window_gen bool)
    (fun ((dt, prev, next), appears) ->
      cover "appears" appears;
      cover "leaves" (not appears);
      let extra = disk "new" (0, 0, None) in
      let got =
        if appears then Munin_sys.disk_rates ~dt ~prev (extra :: next)
        else Munin_sys.disk_rates ~dt ~prev (List.tl next)
      in
      equal pairs [] got)

let disk_cases =
  let d name r w b = disk name (r, w, b) in
  [
    ( "a disk replaced under another name",
      [ d "a" 0 0 None ],
      [ d "b" 10 10 None ],
      [] );
    ("a counter that steps back", [ d "a" 100 0 None ], [ d "a" 50 0 None ], []);
    ("no disk", [], [], []);
    ( "one disk busy all window, one idle",
      [ d "a" 0 0 (Some 0); d "b" 0 0 (Some 0) ],
      [ d "a" 0 0 (Some 2000); d "b" 0 0 (Some 0) ],
      [
        ("sys/disk_read_bytes_per_s", 0.);
        ("sys/disk_write_bytes_per_s", 0.);
        ("sys/disk_busy_pct", 50.);
      ] );
    ( "busy past the window counts as all of it",
      [ d "a" 0 0 (Some 0) ],
      [ d "a" 0 0 (Some 2010) ],
      [
        ("sys/disk_read_bytes_per_s", 0.);
        ("sys/disk_write_bytes_per_s", 0.);
        ("sys/disk_busy_pct", 100.);
      ] );
    ( "bytes per second",
      [ d "a" 1000 0 None ],
      [ d "a" 5000 2000 None ],
      [
        ("sys/disk_read_bytes_per_s", 2000.);
        ("sys/disk_write_bytes_per_s", 1000.);
      ] );
  ]

let test_disk_cases =
  cases "disk windows of 2 s"
    ~id:(fun (n, _, _, _) -> n)
    disk_cases
    (fun (_, prev, next, expected) ->
      equal pairs expected (Munin_sys.disk_rates ~dt:2. ~prev next))

(* Devices *)

let device_gen =
  let open Gen in
  let size = option nat and level = option float in
  let+ allocated = size
  and+ budget = size
  and+ memory = size
  and+ used = size
  and+ busy = level
  and+ activity = level
  and+ temperature = level
  and+ power = level in
  {
    Munin_sys.allocated;
    budget;
    memory;
    used;
    busy;
    activity;
    temperature;
    power;
  }

let test_device_fields =
  prop "device gives its Some fields" device_gen (fun d ->
      let size k v =
        Option.map (fun v -> ("dev/gpu0/" ^ k, Float.of_int v)) v
      in
      let level k v = Option.map (fun v -> ("dev/gpu0/" ^ k, v)) v in
      let expected =
        List.filter_map Fun.id
          [
            size "allocated_bytes" d.allocated;
            size "budget_bytes" d.budget;
            size "memory_bytes" d.memory;
            size "used_bytes" d.used;
            level "busy_pct" d.busy;
            level "activity_pct" d.activity;
            level "temperature_c" d.temperature;
            level "power_w" d.power;
          ]
      in
      equal pairs expected (Munin_sys.device "gpu0" (fun () -> d) ()))

let name_char =
  Gen.of_list
    ~pp:(fun ppf c -> Format.fprintf ppf "%C" c)
    [
      'a';
      'Z';
      '0';
      '-';
      ':';
      ' ';
      '.';
      '/';
      '\x00';
      '\x1f';
      '\x7f';
      '\xc3';
      '\xa9';
    ]

let is_control c = Char.code c < 0x20 || Char.code c = 0x7f

let full =
  {
    Munin_sys.allocated = Some 1;
    budget = Some 2;
    memory = Some 3;
    used = Some 4;
    busy = Some 5.;
    activity = Some 6.;
    temperature = Some 7.;
    power = Some 8.;
  }

let test_device_names =
  prop ~count:500 "device validates its name"
    Gen.(string_of ~size:(int_range 0 5) name_char)
    (fun name ->
      let empty = name = "" in
      let dot = (not empty) && name.[0] = '.' in
      let slash = String.contains name '/' in
      let control = String.exists is_control name in
      let not_utf8 = not (String.is_valid_utf_8 name) in
      cover "empty" empty;
      cover "starts with ." dot;
      cover "holds /" slash;
      cover "holds a control character" control;
      cover "not UTF-8" not_utf8;
      let invalid = empty || dot || slash || control || not_utf8 in
      cover "valid" (not invalid);
      if invalid then
        raises_match Exn.invalid_arg (fun () ->
            Munin_sys.device name (fun () -> full))
      else begin
        let prefix = "dev/" ^ name ^ "/" in
        let got = Munin_sys.device name (fun () -> full) () in
        equal int 8 (List.length got);
        List.iter (fun (k, _) -> starts_with ~prefix k) got
      end)

let test_device_unknown () =
  equal pairs [] (Munin_sys.device "gpu0" (fun () -> Munin_sys.unknown) ())

let test_device_raises () =
  let read () = raise (Sys_error "lost") in
  raises (Sys_error "lost") (Munin_sys.device "gpu0" read)

let () =
  exit
    (run "Munin_sys"
       [
         group "host"
           [
             test "first call gives nothing" test_host_first;
             test "second call" test_host_second;
           ];
         group "process"
           [
             test "first call gives nothing" test_process_first;
             test "second call" test_process_second;
           ];
         group "cpu busy" [ test_busy_cases; test_guest_ignored ];
         group "disk rates"
           [ test_disk_sums; test_disk_set_changes; test_disk_cases ];
         group "device"
           [
             test_device_fields;
             test_device_names;
             test "unknown gives nothing" test_device_unknown;
             test "raises what read raises" test_device_raises;
           ];
       ])
