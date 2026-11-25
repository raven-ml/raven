(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The readings read the live machine, whose values no test can know: they are
   held to their keys and ranges. The formulas of the rates are held to their
   values through the (**/**) section. *)

open Windtrap

let pairs = list (pair string float_exact)
let keys pairs = List.sort String.compare (List.map fst pairs)
let value key pairs = List.assoc key pairs
let by_key pairs = List.sort (fun (a, _) (b, _) -> String.compare a b) pairs

(* The kernel's name, as [uname -s] prints it. *)
let kernel =
  lazy
    (if Sys.win32 then "Windows"
     else
       let ic = Unix.open_process_args_in "uname" [| "uname"; "-s" |] in
       let name = input_line ic in
       match Unix.close_process_in ic with
       | WEXITED 0 -> name
       | _ -> failwith "uname -s failed")

let linux () = Lazy.force kernel = "Linux"
let macos () = Lazy.force kernel = "Darwin"

(* Linux, macOS and Windows: the systems munin.sys reads beyond the load
   average. *)
let read_whole () = linux () || macos () || Sys.win32

let in_range ~low ~high key pairs =
  let v = value key pairs in
  at_least ~msg:key float_exact ~than:low v;
  at_most ~msg:key float_exact ~than:high v

let non_negative key pairs =
  at_least ~msg:key float_exact ~than:0. (value key pairs)

let cpu_seconds () =
  let t = Unix.times () in
  t.Unix.tms_utime +. t.Unix.tms_stime

(* [burn s] computes until the process spent more than [s] seconds of CPU. *)
let burn s =
  let start = cpu_seconds () in
  while cpu_seconds () -. start <= s do
    ignore (Sys.opaque_identity (Array.make 64 0))
  done

(* Two hundredths of a second: Linux's /proc/stat counts CPU time in hundredths,
   so a window in which this process computed that long counts some, whatever
   the kernel's own tick. *)
let ticks = 0.02

(* [window read] calls the new reading [read] twice, computing [ticks] seconds
   between the calls, and is the second call's pairs with the CPU seconds the
   process spent between them. *)
let window read =
  ignore (read ());
  let a = cpu_seconds () in
  burn ticks;
  let b = cpu_seconds () in
  (read (), b -. a)

(* This process computes on one thread, so the CPU seconds it spent between the
   calls cannot pass their wall time, which the window spans. [Unix.times]
   rounds each reading to the microsecond. *)
let at_least_cpu key (got, spent) =
  at_least ~msg:key float_exact ~than:(spent -. 2e-6) (value key got)

(* Readings *)

let rate_readings = [ ("host", Munin_sys.host); ("process", Munin_sys.process) ]

let test_first_call =
  cases "a first call gives nothing" ~id:fst rate_readings (fun (_, reading) ->
      equal pairs [] (reading () ()))

let test_own_window =
  cases "a new reading starts its own window" ~id:fst rate_readings
    (fun (_, reading) ->
      let a = reading () in
      ignore (a ());
      let b = reading () in
      equal pairs [] (b ()))

(* The machine *)

(* The keys of a system munin.sys reads whole. Windows keeps no load average; a
   Linux machine or a Windows VM may have no physical disk that gives counters,
   and macOS always has one. *)
let host_keys ~disks =
  let disk_keys =
    if not disks then []
    else
      [ "sys/disk_read_bytes_per_s"; "sys/disk_write_bytes_per_s" ]
      @ if linux () then [ "sys/disk_busy_pct" ] else []
  in
  List.sort String.compare
    ([
       "sys/cpu_busy_pct";
       "sys/mem_used_bytes";
       "sys/mem_total_bytes";
       "sys/swap_used_bytes";
       "sys/interval_s";
     ]
    @ (if Sys.win32 then [] else [ "sys/load_per_core" ])
    @ disk_keys)

let test_host_keys () =
  let got, _ = window (Munin_sys.host ()) in
  if not (read_whole ()) then
    equal (list string) [ "sys/interval_s"; "sys/load_per_core" ] (keys got)
  else
    let disks = macos () || List.mem_assoc "sys/disk_read_bytes_per_s" got in
    equal (list string) (host_keys ~disks) (keys got)

let test_host_ranges () =
  let got, _ = window (Munin_sys.host ()) in
  let have k = List.mem_assoc k got in
  if have "sys/load_per_core" then non_negative "sys/load_per_core" got;
  if read_whole () then begin
    in_range ~low:0. ~high:100. "sys/cpu_busy_pct" got;
    greater ~msg:"sys/mem_total_bytes" float_exact ~than:0.
      (value "sys/mem_total_bytes" got);
    in_range ~low:0.
      ~high:(value "sys/mem_total_bytes" got)
      "sys/mem_used_bytes" got;
    non_negative "sys/swap_used_bytes" got
  end;
  if have "sys/disk_read_bytes_per_s" then begin
    non_negative "sys/disk_read_bytes_per_s" got;
    non_negative "sys/disk_write_bytes_per_s" got
  end;
  if have "sys/disk_busy_pct" then
    in_range ~low:0. ~high:100. "sys/disk_busy_pct" got

let test_host_interval () =
  at_least_cpu "sys/interval_s" (window (Munin_sys.host ()))

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

(* From the key's sentence and proc(5)'s fields. Linux counts guest time in user
   (guest_nice in nice) as well: a window half running a virtual machine and
   half idle is 50% busy. A window that counted no time gives no percent, and
   kernels before 2.6.11 write four fields. *)
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
  cases "sys/cpu_busy_pct over a /proc/stat window is the stated percent"
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
  prop "sys/cpu_busy_pct is the same whatever guest time ran"
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

let test_busy_percent =
  prop "sys/cpu_busy_pct is a percent for any two texts"
    Gen.(pair counters counters)
    (fun (prev, next) ->
      cover "a counter stepped back" (List.exists2 ( > ) prev next);
      satisfies ~claim:"is none or from 0 to 100" busy
        (function None -> true | Some b -> b >= 0. && b <= 100.)
        (Munin_sys.proc_stat_busy (proc_stat prev) (proc_stat next)))

let disk name (read_bytes, written_bytes, busy_ms) =
  { Munin_sys.name; read_bytes; written_bytes; busy_ms }

let counter = Gen.int_range 0 (1 lsl 40)
let change = Gen.int_range 0 (1 lsl 30)

(* Disks d0, d1, ... and their counters at the start and end of a window of [dt]
   seconds, busy for up to twice the window, as Linux can count. *)
let window_gen =
  let open Gen in
  let* dt = float_range 0.001 100. in
  let+ linux = bool
  and+ ds =
    list ~size:(int_range 1 4)
      (pair
         (triple counter counter counter)
         (triple change change (float_range 0. 2.)))
  in
  let busy b = if linux then Some b else None in
  let prev, next =
    List.split
      (List.mapi
         (fun i ((r, w, b), (dr, dw, share)) ->
           let name = "d" ^ string_of_int i in
           let db = Float.to_int (share *. dt *. 1000.) in
           (disk name (r, w, busy b), disk name (r + dr, w + dw, busy (b + db))))
         ds)
  in
  (dt, prev, next)

let rate = float_rel ~rel:1e-12 ~abs:0.
let sum f l = List.fold_left (fun acc d -> acc + f d) 0 l

(* The pairs of the key's sentence: each sum of changes over the window, and the
   mean over the disks of each one's busy percent, capped at 100. *)
let disk_reference ~dt prev next =
  let per_s f = Float.of_int (sum f next - sum f prev) /. dt in
  let pct p n =
    match (p.Munin_sys.busy_ms, n.Munin_sys.busy_ms) with
    | Some p, Some n ->
        Float.min 100. (100. *. Float.of_int (n - p) /. (dt *. 1000.))
    | _ -> Float.nan
  in
  let busy =
    match (List.hd next).Munin_sys.busy_ms with
    | None -> []
    | Some _ ->
        let pcts = List.map2 pct prev next in
        [
          ( "sys/disk_busy_pct",
            List.fold_left ( +. ) 0. pcts /. Float.of_int (List.length pcts) );
        ]
  in
  ("sys/disk_read_bytes_per_s", per_s (fun d -> d.read_bytes))
  :: ("sys/disk_write_bytes_per_s", per_s (fun d -> d.written_bytes))
  :: busy

let test_disk_rates =
  prop ~count:300
    "disk rates sum each disk's change, busy its mean capped percent" window_gen
    (fun (dt, prev, next) ->
      let capped d p =
        match (d.Munin_sys.busy_ms, p.Munin_sys.busy_ms) with
        | Some n, Some p -> Float.of_int (n - p) > dt *. 1000.
        | _ -> false
      in
      let linux = (List.hd next).Munin_sys.busy_ms <> None in
      cover "a disk busy past the window"
        (linux && List.exists2 capped next prev);
      cover "a disk busy within the window"
        (linux && List.exists2 (fun n p -> not (capped n p)) next prev);
      cover "several disks" (List.length next > 1);
      (* The order of the disks does not matter. *)
      let got = Munin_sys.disk_rates ~dt ~prev (List.rev next) in
      equal
        (list (pair string rate))
        (by_key (disk_reference ~dt prev next))
        (by_key got))

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

(* From the keys' sentences, over a window of 2 s. *)
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
  cases "disk rates over a window are the stated pairs"
    ~id:(fun (n, _, _, _) -> n)
    disk_cases
    (fun (_, prev, next, expected) ->
      equal pairs (by_key expected)
        (by_key (Munin_sys.disk_rates ~dt:2. ~prev next)))

(* This process *)

let test_process_keys () =
  let got, _ = window (Munin_sys.process ()) in
  let rss = if read_whole () then [ "sys/proc_rss_bytes" ] else [] in
  equal (list string)
    (List.sort String.compare
       ([ "sys/proc_cpu_cores"; "sys/proc_interval_s" ] @ rss))
    (keys got)

(* The reading takes the process's CPU time once in each call: between the
   readings [t0] and [a] around the first call, and [b] and [t1] around the
   second. So the CPU seconds of its window, cores times interval, lie from [b -
   a] to [t1 - t0]. The product of a quotient and its divisor is within two
   units in the last place of the CPU seconds, well inside 1e-12. *)
let test_process_cores () =
  let read = Munin_sys.process () in
  let t0 = cpu_seconds () in
  ignore (read ());
  let a = cpu_seconds () in
  burn 0.;
  let b = cpu_seconds () in
  let got = read () in
  let t1 = cpu_seconds () in
  let spent =
    value "sys/proc_cpu_cores" got *. value "sys/proc_interval_s" got
  in
  at_least ~msg:"cores times interval" float_exact
    ~than:((b -. a) *. (1. -. 1e-12))
    spent;
  at_most ~msg:"cores times interval" float_exact
    ~than:((t1 -. t0) *. (1. +. 1e-12))
    spent

let test_process_interval () =
  at_least_cpu "sys/proc_interval_s" (window (Munin_sys.process ()))

let block_bytes = 32 * 1024 * 1024

let test_process_rss () =
  if not (read_whole ()) then
    skip ~reason:"this system gives no resident memory" ();
  let read = Munin_sys.process () in
  ignore (read ());
  (* Written, so resident. *)
  let block = Bytes.make block_bytes 'x' in
  burn 0.;
  let got = read () in
  ignore (Sys.opaque_identity block);
  at_least float_exact ~than:(Float.of_int block_bytes)
    (value "sys/proc_rss_bytes" got)

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
  prop "device gives its Some fields under their keys" device_gen (fun d ->
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
      equal pairs (by_key expected)
        (by_key (Munin_sys.device "gpu0" (fun () -> d) ())))

let test_device_unknown () =
  equal pairs [] (Munin_sys.device "gpu0" (fun () -> Munin_sys.unknown) ())

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

let field_keys =
  [
    "activity_pct";
    "allocated_bytes";
    "budget_bytes";
    "busy_pct";
    "memory_bytes";
    "power_w";
    "temperature_c";
    "used_bytes";
  ]

let test_device_names =
  prop ~count:500 "device refuses a name that is not one key segment"
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
      else
        equal (list string)
          (List.map (fun k -> "dev/" ^ name ^ "/" ^ k) field_keys)
          (keys (Munin_sys.device name (fun () -> full) ())))

let test_device_raises () =
  let read () = raise (Sys_error "lost") in
  raises (Sys_error "lost") (Munin_sys.device "gpu0" read)

let () =
  exit
    (run "Munin_sys"
       [
         group "readings" [ test_first_call; test_own_window ];
         group "host"
           [
             test "a second call gives the system's keys" test_host_keys;
             test "its levels and rates lie in their ranges" test_host_ranges;
             test "sys/interval_s spans the CPU time spent within it"
               test_host_interval;
             test_busy_cases;
             test_guest_ignored;
             test_busy_percent;
             test_disk_rates;
             test_disk_set_changes;
             test_disk_cases;
           ];
         group "process"
           [
             test "a second call gives the process's keys" test_process_keys;
             test "proc_cpu_cores times the window is the CPU time it spent"
               test_process_cores;
             test "proc_interval_s spans the CPU time spent within it"
               test_process_interval;
             test "proc_rss_bytes counts memory the process wrote"
               test_process_rss;
           ];
         group "device"
           [
             test_device_fields;
             test "unknown gives nothing" test_device_unknown;
             test_device_names;
             test "raises what read raises" test_device_raises;
           ];
       ])
