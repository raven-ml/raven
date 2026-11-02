(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type disk = Sysstat.disk = {
  name : string;
  read_bytes : int;
  written_bytes : int;
  busy_ms : int option;
}

let float = Float.of_int
let attempt f = match f () with v -> Some v | exception Sys_error _ -> None

(* Rates *)

(* Linux's aggregate iowait can step back, as it sums per-CPU counters that do;
   a negative change counts as none. *)
let cpu_busy ~(prev : Sysstat.cpu) (next : Sysstat.cpu) =
  let busy = max 0 (next.busy - prev.busy) in
  let idle = max 0 (next.idle - prev.idle) in
  if busy + idle = 0 then None
  else Some (100. *. float busy /. float (busy + idle))

let proc_stat_busy prev next =
  cpu_busy ~prev:(Sysstat.cpu_of_proc_stat prev) (Sysstat.cpu_of_proc_stat next)

let by_name = List.sort (fun a b -> String.compare a.name b.name)
let same_name a b = String.equal a.name b.name

(* A disk that attached or left during the window would bring or take with it
   all it did before, so a window over which the set of disks changed gives no
   disk rates. A disk whose counter stepped back was replaced under its name,
   which changes the set too. Linux updates a disk's busy time when a request
   starts or ends, so a window can see a little more than its length; each
   disk's percent is capped at 100. *)
let disk_rates ~dt ~prev next =
  let prev = by_name prev and next = by_name next in
  if next = [] || not (List.equal same_name prev next) then []
  else
    let delta f = List.map2 (fun p n -> f n - f p) prev next in
    let read = delta (fun d -> d.read_bytes) in
    let written = delta (fun d -> d.written_bytes) in
    let busy_delta p n =
      match (p.busy_ms, n.busy_ms) with
      | Some p, Some n -> Some (n - p)
      | _ -> None
    in
    let busy = List.filter_map Fun.id (List.map2 busy_delta prev next) in
    if List.exists (fun d -> d < 0) (read @ written @ busy) then []
    else
      let rate ds = float (List.fold_left ( + ) 0 ds) /. dt in
      let pct ms = Float.min 100. (100. *. float ms /. (dt *. 1000.)) in
      let busy =
        if List.compare_lengths busy next <> 0 then []
        else
          let sum = List.fold_left (fun acc ms -> acc +. pct ms) 0. busy in
          [ ("sys/disk_busy_pct", sum /. float (List.length busy)) ]
      in
      ("sys/disk_read_bytes_per_s", rate read)
      :: ("sys/disk_write_bytes_per_s", rate written)
      :: busy

(* The machine *)

type counters = { cpu : Sysstat.cpu option; disks : disk list option }

let memory_pairs (m : Sysstat.memory) =
  let used =
    match m.available with
    | Some a -> [ ("sys/mem_used_bytes", float (m.total - a)) ]
    | None -> []
  in
  let swap =
    match m.swap_used with
    | Some s -> [ ("sys/swap_used_bytes", float s) ]
    | None -> []
  in
  used @ (("sys/mem_total_bytes", float m.total) :: swap)

let host_pairs ~dt ~prev next memory load =
  let cpu =
    match (prev.cpu, next.cpu) with
    | Some prev, Some next -> (
        match cpu_busy ~prev next with
        | Some b -> [ ("sys/cpu_busy_pct", b) ]
        | None -> [])
    | _ -> []
  in
  let load =
    match load with Some l -> [ ("sys/load_per_core", l) ] | None -> []
  in
  let memory = match memory with Some m -> memory_pairs m | None -> [] in
  let disks =
    match (prev.disks, next.disks) with
    | Some prev, Some next -> disk_rates ~dt ~prev next
    | _ -> []
  in
  List.concat [ cpu; load; memory; disks; [ ("sys/interval_s", dt) ] ]

(* A call whose clock has not moved gives nothing and reads nothing, so the next
   window spans both calls. *)
let host () =
  let last = ref None in
  fun () ->
    let at = Sysstat.now () in
    match !last with
    | Some (t, _) when not (at -. t > 0.) -> []
    | prev -> (
        let next =
          { cpu = attempt Sysstat.cpu; disks = attempt Sysstat.disks }
        in
        let memory = attempt Sysstat.memory in
        let load = attempt Sysstat.load_per_core in
        last := Some (at, next);
        if next.cpu = None && next.disks = None && memory = None && load = None
        then raise (Sys_error "Munin_sys.host: cannot read the machine");
        match prev with
        | None -> []
        | Some (t, prev) -> host_pairs ~dt:(at -. t) ~prev next memory load)

(* This process *)

let process () =
  let last = ref None in
  fun () ->
    let at = Sysstat.now () in
    match !last with
    | Some (t, _) when not (at -. t > 0.) -> []
    | prev -> (
        let times = Unix.times () in
        let cpu = times.Unix.tms_utime +. times.Unix.tms_stime in
        let rss = attempt Sysstat.rss in
        last := Some (at, cpu);
        match prev with
        | None -> []
        | Some (t, cpu0) ->
            let dt = at -. t in
            let rss =
              match rss with
              | Some r -> [ ("sys/proc_rss_bytes", float r) ]
              | None -> []
            in
            (("sys/proc_cpu_cores", (cpu -. cpu0) /. dt) :: rss)
            @ [ ("sys/proc_interval_s", dt) ])

(* Devices *)

type device = {
  allocated : int option;
  budget : int option;
  memory : int option;
  used : int option;
  busy : float option;
  activity : float option;
  temperature : float option;
  power : float option;
}

let unknown =
  {
    allocated = None;
    budget = None;
    memory = None;
    used = None;
    busy = None;
    activity = None;
    temperature = None;
    power = None;
  }

(* A device name is one segment of a key. *)
let is_control c = Char.code c < 0x20 || Char.code c = 0x7f

let check_name name =
  if
    name = ""
    || name.[0] = '.'
    || (not (String.is_valid_utf_8 name))
    || String.exists (fun c -> c = '/' || is_control c) name
  then
    invalid_arg
      (Printf.sprintf "Munin_sys.device: %S is not a device name" name)

let device name read =
  check_name name;
  let key k = "dev/" ^ name ^ "/" ^ k in
  let bytes k v = Option.map (fun v -> (key k, float v)) v in
  let level k v = Option.map (fun v -> (key k, v)) v in
  fun () ->
    let d = read () in
    List.filter_map Fun.id
      [
        bytes "allocated_bytes" d.allocated;
        bytes "budget_bytes" d.budget;
        bytes "memory_bytes" d.memory;
        bytes "used_bytes" d.used;
        level "busy_pct" d.busy;
        level "activity_pct" d.activity;
        level "temperature_c" d.temperature;
        level "power_w" d.power;
      ]
