(** System metrics collection library. Implementation for sysstat.mli.

    Design:
    - Stateless sampling: callers store previous samples and provide dt.
    - Linux: read from /proc in OCaml.
    - macOS: use C stubs for kernel / libproc / IOKit access. *)

(* ---------- C Binding Types ---------- *)

type raw_memory = {
  total : int64;
  page_size : int64;
  active : int64;
  inactive : int64;
  speculative : int64;
  wired : int64;
  compressor : int64;
  purgeable : int64;
  external_ : int64;
  free : int64;
  swap_total : int64;
  swap_used : int64;
}

type raw_network_io = {
  bytes_rx : int64;
  packets_rx : int64;
  bytes_tx : int64;
  packets_tx : int64;
}

type raw_disk_io = {
  bytes_read : int64;
  bytes_written : int64;
  time_ms : int64;
  num_disks : int64;
}

type raw_proc_info = {
  name : string;
  ppid : int;
  state : char;
  priority : int;
  nice : int;
  cmdline : string;
  user : string;
  user_time : int64;
  system_time : int64;
  resident_size : int64;
  virtual_size : int64;
  num_threads : int;
  num_running : int;
  faults : int64;
}

type raw_statvfs = { total : int64; free : int64; avail : int64 }
type raw_self_mem = { rss : int64; vsize : int64 }
type raw_timebase = { numer : int; denom : int }

(* ---------- C Bindings ---------- *)

external c_clk_tck : unit -> int = "caml_sysstat_clk_tck"
external c_getpagesize : unit -> int = "caml_sysstat_getpagesize"

external c_get_cpu_load : unit -> int64 array array
  = "caml_sysstat_get_cpu_load"

external c_get_memory : unit -> raw_memory = "caml_sysstat_get_memory"

external c_get_network_io : unit -> raw_network_io
  = "caml_sysstat_get_network_io"

external c_get_disk_io : unit -> raw_disk_io = "caml_sysstat_get_disk_io"
external c_list_pids : unit -> int array = "caml_sysstat_list_pids"

external c_get_proc_info : int -> raw_proc_info option
  = "caml_sysstat_get_proc_info"

external c_statvfs : string -> raw_statvfs = "caml_sysstat_statvfs"
external c_proc_self_mem : unit -> raw_self_mem = "caml_sysstat_proc_self_mem"
external c_get_timebase : unit -> raw_timebase = "caml_sysstat_get_timebase"

external c_getmounts : unit -> (string * string * string) array
  = "caml_sysstat_getmounts"

external c_get_loadavg : unit -> float * float * float
  = "caml_sysstat_get_loadavg"

external c_get_uptime : unit -> int64 = "caml_sysstat_get_uptime"

(* ---------- Small, boring helpers ---------- *)

let is_whitespace = function ' ' | '\t' | '\r' | '\n' -> true | _ -> false

let split_whitespace (s : string) : string list =
  let len = String.length s in
  let rec skip i = if i < len && is_whitespace s.[i] then skip (i + 1) else i in
  let rec take i j =
    if j < len && not (is_whitespace s.[j]) then take i (j + 1) else j
  in
  let rec loop i acc =
    let i = skip i in
    if i >= len then List.rev acc
    else
      let j = take i i in
      loop j (String.sub s i (j - i) :: acc)
  in
  loop 0 []

let with_in_file path f =
  let ic = open_in path in
  match f ic with
  | v ->
      close_in_noerr ic;
      v
  | exception e ->
      close_in_noerr ic;
      raise e

let fold_lines path ~init ~f =
  with_in_file path (fun ic ->
      let rec loop acc =
        match input_line ic with
        | line -> loop (f acc line)
        | exception End_of_file -> acc
      in
      loop init)

let int64_of_string_opt s = try Some (Int64.of_string s) with _ -> None
let int_of_string_opt s = try Some (int_of_string s) with _ -> None
let i64_max0 x = if Int64.compare x 0L < 0 then 0L else x

let i64_delta ~prev ~next =
  let d = Int64.sub next prev in
  if Int64.compare d 0L < 0 then 0L else d

let f_delta_i64 ~prev ~next = Int64.to_float (i64_delta ~prev ~next)
let page_size_bytes : int64 Lazy.t = lazy (Int64.of_int (c_getpagesize ()))
let page_size () = Lazy.force page_size_bytes

(* ---------- Time conversion helpers ---------- *)

let clk_tck : int Lazy.t = lazy (c_clk_tck ())

let jiffies_to_ns (jiffies : int64) : int64 =
  let hz = Int64.of_int (Lazy.force clk_tck) in
  if Int64.compare hz 0L <= 0 then 0L
  else
    (* Avoid overflow: (q*1e9) + (r*1e9)/hz where jiffies = q*hz + r. *)
    let q = Int64.div jiffies hz in
    let r = Int64.rem jiffies hz in
    Int64.add
      (Int64.mul q 1_000_000_000L)
      (Int64.div (Int64.mul r 1_000_000_000L) hz)

let mach_timebase : (float * float) Lazy.t =
  lazy
    (let tb = c_get_timebase () in
     (float_of_int tb.numer, float_of_int tb.denom))

let mach_ticks_to_ns (ticks : int64) : float =
  let numer, denom = Lazy.force mach_timebase in
  Int64.to_float ticks *. numer /. denom

let proc_ticks_to_ns (ticks : int64) : float =
  if Sys.file_exists "/proc" then Int64.to_float (jiffies_to_ns ticks)
  else mach_ticks_to_ns ticks

(* ---------- CPU Module ---------- *)

module Cpu = struct
  type t = {
    user : int64;
    nice : int64;
    system : int64;
    idle : int64;
    iowait : int64;
    irq : int64;
    softirq : int64;
    steal : int64;
    guest : int64;
  }

  type stats = {
    user : float;
    nice : float;
    system : float;
    idle : float;
    iowait : float;
    irq : float;
    softirq : float;
    steal : float;
    guest : float;
  }

  let zero : t =
    {
      user = 0L;
      nice = 0L;
      system = 0L;
      idle = 0L;
      iowait = 0L;
      irq = 0L;
      softirq = 0L;
      steal = 0L;
      guest = 0L;
    }

  let zero_stats =
    {
      user = 0.0;
      nice = 0.0;
      system = 0.0;
      idle = 100.0;
      iowait = 0.0;
      irq = 0.0;
      softirq = 0.0;
      steal = 0.0;
      guest = 0.0;
    }

  let of_linux_fields (fields : string list) : t option =
    match fields with
    | _name :: user :: nice :: system :: idle :: rest -> (
        match
          ( int64_of_string_opt user,
            int64_of_string_opt nice,
            int64_of_string_opt system,
            int64_of_string_opt idle )
        with
        | Some user, Some nice, Some system, Some idle ->
            let get idx =
              match List.nth_opt rest idx with
              | None -> 0L
              | Some s -> Option.value (int64_of_string_opt s) ~default:0L
            in
            Some
              {
                user;
                nice;
                system;
                idle;
                iowait = get 0;
                irq = get 1;
                softirq = get 2;
                steal = get 3;
                guest = get 4;
              }
        | _ -> None)
    | _ -> None

  let of_macos_row (row : int64 array) : t option =
    if Array.length row < 4 then None
    else
      Some
        {
          user = row.(0);
          nice = row.(1);
          system = row.(2);
          idle = row.(3);
          iowait = 0L;
          irq = 0L;
          softirq = 0L;
          steal = 0L;
          guest = 0L;
        }

  let is_cpu_core_name (name : string) =
    (* "cpu0", "cpu1", ... *)
    String.length name >= 4
    && String.starts_with ~prefix:"cpu" name
    && match name.[3] with '0' .. '9' -> true | _ -> false

  let read_linux () : (t * t list) option =
    let path = "/proc/stat" in
    if not (Sys.file_exists path) then None
    else
      let total = ref None in
      let cores = ref [] in
      let handle_line () line =
        match split_whitespace line with
        | [] -> ()
        | name :: _ as fields ->
            if name = "cpu" then total := of_linux_fields fields
            else if is_cpu_core_name name then
              match of_linux_fields fields with
              | Some t -> cores := t :: !cores
              | None -> ()
            else ()
      in
      (try
         fold_lines path ~init:() ~f:(fun () line ->
             handle_line () line;
             ())
       with _ -> ());
      match !total with None -> None | Some t -> Some (t, List.rev !cores)

  let read_macos () : (t * t list) option =
    try
      let rows = c_get_cpu_load () in
      if Array.length rows = 0 then None
      else
        match of_macos_row rows.(0) with
        | None -> None
        | Some total ->
            let cores =
              if Array.length rows <= 1 then []
              else
                Array.to_list
                  (Array.init
                     (Array.length rows - 1)
                     (fun i ->
                       match of_macos_row rows.(i + 1) with
                       | Some t -> t
                       | None -> zero))
            in
            Some (total, cores)
    with _ -> None

  let read () =
    if Sys.file_exists "/proc/stat" then read_linux () else read_macos ()

  let sample () : t =
    match read () with
    | Some (t, _) -> t
    | None -> raise (Sys_error "Cpu.sample: failed to read CPU statistics")

  let sample_per_core () : t array =
    match read () with
    | Some (_, cores) -> Array.of_list cores
    | None ->
        raise
          (Sys_error
             "Cpu.sample_per_core: failed to read per-core CPU statistics")

  let compute ~(prev : t) ~(next : t) : stats =
    let du = i64_delta ~prev:prev.user ~next:next.user in
    let dn = i64_delta ~prev:prev.nice ~next:next.nice in
    let ds = i64_delta ~prev:prev.system ~next:next.system in
    let di = i64_delta ~prev:prev.idle ~next:next.idle in
    let diw = i64_delta ~prev:prev.iowait ~next:next.iowait in
    let dirq = i64_delta ~prev:prev.irq ~next:next.irq in
    let dsi = i64_delta ~prev:prev.softirq ~next:next.softirq in
    let dst = i64_delta ~prev:prev.steal ~next:next.steal in
    let dg = i64_delta ~prev:prev.guest ~next:next.guest in

    let total =
      List.fold_left Int64.add 0L [ du; dn; ds; di; diw; dirq; dsi; dst; dg ]
    in
    if Int64.compare total 0L <= 0 then zero_stats
    else
      let total_f = Int64.to_float total in
      let pct v = Int64.to_float v /. total_f *. 100.0 in
      {
        user = pct du;
        nice = pct dn;
        system = pct ds;
        idle = pct di;
        iowait = pct diw;
        irq = pct dirq;
        softirq = pct dsi;
        steal = pct dst;
        guest = pct dg;
      }
end

(* ---------- Memory Module ---------- *)

module Mem = struct
  type t = {
    total : int64;
    used : int64;
    free : int64;
    available : int64;
    compressed : int64;
    wired : int64;
    active : int64;
    inactive : int64;
    purgeable : int64;
    speculative : int64;
    external_ : int64;
    page_size : int64;
    swap_total : int64;
    swap_used : int64;
  }

  let read_linux_meminfo () =
    let path = "/proc/meminfo" in
    if not (Sys.file_exists path) then None
    else
      let mem_total = ref None in
      let mem_free = ref None in
      let mem_available = ref None in
      let buffers = ref None in
      let cached = ref None in
      let swap_total = ref None in
      let swap_free = ref None in
      let shmem = ref None in
      let sreclaimable = ref None in
      let set_kb (r : int64 option ref) (line : string) =
        match split_whitespace line with
        | _key :: value :: _ -> (
            match int64_of_string_opt value with
            | None -> ()
            | Some kb -> r := Some (Int64.mul kb 1024L))
        | _ -> ()
      in
      let handle_line line =
        if String.starts_with ~prefix:"MemTotal:" line then
          set_kb mem_total line
        else if String.starts_with ~prefix:"MemFree:" line then
          set_kb mem_free line
        else if String.starts_with ~prefix:"MemAvailable:" line then
          set_kb mem_available line
        else if String.starts_with ~prefix:"Buffers:" line then
          set_kb buffers line
        else if String.starts_with ~prefix:"Cached:" line then
          set_kb cached line
        else if String.starts_with ~prefix:"SwapTotal:" line then
          set_kb swap_total line
        else if String.starts_with ~prefix:"SwapFree:" line then
          set_kb swap_free line
        else if String.starts_with ~prefix:"Shmem:" line then set_kb shmem line
        else if String.starts_with ~prefix:"SReclaimable:" line then
          set_kb sreclaimable line
        else ()
      in
      (try fold_lines path ~init:() ~f:(fun () line -> handle_line line)
       with _ -> ());
      match !mem_total with
      | None -> None
      | Some total ->
          let free = Option.value !mem_free ~default:0L in
          let buffers_b = Option.value !buffers ~default:0L in
          let cached_b = Option.value !cached ~default:0L in
          let shmem_b = Option.value !shmem ~default:0L in
          let sreclaimable_b = Option.value !sreclaimable ~default:0L in
          let swap_total_b = Option.value !swap_total ~default:0L in
          let swap_free_b = Option.value !swap_free ~default:0L in

          let cache_used =
            i64_max0 (Int64.sub (Int64.add cached_b sreclaimable_b) shmem_b)
          in
          let used =
            i64_max0
              (Int64.sub
                 (Int64.sub (Int64.sub total free) buffers_b)
                 cache_used)
          in
          let swap_used = i64_max0 (Int64.sub swap_total_b swap_free_b) in
          (* MemAvailable is available on Linux 3.14+; fallback to free + buffers + cached *)
          let available =
            match !mem_available with
            | Some v -> v
            | None -> Int64.add free (Int64.add buffers_b cached_b)
          in

          Some
            {
              total;
              used;
              free;
              available;
              compressed = 0L;
              wired = 0L;
              active = 0L;
              inactive = cache_used;
              purgeable = buffers_b;
              speculative = 0L;
              external_ = 0L;
              page_size = page_size ();
              swap_total = swap_total_b;
              swap_used;
            }

  let read_macos () =
    try
      let raw = c_get_memory () in
      let ps = raw.page_size in
      let to_bytes pages = Int64.mul pages ps in
      let active = to_bytes raw.active in
      let inactive = to_bytes raw.inactive in
      let speculative = to_bytes raw.speculative in
      let wired = to_bytes raw.wired in
      let compressed = to_bytes raw.compressor in
      let purgeable = to_bytes raw.purgeable in
      let external_ = to_bytes raw.external_ in
      let free = to_bytes raw.free in

      let used_raw =
        Int64.add active
          (Int64.add inactive
             (Int64.add speculative (Int64.add wired compressed)))
      in
      let used_raw = i64_max0 (Int64.sub used_raw purgeable) in
      let used_raw = i64_max0 (Int64.sub used_raw external_) in
      let used_display = i64_max0 (Int64.sub used_raw compressed) in
      (* Approximate available as free + inactive + purgeable *)
      let available = Int64.add free (Int64.add inactive purgeable) in

      Some
        {
          total = raw.total;
          used = used_display;
          free;
          available;
          compressed;
          wired;
          active;
          inactive;
          purgeable;
          speculative;
          external_;
          page_size = ps;
          swap_total = raw.swap_total;
          swap_used = raw.swap_used;
        }
    with _ -> None

  let sample () : t =
    let result =
      if Sys.file_exists "/proc/meminfo" then read_linux_meminfo ()
      else read_macos ()
    in
    match result with
    | Some t -> t
    | None -> raise (Sys_error "Mem.sample: failed to read memory statistics")
end

(* ---------- Network I/O Module ---------- *)

module Net = struct
  type t = {
    bytes_rx : int64;
    packets_rx : int64;
    bytes_tx : int64;
    packets_tx : int64;
  }

  type stats = {
    rx_bytes_per_sec : float;
    rx_packets_per_sec : float;
    tx_bytes_per_sec : float;
    tx_packets_per_sec : float;
  }

  let read_linux () : t option =
    let path = "/proc/net/dev" in
    if not (Sys.file_exists path) then None
    else
      let add acc (rx_b, rx_p, tx_b, tx_p) =
        {
          bytes_rx = Int64.add acc.bytes_rx rx_b;
          packets_rx = Int64.add acc.packets_rx rx_p;
          bytes_tx = Int64.add acc.bytes_tx tx_b;
          packets_tx = Int64.add acc.packets_tx tx_p;
        }
      in
      let parse_line line =
        match String.split_on_char ':' line with
        | [ lhs; rhs ] -> (
            let iface = String.trim lhs in
            if iface = "" || iface = "lo" then None
            else
              let fields = split_whitespace rhs |> Array.of_list in
              if Array.length fields < 10 then None
              else
                match
                  ( int64_of_string_opt fields.(0),
                    int64_of_string_opt fields.(1),
                    int64_of_string_opt fields.(8),
                    int64_of_string_opt fields.(9) )
                with
                | Some rx_b, Some rx_p, Some tx_b, Some tx_p ->
                    Some (rx_b, rx_p, tx_b, tx_p)
                | _ -> None)
        | _ -> None
      in
      let init =
        { bytes_rx = 0L; packets_rx = 0L; bytes_tx = 0L; packets_tx = 0L }
      in
      let total =
        fold_lines path ~init ~f:(fun acc line ->
            match parse_line line with
            | None -> acc
            | Some tuple -> add acc tuple)
      in
      Some total

  let read_macos () : t option =
    try
      let raw = c_get_network_io () in
      Some
        {
          bytes_rx = raw.bytes_rx;
          packets_rx = raw.packets_rx;
          bytes_tx = raw.bytes_tx;
          packets_tx = raw.packets_tx;
        }
    with _ -> None

  let sample () : t =
    let result =
      if Sys.file_exists "/proc/net/dev" then read_linux () else read_macos ()
    in
    match result with
    | Some t -> t
    | None -> raise (Sys_error "Net.sample: failed to read network statistics")

  let compute ~(prev : t) ~(next : t) ~(dt : float) : stats =
    if dt <= 0.0 then invalid_arg "Net.compute: dt must be positive";
    {
      rx_bytes_per_sec =
        f_delta_i64 ~prev:prev.bytes_rx ~next:next.bytes_rx /. dt;
      rx_packets_per_sec =
        f_delta_i64 ~prev:prev.packets_rx ~next:next.packets_rx /. dt;
      tx_bytes_per_sec =
        f_delta_i64 ~prev:prev.bytes_tx ~next:next.bytes_tx /. dt;
      tx_packets_per_sec =
        f_delta_i64 ~prev:prev.packets_tx ~next:next.packets_tx /. dt;
    }
end

(* ---------- Disk I/O Module ---------- *)

module Disk_io = struct
  type t = {
    bytes_read : int64;
    bytes_written : int64;
    time_ms : int64;
    num_disks : int64;
  }

  type stats = {
    read_bytes_per_sec : float;
    write_bytes_per_sec : float;
    utilization_percent : float;
  }

  type linux_disk = {
    name : string;
    sectors_read : int64;
    sectors_written : int64;
    io_time_ms : int64;
  }

  let is_excluded_name name =
    String.starts_with ~prefix:"dm-" name
    || String.starts_with ~prefix:"loop" name
    || String.starts_with ~prefix:"md" name
    || String.starts_with ~prefix:"zram" name

  let partition_base (name : string) : string option =
    (* If name ends with digits, it's *possibly* a partition.
       We return the candidate base device name. *)
    let len = String.length name in
    let rec find_first_non_digit i =
      if i < 0 then None
      else
        match name.[i] with
        | '0' .. '9' -> find_first_non_digit (i - 1)
        | _ -> Some i
    in
    match find_first_non_digit (len - 1) with
    | None -> None
    | Some i when i = len - 1 -> None (* no trailing digits *)
    | Some i ->
        let digit_start = i + 1 in
        if digit_start <= 0 then None
        else if name.[i] = 'p' then
          (* nvme0n1p1 / mmcblk0p1 style *)
          if i <= 0 then None else Some (String.sub name 0 i)
        else
          (* sda1 style *)
          Some (String.sub name 0 (i + 1))

  let read_linux () : t option =
    let path = "/proc/diskstats" in
    if not (Sys.file_exists path) then None
    else
      let parsed =
        fold_lines path ~init:[] ~f:(fun acc line ->
            match split_whitespace line with
            | _major :: _minor :: name :: _reads_ok :: _reads_merged
              :: sectors_read :: _time_read :: _writes_ok :: _writes_merged
              :: sectors_written :: _time_write :: _in_flight :: io_time_ms
              :: _weighted_time :: _ -> (
                if is_excluded_name name then acc
                else
                  match
                    ( int64_of_string_opt sectors_read,
                      int64_of_string_opt sectors_written,
                      int64_of_string_opt io_time_ms )
                  with
                  | Some sr, Some sw, Some t ->
                      {
                        name;
                        sectors_read = sr;
                        sectors_written = sw;
                        io_time_ms = t;
                      }
                      :: acc
                  | _ -> acc)
            | _ -> acc)
      in
      let candidates = List.rev parsed in
      let name_set = Hashtbl.create (List.length candidates) in
      List.iter (fun d -> Hashtbl.replace name_set d.name ()) candidates;

      let is_partition d =
        match partition_base d.name with
        | None -> false
        | Some base -> Hashtbl.mem name_set base
      in

      let bytes_read = ref 0L in
      let bytes_written = ref 0L in
      let time_ms = ref 0L in
      let num_disks = ref 0L in

      List.iter
        (fun d ->
          if not (is_partition d) then (
            bytes_read := Int64.add !bytes_read (Int64.mul d.sectors_read 512L);
            bytes_written :=
              Int64.add !bytes_written (Int64.mul d.sectors_written 512L);
            time_ms := Int64.add !time_ms d.io_time_ms;
            num_disks := Int64.add !num_disks 1L))
        candidates;

      Some
        {
          bytes_read = !bytes_read;
          bytes_written = !bytes_written;
          time_ms = !time_ms;
          num_disks = !num_disks;
        }

  let read_macos () : t option =
    try
      let raw = c_get_disk_io () in
      Some
        {
          bytes_read = raw.bytes_read;
          bytes_written = raw.bytes_written;
          time_ms = raw.time_ms;
          num_disks = raw.num_disks;
        }
    with _ -> None

  let sample () : t =
    let result =
      if Sys.file_exists "/proc/diskstats" then read_linux () else read_macos ()
    in
    match result with
    | Some t -> t
    | None ->
        raise (Sys_error "Disk_io.sample: failed to read disk I/O statistics")

  let compute ~(prev : t) ~(next : t) ~(dt : float) : stats =
    if dt <= 0.0 then invalid_arg "Disk_io.compute: dt must be positive";

    let read_bps =
      f_delta_i64 ~prev:prev.bytes_read ~next:next.bytes_read /. dt
    in
    let write_bps =
      f_delta_i64 ~prev:prev.bytes_written ~next:next.bytes_written /. dt
    in
    let time_delta_ms = f_delta_i64 ~prev:prev.time_ms ~next:next.time_ms in

    let utilization =
      if Int64.compare next.num_disks 0L <= 0 then 0.0
      else
        let denom = dt *. 1000.0 *. Int64.to_float next.num_disks in
        if denom <= 0.0 then 0.0 else 100.0 *. time_delta_ms /. denom
    in
    {
      read_bytes_per_sec = read_bps;
      write_bytes_per_sec = write_bps;
      utilization_percent = min utilization 100.0;
    }
end

(* ---------- Filesystem Module ---------- *)

module Fs = struct
  type partition = {
    mount_point : string;
    total_bytes : int64;
    used_bytes : int64;
    avail_bytes : int64;
  }

  type t = {
    total_bytes : int64;
    used_bytes : int64;
    avail_bytes : int64;
    partitions : partition list;
  }

  let is_excluded_mount ~mount_point ~fstype =
    let excluded_fstypes =
      [
        "devfs";
        "devtmpfs";
        "tmpfs";
        "proc";
        "sysfs";
        "cgroup";
        "autofs";
        "overlay";
      ]
    in
    let excluded_mount_prefixes = [ "/dev"; "/proc"; "/sys"; "/run" ] in
    List.exists (fun p -> String.starts_with ~prefix:p fstype) excluded_fstypes
    || List.exists
         (fun p -> String.starts_with ~prefix:p mount_point)
         excluded_mount_prefixes

  let stat_partition (mount_point : string) : partition option =
    try
      let raw = c_statvfs mount_point in
      if Int64.compare raw.total 0L <= 0 then None
      else if Int64.compare raw.total 100_000_000L < 0 then None
      else
        let used = i64_max0 (Int64.sub raw.total raw.free) in
        Some
          {
            mount_point;
            total_bytes = raw.total;
            used_bytes = used;
            avail_bytes = raw.avail;
          }
    with _ -> None

  let partitions () : partition list =
    try
      c_getmounts () |> Array.to_list
      |> List.filter_map (fun (mount_point, _device, fstype) ->
          if is_excluded_mount ~mount_point ~fstype then None
          else stat_partition mount_point)
    with _ -> []

  let sample ?(path = "/") () : t =
    let result =
      try
        let raw = c_statvfs path in
        if Int64.compare raw.total 0L < 0 then None
        else
          let used = i64_max0 (Int64.sub raw.total raw.free) in
          Some
            {
              total_bytes = raw.total;
              used_bytes = used;
              avail_bytes = raw.avail;
              partitions = partitions ();
            }
      with _ -> None
    in
    match result with
    | Some t -> t
    | None ->
        raise
          (Sys_error
             ("Fs.sample: failed to read filesystem statistics for " ^ path))
end

(* ---------- Process Module ---------- *)

module Proc = struct
  type state =
    | Running
    | Sleeping
    | Disk_sleep
    | Stopped
    | Zombie
    | Idle
    | Unknown

  let state_of_char = function
    | 'R' -> Running
    | 'S' -> Sleeping
    | 'D' -> Disk_sleep
    | 'T' | 't' -> Stopped
    | 'Z' -> Zombie
    | 'I' -> Idle
    | 'X' -> Unknown (* dead *)
    | _ -> Unknown

  module Self = struct
    type t = {
      utime : float;
      stime : float;
      rss_bytes : int64;
      vsize_bytes : int64;
    }

    type stats = { cpu_percent : float; rss_bytes : int64; vsize_bytes : int64 }

    let read_proc_stat_tail (path : string) : string list option =
      try
        let line = with_in_file path input_line in
        match String.rindex_opt line ')' with
        | None -> None
        | Some k ->
            let start = k + 2 in
            if start >= String.length line then None
            else
              Some
                (split_whitespace
                   (String.sub line start (String.length line - start)))
      with _ -> None

    let linux_rss_vsize () : (int64 * int64) option =
      match read_proc_stat_tail "/proc/self/stat" with
      | None -> None
      | Some fields -> (
          match (List.nth_opt fields 20, List.nth_opt fields 21) with
          | Some vsize_s, Some rss_pages_s -> (
              match
                (int64_of_string_opt vsize_s, int64_of_string_opt rss_pages_s)
              with
              | Some vsize, Some rss_pages ->
                  let rss = Int64.mul rss_pages (page_size ()) in
                  Some (rss, vsize)
              | _ -> None)
          | _ -> None)

    let sample () : t =
      let result =
        try
          let times = Unix.times () in
          let utime = times.Unix.tms_utime in
          let stime = times.Unix.tms_stime in

          let rss_bytes, vsize_bytes =
            if Sys.file_exists "/proc/self/stat" then
              match linux_rss_vsize () with
              | Some (rss, vsz) -> (rss, vsz)
              | None -> (0L, 0L)
            else
              let raw = c_proc_self_mem () in
              let rss = if Int64.compare raw.rss 0L < 0 then 0L else raw.rss in
              let vsz =
                if Int64.compare raw.vsize 0L < 0 then 0L else raw.vsize
              in
              (rss, vsz)
          in
          Some { utime; stime; rss_bytes; vsize_bytes }
        with _ -> None
      in
      match result with
      | Some t -> t
      | None ->
          raise
            (Sys_error "Proc.Self.sample: failed to read process statistics")

    let compute ~(prev : t) ~(next : t) ~(dt : float) ~(num_cores : int option)
        : stats =
      if dt <= 0.0 then invalid_arg "Proc.Self.compute: dt must be positive";

      let dt_ok = dt > 0.01 && dt < 10.0 in
      let cpu_delta = next.utime -. prev.utime +. (next.stime -. prev.stime) in
      let cpu_delta = max 0.0 cpu_delta in

      let cpu_percent =
        if not dt_ok then 0.0
        else
          let raw = cpu_delta /. dt *. 100.0 in
          match num_cores with
          | Some n when n > 0 && n <= 128 -> min (raw /. float_of_int n) 100.0
          | _ -> min raw 800.0
      in
      {
        cpu_percent;
        rss_bytes = next.rss_bytes;
        vsize_bytes = next.vsize_bytes;
      }
  end

  module Table = struct
    type t = {
      pid : int;
      ppid : int;
      name : string;
      cmdline : string;
      state : state;
      user : string;
      priority : int;
      nice : int;
      user_time : int64;
      system_time : int64;
      resident_size : int64;
      virtual_size : int64;
      num_threads : int;
      num_running : int;
      faults : int64;
      mem_percent : float;
    }

    type stats = {
      pid : int;
      name : string;
      cpu_percent : float;
      mem_percent : float;
      rss_bytes : int64;
    }

    let read_total_mem_bytes_linux () : int64 option =
      let path = "/proc/meminfo" in
      if not (Sys.file_exists path) then None
      else
        let mem_total = ref None in
        let handle line =
          if String.starts_with ~prefix:"MemTotal:" line then
            match split_whitespace line with
            | _key :: value :: _ -> (
                match int64_of_string_opt value with
                | Some kb -> mem_total := Some (Int64.mul kb 1024L)
                | None -> ())
            | _ -> ()
        in
        (try fold_lines path ~init:() ~f:(fun () line -> handle line)
         with _ -> ());
        !mem_total

    let total_mem_bytes () : int64 =
      if Sys.file_exists "/proc/meminfo" then
        Option.value (read_total_mem_bytes_linux ()) ~default:0L
      else try (c_get_memory ()).total with _ -> 0L

    let read_proc_stat_tail (path : string) : string list option =
      try
        let line = with_in_file path input_line in
        match String.rindex_opt line ')' with
        | None -> None
        | Some k ->
            let start = k + 2 in
            if start >= String.length line then None
            else
              Some
                (split_whitespace
                   (String.sub line start (String.length line - start)))
      with _ -> None

    let read_comm (path : string) : string option =
      try
        let s = with_in_file path input_line |> String.trim in
        if s = "" then None else Some s
      with _ -> None

    let read_cmdline (pid : int) : string =
      let path = Printf.sprintf "/proc/%d/cmdline" pid in
      try
        let ic = open_in path in
        let len = in_channel_length ic in
        if len = 0 then (
          close_in_noerr ic;
          "")
        else
          let buf = Bytes.create len in
          let n = input ic buf 0 len in
          close_in_noerr ic;
          (* Replace null bytes with spaces *)
          for i = 0 to n - 1 do
            if Bytes.get buf i = '\000' then Bytes.set buf i ' '
          done;
          String.trim (Bytes.sub_string buf 0 n)
      with _ -> ""

    let get_username (uid : int) : string =
      try (Unix.getpwuid uid).Unix.pw_name with _ -> ""

    let sample_linux () : t list =
      let total_mem = total_mem_bytes () in
      let dir = "/proc" in
      let dh = try Some (Unix.opendir dir) with _ -> None in
      match dh with
      | None -> []
      | Some dh ->
          let procs = ref [] in
          let add pid fields =
            (* Fields are tokens starting at /proc/[pid]/stat field 3 (state).
               Index 0 = state, 1 = ppid, ... 11 = utime, 12 = stime, 15 = priority, 16 = nice,
               17 = num_threads, 20 = vsize, 21 = rss *)
            let state_char =
              match List.nth_opt fields 0 with
              | Some s when String.length s > 0 -> s.[0]
              | _ -> '?'
            in
            let ppid =
              match List.nth_opt fields 1 with
              | Some s -> Option.value (int_of_string_opt s) ~default:0
              | None -> 0
            in
            let priority =
              match List.nth_opt fields 15 with
              | Some s -> Option.value (int_of_string_opt s) ~default:0
              | None -> 0
            in
            let nice =
              match List.nth_opt fields 16 with
              | Some s -> Option.value (int_of_string_opt s) ~default:0
              | None -> 0
            in
            let num_threads =
              match List.nth_opt fields 17 with
              | Some s -> Option.value (int_of_string_opt s) ~default:1
              | None -> 1
            in
            match
              ( List.nth_opt fields 11,
                List.nth_opt fields 12,
                List.nth_opt fields 20,
                List.nth_opt fields 21 )
            with
            | Some ut_s, Some st_s, Some vsz_s, Some rss_pages_s -> (
                match
                  ( int64_of_string_opt ut_s,
                    int64_of_string_opt st_s,
                    int64_of_string_opt vsz_s,
                    int64_of_string_opt rss_pages_s )
                with
                | Some utime, Some stime, Some vsize, Some rss_pages ->
                    let rss = Int64.mul rss_pages (page_size ()) in
                    let mem_percent =
                      if Int64.compare total_mem 0L > 0 then
                        Int64.to_float rss *. 100.0 /. Int64.to_float total_mem
                      else 0.0
                    in
                    let name =
                      match read_comm (Printf.sprintf "/proc/%d/comm" pid) with
                      | Some n -> n
                      | None -> string_of_int pid
                    in
                    let cmdline = read_cmdline pid in
                    let uid =
                      try
                        (Unix.stat (Printf.sprintf "/proc/%d" pid)).Unix.st_uid
                      with _ -> -1
                    in
                    let user = if uid >= 0 then get_username uid else "" in
                    procs :=
                      {
                        pid;
                        ppid;
                        name;
                        cmdline;
                        state = state_of_char state_char;
                        user;
                        priority;
                        nice;
                        user_time = utime;
                        system_time = stime;
                        resident_size = rss;
                        virtual_size = vsize;
                        num_threads;
                        num_running = 0;
                        faults = 0L;
                        mem_percent;
                      }
                      :: !procs
                | _ -> ())
            | _ -> ()
          in
          (try
             while true do
               let entry = Unix.readdir dh in
               match int_of_string_opt entry with
               | None -> ()
               | Some pid -> (
                   let stat_path = Printf.sprintf "/proc/%d/stat" pid in
                   if Sys.file_exists stat_path then
                     match read_proc_stat_tail stat_path with
                     | Some fields -> add pid fields
                     | None -> ())
             done
           with End_of_file -> ());
          Unix.closedir dh;
          !procs

    let sample_macos () : t list =
      let total_mem = total_mem_bytes () in
      let pids =
        try c_list_pids () |> Array.to_list |> List.filter (fun pid -> pid > 0)
        with _ -> []
      in
      let add acc pid =
        match c_get_proc_info pid with
        | None -> acc
        | Some info ->
            let mem_percent =
              if Int64.compare total_mem 0L > 0 then
                Int64.to_float info.resident_size
                *. 100.0 /. Int64.to_float total_mem
              else 0.0
            in
            {
              pid;
              ppid = info.ppid;
              name = info.name;
              cmdline = info.cmdline;
              state = state_of_char info.state;
              user = info.user;
              priority = info.priority;
              nice = info.nice;
              user_time = info.user_time;
              system_time = info.system_time;
              resident_size = info.resident_size;
              virtual_size = info.virtual_size;
              num_threads = info.num_threads;
              num_running = info.num_running;
              faults = info.faults;
              mem_percent;
            }
            :: acc
      in
      List.fold_left add [] pids

    let sample () : t list =
      try if Sys.file_exists "/proc" then sample_linux () else sample_macos ()
      with _ -> []

    let compute ~(prev : t list) ~(next : t list) ~(dt : float) : stats list =
      if dt <= 0.0 then invalid_arg "Proc.Table.compute: dt must be positive";

      let prev_by_pid : (int, t) Hashtbl.t =
        Hashtbl.create (List.length prev)
      in
      List.iter (fun (p : t) -> Hashtbl.replace prev_by_pid p.pid p) prev;

      let interval_ns = dt *. 1e9 in
      let cpu_percent (prev_p : t) (curr_p : t) =
        let prev_total = Int64.add prev_p.user_time prev_p.system_time in
        let curr_total = Int64.add curr_p.user_time curr_p.system_time in
        if Int64.compare curr_total prev_total <= 0 then 0.0
        else
          let delta_ticks = Int64.sub curr_total prev_total in
          let delta_ns = proc_ticks_to_ns delta_ticks in
          if interval_ns <= 0.0 then 0.0 else delta_ns /. interval_ns *. 100.0
      in

      let make_stats ~pid ~name ~cpu_percent ~mem_percent ~rss_bytes : stats =
        { pid; name; cpu_percent; mem_percent; rss_bytes }
      in
      next
      |> List.filter_map (fun (curr : t) ->
          match Hashtbl.find_opt prev_by_pid curr.pid with
          | Some prev_p ->
              let cpu = cpu_percent prev_p curr in
              if cpu > 0.0 || curr.mem_percent > 0.0 then
                Some
                  (make_stats ~pid:curr.pid ~name:curr.name ~cpu_percent:cpu
                     ~mem_percent:curr.mem_percent ~rss_bytes:curr.resident_size)
              else None
          | None ->
              if curr.mem_percent > 0.0 then
                Some
                  (make_stats ~pid:curr.pid ~name:curr.name ~cpu_percent:0.0
                     ~mem_percent:curr.mem_percent ~rss_bytes:curr.resident_size)
              else None)
  end
end

(* ---------- System Info Functions ---------- *)

let loadavg () = c_get_loadavg ()
let uptime () = c_get_uptime ()
