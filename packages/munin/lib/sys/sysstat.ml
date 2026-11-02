(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type system = Linux | Macos | Windows | Other

external c_system : unit -> int = "caml_munin_sys_system" [@@noalloc]

external now : unit -> (float[@unboxed])
  = "caml_munin_sys_now_byte" "caml_munin_sys_now"
[@@noalloc]

external c_cpu : unit -> int * int = "caml_munin_sys_cpu"
external c_memory : unit -> int * int * int = "caml_munin_sys_memory"
external c_load : unit -> float * int = "caml_munin_sys_load"
external c_rss : unit -> int = "caml_munin_sys_rss"
external c_disks : unit -> (string * int * int) array = "caml_munin_sys_disks"

let system =
  match c_system () with 0 -> Linux | 1 -> Macos | 2 -> Windows | _ -> Other

let kib = 1024

(* Linux's /proc *)

let read path = In_channel.with_open_bin path In_channel.input_all
let lines text = String.split_on_char '\n' text
let fields line = List.filter (( <> ) "") (String.split_on_char ' ' line)
let bad path what = raise (Sys_error (Printf.sprintf "%s: %s" path what))

let int path s =
  match int_of_string_opt s with Some v -> v | None -> bad path s

(* "Key: 1234 kB", as /proc/meminfo and /proc/self/status lay out a size. *)
let kb_field path key text =
  let prefix = key ^ ":" in
  match List.find_opt (String.starts_with ~prefix) (lines text) with
  | None -> None
  | Some line -> (
      match fields line with
      | _ :: v :: _ -> Some (int path v * kib)
      | _ -> bad path line)

(* CPU *)

type cpu = { busy : int; idle : int }

let proc_stat = "/proc/stat"

(* The fields after [cpu] are user, nice, system, idle, iowait, irq, softirq,
   steal, guest and guest_nice; an older kernel gives fewer. *)
let cpu_of_proc_stat text =
  match List.find_opt (String.starts_with ~prefix:"cpu ") (lines text) with
  | None -> bad proc_stat "no cpu line"
  | Some line -> (
      match List.map (int proc_stat) (List.tl (fields line)) with
      | user :: nice :: system :: idle :: rest ->
          let at i = Option.value (List.nth_opt rest i) ~default:0 in
          let iowait = at 0
          and irq = at 1
          and softirq = at 2
          and steal = at 3 in
          {
            busy = user + nice + system + irq + softirq + steal;
            idle = idle + iowait;
          }
      | _ -> bad proc_stat line)

let cpu () =
  match system with
  | Linux -> cpu_of_proc_stat (read proc_stat)
  | Macos | Windows | Other ->
      let busy, idle = c_cpu () in
      { busy; idle }

(* Memory *)

type memory = { total : int; available : int option; swap_used : int option }

let meminfo = "/proc/meminfo"

let memory_linux () =
  let text = read meminfo in
  let field key = kb_field meminfo key text in
  match field "MemTotal" with
  | None -> bad meminfo "no MemTotal"
  | Some total ->
      let swap_used =
        match (field "SwapTotal", field "SwapFree") with
        | Some t, Some f -> Some (t - f)
        | _ -> None
      in
      { total; available = field "MemAvailable"; swap_used }

let memory () =
  match system with
  | Linux -> memory_linux ()
  | Macos | Windows | Other ->
      let total, available, swap_used = c_memory () in
      { total; available = Some available; swap_used = Some swap_used }

(* Load *)

let is_core line =
  String.length line > 3
  && String.starts_with ~prefix:"cpu" line
  && match line.[3] with '0' .. '9' -> true | _ -> false

let load_linux () =
  let path = "/proc/loadavg" in
  let load =
    match fields (read path) with
    | one :: _ -> (
        match float_of_string_opt one with Some l -> l | None -> bad path one)
    | [] -> bad path "empty"
  in
  match List.length (List.filter is_core (lines (read proc_stat))) with
  | 0 -> bad proc_stat "no CPU"
  | n -> load /. float_of_int n

let load_per_core () =
  match system with
  | Linux -> load_linux ()
  | Macos | Windows | Other ->
      let load, n = c_load () in
      load /. float_of_int n

(* Disks *)

type disk = {
  name : string;
  read_bytes : int;
  written_bytes : int;
  busy_ms : int option;
}

let diskstats = "/proc/diskstats"

(* /proc/diskstats counts sectors of 512 bytes, whatever the disk's own. *)
let sector = 512

(* A physical disk is backed by a device; /sys/block names a disk whose name
   holds a '/', as cciss/c0d0, with a '!'. Partitions are not in /sys/block and
   virtual devices have no device link. *)
let is_physical name =
  let name = String.map (function '/' -> '!' | c -> c) name in
  Sys.file_exists (Filename.concat "/sys/block" name ^ "/device")

(* After major, minor and name: reads, reads merged, sectors read, read time,
   writes, writes merged, sectors written, write time, in flight, io_ticks, the
   time in milliseconds the disk had requests in flight. *)
let disk_of_line line =
  match fields line with
  | _ :: _ :: name :: _ :: _ :: read :: _ :: _ :: _ :: written :: _ :: _
    :: ticks :: _
    when is_physical name ->
      let int = int diskstats in
      Some
        {
          name;
          read_bytes = int read * sector;
          written_bytes = int written * sector;
          busy_ms = Some (int ticks);
        }
  | _ -> None

let disks () =
  match system with
  | Linux -> List.filter_map disk_of_line (lines (read diskstats))
  | Macos | Windows | Other ->
      Array.to_list (c_disks ())
      |> List.map (fun (name, read_bytes, written_bytes) ->
          { name; read_bytes; written_bytes; busy_ms = None })

(* This process *)

let rss () =
  match system with
  | Linux -> (
      let path = "/proc/self/status" in
      match kb_field path "VmRSS" (read path) with
      | Some rss -> rss
      | None -> bad path "no VmRSS")
  | Macos | Windows | Other -> c_rss ()
