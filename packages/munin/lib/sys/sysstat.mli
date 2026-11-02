(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The counters and levels the readings take from the system.

    Linux is read from [/proc] and [/sys], macOS and Windows through their
    stubs. Each function raises [Sys_error] when its system cannot give what it
    returns. *)

val now : unit -> float
(** [now ()] is the time in seconds on a clock that never steps back, from an
    arbitrary origin. *)

(** {1:cpu CPU} *)

type cpu = { busy : int; idle : int }
(** The type for the time all CPUs spent since boot, in ticks of the system's
    own unit. [idle] is the time spent idle or waiting for I/O, [busy] the rest.
*)

val cpu : unit -> cpu
(** [cpu ()] is the time all CPUs spent since boot. *)

val cpu_of_proc_stat : string -> cpu
(** [cpu_of_proc_stat text] is the time of the aggregate [cpu] line of [text],
    laid out as Linux's [/proc/stat]. Linux counts the time a CPU ran a virtual
    machine in [user] and [nice] as well as in [guest] and [guest_nice], so the
    latter two are left out. *)

(** {1:memory Memory} *)

type memory = { total : int; available : int option; swap_used : int option }
(** The type for the machine's memory, in bytes. [available] is what the system
    can give to new allocations without swapping: Linux's [MemAvailable],
    macOS's free and inactive pages, Windows's available physical memory. *)

val memory : unit -> memory
(** [memory ()] is the machine's memory now. *)

val load_per_core : unit -> float
(** [load_per_core ()] is the one-minute load average over the number of logical
    CPUs online. Raises [Sys_error] on Windows, which keeps no load average. *)

(** {1:disks Disks} *)

type disk = {
  name : string;
  read_bytes : int;
  written_bytes : int;
  busy_ms : int option;
}
(** The type for a physical disk's counters since it attached: bytes read, bytes
    written, and on Linux the milliseconds it had requests in flight. [name] is
    unique among the disks of one call. *)

val disks : unit -> disk list
(** [disks ()] are the physical disks' counters. On Linux a physical disk is a
    block device of [/proc/diskstats] backed by a device in [/sys/block];
    partitions and virtual devices (loop, device-mapper, RAID, zram) are left
    out. *)

(** {1:process This process} *)

val rss : unit -> int
(** [rss ()] is this process's resident memory in bytes. *)
