(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Readings of the machine, of this process and of devices.

    A {e reading} is a function that reads some facts now and gives them as
    pairs of a key and a number. [Munin.Session.sample] calls readings on a
    schedule and logs what they give into a run. The values here make
    readings, and depend on no part of munin, so any logger may call them.

    A {e level} is a fact's value at the call, such as memory in use. A
    {e rate} is a counter's change per second over the {e window} since
    the reading's previous call, timed by a monotonic clock, such as bytes
    read from disk per second. A reading that gives rates gives its
    window's length in seconds under a key of its own. Its first call
    starts the window and gives nothing, so every pair a reading gives
    comes with its rates, and a sampler logs the first pairs of such a
    reading one period after it starts. A reading keeps its own counters,
    so it must be called from one thread at a time, and given to one
    sampler. The monotonic clock counts time asleep on macOS and not on
    Linux, so a window across a suspend is longer on macOS.

    Keys start with [sys/] for the machine and this process, and with
    [dev/<name>/] for a device. A program keeps its own keys out of these
    prefixes. Sizes are in bytes. A key the system cannot read is left out
    of the pairs, and a reading raises [Sys_error] only when it can read
    nothing.

    A program samples its machine, its process and the devices its arrays
    live on. A device it closed or lost gives nothing that time, since its
    reading raises [Sys_error], which a sampler skips; any other exception
    would stop the sampler.
    {[
      let held d () : Munin_sys.device =
        match (Rig.allocated d, Rig.budget d) with
        | exception Rig.Lost (_, why) -> raise (Sys_error why)
        | allocated, budget ->
            { Munin_sys.unknown with
              allocated = Some allocated;
              budget = (if budget = max_int then None else Some budget) }
      in
      let reading d = Munin_sys.device (Rig.name d) (held d) in
      Munin.Session.sample s ~every:15.
        (Munin_sys.host () :: Munin_sys.process ()
         :: List.map reading (Nx.Devices.to_list devices))
    ]} *)

(** {1:host The machine} *)

val host : unit -> unit -> (string * float) list
(** [host ()] is a new reading of the machine. Its keys:
    - [sys/cpu_busy_pct], a rate: the percent of all CPUs' time spent
      neither idle nor waiting for I/O. On Linux, time that a CPU ran a
      virtual machine counts once, as user time.
    - [sys/load_per_core], a level: the one-minute load average over the
      number of logical CPUs. Windows keeps no load average.
    - [sys/mem_used_bytes], a level: total memory less available memory,
      where available memory is Linux's [MemAvailable], macOS's free and
      inactive pages, and Windows's available physical memory.
    - [sys/mem_total_bytes] and [sys/swap_used_bytes], levels.
    - [sys/disk_read_bytes_per_s] and [sys/disk_write_bytes_per_s], rates
      summed over the physical disks, each disk's own change; absent for a
      window over which the set of disks changed, or a disk's counter went
      back, which reads as a disk replaced.
    - [sys/disk_busy_pct], a rate on Linux alone: the mean over the disks
      of the percent of the window each had requests in flight, each disk's
      capped at [100.], since Linux counts busy time when a request starts
      or ends.
    - [sys/interval_s]: the window's length. *)

(** {1:process This process} *)

val process : unit -> unit -> (string * float) list
(** [process ()] is a new reading of this process. Its keys:
    - [sys/proc_cpu_cores], a rate: the process's CPU seconds per second,
    so [2.] is two cores busy.
    - [sys/proc_rss_bytes], a level: its resident memory.
    - [sys/proc_interval_s]: the window's length. *)

(** {1:devices Devices}

    munin reads no device. A program builds a device's reading from what
    the device's library reports, through {!val-device}, which fixes the keys
    the run records and a reader looks for. Each field holds one quantity
    under one key, whichever vendor made the device. rig reports
    [allocated] and [budget] of every device ([Rig.allocated],
    [Rig.budget]), and [memory], [used] and [busy] of an NVIDIA GPU on
    NVIDIA's kernel driver ([Rig_nv_nvidia.usage]); a field that the
    device's library does not report stays [None]. *)

type device = {
  allocated : int option;
      (** The bytes this process holds on the device, under
          [allocated_bytes]. *)
  budget : int option;
      (** The bytes this process may hold on the device before its runtime
          reclaims memory, if they are bounded, under [budget_bytes]. *)
  memory : int option;
      (** The bytes of the device's own memory from which its driver
          allocates, for every process and for itself, under
          [memory_bytes]. *)
  used : int option;
      (** Of [memory], the bytes that allocations of every process and of
          the driver hold, under [used_bytes]. *)
  busy : float option;
      (** The percent of the window's time during which the device's
          compute engine ran work of any process, under [busy_pct]. It
          measures time, so one small kernel that runs through the window
          makes the device [100.] busy. *)
  activity : float option;
      (** The percent of activity that the device's firmware measures
          across its compute units, over a window of its own, under
          [activity_pct]. The firmware defines it, and it is a different
          quantity from [busy]. *)
  temperature : float option;
      (** The temperature at the device's edge sensor, in degrees Celsius,
          under [temperature_c]. A device without one reports none. *)
  power : float option;
      (** The power the device drew, in watts, averaged by its driver over
          a window of the driver's own, under [power_w]. *)
}
(** The type for what is known of a device at a call. *)

val unknown : device
(** [unknown] is a device of which nothing is known, every field
    [None]. *)

val device : string -> (unit -> device) -> unit -> (string * float) list
(** [device name read] is a reading that calls [read] and gives each field
    that is [Some] as a level, under [dev/<name>/] and the key its field
    names. It raises what [read] raises.

    Raises [Invalid_argument] if [name] is empty, is not UTF-8, holds a [/]
    or a control character, or starts with [.]. *)

(**/**)

(* For the tests: the formulas of the rates, apart from the system. *)

type disk = {
  name : string;
  read_bytes : int;
  written_bytes : int;
  busy_ms : int option;
}

val disk_rates :
  dt:float -> prev:disk list -> disk list -> (string * float) list
(** [disk_rates ~dt ~prev next] are the disk pairs of {!val-host} for a window
    of [dt > 0.] seconds from the disks [prev] to the disks [next]. *)

val proc_stat_busy : string -> string -> float option
(** [proc_stat_busy prev next] is [sys/cpu_busy_pct] between two texts of
    Linux's [/proc/stat], or [None] if no CPU time passed. *)

(**/**)
