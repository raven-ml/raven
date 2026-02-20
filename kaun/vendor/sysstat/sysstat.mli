(** System metrics collection library.

    Sysstat provides stateless, poll-based system monitoring. The caller manages
    state and sampling intervals. Each module samples instantaneous or
    cumulative values from the operating system. CPU, network, and disk I/O
    statistics are cumulative since boot and require two samples to compute
    usage percentages; memory statistics are instantaneous.

    {1 Platform Support}

    Supported platforms: Linux and macOS. Platform-specific behavior is
    documented per module. Some metrics have limited availability on certain
    platforms (e.g., macOS CPU counters populate only user/nice/system/idle
    fields). *)

(** {1 Sysstat Modules} *)

(** CPU statistics. *)
module Cpu : sig
  type t = {
    user : int64;  (** Time spent in user mode. *)
    nice : int64;  (** Time spent in user mode with low priority (nice). *)
    system : int64;  (** Time spent in system (kernel) mode. *)
    idle : int64;  (** Time spent idle. *)
    iowait : int64;  (** Time spent waiting for I/O to complete (Linux only). *)
    irq : int64;  (** Time spent servicing interrupts (Linux only). *)
    softirq : int64;  (** Time spent servicing soft interrupts (Linux only). *)
    steal : int64;
        (** Time stolen by other VMs in virtualized environments (Linux only).
        *)
    guest : int64;  (** Time spent running guest VMs (Linux only). *)
  }
  (** Cumulative CPU time counters in ticks since boot.

      All fields represent cumulative time spent in each CPU state. The unit is
      platform-specific ticks (Linux jiffies or macOS Mach ticks) but is
      abstracted away by {!compute}.

      Platform behavior:
      - {b macOS}: Only [user], [nice], [system], and [idle] are available;
        other fields are [0L]. *)

  type stats = {
    user : float;  (** Percentage of time spent in user mode. *)
    nice : float;
        (** Percentage of time spent in user mode with low priority. *)
    system : float;  (** Percentage of time spent in system mode. *)
    idle : float;  (** Percentage of time spent idle. *)
    iowait : float;  (** Percentage of time spent waiting for I/O. *)
    irq : float;  (** Percentage of time spent servicing interrupts. *)
    softirq : float;  (** Percentage of time spent servicing soft interrupts. *)
    steal : float;  (** Percentage of time stolen by hypervisor. *)
    guest : float;  (** Percentage of time spent running guest VMs. *)
  }
  (** CPU usage percentages between two samples.

      All fields are in the range [0.0] to [100.0], where [100.0] represents
      full utilization. The sum of all fields equals [100.0]. *)

  val sample : unit -> t
  (** [sample ()] returns aggregate CPU counters across all cores.

      @raise Sys_error
        if CPU statistics are unavailable on the current platform or an error
        occurs during sampling. *)

  val sample_per_core : unit -> t array
  (** [sample_per_core ()] returns per-core CPU counters.

      The array length equals the number of logical CPU cores.

      @raise Sys_error
        if per-core statistics are unavailable or an error occurs. *)

  val compute : prev:t -> next:t -> stats
  (** [compute ~prev ~next] calculates CPU usage percentages between two
      samples.

      Computes the delta for each counter field and converts to percentages of
      total CPU time elapsed between samples. The sum of all fields in the
      returned [stats] equals [100.0].

      Returns {!zero_stats} if no time has elapsed between samples (i.e., all
      counters are identical). This is not an error condition. *)
end

(** Memory statistics (instantaneous). *)
module Mem : sig
  type t = {
    total : int64;  (** Total physical memory. *)
    used : int64;
        (** Memory in use (calculated via platform-specific formula). *)
    free : int64;  (** Free memory available for allocation. *)
    available : int64;
        (** Memory available for starting new applications without swapping.

            On macOS, approximated as [free + inactive + purgeable]. *)
    compressed : int64;  (** Compressed memory (macOS only, [0L] on Linux). *)
    wired : int64;
        (** Wired (non-pageable) memory (macOS only, [0L] on Linux). *)
    active : int64;  (** Active memory pages (macOS only, [0L] on Linux). *)
    inactive : int64;
        (** Inactive memory pages (Linux: cached + sreclaimable - shmem). *)
    purgeable : int64;  (** Purgeable memory (macOS) or buffers (Linux). *)
    speculative : int64;  (** Speculative memory (macOS only, [0L] on Linux). *)
    external_ : int64;  (** External memory (macOS only, [0L] on Linux). *)
    page_size : int64;  (** System page size in bytes. *)
    swap_total : int64;  (** Total swap space. *)
    swap_used : int64;  (** Used swap space. *)
  }
  (** Memory usage statistics.

      All size fields are in bytes. Memory statistics are instantaneous
      snapshots, not cumulative counters.

      Platform behavior:
      - {b Linux}: The [used] field is computed as
        [total - free - buffers - (cached + sreclaimable - shmem)].
      - {b macOS}: The [used] field is computed as
        [active + inactive + speculative + wired + compressed - purgeable -
         external], with [compressed] subtracted from the display value. *)

  val sample : unit -> t
  (** [sample ()] returns current memory usage statistics.

      @raise Sys_error
        if memory statistics are unavailable on the current platform or an error
        occurs during sampling. *)
end

(** Network I/O statistics. *)
module Net : sig
  type t = {
    bytes_rx : int64;  (** Total bytes received. *)
    packets_rx : int64;  (** Total packets received. *)
    bytes_tx : int64;  (** Total bytes transmitted. *)
    packets_tx : int64;  (** Total packets transmitted. *)
  }
  (** Cumulative network I/O counters since boot.

      Aggregates all network interfaces except loopback. Counters are cumulative
      and monotonically increasing (until system reboot or counter overflow). *)

  type stats = {
    rx_bytes_per_sec : float;  (** Receive rate in bytes per second. *)
    rx_packets_per_sec : float;  (** Receive rate in packets per second. *)
    tx_bytes_per_sec : float;  (** Transmit rate in bytes per second. *)
    tx_packets_per_sec : float;  (** Transmit rate in packets per second. *)
  }
  (** Network I/O rates computed between two samples. *)

  val sample : unit -> t
  (** [sample ()] returns cumulative network I/O counters.

      @raise Sys_error
        if network statistics are unavailable on the current platform or an
        error occurs during sampling. *)

  val compute : prev:t -> next:t -> dt:float -> stats
  (** [compute ~prev ~next ~dt] calculates network I/O rates between two
      samples.

      Computes the delta for each counter and divides by [dt] to obtain rates
      per second. Negative deltas (e.g., from counter overflow or reboot) are
      treated as zero.

      @raise Invalid_argument if [dt <= 0.0]. *)
end

(** Disk I/O statistics. *)
module Disk_io : sig
  type t = {
    bytes_read : int64;  (** Total bytes read from disk. *)
    bytes_written : int64;  (** Total bytes written to disk. *)
    time_ms : int64;  (** Cumulative I/O time in milliseconds. *)
    num_disks : int64;  (** Number of physical disks included in aggregation. *)
  }
  (** Cumulative disk I/O counters since boot.

      Aggregates physical disks only, excluding virtual devices, partitions, and
      metadata devices. Counters are cumulative and monotonically increasing
      until system reboot.

      Platform behavior:
      - {b Linux}: Excludes virtual devices ([dm-*], [loop*], [md*], [zram*])
        and partitions (detected by prefix matching the parent device name). *)

  type stats = {
    read_bytes_per_sec : float;  (** Read rate in bytes per second. *)
    write_bytes_per_sec : float;  (** Write rate in bytes per second. *)
    utilization_percent : float;
        (** Disk utilization percentage (0.0 to 100.0). *)
  }
  (** Disk I/O rates and utilization computed between two samples. *)

  val sample : unit -> t
  (** [sample ()] returns cumulative disk I/O counters.

      Only physical disks are included; partitions and virtual devices are
      excluded.

      @raise Sys_error
        if disk statistics are unavailable on the current platform or an error
        occurs during sampling. *)

  val compute : prev:t -> next:t -> dt:float -> stats
  (** [compute ~prev ~next ~dt] calculates disk I/O rates and utilization
      between two samples.

      Computes the delta for each counter and divides by [dt] to obtain rates
      per second. Utilization is calculated as
      [(time_delta / (dt * 1000 * num_disks)) * 100], representing the
      percentage of time disks were actively performing I/O, capped at [100.0].

      If [num_disks] is [0L], [utilization_percent] is [0.0].

      @raise Invalid_argument if [dt <= 0.0]. *)
end

(** Filesystem statistics (instantaneous). *)
module Fs : sig
  type partition = {
    mount_point : string;  (** Mount point path (e.g., ["/"], ["/home"]). *)
    total_bytes : int64;  (** Total filesystem size. *)
    used_bytes : int64;  (** Used space (calculated as [total - free]). *)
    avail_bytes : int64;  (** Available space for unprivileged users. *)
  }
  (** Partition information for a single mounted filesystem.

      All size fields are in bytes. Represents a snapshot of filesystem usage at
      the time of sampling. *)

  type t = {
    total_bytes : int64;  (** Total filesystem size for the queried path. *)
    used_bytes : int64;  (** Used space for the queried path. *)
    avail_bytes : int64;  (** Available space for the queried path. *)
    partitions : partition list;
        (** All mounted partitions (excluding virtual filesystems). *)
  }
  (** Filesystem statistics for a specific path.

      All size fields are in bytes. Contains statistics for the filesystem
      containing the specified path, plus a list of all mounted partitions. *)

  val sample : ?path:string -> unit -> t
  (** [sample ?path ()] returns filesystem statistics for the specified path.

      Returns statistics for the filesystem containing [path] (default: ["/"]),
      along with a list of all mounted partitions via {!partitions}.

      Virtual and system filesystems (e.g., [devfs], [tmpfs], [proc], [sysfs])
      are excluded from the partitions list. Filesystems smaller than
      approximately 100 MB are also excluded.

      @raise Sys_error if the path does not exist or an error occurs. *)

  val partitions : unit -> partition list
  (** [partitions ()] returns a list of all mounted partitions.

      Enumerates mounted filesystems and queries their usage. Excludes
      virtual/system filesystems and small filesystems (< 100 MB).

      Returns an empty list if no partitions are found or an error occurs. *)
end

(** Process statistics. *)
module Proc : sig
  (** Process state. *)
  type state =
    | Running  (** Currently executing on CPU. *)
    | Sleeping  (** Interruptible sleep (waiting for event). *)
    | Disk_sleep  (** Uninterruptible sleep (waiting for I/O). *)
    | Stopped  (** Stopped by signal (e.g., SIGSTOP). *)
    | Zombie  (** Terminated but not yet reaped by parent. *)
    | Idle  (** Idle kernel thread. *)
    | Unknown  (** State could not be determined. *)

  (** Current process (self) statistics. *)
  module Self : sig
    type t = {
      utime : float;  (** Cumulative user-mode CPU time in seconds. *)
      stime : float;  (** Cumulative system-mode CPU time in seconds. *)
      rss_bytes : int64;  (** Resident set size (physical memory) in bytes. *)
      vsize_bytes : int64;  (** Virtual memory size in bytes. *)
    }
    (** Raw process snapshot for delta calculation.

        Contains cumulative CPU time and instantaneous memory usage for the
        current process. CPU times are in seconds (converted from
        platform-specific units by [Unix.times]). *)

    type stats = {
      cpu_percent : float;
          (** CPU usage percentage (0.0 to 100.0 per core, or total if
              [num_cores] provided). *)
      rss_bytes : int64;  (** Resident set size in bytes. *)
      vsize_bytes : int64;  (** Virtual memory size in bytes. *)
    }
    (** Computed process statistics. *)

    val sample : unit -> t
    (** [sample ()] returns raw CPU times and memory usage for the current
        process.

        Uses [Unix.times] for CPU times.

        @raise Sys_error if an error occurs during sampling. *)

    val compute : prev:t -> next:t -> dt:float -> num_cores:int option -> stats
    (** [compute ~prev ~next ~dt ~num_cores] computes CPU usage percentage
        between two samples.

        CPU percentage is calculated as
        [((utime_delta + stime_delta) / dt) * 100]. If [num_cores] is provided,
        the percentage is normalized by dividing by the number of cores,
        yielding a value in [0.0] to [100.0]. Without normalization, the value
        can exceed [100.0] on multi-core systems.

        The result is clamped to prevent spurious values from timing anomalies:
        - With [num_cores]: capped at [100.0]
        - Without [num_cores]: capped at [800.0]

        If [dt] is outside the range [(0.01, 10.0)], returns [0.0] for
        [cpu_percent] to avoid division by near-zero or implausibly large
        intervals.

        @raise Invalid_argument if [dt <= 0.0]. *)
  end

  (** Process table statistics. *)
  module Table : sig
    type t = {
      pid : int;  (** Process ID. *)
      ppid : int;  (** Parent process ID. *)
      name : string;  (** Process name (comm). *)
      cmdline : string;
          (** Full command line with arguments. Empty if unavailable. *)
      state : state;  (** Current process state. *)
      user : string;  (** Owner username. Empty if UID lookup fails. *)
      priority : int;  (** Scheduling priority. *)
      nice : int;  (** Nice value (-20 to 19). *)
      user_time : int64;  (** Cumulative user-mode CPU time in ticks. *)
      system_time : int64;  (** Cumulative system-mode CPU time in ticks. *)
      resident_size : int64;  (** Resident set size in bytes. *)
      virtual_size : int64;  (** Virtual memory size in bytes. *)
      num_threads : int;  (** Number of threads. *)
      num_running : int;
          (** Number of running threads (macOS only, [0] on Linux). *)
      faults : int64;  (** Page faults (macOS only, [0L] on Linux). *)
      mem_percent : float;
          (** Memory usage as percentage of total physical memory. *)
    }
    (** Raw process snapshot for delta calculation.

        Contains cumulative CPU time, state, and instantaneous memory/thread
        information for a process. CPU times are in platform-specific ticks
        (Linux jiffies or macOS Mach ticks). *)

    type stats = {
      pid : int;  (** Process ID. *)
      name : string;  (** Process name. *)
      cpu_percent : float;  (** CPU usage percentage between samples. *)
      mem_percent : float;
          (** Memory usage as percentage of total physical memory. *)
      rss_bytes : int64;  (** Resident set size in bytes. *)
    }
    (** Computed process statistics.

        Contains derived CPU percentage and filtered memory information. Only
        processes with non-zero CPU or memory usage are included. *)

    val sample : unit -> t list
    (** [sample ()] returns raw process snapshots for all running processes.

        Enumerates all processes visible to the current user and reads their
        statistics.

        Returns an empty list if an error occurs during enumeration. Individual
        process errors (e.g., process termination during sampling) are silently
        skipped. *)

    val compute : prev:t list -> next:t list -> dt:float -> stats list
    (** [compute ~prev ~next ~dt] calculates CPU percentages and filters
        processes.

        Matches processes by PID between [prev] and [next] samples. For matched
        processes, computes CPU percentage as:
        [(cpu_time_delta_ns / interval_ns) * 100].

        Only processes with non-zero [cpu_percent] or [mem_percent] are included
        in the result. New processes (in [next] but not [prev]) are included if
        their [mem_percent] is non-zero, with [cpu_percent] set to [0.0].

        @raise Invalid_argument if [dt <= 0.0]. *)
  end
end

(** {1 System Information} *)

val loadavg : unit -> float * float * float
(** [loadavg ()] returns the 1, 5, and 15 minute load averages. *)

val uptime : unit -> int64
(** [uptime ()] returns system uptime in seconds. *)
