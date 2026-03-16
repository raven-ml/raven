(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** System monitoring.

    Provides stateless, poll-based system metrics sampling and background
    session monitoring. Each module samples instantaneous or cumulative values
    from the operating system. CPU, network, and disk I/O statistics are
    cumulative since boot and require two samples to compute usage percentages;
    memory statistics are instantaneous.

    {1:platform Platform support}

    Supported platforms: Linux and macOS. Platform-specific behavior is
    documented per module. Some metrics have limited availability on certain
    platforms (e.g., macOS CPU counters populate only user/nice/system/idle
    fields).

    {1:background Background monitoring}

    {!start} spawns a background thread that periodically samples CPU, memory,
    and process statistics and logs them as scalar metrics via {!Munin.Session}.

    Logged metrics (all with [sys/] prefix):

    {b System-wide:}
    - [sys/cpu_user] — user CPU percentage (0–100)
    - [sys/cpu_system] — system CPU percentage (0–100)
    - [sys/mem_used_pct] — memory usage percentage (0–100)
    - [sys/mem_used_gb] — memory used in GB

    {b Per-process:}
    - [sys/proc_cpu_pct] — process CPU percentage
    - [sys/proc_mem_mb] — process resident set size in MB

    {b Disk I/O:}
    - [sys/disk_read_mbs] — disk read rate in MB/s
    - [sys/disk_write_mbs] — disk write rate in MB/s
    - [sys/disk_util_pct] — disk utilization percentage *)

include module type of Sysstat
(** @inline *)

(** {1 Background monitoring} *)

type t
(** The type for background monitors. *)

val start : Session.t -> ?interval:float -> unit -> t
(** [start session ~interval ()] begins periodic system monitoring.

    All [sys/] metrics are defined with [~summary:`Last] so the final sampled
    value appears in run summaries.

    [interval] defaults to [2.0] seconds. The first sample is taken after one
    interval. The monitor thread is a daemon thread. *)

val stop : t -> unit
(** [stop t] signals the monitoring thread to exit and blocks until it
    terminates. Safe to call multiple times. *)
