(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** System monitoring.

    {!Stat} samples operating-system statistics on demand. {!start} runs those
    samplers on a background thread and logs the results to a {!Munin.Session}.

    This library owns the [sys/] metric-key prefix: every metric it logs is
    named [sys/...]. Keep your own metric keys out of that prefix.

    {1:platform Platform support}

    Supported platforms: Linux and macOS. Platform-specific behavior is
    documented per module. Some metrics have limited availability on certain
    platforms (e.g., macOS CPU counters populate only user/nice/system/idle
    fields). *)

(** {1:sampling Sampling} *)

module Stat : module type of Sysstat
(** Stateless, poll-based sampling of operating-system statistics.

    Each module samples instantaneous or cumulative values. CPU, network, and
    disk I/O statistics are cumulative since boot and require two samples to
    compute usage percentages; memory statistics are instantaneous. The caller
    manages state and sampling intervals. *)

(** {1:monitor Background monitoring}

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

type t
(** The type for background monitors. *)

val start : ?interval:float -> Munin.Session.t -> t
(** [start session] begins periodic system monitoring of [session].

    All [sys/] metrics are defined with [~summary:`Last] so the final sampled
    value appears in run summaries.

    [interval] defaults to [2.0] seconds. The first sample is taken after one
    interval. The monitor thread is a daemon thread. *)

val stop : t -> unit
(** [stop t] signals the monitoring thread to exit and blocks until it
    terminates. Safe to call multiple times. *)
