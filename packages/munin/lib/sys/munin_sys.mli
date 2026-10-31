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

    Every sampler reads the system on Linux and macOS. Platform-specific
    behavior is documented per module. Some metrics have limited availability
    on certain platforms (e.g., macOS CPU counters populate only
    user/nice/system/idle fields). Windows keeps no load average, and munin
    reads no GPU statistics there. *)

(** {1:sampling Sampling} *)

module Stat : module type of Sysstat
(** Stateless, poll-based sampling of operating-system statistics.

    Each module samples instantaneous or cumulative values. CPU, network, and
    disk I/O statistics are cumulative since boot and require two samples to
    compute usage percentages; memory statistics are instantaneous. The caller
    manages state and sampling intervals. *)

(** {1:monitor Background monitoring}

    {!start} spawns a background thread that periodically samples CPU, memory,
    process, disk I/O and GPU statistics and logs them as scalar metrics via
    {!Munin.Session}.

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
    - [sys/disk_util_pct] — disk utilization percentage

    {b GPU} [i], the [i]th GPU the monitor saw by {!Stat.Gpu.field-id}: in
    the order of {!Stat.Gpu.sample} when it starts, then in the order GPUs
    appear:
    - [sys/gpu]{i i}[_util_pct] — how busy the GPU is (0–100), on NVIDIA a
      mean over the interval
    - [sys/gpu]{i i}[_mem_used_gb] — GPU memory used in GB
    - [sys/gpu]{i i}[_mem_used_pct] — GPU memory used, percent of its own
    - [sys/gpu]{i i}[_temp_c] — temperature in Celsius
    - [sys/gpu]{i i}[_power_w] — power in watts

    Each group of metrics above comes from one sampler of {!Stat}, the
    per-process group from {!Stat.Proc.Self.sample}. A group whose sampler
    raises [Sys_error] when the monitor starts is neither defined nor logged.
    A GPU's metric is defined when the monitor starts if the first sample has
    its value, else the first time a sample has it, as the load of a GPU that
    was suspended at the start. A sample that raises later leaves its group
    out of that interval's samples, and the group's next rates are means over
    both intervals; a GPU missing from a sample, or a value its sample lacks,
    is left out of that interval. An interval over which the wall clock
    stepped back logs no rates. *)

val gpu_key : int -> string -> string
(** [gpu_key i name] is GPU [i]'s metric [name], [sys/gpu]{i i}[_]{i name},
    such as ["sys/gpu1_util_pct"] for [gpu_key 1 "util_pct"]. *)

val gpu_metric : string -> (int * string) option
(** [gpu_metric key] is [Some (i, name)] if [key] is {!gpu_key}[ i name],
    and [None] for any other key. *)

type t
(** The type for background monitors. *)

val start : ?interval:float -> Munin.Session.t -> t
(** [start session] begins periodic system monitoring of [session].

    Every [interval] seconds the monitor logs one sample per metric. Rates
    ([sys/cpu_*], [sys/proc_cpu_pct], [sys/disk_*]) are means over the interval,
    computed from cumulative counters; memory and GPU metrics are the value
    at the sample's instant.

    [interval] defaults to [15.0] seconds. The first sample is taken after one
    interval. The monitor thread is a daemon thread. *)

val stop : t -> unit
(** [stop t] wakes the monitoring thread and blocks until it terminates. It
    waits for a sample in progress, never for the rest of the interval. Safe to
    call multiple times. *)
