(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The dashboard's panel of system usage. *)

type gpu = {
  util_pct : float option;
  mem_used_gb : float option;
  mem_used_pct : float option;
  temp_c : float option;
  power_w : float option;
}
(** The type for a GPU's latest values, [None] for a metric the run has not
    logged. *)

type values = {
  cpu_user : float option;  (** Percent. *)
  cpu_system : float option;  (** Percent. *)
  mem_pct : float option;
  mem_gb : float option;
  proc_cpu : float option;  (** The run's process, in percent. *)
  proc_mem_mb : float option;  (** The run's process, resident. *)
  disk_read_mbs : float option;  (** MB per second. *)
  disk_write_mbs : float option;  (** MB per second. *)
  disk_util_pct : float option;
  gpus : gpu list;  (** GPU [i] is the run's [sys/gpu]{i i}[_*] metrics. *)
}
(** The type for the latest system values, [None] for a metric the run has
    not logged. *)

val view :
  values -> history_for_tag:(string -> (int * float) list) -> 'a Mosaic.t
(** [view v ~history_for_tag] is up to seven lines, and three per GPU, that
    show [v] as bars and values, each followed by a sparkline of the last 10
    samples of its [sys/] metric in [history_for_tag]. A line whose values are
    [None] is left out. *)
