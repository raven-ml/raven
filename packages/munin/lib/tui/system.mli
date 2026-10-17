(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The dashboard's panel of system usage. *)

type values = {
  cpu_user : float;  (** Percent. *)
  cpu_system : float;  (** Percent. *)
  mem_pct : float;
  mem_gb : float;
  proc_cpu : float;  (** The run's process, in percent. *)
  proc_mem_mb : float;  (** The run's process, resident. *)
  disk_read_mbs : float;  (** MB per second. *)
  disk_write_mbs : float;  (** MB per second. *)
  disk_util_pct : float;
}
(** The type for the latest system values. *)

val view :
  values -> history_for_tag:(string -> (int * float) list) -> 'a Mosaic.t
(** [view v ~history_for_tag] is seven lines that show [v] as bars and
    values, each followed by a sparkline of the last 10 samples of its
    [sys/] metric in [history_for_tag]. *)
