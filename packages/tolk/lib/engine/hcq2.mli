(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. ISC License.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Compilation of queue dependencies and timeline instructions. *)

type call = { call : Tolk_uop.Uop.t; device : string; queue : string }
(** A kernel or bulk store assigned to its submission device and queue. *)

type plan = {
  queues : (string * string * Tolk_uop.Uop.t list) list;
      (** Device, queue and ordered instructions, in first-use submission order. *)
  timestamps : (string * Tolk_uop.Uop.t * Tolk_uop.Uop.t) list;
      (** Per-call device and start/end signal slots when profiling. *)
  timelines : Tolk_uop.Uop.t list;
      (** Per-batch timeline slots to fence before reusing command storage. *)
  independent_accesses : (Tolk_uop.Uop.t * Tolk_uop.Uop.t) list;
      (** Access pairs from unordered calls, with at least one write. *)
  signals : Tolk_uop.Uop.t list;
      (** Queue signals to re-arm before submission. *)
}

val timeline : string -> Tolk_uop.Uop.t
(** [timeline device] names the shared device signal and submitted value. *)

val plan : ?profile:bool -> call list -> plan
(** [plan ?profile calls] orders overlapping byte accesses across queues,
    emits necessary signals and waits, and advances each device timeline after
    its queues and peers finish. Same-queue calls use FIFO order. [profile]
    adds two timestamp instructions per call and defaults to [false]. *)

val ccall :
  ?host:string -> ?libs:string list -> ?after:Tolk_uop.Uop.t list -> name:string -> dtype:Tolk_uop.Dtype.t ->
  Tolk_uop.Uop.t list -> Tolk_uop.Uop.t
(** [ccall ?host ?libs ?after ~name ~dtype args] calls a C symbol resolved at link time.
    [host] defaults to ["CPU"]. [after] sequences effects before the call. *)

val patch :
  ?blob:string -> ?after:Tolk_uop.Uop.t list -> Tolk_uop.Uop.t -> (int * Tolk_uop.Uop.t) list -> Tolk_uop.Uop.t
(** [patch ?blob ?after buffer rows] sequences byte initialization and typed word
    stores at the byte offsets in [rows]. Static stores move to link time.
    Runtime stores wait for [after] before modifying command storage. *)

val compile : ?profile:bool -> Tolk_uop.Uop.t -> Tolk_uop.Uop.t
(** [compile ?profile linear] encodes calls supported by their device's queue hooks into
    host programs, combining adjacent calls from the same peer group. Other
    calls remain individual dispatches. *)
