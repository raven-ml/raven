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
  timelines : Tolk_uop.Uop.t list;
      (** Per-batch timeline slots to fence before reusing command storage. *)
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
