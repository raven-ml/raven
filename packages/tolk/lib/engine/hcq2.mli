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

val stage_copies :
  resolve:(Tolk_uop.Uop.t -> Device.Buffer.t) -> Tolk_uop.Uop.t -> Tolk_uop.Uop.t option
(** [stage_copies ~resolve linear] replaces bulk copies with unsupported queue
    imports by two queue legs through alternating 64 MiB host slots. Storage is
    private to the returned schedule. [resolve] supplies current call bindings.
    Returns [None] if no copy needs staging or staging cannot be imported.
    Other allocation and device failures propagate. *)

val lower_call : devices:string list -> Tolk_uop.Uop.t -> Tolk_uop.Uop.t
(** [lower_call ~devices sink] lowers the explicit queue submissions in [sink]
    to a host program call through the same encoding, linking and address-table
    rules as {!compile}. The call has no tensor arguments or execution fallback.
    [devices] lists the nonempty set of devices whose queues must complete.
    The returned call must be linked before execution. *)

val compile :
  to_program:(Device.t -> Tolk_uop.Uop.t -> Tolk_uop.Uop.t) ->
  ?profile:bool -> Tolk_uop.Uop.t -> Tolk_uop.Uop.t
(** [compile ~to_program ?profile linear] encodes supported queue calls into
    host programs. Interleaved peer groups combine when byte dependencies
    permit reordering; ordinary calls remain dispatch boundaries. Reordered
    writable accesses participate in the runtime alias checks, including
    bindings used only by another group. [to_program device sink] compiles
    byte-copy kernels for stores assigned to compute queues. *)
