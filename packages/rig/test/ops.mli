(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Operations on a Polled device, which the failure walks ([test_walk]) and
    the interrupt walks ([test_interrupt]) run. *)

val why : string
(** [why] is the reason of the faults the walks inject. *)

(** How a walk fails an operation: from its [n]-th fallible call on, every
    call faults; its [n]-th call refuses; from its [n]-th call on, every call
    refuses, as a device out of memory does; or a function it gave rig raises
    [Exit]. *)
type failure = Fault of int | Refuse of int | Exhaust of int | Raise

(** An operation on a device: [prepare d] makes what it needs, and is the
    operation, told the failure it meets. It checks the values it gets, and
    raises on one rig.mli does not allow. [raises] says whether it gives rig
    a function, which raises under [Raise]. *)
type op = {
  name : string;
  opener : string -> Rig.t * Rig_support.Polled.t;
  prepare : Rig.t -> failure option -> unit;
  raises : bool;
}

val op :
  ?opener:(string -> Rig.t * Rig_support.Polled.t) ->
  ?raises:bool ->
  string ->
  (Rig.t -> failure option -> unit) ->
  op
(** [op ~opener ~raises name prepare] is an operation. [opener] defaults to
    {!Rig_support.Polled.open_}. *)

val fresh : string -> string
(** [fresh what] is a device name no other call returns. *)

val page : int
(** [page] is the size of the operations' larger buffers. *)

val collect : unit -> unit
(** [collect ()] runs two full collections, so chains of finalisers run. *)

val once : ?buffers:Rig.Buffer.t array -> Rig.Submission.Run.t -> Rig.Point.t
(** [once ~buffers r] submits [r]. *)

val ops : op list
(** [ops] are the operations on a device that walks fail. *)

val staged : op list
(** [staged] are the copies through staging memory. *)

val queued : op * op list
(** [queued] are the queued copies and fill, each waited for: a copy of boxes
    on a device's copy engine, and those through staging memory. *)

val count : string -> Rig_support.Polled.t -> int
(** [count call p] is the number of [p]'s logged calls named [call]. *)

val peer : unit -> Rig.t
(** [peer ()] is the device whose memory the borrow operations borrow, which
    this process opens at its first call, and again after a loss. *)

val peer_devices : unit -> Rig.t list
(** [peer_devices ()] is the device whose memory the borrow operations
    borrow, if this process opened it. *)

val settle : Rig.t -> unit
(** [settle d] collects what the caller dropped and returns memory in the
    order it comes back: drains of the host, the peer and [d], their caches
    freed, a collection, and a drain of the host again. *)
