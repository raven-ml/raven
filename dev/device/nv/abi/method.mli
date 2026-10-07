(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Channel methods.

    A channel runs pushbuffers. In a pushbuffer, a header word names a
    subchannel, a method and a count; the [count] words after it are the
    arguments of that method and of the methods after it, one each. A channel
    has eight subchannels, each bound to an engine by {!set_object}. The host's
    methods, its semaphores, run on every channel.

    Each value of this module is the words of one operation, headers included.
    An operand that may be unknown when encoding, such as an address or a
    payload, is of the caller's type ['v]; an operand the operation fixes, such
    as a class, is an integer.

    Addresses are the GPU's virtual addresses. *)

(** {1:engines Engines} *)

(** The type for the engines a channel binds. *)
type engine =
  | Compute  (** The compute engine, on subchannel [1]. *)
  | Copy  (** The copy engine, on subchannel [4]. *)

val set_object : engine -> int -> 'v Packet.t
(** [set_object e cls] binds the engine of class [cls] to [e]'s subchannel. A
    channel runs no method of [e] before it. *)

(** {1:host Semaphores}

    A semaphore is a 64-bit word in memory, 8-byte aligned, below [2{^40}]. *)

val acquire : 'v -> 'v -> 'v Packet.t
(** [acquire addr v] stops the channel until the semaphore [w] at [addr] is at
    least [v] in circular order: until [w - v], computed modulo [2{^64}], is
    below [2{^63}]. *)

val release : 'v -> 'v -> 'v Packet.t
(** [release addr v] waits for the channel's earlier work to complete, every
    engine idle, then writes [v] into the semaphore at [addr] and raises a
    non-stalling interrupt. The channel's later methods run after the write. *)

val release_stamp : 'v -> 'v -> 'v Packet.t
(** [release_stamp addr v] is {!release} without the interrupt, also writing the
    GPU's timer, in nanoseconds, into the 8 bytes at [addr + 8]. [addr] is
    16-byte aligned. *)

(** {1:compute Compute} *)

val shared_memory_window : 'v -> 'v Packet.t
(** [shared_memory_window addr] makes kernels' shared memory appear at the
    address [addr]. *)

val local_memory_window : 'v -> 'v Packet.t
(** [local_memory_window addr] makes kernels' local memory appear at the address
    [addr]. *)

val local_memory : 'v -> per_tpc:'v -> 'v Packet.t
(** [local_memory addr ~per_tpc] gives kernels the local memory at [addr]:
    [per_tpc] bytes for each texture processing cluster, for every streaming
    multiprocessor. *)

val invalidate_caches : 'v Packet.t
(** [invalidate_caches] invalidates the compute engine's instruction, data and
    constant caches, without waiting for its work to complete. *)

val schedule : 'v -> 'v Packet.t
(** [schedule addr] schedules the launch descriptor at [addr], 256-byte aligned
    and below [2{^40}], and the descriptors chained to it ({!Qmd.chain}). The
    channel goes on without waiting for the launches: launches scheduled one
    after another run at the same time, and only a chain orders them. *)

(** {1:copy Copies} *)

val max_copy : int
(** [max_copy] is the most bytes one {!copy} moves, [2{^31}]. *)

val copy : dst:'v -> src:'v -> 'v -> 'v Packet.t
(** [copy ~dst ~src n] copies the [n] bytes at [src] to [dst] on the copy
    engine. [n] is at most {!max_copy}; a longer copy is several, at increasing
    offsets. *)

val copy_release : 'v -> 'v -> 'v Packet.t
(** [copy_release addr v] writes the low 32 bits of [v] into the 4 bytes at
    [addr], 4-byte aligned, once the copy engine's earlier copies are complete
    and their writes visible. *)

val copy_stamp : 'v -> 'v Packet.t
(** [copy_stamp addr] writes [0] into the 8 bytes at [addr], 16-byte aligned,
    and the GPU's timer, in nanoseconds, into the 8 bytes after, once the copy
    engine's earlier copies are complete. *)
