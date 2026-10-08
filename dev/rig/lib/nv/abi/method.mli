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

    Addresses are the GPU's virtual addresses. Compute methods are those of the
    compute classes from AMPERE_COMPUTE_B ([0xc7c0]) on, copy methods those of
    the copy classes from AMPERE_DMA_COPY_B ([0xc7b5]) on.

    {b References.} NVIDIA's class headers in open-gpu-kernel-modules 570.144:
    [clc56f.h] (the host's methods, pushbuffer headers), [clc7c0.h] (compute)
    and [clc7b5.h] (copy). *)

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

val release : Packet.scope -> 'v -> 'v -> 'v Packet.t
(** [release s addr v] waits for the channel's earlier work to complete, every
    engine idle, then writes the 64-bit [v] into the semaphore at [addr], so
    that readers of scope [s] who see [v] see the work's writes. The channel's
    later methods run after the write. *)

val release_stamp : Packet.scope -> 'v -> 'v -> 'v Packet.t
(** [release_stamp s addr v] is [release s addr v], also writing the GPU's
    timer, in nanoseconds, into the 8 bytes at [addr + 8]. [addr] is 16-byte
    aligned. *)

val interrupt : 'v Packet.t
(** [interrupt] raises a non-stalling interrupt when the channel reaches it. It
    waits for nothing: it may reach the host before the write of a release just
    before it is visible. *)

(** {1:compute Compute} *)

val shared_memory_window : 'v -> 'v Packet.t
(** [shared_memory_window addr] makes kernels' shared memory appear at the
    address [addr]. *)

val local_memory_window : 'v -> 'v Packet.t
(** [local_memory_window addr] makes kernels' local memory appear at the address
    [addr]. *)

val local_memory : 'v -> per_tpc:'v -> 'v Packet.t
(** [local_memory addr ~per_tpc] gives kernels the local memory at [addr],
    [per_tpc] bytes for each texture processing cluster
    ({!Local_memory.field-per_tpc}), and lets every streaming multiprocessor use
    it. Launches scheduled after it use it. A launch scheduled before it must
    have completed: a {!release} separates them, and the caller keeps the memory
    until the launches that use it complete. *)

val invalidate_caches : Packet.scope -> 'v Packet.t
(** [invalidate_caches s] invalidates the compute engine's caches that would
    hide writes of scope [s] from launches scheduled after it: at [Agent], at
    least its data and constant caches; at [System], its instruction cache too.
    It does not wait for the engine's work to complete. *)

val wait_for_idle : 'v Packet.t
(** [wait_for_idle] holds the compute engine's later methods until the launches
    scheduled before it completed. A launch scheduled after it reads what those
    wrote. *)

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

val copy_release : Packet.scope -> 'v -> 'v -> 'v Packet.t
(** [copy_release s addr v] writes the 64-bit [v] into the 8 bytes at [addr],
    8-byte aligned, once the copy engine's earlier copies are complete, so that
    readers of scope [s] who see [v] see their writes. *)

val copy_release_stamp : Packet.scope -> 'v -> 'v -> 'v Packet.t
(** [copy_release_stamp s addr v] is [copy_release s addr v], also writing the
    GPU's timer, in nanoseconds, into the 8 bytes at [addr + 8]. [addr] is
    16-byte aligned. *)
