(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The formats of a GPU's GC that its driver writes: the descriptors (MQD)
    its compute queues load, and the index that selects the shader arrays
    its registers reach. Every value is pure, and any domain may call it. *)

(** {1:mqd Queue descriptors} *)

type queue = {
  ring : int;  (** The ring's GPU address, 256-byte aligned. *)
  ring_bytes : int;  (** Its size, a power of two. *)
  read : int;  (** Where the queue writes its read position. *)
  write : int;  (** Where it polls its write position. *)
  eop : int;  (** Its end-of-pipe buffer's GPU address, 256-byte aligned. *)
  eop_bytes : int;  (** Its size, a power of two. *)
  doorbell : int;  (** Its doorbell index. *)
}
(** The type for what a compute queue is made of. *)

val mqd :
  Gpu.t -> queue -> base:int -> [ `Kiq | `Pm4 | `Aql ] -> xcc:int -> string
(** [mqd g q ~base kind ~xcc] is the descriptor of queue [q] on die [xcc] of
    [g]'s dies, laid out for [g]'s GC. The hardware reads it at the address
    the memory controller gives [base]. The queue is of [kind]:
    {ul
    {- [`Pm4] reads PM4 packets.}
    {- [`Kiq] reads PM4 packets, and is privileged and of the kernel
       driver's kind.}
    {- [`Aql] reads AQL packets, and a GPU of several dies spreads its work
       across them, its dies' descriptors 4 KiB apart.}}

    Raises [Invalid_argument] if [g]'s GC is not of major 9, 11 or 12. *)

(** {1:arrays Shader arrays} *)

val index : Gpu.t -> [ `Array of int * int | `All ] -> int
(** [index g sel] is the value of [g]'s [GRBM_GFX_INDEX] that directs
    register accesses to shader array [a] of shader engine [e] on every
    instance ([`Array (e, a)]), or broadcasts them to all ([`All]). *)
