(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Scratch memory: the private memory of kernels' work-items.

    A GPU gives each wave that runs with scratch a slice of one buffer, sized
    for the waves every compute unit runs at once ({!Gpu.scratch_slots}). A
    lane's share is the kernel's private segment
    ({!Code_object.kernel.private_segment}), at least 128 bytes, rounded up to
    the generation's granule: 16 bytes on GFX9, 4 after. *)

val size : Gpu.t -> int -> int
(** [size g n] is the bytes of the scratch buffer of [g] for kernels of [n]
    bytes per lane: a wave's 64 lanes, for every scratch slot of every compute
    unit of every die. *)

val tmpring : Gpu.t -> int -> int
(** [tmpring g n] is the word of [COMPUTE_TMPRING_SIZE] for kernels of [n] bytes
    per lane: the size of a wave's scratch, in the generation's units, and the
    waves one die's scratch serves, divided among its shader engines after GFX9
    and at most every slot's.

    Raises [Invalid_argument] if [g]'s GC has no such register. *)

val descriptor : Gpu.t -> base:int -> int -> string
(** [descriptor g ~base n] is the 16 bytes of the buffer descriptor of the
    scratch buffer of [n] bytes at address [base], split evenly among [g]'s
    dies, as a queue that dispatches AQL packets hands it to kernels.

    Raises [Invalid_argument] if [g]'s GC has no buffer descriptor layout. *)
