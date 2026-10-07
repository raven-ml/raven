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
    per lane: the size of a wave's scratch, in units of 1024 bytes on GFX9 and
    of 256 bytes after, and the waves one die's scratch serves, divided among
    its shader engines after GFX9 and at most every slot's.

    Raises [Invalid_argument] if [g]'s GC has no such register. *)

val descriptor : Gpu.t -> base:int -> int -> string
(** [descriptor g ~base n] is the 16 bytes of the buffer descriptor of the
    scratch buffer of [n] bytes at address [base], split evenly among [g]'s
    dies, [n / g.xccs] bytes each, as a queue that dispatches AQL packets hands
    it to kernels. A die's share is the descriptor's 32-bit record count:
    [n / g.xccs] is less than [2{^32}].

    Raises [Invalid_argument] if [n] is negative, if [n / g.xccs] is [2{^32}] or
    more, or if [g]'s GC has no buffer descriptor layout. *)
