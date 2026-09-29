(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Two-level segregated fit allocators of address ranges.

    An allocator hands out blocks of a range of addresses: a device's physical
    memory or a virtual address space. It indexes free blocks by the most
    significant bit of their size, then by a fixed number of subdivisions of
    that power of two, so an allocation finds a block that fits in a bounded
    number of steps. It splits the block it takes, and a freed block merges with
    its free neighbours.

    An allocator is not synchronized: its owner serializes calls. *)

type t
(** The type for allocators. *)

val create : ?block:int -> ?subdivisions:int -> base:int -> int -> t
(** [create ~base n] is an allocator of the [n] addresses from [base] on, all
    free. [block] (defaults to [16]) is the smallest block it hands out, and
    [subdivisions] (defaults to [16], a power of two) the number of second level
    buckets per power of two.

    Raises [Invalid_argument] if [n < 0], [block <= 0], or [subdivisions] is not
    a positive power of two. *)

val base : t -> int
(** [base a] is the first address [a] manages. *)

val length : t -> int
(** [length a] is the number of addresses [a] manages. *)

val alloc : ?align:int -> t -> int -> int option
(** [alloc a n] is the first address of a new block of at least [n] bytes, a
    multiple of [align] (defaults to [1]), or [None] if no free block fits.

    Raises [Invalid_argument] if [n < 0] or [align] is not positive. *)

val free : t -> int -> unit
(** [free a x] frees the block {!alloc} returned at [x].

    Raises [Invalid_argument] if no block of [a] starts at [x] or it is free
    already. *)
