(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Two-level segregated fit allocators of address ranges.

    An allocator hands out blocks of a range of addresses. Free blocks are
    indexed by the most significant bit of their size, then by the next four
    bits, in bitmaps searched with find-first-set; each class keeps a doubly
    linked list. A request is rounded up to the next class before the search, so
    any block found fits, and allocating and freeing take a time bounded
    independently of what the allocator holds. A freed block merges with its
    free neighbours immediately.

    The bound costs fit. A request of [n] bytes aligned to [a] succeeds while a
    free block of [2 * (max n 16 + a)] bytes exists, and may fail with a smaller
    one that would fit.

    The allocator's owner calls one function at a time. An exception inside a
    call, such as an asynchronous one where it allocates or polls, leaves
    the allocator as the call found it or as the call made it.

    {b Reference.} M. Masmano, I. Ripoll, A. Crespo and J. Real. "TLSF: a new
    dynamic memory allocator for real-time systems".
    {e Proceedings of the 16th Euromicro Conference on Real-Time Systems}, 2004.
*)

type t
(** The type for allocators. *)

val create : base:int -> int -> t
(** [create ~base n] is an allocator of the [n] addresses from [base] on, all
    free. The smallest block it hands out is 16 bytes.

    Raises [Invalid_argument] if [n < 0]. *)

val base : t -> int
(** [base a] is the first address [a] manages. *)

val length : t -> int
(** [length a] is the number of addresses [a] manages. *)

val alloc : ?align:int -> ?below:int -> t -> int -> int option
(** [alloc a n] is the first address of a new block of at least [n] bytes, a
    multiple of [align] (defaults to [1]), or [None] as the fit bound allows.
    With [below], the block ends at or below that address. A bound below the
    range's end gives the first such block in address order, or [None] if no
    free block holds it there, and that search walks the blocks. A bound at or
    past the end bounds nothing.

    Raises [Invalid_argument] if [n < 0] or [align <= 0]. *)

val free : t -> int -> bool
(** [free a x] frees the block {!alloc} returned at [x], and is [true] iff there
    was one. It is [false], changing nothing, if no block of [a] starts at [x]
    or that block is free already. *)
