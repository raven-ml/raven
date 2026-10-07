(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Two-level segregated fit allocators of address ranges (private).

    An allocator hands out blocks of a range of addresses. Free blocks are
    indexed by the most significant bit of their size, then by the next four
    bits, in bitmaps searched with find-first-set; each class keeps a doubly
    linked list. A request is rounded up to the next class before the search, so
    any block found fits, and allocating and freeing take a time bounded
    independently of what the allocator holds. A freed block merges with its
    free neighbours at once.

    The price of the bound is the fit: a request of [n] bytes aligned to [a]
    succeeds while a free block of [2 * (n + a)] bytes exists, and may fail with
    a smaller one that would fit.

    Not synchronized: the owner serializes calls.

    {b Reference.} M. Masmano, I. Ripoll, A. Crespo and J. Real. "TLSF: a new
    dynamic memory allocator for real-time systems".
    {e Proceedings of the 16th Euromicro Conference on Real-Time Systems}, 2004.
*)

type t
(** The type for allocators. *)

val create : ?block:int -> base:int -> int -> t
(** [create ~base n] is an allocator of the [n] addresses from [base] on, all
    free. [block] (defaults to [16]) is the smallest block it hands out.

    Raises [Invalid_argument] if [n < 0] or [block <= 0]. *)

val base : t -> int
(** [base a] is the first address [a] manages. *)

val length : t -> int
(** [length a] is the number of addresses [a] manages. *)

val alloc : ?align:int -> t -> int -> int option
(** [alloc a n] is the first address of a new block of at least [n] bytes, a
    multiple of [align] (defaults to [1]), or [None] as the fit bound allows.

    Raises [Invalid_argument] if [n < 0] or [align <= 0]. *)

val free : t -> int -> unit
(** [free a x] frees the block {!alloc} returned at [x].

    Raises [Invalid_argument] if no block of [a] starts at [x] or it is free
    already. *)
