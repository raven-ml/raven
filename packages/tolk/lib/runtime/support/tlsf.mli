(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Two-Level Segregated Fit allocator.

    Manages a contiguous address range with O(1) best-fit allocation and
    O(1) deallocation with coalescing. Free blocks are indexed by two
    levels of buckets:
    {ul
    {- Level 1 is the most significant bit of the block size.}
    {- Level 2 subdivides each L1 range into [2{^l2_cnt}] entries.}}

    Allocation finds the smallest free block that fits, splitting the
    remainder. Deallocation merges the freed block with its neighbours. *)

(** {1:types Types} *)

type t
(** The type for TLSF allocators. Mutable. *)

(** {1:constructors Constructors} *)

val create :
  size:int ->
  ?base:int ->
  ?block_size:int ->
  ?lv2_cnt:int ->
  unit ->
  t
(** [create ~size ?base ?block_size ?lv2_cnt ()] is a TLSF allocator
    managing [size] bytes starting at virtual address [base].

    [base] defaults to [0]. [block_size] is the minimum allocation
    granularity and defaults to [16]. [lv2_cnt] is the number of
    level-2 subdivisions per level-1 bucket and defaults to [16]. *)

(** {1:operations Operations} *)

val alloc : t -> int -> ?align:int -> unit -> int
(** [alloc t size ?align ()] is the start address of a newly allocated
    region of [size] bytes. The returned address is a multiple of [align].

    [align] defaults to [1]. The actual allocation is at least
    [block_size] bytes.

    Raises [Out_of_memory] if no free block can satisfy the request. *)

val free : t -> int -> unit
(** [free t addr] returns the block at [addr] to the free pool and
    merges it with any adjacent free blocks. [addr] must have been
    previously returned by {!alloc} on the same allocator. *)
