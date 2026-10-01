(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Memory planning.

    Placing blocks in a range of addresses, without touching memory: the
    addresses are numbers, for whoever allocates the memory they stand for. *)

(** Two-level segregated fit allocators.

    An allocator hands out blocks of a range of addresses. It indexes its free
    blocks by the most significant bit of their size, then by a fixed number of
    subdivisions of that power of two, so that an allocation finds the smallest
    block that fits in a bounded number of steps. It splits the block it takes,
    and a freed block merges with its free neighbours. *)
module Tlsf_allocator : sig
  type t
  (** The type for allocators. Allocators are mutable. *)

  val create : ?base:int -> ?block_size:int -> ?lv2_cnt:int -> int -> t
  (** [create ~base ~block_size ~lv2_cnt size] is an allocator of the [size]
      addresses from [base] (default [0]) on, all free. [block_size] (default
      [16]) is the smallest block it hands out. [lv2_cnt] (default [16]) sets
      the subdivisions of each power of two: there are [2{^b - 1}], where [b] is
      the number of bits of [lv2_cnt].

      Raises [Invalid_argument] if [size] is negative, [block_size] or [lv2_cnt]
      is not positive, or [block_size] has fewer bits than [lv2_cnt]. *)

  val alloc : ?align:int -> t -> int -> int option
  (** [alloc ~align a n] is the first address of a new block of
      [max block_size n] addresses, a multiple of [align] (default [1]) after
      [base]. The block is cut from a free block of the smallest subdivision
      whose blocks all fit it, the one that joined that subdivision first; it is
      [None] if there is none.

      Raises [Invalid_argument] if [align] is not positive. *)

  val free : t -> int -> unit
  (** [free a x] frees the block that {!alloc} returned at [x].

      Raises [Invalid_argument] if no block of [a] starts at [x], or it is free
      already. *)
end
