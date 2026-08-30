(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Bump allocator.

    Allocates from a contiguous address range by advancing a pointer.
    Individual regions are never freed; when the range is exhausted the
    pointer wraps back to the start and earlier addresses are reused
    (or allocation fails when wrapping is disabled). *)

(** {1:types Types} *)

type t
(** The type for bump allocators. Mutable. *)

(** {1:constructors Constructors} *)

val create : size:int -> ?base:int -> ?wrap:bool -> unit -> t
(** [create ~size ?base ?wrap ()] is a bump allocator managing [size]
    bytes starting at virtual address [base].

    [base] defaults to [0]. [wrap] defaults to [true]: an allocation
    that does not fit in the remaining space restarts from the
    beginning of the range, handing out addresses that overlap earlier
    allocations. *)

(** {1:operations Operations} *)

val alloc : t -> int -> ?align:int -> unit -> int
(** [alloc t size ?align ()] is the start address of a newly allocated
    region of [size] bytes. The returned address is [base] plus an
    offset into the range that is a multiple of [align].

    [align] defaults to [1].

    Raises [Out_of_memory] if wrapping is disabled and the aligned
    region does not fit in the remaining space. *)
