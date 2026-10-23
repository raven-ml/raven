(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Virtual address spaces of GPUs.

    A space is a range of virtual addresses that the GPUs of one vendor share.
    An address in it means the same memory on each GPU that maps it and, for
    system memory a GPU allocates, in the process. A space hands out parts of
    the range; {!Page_table} maps them. Any domain may use a space. *)

type t
(** The type for virtual address spaces. *)

val create : base:int -> int -> t
(** [create ~base n] is the [n] virtual addresses from [base] on, all free. It
    allocates nothing until the first {!alloc}, so a vendor's space costs a
    program that drives none of its GPUs nothing.

    Raises [Invalid_argument] if [base < 0], [n < 0] or [base + n > max_int]. *)

val base : t -> int
(** [base s] is [s]'s first address. *)

val length : t -> int
(** [length s] is [s]'s number of addresses. *)

val alloc : ?align:int -> t -> int -> int option
(** [alloc s n] is the first of [n] free addresses of [s], aligned to [a], the
    larger of [align] (defaults to 4096) and the largest power of two not above
    [n], so that large ranges map with large pages. It is [Some _] while [s]
    has a free range of [2 * (max n 16 + a)] addresses, and may be [None] with
    a smaller free range that would fit. An allocation and a free take a time
    bounded independently of what [s] holds.

    Raises [Invalid_argument] if [n <= 0] or [align] is not a positive power of
    two. *)

val free : t -> int -> unit
(** [free s a] frees the range {!alloc} returned at [a].

    Raises [Invalid_argument] if no allocated range of [s] starts at [a]. *)

(**/**)

val taken : t -> int
(* [taken s] is the number of ranges of [s] allocated and not freed, for the
   suites that check that a driver gives back what it takes, in a time that
   grows with the ranges [s] holds. *)
