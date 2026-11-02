(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Virtual address spaces of GPUs.

    A space is a range of virtual addresses that the GPUs of one vendor share.
    An address in it means the same memory on each GPU that maps it and, for
    system memory a GPU allocates, in the process. A space hands out parts of
    the range to holders ({!holders}); {!Page_table} maps them. Any domain may
    use a space. *)

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

(** {1:holders Holders}

    Every range a space hands out is held. A {e holder} holds ranges: the
    function of a GPU's open holds the ranges its memory lies at, and another
    GPU that maps that memory at the same addresses holds them too. A range
    goes back to its space once no holder holds it. *)

type holder
(** The type for holders of ranges. *)

type hold
(** The type for one holder's hold on one range. *)

val holder : unit -> holder
(** [holder ()] holds no range. *)

val alloc : ?align:int -> t -> holder -> int -> hold option
(** [alloc s h n] is [h]'s hold on [n] free addresses of [s], aligned to [a],
    the larger of [align] (defaults to 4096) and the largest power of two not
    above [n], so that large ranges map with large pages. It is [Some _] while
    [s] has a free range of [2 * (max n 16 + a)] addresses, and may be [None]
    with a smaller free range that would fit. An allocation and a free take a
    time bounded independently of what [s] holds.

    The range is [h]'s from the step that takes it: an exception that ends
    the call after it, such as an asynchronous one, leaves the range to
    {!release} of [h].

    Raises [Invalid_argument] if [n <= 0] or [align] is not a positive power of
    two. *)

val address : hold -> int
(** [address x] is the first address of [x]'s range. *)

val share : holder -> hold -> hold option
(** [share h x] is [h]'s hold on the range of [x], which then goes back once
    neither holds it, or [None], holding nothing, if [x] was given back, as by
    a release of its holder on another thread. *)

val free : hold -> unit
(** [free x] gives back [x]: its range goes back with the last hold on it.
    Freeing [x] again, or after {!release} of its holder, does nothing. *)

val release : holder -> unit
(** [release h] gives back every hold of [h], as {!free} does. Each goes as
    one step, so a release called again after an exception ended one gives
    back the rest. *)

(**/**)

val taken : t -> int
(* [taken s] is the number of ranges of [s] allocated and not freed, for the
   suites that check that a driver gives back what it takes, in a time that
   grows with the ranges [s] holds. *)
