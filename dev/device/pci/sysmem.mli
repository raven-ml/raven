(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** This machine's memory for functions that reach it (private).

    A function taken physically reaches system memory at physical addresses, so
    that memory must stay where it is: resident, locked, and at the pages the
    process read for it. A function behind an IOMMU pins what the IOMMU maps, so
    its memory needs neither ({!map}).

    Physical addresses need Linux, the privileges for [/proc/self/pagemap] and
    [mlock], and [vm.compact_unevictable_allowed = 0]; a missing one raises
    [Failure] naming it. On other systems every function but {!page} raises
    [Failure]. Synchronized: any domain may call. *)

val page : int
(** [page] is the system's page size in bytes. *)

val reserve : base:int -> int -> unit
(** [reserve ~base n] reserves the [n] addresses from [base] on in the process,
    once per range for the life of the process, so that only {!alloc} and {!map}
    at addresses in it map memory there.

    Raises [Failure] if part of the range is in use. *)

val alloc : ?contiguous:bool -> ?va:int -> int -> Window.t * int list
(** [alloc n] is [n] bytes, rounded up to {!page}, of new, zeroed, locked memory
    at [va], inside a range {!reserve} reserved, or where the system chooses
    without [va], with the physical address of each of its pages. It holds one
    {!pin} of each page until {!free}. With [~contiguous:true] (defaults to
    [false]) the memory is one physical block and the list holds its address
    alone: larger than a page, it is a huge page of 2 MiB, which the system must
    have free ([vm.nr_hugepages]).

    Raises [Invalid_argument] if [n <= 0], if [va] is not on a page, if
    [contiguous] memory is larger than 2 MiB, or if [va] is not on 2 MiB for a
    huge page, and [Failure] if the system cannot. *)

val map : ?va:int -> int -> Window.t
(** [map n] is [n] bytes, rounded up to {!page}, of new, zeroed memory at [va],
    inside a range {!reserve} reserved, or where the system chooses. It is
    neither locked nor are its addresses read, and needs no privilege.

    Raises [Invalid_argument] if [n <= 0] or [va] is not on a page, and
    [Failure] if the system cannot. *)

val free : Window.t -> unit
(** [free w] releases memory {!alloc} or {!map} returned, with the pins {!alloc}
    held, and returns its addresses to their reservation if they had one. *)

val pin : int -> int -> int list
(** [pin a n] locks the [n] bytes of process memory at [a], which starts on a
    page, and is the physical address of each of their pages. Pins are counted
    per page: a page stays locked until each of its pins is {!unpin}ned.

    Raises [Invalid_argument] if [a] is not on a page, and [Failure] if the
    pages cannot be locked or their addresses read. *)

val unpin : int -> int -> unit
(** [unpin a n] releases one pin of each page of the [n] bytes at [a], and
    unlocks those whose last pin it releases.

    Raises [Invalid_argument] if a page of them is not pinned. *)
