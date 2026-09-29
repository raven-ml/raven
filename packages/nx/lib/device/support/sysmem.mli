(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** System memory that devices address physically.

    A GPU that the process drives itself reaches system memory by physical
    address, through page tables the process writes. That memory must stay where
    it is: resident, locked, and at the physical pages the process read for it.
    This module allocates such memory, and pins memory the process already has.

    Reading physical addresses and locking pages need Linux, the privileges for
    [/proc/self/pagemap] and [mlock], and the kernel setting
    [vm.compact_unevictable_allowed = 0], without which the kernel may move
    locked pages. A missing privilege or setting raises [Failure] naming what to
    change. On other systems every function raises [Failure]. *)

val page : int
(** [page] is the system's page size in bytes. *)

val reserve : base:int -> int -> unit
(** [reserve ~base n] reserves the [n] addresses from [base] on in the process's
    address space, once per range for the life of the process, so that nothing
    but {!alloc} maps memory there.

    Raises [Failure] if part of the range is in use. *)

val alloc : ?contiguous:bool -> va:int -> int -> Mmio.t * int list
(** [alloc ~va n] maps [n] bytes, rounded up to {!page}, of new, zeroed, locked
    memory at [va], inside a range {!reserve} reserved, and is that memory with
    the physical address of each of its pages. It holds one {!pin} of each page
    until {!free}. With [~contiguous:true] (defaults to [false]) the memory is
    one physical block and the list holds its address alone: larger than a page,
    it is a huge page of 2 MiB at a [va] on 2 MiB, which the system must have
    free ([vm.nr_hugepages]).

    Raises [Failure] if the system cannot, and [Invalid_argument] if [va] is not
    on a page, if [contiguous] memory is larger than 2 MiB, or if [va] is not on
    2 MiB for [contiguous] memory larger than a page. *)

val free : Mmio.t -> unit
(** [free m] releases memory {!alloc} returned, with the pins it held, and
    returns its addresses to their reservation. *)

val pin : nativeint -> int -> int list
(** [pin a n] locks the [n] bytes of process memory at [a], which starts on a
    page, and is the physical address of each of their pages. Pins are counted
    per page across the process: a page stays locked until each of its pins is
    {!unpin}ned.

    Raises [Failure] if the pages cannot be locked or their addresses read, and
    [Invalid_argument] if [a] is not on a page. *)

val unpin : nativeint -> int -> unit
(** [unpin a n] releases one pin of each page of the [n] bytes at [a], which
    {!pin} pinned, and unlocks the pages whose last pin it releases. *)
