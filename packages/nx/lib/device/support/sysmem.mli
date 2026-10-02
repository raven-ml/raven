(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** System memory that devices address.

    A GPU that the process drives itself reaches system memory by physical
    address, through page tables the process writes, unless an IOMMU stands
    between them ({!Pci.addressing}). Memory reached physically must stay where
    it is: resident, locked, and at the physical pages the process read for it.
    This module allocates such memory, pins memory the process already has, and
    maps memory for an IOMMU to pin ({!map}).

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

val unreserve : base:int -> int -> unit
(** [unreserve ~base n] returns the range {!reserve} reserved to the system,
    which may map anything there again. The memory {!alloc} mapped in it must
    have been freed. It does nothing for a range not reserved. *)

val extent : ?contiguous:bool -> int -> int
(** [extent ?contiguous n] is the bytes {!alloc} maps for [n]. *)

val alloc : ?contiguous:bool -> ?va:int -> int -> Mmio.t * int list
(** [alloc ~va n] maps [n] bytes, rounded up to {!page}, of new, zeroed, locked
    memory at [va], inside a range {!reserve} reserved, or where the system
    chooses without [va], and is that memory with the physical address of each
    of its pages. It holds one {!pin} of each page until {!free}. With
    [~contiguous:true] (defaults to [false]) the memory is one physical block
    and the list holds its address alone: larger than a page, it is a huge page
    of 2 MiB, at a [va] on 2 MiB if given, which the system must have free
    ([vm.nr_hugepages]).

    Raises [Failure] if the system cannot, and [Invalid_argument] if [va] is not
    on a page, if [contiguous] memory is larger than 2 MiB, or if [va] is not on
    2 MiB for [contiguous] memory larger than a page. *)

val map : ?va:int -> int -> Mmio.t
(** [map ~va n] maps [n] bytes, rounded up to {!page}, of new, zeroed memory at
    [va], inside a range {!reserve} reserved, or where the system chooses
    without [va]. The memory is neither locked nor its addresses read: a
    function behind an IOMMU pins it when the IOMMU maps it. It needs no
    privilege.

    Raises [Failure] if the system cannot, and [Invalid_argument] if [va] is not
    on a page. *)

val free : Mmio.t -> unit
(** [free m] releases memory {!alloc} or {!map} returned, with the pins it held,
    and returns its addresses to their reservation if they had one. *)

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
