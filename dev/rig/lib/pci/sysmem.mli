(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** This machine's memory for functions that reach it.

    {!Function} checks every argument first: sizes are positive, addresses are
    on a page, and contiguous memory is at most 2 MiB, a huge page that starts
    on 2 MiB. What the system refuses raises {!Fail.Failed}; off Linux every
    function but {!page} does. Any domain may call. *)

val page : int
(** [page] is the system's page size in bytes. *)

val huge : int
(** [huge] is the huge page of 2 MiB that contiguous memory larger than a page
    is. *)

val reserve : base:int -> int -> unit
(** [reserve ~base n] reserves [n] addresses from [base] once per range, for
    {!map} and {!alloc} only. *)

val map : ?va:int -> int -> Window.t
(** [map n] is [n] bytes, rounded up to {!page}, of new, zeroed, shared memory
    of the process, neither locked nor read for its addresses: a function behind
    an IOMMU reaches it through mappings VFIO removes when the process's files
    close. At [va], or where the system chooses. *)

val unmap : Window.t -> unit
(** [unmap w] releases {!map} memory. *)

(** {1:store Memory that outlives the process}

    A function taken physically writes physical pages, and keeps writing after
    its process dies. Its memory lies in huge pages of a file of the process
    under the machine's root, [dev/hugepages], which keep their pages when the
    process dies and which the kernel neither swaps nor compacts. Each 2 MiB
    block of addresses that holds memory maps one huge page, which the
    allocations of every function of the machine in that block share. *)

type store
(** The type for the memory of one function taken physically. *)

val store : root:string -> bus:string -> store
(** [store ~root ~bus] is the memory of the function at [bus] of the machine
    whose root is [root]. It makes no file until {!alloc}. *)

val alloc :
  ?contiguous:bool ->
  ?va:int ->
  store ->
  int ->
  (Window.t * (int * int) list) option
(** [alloc s n] is [n] bytes, rounded up to {!page}, of new, zeroed memory in
    huge pages of the file under [s]'s root, with its physical runs read from
    [proc/self/pagemap] there: one a 2 MiB block, merged where the blocks'
    frames follow. At [va], or at addresses of the process's own, on 2 MiB if
    [contiguous] memory is larger than a page. [va] and the 2 MiB blocks around
    it lie in a range {!reserve} reserved, as {!Function.alloc_dma} checks.
    [None] if the machine has no free huge page for a new block, or the process
    no free addresses of its own: freeing memory makes room. Raises
    {!Fail.Failed} if a block holds memory of another root. *)

val free : store -> Window.t -> unit
(** [free s w] unmaps {!alloc}'s memory [w]. A huge page goes back once no
    allocation holds it and no function maps it as a peer's ({!reach}). *)

val close : store -> unit
(** [close s] records that [s]'s function reaches no memory any more, its bus
    mastering off: it leaves the list of the process's file, which goes once it
    lists no function and holds no memory. The process's mappings keep their
    pages until {!free}. Files a process that died left stay until {!forget}.
    Closing it again does nothing. *)

val reach : a:int -> n:int -> bus:string -> unit
(** [reach ~a ~n ~bus] records that the function at [bus] maps, as a peer's, the
    [n] bytes at [a] that {!alloc} gave: their huge pages stay until {!unreach},
    and the file until that function no longer reaches it either. Nothing if no
    memory {!alloc} gave is at [a].

    Raises {!Fail.Failed}, recording nothing, if the record cannot be written
    for after the process's death. *)

val unreach : a:int -> n:int -> unit
(** [unreach ~a ~n] records that a peer's mapping {!reach} recorded is gone. *)

val forget_dead : root:string -> bus:string -> unit
(** [forget_dead ~root ~bus] is {!forget} for the files processes that died left
    alone: the process's own file keeps listing the function at [bus], which
    still reaches its memory. *)

val left : root:string -> bus:string -> bool
(** [left ~root ~bus] is [true] iff a process that died left memory under [root]
    that the function at [bus] reaches: the GPU may still write it, and holds
    its pages until {!forget}. Raises {!Fail.Failed} if a file under [root]
    cannot be opened for another reason than its end, naming it. *)

val forget : root:string -> bus:string -> unit
(** [forget ~root ~bus] records that the GPU of the function at [bus] was reset,
    so it reaches no memory: it leaves the list of the process's file and of
    every file under [root] that a process that died left, which a shared flock
    its process held while it lived tells, and each such file no function
    reaches then goes. Called with the function taken. *)
