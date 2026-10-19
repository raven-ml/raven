(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** This machine's memory for functions that reach it.

    {!Function} checks every argument first. Sizes are positive, addresses are
    on a page, and contiguous memory is at most 2 MiB, a huge page that starts
    on 2 MiB. What the system refuses raises {!Fail.Failed}, and off Linux
    every value but {!page} and {!set_own_length} raises it. Any domain may
    use this module. *)

val page : int
(** [page] is the system's page size in bytes. *)

val huge : int
(** [huge] is the size of a huge page, 2 MiB. Contiguous memory larger than a
    page is one huge page. *)

val reserve : base:int -> int -> unit
(** [reserve ~base n] reserves [n] addresses from [base] once per range, for
    {!map} and {!alloc} only. *)

val map : ?va:int -> int -> Window.t
(** [map n] is [n] bytes, rounded up to {!page}, of new, zeroed, shared memory
    of the process, at [va] or where the system chooses. It is neither locked
    nor read for its addresses, since a function behind an IOMMU reaches it
    through mappings that VFIO removes when the process's files close. *)

val unmap : Window.t -> unit
(** [unmap w] releases {!map} memory. *)

(** {1:store Memory that outlives the process}

    The memory of functions taken physically, which outlives the process as
    {{!Function.dma}system memory} states. It lies in huge pages of a file of
    the process under the machine's root, in [dev/hugepages], laid out as
    [sysmem.ml] comments. *)

type store
(** The type for the memory of one function taken physically. *)

val store : root:string -> bus:string -> store
(** [store ~root ~bus] is the memory of the function at [bus] of the machine
    whose root is [root]. It makes its file at the first {!alloc}. *)

val root : store -> string
(** [root s] is the root of [s]'s machine. *)

val alloc :
  ?contiguous:bool ->
  ?va:int ->
  store ->
  int ->
  (Window.t * (int * int) list) option
(** [alloc s n] is [n] bytes, rounded up to {!page}, of new, zeroed memory in
    huge pages of the file under [s]'s root, with its physical runs read from
    [proc/self/pagemap] there. Each run lies in one 2 MiB block, and the runs of
    neighbouring blocks merge where their physical pages follow. The memory lies
    at [va], or at addresses of the process's own, on 2 MiB if [contiguous]
    memory is larger than a page. [va] and the 2 MiB blocks around it lie in a
    range {!reserve} reserved, as {!Function.alloc_dma} checks. It is [None] if
    the machine has no free huge page for a new block, or the process no free
    addresses of its own. Raises {!Fail.Failed} if a block holds memory of
    another root, or if the root's [proc/self/mounts] shows its [dev/hugepages]
    on no hugetlbfs of 2 MiB pages. *)

val free : Window.t -> unit
(** [free w] unmaps {!alloc}'s live memory [w]. A huge page goes back once no
    allocation holds it and no function maps it as a peer's ({!reach}). *)

val close : store -> unit
(** [close s] records that [s]'s function, with its bus mastering off, reaches
    no memory any more. The function leaves the lists of the process's files,
    each of which goes once it lists no function and holds no memory. The
    process's mappings keep their pages until {!free}. A file a dead process
    left goes at {!collect_dead} once it lists no function, or at {!forget} of
    each function it lists. Called once per store. *)

val set_own_length : int -> int
(** [set_own_length n] makes each range of the process's own addresses that
    {!alloc} reserves from then on, for memory without an address, [n] bytes,
    or as many as a request needs. It is the length it replaces. The first
    is 1 GiB. *)

val reach : root:string -> bus:string -> int -> int -> (int * int) list option
(** [reach ~root ~bus a n] is the physical runs of the [n] bytes at [a], if
    {!alloc} gave every 2 MiB block that holds them under [root]. It records
    that the function at [bus] reaches them, so their huge pages stay until
    {!unreach}, and their file stays until that function no longer reaches it
    either. It is [None], recording nothing, otherwise.

    Raises {!Fail.Failed}, recording nothing, if the record cannot be written
    for after the process's death. *)

val unreach : a:int -> n:int -> unit
(** [unreach ~a ~n] ends one {!reach} of the [n] bytes at [a]. *)

val exit : unit -> unit
(** [exit ()] deletes the process's files that list no function, blocks and
    all, since nothing else frees them once the process exits. The exit
    function of {!Local} calls it, after its takes are given back. *)

val collect_dead : root:string -> unit
(** [collect_dead ~root] deletes the files under [root] that dead processes
    left and that list no function, whatever memory they hold. Each function
    that reached such a file has been released with its bus mastering off, or
    its GPU has been reset. Called at each take of a function of the machine. *)

val left : root:string -> bus:string -> bool
(** [left ~root ~bus] is [true] iff a dead process left memory under [root]
    that the function at [bus] reaches. The GPU may still write it, and
    holds its pages until {!forget}. Raises {!Fail.Failed}, naming the file, if
    a file under [root] cannot be opened for another reason than its end. *)

val forget : root:string -> bus:string -> unit
(** [forget ~root ~bus] records that the GPU of the function at [bus] was
    reset, so that it reaches none of the memory that dead processes left. The
    function leaves the list of every file under [root] that a dead process
    left, which the file's shared flock tells, since its process held that
    lock while it lived. Each such file that no function reaches then goes. The
    process's own files keep listing the function, since its driver may reach
    their memory again, until its release ({!close}). Called with the function
    taken. *)
