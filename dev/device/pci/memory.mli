(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The memory a GPU driven through its PCI function addresses.

    A GPU's work addresses memory through the GPU's page tables ({!Page_table}):
    the GPU's own memory, which the process reaches through the memory BAR;
    system memory of the GPU's machine, at the same address for the process and
    the GPU; memory of that machine the GPU borrows; and another GPU's memory,
    through that GPU's memory BAR or a direct link. Every address is of the
    GPU's machine.

    This module is where a driver places what it allocates. What a driver
    allocates or maps is a {!region}, whichever vendor's the GPU is.

    {b After the GPU is given back.} Once the GPU's function is released
    ({!Function.release}, {!Gpus.release}, {!Gpus.lose}), {!free} and {!unmap}
    touch none of the GPU's memory: they return system memory, pins and virtual
    addresses only, and {!alloc}, {!map_host} and {!map_peer} raise
    [Invalid_argument]. A GPU reset and opened again by another instance is not
    written by this one.

    A call whose page tables' format raises ({!Page_table.map}) raises the same
    exception, having given back what it took.

    The GPU's owner serializes calls on one GPU. *)

type t
(** The type for the memory of one GPU. *)

val create :
  ?peer:((int * int) list -> (int * int) list * Page_table.target) ->
  Function.t ->
  Page_table.t ->
  bar:int ->
  t
(** [create f tables ~bar] is the memory of the GPU of [f], whose page tables
    are [tables] and whose memory BAR is [bar]. [peer ranges] is how another GPU
    reaches the physical [ranges] of this one's memory, and in which
    {!Page_table.target}; it defaults to through the memory BAR's bus address
    ({!Function.bar}), as {!Page_table.System} memory. System memory goes at
    addresses of [tables]' space ({!Page_table.space}), which the driver
    reserves on [f]'s machine first ({!Machine.reserve}).

    Raises [Invalid_argument] if [f] has no BAR [bar]. *)

val small_bar : t -> bool
(** [small_bar m] is [true] iff the memory BAR is smaller than the GPU's memory
    ({!Page_table.memory}), so that the process does not reach all of it. *)

(** {1:alloc Allocating} *)

(** The type for where memory is and who reaches it. *)
type kind =
  | Gpu  (** The GPU's memory, which the process does not reach. *)
  | Bar
      (** The GPU's memory, one block the process reaches through the memory
          BAR, for structures the GPU requires in its own memory. Its window is
          mapped with [~combine:true] ({!Function.map}). *)
  | Host
      (** System memory of the GPU's machine, which the GPU reaches snooped and
          uncached. *)
  | Visible
      (** Memory both reach: {!Bar} memory, or {!Host} memory when the BAR is
          {!small_bar}. *)

(** The type for how memory came to the GPU. *)
type source =
  | Allocated  (** By {!alloc}. *)
  | Borrowed  (** Memory of the machine, by {!map_host}. *)
  | Peer  (** Another GPU's memory, by {!map_peer}. *)

type region = private {
  mapping : Page_table.mapping;  (** Its virtual range and pages. *)
  host : Window.t option;  (** The process's window on it, if it has one. *)
  source : source;  (** How it came to the GPU. *)
}
(** The type for ranges of memory the GPU addresses. *)

val alloc : ?uncached:bool -> t -> kind -> int -> (region option, string) result
(** [alloc m k n] is [n] new bytes of kind [k]: rounded up to the machine's page
    in system memory, and in the GPU's memory to 4 KiB, or to 2 MiB from 8 MiB
    on so that large ones map with large pages. With [~uncached:true] (defaults
    to [false]) the GPU bypasses its caches for them; {!Host} memory is always
    uncached. [Ok None] if the GPU's memory, a page table or the address space
    has no room, as {!Page_table.alloc} bounds it, or for {!Bar} memory, if the
    BAR does not reach a block that fits: freeing memory makes room. [Error why]
    if the machine refuses system memory or a window on the BAR, having freed
    what it took, [why] naming what is missing as {!Function.alloc_dma} does:
    such a limit is cured by a setting, rarely by freeing memory.

    Raises [Invalid_argument] if [n <= 0], if system memory goes at addresses
    {!Machine.reserve} did not reserve, or for {!Bar} memory if a live window of
    the memory BAR does not combine ({!Function.map}). *)

val free : t -> region -> unit
(** [free m mem] unmaps and frees [mem] and returns its addresses.

    Raises [Invalid_argument] if [mem] is not {!Allocated} by [m], or freed
    already. *)

(** {1:maps Mapping} *)

val map_host : t -> int -> int -> (region, string) result
(** [map_host m a n] maps the [n] bytes at [a] of the GPU's machine for the GPU,
    at [a]: it {!Function.pin}s them and maps their pages, snooped and uncached.
    [Error why] if [a] is not on a page of the machine ({!Machine.page}), lies
    outside the GPU's virtual addresses, or cannot be pinned, [why] being
    {!Function.pin}'s reason, or if a page table has no room.

    Raises [Invalid_argument] if [n <= 0], or if part of the range is mapped for
    the GPU already, pinning nothing. *)

val map_peer : t -> owner:t -> region -> (region, string) result
(** [map_peer m ~owner mem] maps [mem], which {!alloc} allocated on the GPU of
    [owner], for the GPU of [m], at its address on [owner]: the GPU's memory
    through [owner]'s memory BAR or link, and system memory at its pages, which
    stay [owner]'s. [Error why] if the GPUs are on different machines, if either
    is behind an IOMMU ({!Machine.Iommu}), if [mem] is in the GPU's memory and
    [owner]'s BAR is {!small_bar}, or if a page table has no room.

    Raises [Invalid_argument] if [mem] is not {!Allocated} by [owner], or if the
    GPU of [m] maps its addresses already. *)

val unmap : t -> region -> unit
(** [unmap m mem] unmaps [mem] and unpins the memory {!map_host} pinned.

    Raises [Invalid_argument] if [mem] is not {!Borrowed} or {!Peer} memory of
    [m], or unmapped already. *)
