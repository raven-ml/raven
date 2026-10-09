(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The memory a GPU driven through its PCI function addresses.

    A GPU's work addresses memory through the GPU's page tables ({!Page_table}):
    the GPU's own memory, which the process reaches through the memory BAR;
    system memory of the GPU's machine, at the same address for the process and
    the GPU; memory of that machine the GPU borrows, at addresses its space
    gives it; and another GPU's memory, through that GPU's memory BAR or a link
    of their fabric. Every address is of the GPU's machine.

    This module is where a driver places what it allocates. What a driver
    allocates or maps is a {!region}, whichever vendor's the GPU is, and {!free}
    gives any region back.

    {b After the GPU is given back.} Once the GPU's function is released
    ({!Function.release}, which {!Gpus.stop} does), {!free} writes none of the
    GPU's tables: it gives back system memory, pins and virtual addresses only,
    and {!alloc}, {!map_host} and {!map_peer} raise [Invalid_argument]. A GPU
    reset and opened again by another instance is not written by this one.

    {b Unconfirmed flushes.} A {!free} whose flush the GPU does not confirm
    ({!Page_table.unmap}) keeps the region's memory and addresses, which the GPU
    may still reach, until a {!free} once the GPU's function is released gives
    them back.

    {b Peers.} Freeing a region another GPU maps ({!map_peer}) leaves that
    mapping reaching memory the owner may give out again: the peer's {!free}
    comes first.

    The GPU's owner serializes calls on one GPU. *)

type t
(** The type for the memory of one GPU. *)

type link = {
  fabric : int64;
      (** The fabric's identity: two GPUs' are equal iff one fabric joins them.
      *)
  node : int;  (** This GPU's number on it, as {!Page_table.Peer} names it. *)
}
(** The type for a GPU's place in a fabric of GPUs, which reach each other's
    memory over direct links. *)

val create : ?link:link -> Function.t -> Page_table.t -> bar:int -> t
(** [create f tables ~bar] is the memory of the GPU of [f], whose page tables
    are [tables], whose memory BAR is [bar], and which [link] joins to a fabric,
    if given. System memory goes at addresses of [tables]' space
    ({!Page_table.space}), which the driver reserves on [f]'s machine first
    ({!Machine.reserve}).

    Raises [Invalid_argument] if [f] has no BAR [bar]. *)

val reaches : t -> t -> bool
(** [reaches m o] is [true] iff {!map_peer}[ m] maps the GPU memory of [o],
    given room: [o] is another GPU's memory on [m]'s machine whose page tables
    share [m]'s space, and either links of one fabric join them, or neither GPU
    is behind an IOMMU ({!Machine.Iommu}) and [o]'s memory BAR reaches all of
    [o]'s memory. [reaches m m] is [false]: the GPU maps its own memory already.
*)

(** {1:regions Regions} *)

(** The type for where new memory is and who reaches it. *)
type kind =
  | Gpu  (** The GPU's memory, which the process does not reach. *)
  | Bar
      (** The GPU's memory, one block the process reaches through the memory
          BAR, for structures the GPU requires in its own memory. Its window
          maps the BAR as the driver's own windows on it do: it combines if the
          driver mapped the memory BAR with [~combine:true] ({!Function.map}),
          and is uncached otherwise. *)
  | Host
      (** System memory of the GPU's machine, which the GPU reaches snooped and
          uncached. *)

type region
(** The type for ranges of memory a GPU addresses: what {!alloc}, {!map_host} or
    {!map_peer} gave, until {!free}. *)

(** The type for how a region came to its GPU. *)
type source =
  | Allocated  (** By {!alloc}. *)
  | Borrowed of int
      (** Memory of the machine at this address of the process, by {!map_host}.
      *)
  | Peer of region
      (** Another GPU's region, allocated or borrowed there, by {!map_peer}. *)

val source : region -> source
(** [source r] is how [r] came to its GPU. *)

val address : region -> int
(** [address r] is the GPU address of [r]'s first byte. *)

val pages : region -> Page_table.target * (int * int) list
(** [pages r] is the memory [r]'s bytes are in and the physical ranges that hold
    them, as (address, bytes), in order from its first byte: addresses of the
    GPU's memory, of a peer's memory for {!Page_table.Peer}, and bus addresses
    for system memory. *)

val host : region -> Window.t option
(** [host r] is the process's window on [r]'s bytes, if it has one: the system
    memory {!alloc} gave, the BAR window on {!Bar} memory, the bytes {!map_host}
    borrowed, or the window of the region a {!Peer} maps. *)

(** {1:alloc Allocating and mapping} *)

val alloc : ?uncached:bool -> t -> kind -> int -> (region option, string) result
(** [alloc m k n] is [n] new bytes of kind [k]: rounded up to the machine's page
    in system memory, and in the GPU's memory to 4 KiB, or to 2 MiB from 8 MiB
    on so that large ones map with large pages. With [~uncached:true] (defaults
    to [false]) the GPU bypasses its caches for them; {!Host} memory is always
    uncached. [Ok None] if the GPU's memory, a page table or the address space
    has no room, as {!Page_table.alloc} bounds it, for {!Bar} memory if the BAR
    does not reach a block that fits, or for system memory if the machine has
    none free now ({!Function.alloc_dma}): freeing memory makes room.
    [Error why] if the machine refuses system memory or a window on the BAR,
    [why] naming what is missing as {!Function.alloc_dma} does, or gives system
    memory at addresses the GPU's entries do not hold ({!Page_table.pa_bits}),
    having given back what it took.

    Raises [Invalid_argument] if [n <= 0], or if system memory goes at addresses
    {!Machine.reserve} did not reserve. *)

val map_host : t -> int -> int -> (region option, string) result
(** [map_host m a n] maps the pages of the GPU's machine that hold the [n] bytes
    at [a] for the GPU: it {!Function.pin}s them and maps them, snooped and
    uncached, at addresses it takes from the GPU's space ({!Page_table.space}),
    whatever [a]. {!address} is where the GPU reaches byte [a], {!host} is a
    window on the [n] bytes at [a], and the source is [Borrowed a]. Each call
    maps anew: memory mapped already, borrowed or allocated, maps again at other
    addresses, pinned once more. [Ok None] if the space or a page table has no
    room: freeing memory makes room.

    [Error why] if the pages cannot be pinned, [why] being {!Function.pin}'s
    reason, such as pages of the process's own for a GPU taken physically, or if
    the GPU's entries do not hold their addresses ({!Page_table.pa_bits}).

    Raises [Invalid_argument] if [n <= 0]. *)

val map_peer : t -> region -> (region option, string) result
(** [map_peer m r] maps [r], a region of another GPU, for the GPU of [m], at
    [r]'s address there: GPU memory through its owner's memory BAR or their
    fabric's link, and system memory, allocated or borrowed, at its pages, which
    {!Function.pin} keeps for [m]'s GPU. If [r] is itself a {!Peer} region,
    [map_peer] maps the region it maps. The result's source is [Peer r'], [r']
    the region its owner gave. [Ok None] if a page table has no room: freeing
    memory makes room.

    [Error why] if the GPUs are on different machines, if their page tables take
    addresses from different spaces, naming both, if [r] is GPU memory and [m]
    does not {!reaches} its owner's, if the system memory cannot be pinned,
    [why] being {!Function.pin}'s reason, or if [m]'s entries do not hold the
    addresses ({!Page_table.pa_bits}), the peer's BAR's or its system memory's.

    Raises [Invalid_argument] if [r], or the region it maps, was freed, if it is
    [m]'s GPU's own memory, or if the GPU of [m] maps its addresses already. *)

val free : t -> region -> unit
(** [free m r] gives back what {!alloc}, {!map_host} or {!map_peer} gave: it
    unmaps [r], then frees its memory or unpins it and returns its addresses. A
    {!Peer} region's memory stays its owner's. After the function's release it
    writes no table.

    Raises [Invalid_argument] if [r] is not [m]'s or was freed. *)
