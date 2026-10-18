(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The memory a GPU driven through its PCI function addresses.

    A GPU's work addresses memory through the GPU's page tables ({!Page_table}).
    It reaches four kinds of memory:
    - the GPU's own memory, which the process reaches through the memory BAR;
    - system memory of the GPU's machine, at the same address for the process
      and the GPU;
    - memory of that machine that the GPU borrows, at addresses its space gives
      it;
    - another GPU's memory, through that GPU's memory BAR or a link of their
      fabric.

    Every address is of the GPU's machine. A driver allocates and maps a GPU's
    memory here, whatever the GPU's vendor. Each allocation or mapping is a
    {!region}, and {!free} gives any region back.

    {b After the GPU is given back.} Once the GPU's function is released
    ({!Function.release}, which {!Driver.stop} and {!release} do), {!free}
    writes none of the GPU's tables. It gives back system memory, pins and
    virtual addresses only, and {!val-alloc}, {!map_host} and {!map_peer} raise
    [Invalid_argument]. Once the function is released, another open may reset
    the GPU, and this {!t} never writes it again.

    {b Unconfirmed flushes.} A {!free} whose flush the GPU does not confirm
    ({!Page_table.unmap}) keeps the region's memory and addresses, which the GPU
    may still reach. A {!free} after the GPU's function has been released
    gives back the region's system memory, pins and virtual addresses.

    {b Peers.} Freeing a region that another GPU maps ({!map_peer}) leaves that
    mapping reaching memory the owner may give out again, so the peer's {!free}
    must come first.

    The GPU's owner makes one call at a time on one GPU's memory. *)

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
    are [tables] and whose memory BAR is [bar]. [link], if given, joins it to a
    fabric. System memory goes at addresses of [tables]' space
    ({!Page_table.space}), which the driver must first reserve on [f]'s machine
    ({!Machine.val-reserve}). [create] bounds [f]'s system memory to the
    addresses [tables]' entries hold ({!Function.set_dma_bits} with
    {!Page_table.val-pa_bits}).

    Raises [Invalid_argument] if [f] has no BAR [bar]. *)

val release : t -> unit
(** [release m] releases the function of [m]'s GPU ({!Function.release}), then
    drops the page tables {!create} was given, which only a live GPU uses. A
    driver that keeps [m] past the GPU's stop ({!Driver.stop}) releases it
    there, so that [m] holds no page table. Releasing [m] again does nothing. *)

type reach
(** The type for the facts that decide whose GPU memory a GPU maps
    ({!reaches}). They are facts of the GPU, its machine and its driver that
    are the same at every open of the GPU, so a driver may keep them past the
    GPU's stop. *)

val reach : t -> reach
(** [reach m] is the facts that decide whose GPU memory [m]'s GPU maps. *)

val reaches : reach -> reach -> bool
(** [reaches r o] is [true] iff {!map_peer} on a GPU of reach [r] maps the
    GPU memory of another GPU of reach [o], given room. That holds iff both
    are on one machine, their page tables share a space, and either links of
    one fabric join them, or neither GPU is behind an IOMMU
    ({!Machine.Iommu}), [o]'s memory BAR reaches all of [o]'s memory and the
    bus carries [r]'s requests to it. The bus carries them as Linux's p2pdma
    decides, through a bridge above both GPUs, where no device on the way to
    it redirects peer requests to the root complex by ACS, or else through the
    root complex where the CPU is AMD's from Zen on, or where Linux lists both
    GPUs' host bridges, some of them only for GPUs under one host bridge. A
    machine reached through a transport carries none.
    It is [false] for two reaches of one GPU, [reaches r r] included, since
    the GPU maps its own memory already. *)

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
(** The type for ranges of memory a GPU addresses: what {!val-alloc},
    {!map_host} or {!map_peer} gave, until {!free}. *)

val address : region -> int
(** [address r] is the GPU address of [r]'s first byte. *)

val pages : region -> Page_table.target * (int * int) list
(** [pages r] is the memory [r]'s bytes are in and the physical ranges that hold
    them, as (address, bytes), in order from its first byte: addresses of the
    GPU's memory, of a peer's memory for {!Page_table.Peer}, and bus addresses
    for system memory. *)

val host : region -> Window.t option
(** [host r] is the process's window on [r]'s bytes, if it has one: the system
    memory {!val-alloc} gave, the BAR window on {!Bar} memory, the bytes
    {!map_host} borrowed, or the window of the region a {!map_peer} region
    maps. *)

(** {1:alloc Allocating and mapping}

    {!val-alloc}, {!map_host} and {!map_peer} each return [Ok None] if the
    machine's system memory ({!Function.alloc_dma}, {!Function.pin}), the GPU's
    memory, its space or a page table has no room, and freeing memory makes
    room. Each returns [Error why] if the machine refuses, naming what is
    missing as {!Function.alloc_dma} does, or if the GPU's entries cannot hold
    a peer's addresses ({!Page_table.val-pa_bits}). System memory lies within
    them ({!create}). Each gives back what it took before it returns [None] or
    [Error]. *)

val alloc : ?uncached:bool -> t -> kind -> int -> (region option, string) result
(** [alloc m k n] is [n] new bytes of kind [k]. System memory is rounded up to
    the machine's page. The GPU's memory is rounded up to 4 KiB, or to 2 MiB
    from 8 MiB on, so that large allocations map with large pages. With
    [~uncached:true] (defaults to [false]) the GPU bypasses its caches for
    them; {!Host} memory is always uncached.

    [Ok None] also if the BAR reaches no free block that fits {!Bar} memory, or
    if the machine has no system memory free now ({!Function.alloc_dma}), as
    {!Page_table.alloc} bounds each request. [Error why] if the machine refuses
    system memory or a window on the BAR.

    Raises [Invalid_argument] if [n <= 0], or if system memory goes at addresses
    {!Machine.val-reserve} did not reserve. *)

val map_host : t -> int -> int -> (region option, string) result
(** [map_host m a n] maps for the GPU the pages of its machine that hold the
    [n] bytes at [a]. It {!Function.pin}s them and maps them, snooped and
    uncached, at addresses it takes from the GPU's space ({!Page_table.space}),
    whatever [a] is. In the result, {!address} is where the GPU reaches byte
    [a], and {!host} is a window on the [n] bytes at [a]. Each call maps
    anew, so memory mapped already maps again at
    other addresses and is pinned once more.

    [Error why] if the pages cannot be pinned, with {!Function.pin}'s reason,
    such as pages of the process's own for a GPU taken physically.

    Raises [Invalid_argument] if [n <= 0]. *)

val map_peer : t -> region -> (region option, string) result
(** [map_peer m r] maps [r], a region of a GPU, [m]'s own included, for the
    GPU of [m], at the address [r] has there. GPU memory is mapped through its
    owner's memory BAR or their fabric's link. System memory, allocated or
    borrowed, is mapped at its pages, which {!Function.pin} keeps for [m]'s
    GPU. If [map_peer] gave [r] itself, [map_peer] maps the region that [r]
    maps, which its owner gave.

    Where the GPU of [m] maps that memory already, as its own or through
    another region [map_peer] gave, the result is a new region over the same
    mapping, and it writes no table and pins nothing. The mapping stays until
    every region over it is freed.

    [Error why] if the GPUs are on different machines, if their page tables
    take addresses from different spaces, naming both, if [r] is GPU memory
    that [m]'s GPU does not reach ({!reaches}), or if the system memory cannot
    be pinned, with {!Function.pin}'s reason.

    Raises [Invalid_argument] if [r], or the region it maps, was freed, or if
    [m]'s GPU maps other memory at [r]'s addresses, as a free out of the
    order {b Peers} states leaves. *)

val free : t -> region -> unit
(** [free m r] gives back what {!val-alloc}, {!map_host} or {!map_peer} gave.
    Once no other region of [m] over [r]'s mapping is out, it unmaps [r], then
    frees its memory or unpins it, and returns its addresses. The memory of a
    region {!map_peer} gave stays its owner's. After the function's release it
    writes no table.

    Raises [Invalid_argument] if [r] is not [m]'s or was freed. *)

(**/**)

(* How a region came to its GPU: by {!val-alloc}, by {!map_host} at this
   address of the process, or by {!map_peer} from another GPU's region,
   allocated or borrowed there. *)
type source = Allocated | Borrowed of int | Peer of region

(* [source r] is how [r] came to its GPU. *)
val source : region -> source
