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

    This module is where a driver places what it allocates. A driver's region of
    memory is a {!memory}, whichever vendor's the GPU is.

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
    {!Page_table.target}; it defaults to through the memory BAR, as
    {!Page_table.System} memory. *)

val small_bar : t -> bool
(** [small_bar m] is [true] iff the memory BAR is 256 MiB, too small for the
    process to reach all of the GPU's memory. *)

(** {1:alloc Allocating} *)

(** The type for where memory is and who reaches it. *)
type kind =
  | Gpu  (** The GPU's memory, which the process does not reach. *)
  | Bar
      (** The GPU's memory, one block the process reaches through the memory
          BAR, for structures the GPU requires in its own memory. *)
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

type memory = private {
  mapping : Page_table.mapping;  (** Its virtual range and pages. *)
  host : Window.t option;  (** The process's window on it, if it has one. *)
  source : source;  (** How it came to the GPU. *)
}
(** The type for memory the GPU addresses. *)

val alloc : ?uncached:bool -> t -> kind -> int -> memory option
(** [alloc m k n] is [n] new bytes of kind [k]: rounded up to the machine's page
    in system memory, and in the GPU's memory to 4 KiB, or to 2 MiB from 8 MiB
    on so that large ones map with large pages. With [~uncached:true] (defaults
    to [false]) the GPU bypasses its caches for them; {!Host} memory is always
    uncached. [None] if the memory or the address space is exhausted, or for
    {!Bar} memory, if the BAR does not reach a block that fits.

    Raises [Failure] if system memory or a page table cannot be allocated,
    having freed what it took. *)

val free : t -> memory -> unit
(** [free m mem] unmaps and frees [mem], which {!alloc} returned, and returns
    its addresses.

    Raises [Invalid_argument] if [mem] was not {!Allocated} by [m]. *)

(** {1:maps Mapping} *)

val map_host : t -> nativeint -> int -> (memory, string) result
(** [map_host m a n] maps the [n] bytes at [a] of the GPU's machine for the GPU,
    at [a]: it {!Function.pin}s them and maps their pages, snooped and uncached.
    [Error why] if [a] is not on a page, lies outside the GPU's virtual
    addresses, or cannot be pinned. *)

val map_peer : t -> t -> memory -> (memory, string) result
(** [map_peer m m' mem] maps [mem], which {!alloc} allocated on the GPU of [m'],
    for the GPU of [m], at its address on [m']: the GPU's memory through [m']'s
    memory BAR or link, and system memory at its pages, which stay [m']'s.
    [Error why] if the GPUs are on different machines, if either is behind an
    IOMMU ({!Function.Iommu}), or if [mem] is in the GPU's memory and [m']'s BAR
    is {!small_bar}. *)

val unmap : t -> memory -> unit
(** [unmap m mem] unmaps [mem], which {!map_host} or {!map_peer} mapped, and
    unpins the memory {!map_host} pinned.

    Raises [Invalid_argument] if [mem] was {!Allocated}. *)
