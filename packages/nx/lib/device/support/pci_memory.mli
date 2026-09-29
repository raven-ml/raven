(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The memory of a GPU the process drives over PCI.

    Allocates, maps and frees the memory a GPU's work addresses, through the
    GPU's page tables ({!Page_table}): its own memory, which the process reaches
    through the memory BAR when it asks to; system memory of the GPU's machine
    ({!Pci.alloc_sysmem}), at the same address for that machine and the GPU;
    that machine's memory the GPU borrows; and another GPU's memory, through
    that GPU's memory BAR or a direct link. The GPU may be another machine's
    ({!Pci.remote}): its memory is then that machine's, and so are the
    addresses.

    Calls on one GPU are serialized by its owner. *)

type t
(** The type for the memory of one GPU. *)

val create :
  ?peer:((int * int) list -> (int * int) list * Page_table.space) ->
  Pci.t ->
  Page_table.t ->
  bar:int ->
  t
(** [create p tables ~bar] is the memory of the GPU [p], whose page tables are
    [tables] and whose memory BAR is [bar]. [peer ranges] is how another GPU
    reaches the physical [ranges] of this one's memory; it defaults to through
    the memory BAR, as system memory. *)

val small_bar : t -> bool
(** [small_bar m] is [true] iff the memory BAR is 256 MiB, too small for the
    process to reach all of the GPU's memory. *)

(** The type for where memory comes from. *)
type source =
  | Allocated  (** By {!alloc}. *)
  | Pinned  (** Process memory, by {!map_host}. *)
  | Peer  (** Another GPU's memory, by {!map_peer}. *)

type memory = {
  mapping : Page_table.mapping;  (** Its virtual range and pages. *)
  host : Mmio.t option;
      (** The process's view of it, if the process has one. *)
  source : source;  (** Where it comes from. *)
}
(** The type for memory the GPU addresses. *)

val alloc :
  ?host:bool ->
  ?uncached:bool ->
  ?cpu_access:bool ->
  ?devmem:bool ->
  ?zero:bool ->
  t ->
  int ->
  memory option
(** [alloc m n] is [n] new bytes the GPU addresses, rounded up to 2 MiB from 8
    MiB on and to 4 KiB otherwise, so large ones map with large pages:
    - with [~host:true], system memory, snooped and uncached, which the process
      addresses at the GPU's address;
    - otherwise the GPU's memory, bypassing the GPU's caches if [uncached] and
      zeroed if [zero]; with [~cpu_access:true], one physical block the process
      reaches through the memory BAR, or system memory when the BAR is small,
      unless [devmem], which keeps it in the GPU's memory for structures the GPU
      requires there. Every flag defaults to [false]. [None] if the memory or
      the address space is exhausted.

    Raises [Failure] if system memory or a page table cannot be allocated,
    having freed what it took. *)

val free : t -> memory -> unit
(** [free m mem] unmaps and frees [mem], which {!alloc} returned, and returns
    its addresses. *)

val map_host : t -> nativeint -> int -> (memory, string) result
(** [map_host m a n] maps the [n] bytes of memory at [a] of the GPU's machine
    for the GPU, at [a]: it {!Pci.pin}s them and maps their pages, snooped and
    uncached. [Error why] if [a] does not start on a page, lies outside the
    GPU's virtual address range, or cannot be pinned. *)

val map_peer : t -> t -> memory -> (memory, string) result
(** [map_peer m m' mem] maps [mem], memory {!alloc} allocated on the GPU of
    [m'], for the GPU of [m], at its address on [m']: the GPU's memory through
    [m']'s memory BAR or link, and system memory at its pages, which stay
    [m']'s. [Error why] if [mem] is the GPU's memory and [m']'s BAR is small. *)

val unmap : t -> memory -> unit
(** [unmap m mem] unmaps [mem], which {!map_host} or {!map_peer} mapped, and
    unpins the process memory {!map_host} pinned. *)
