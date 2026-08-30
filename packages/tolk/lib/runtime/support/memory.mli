(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Device memory manager over multi-level page tables.

    Manages a device's physical memory and virtual address space.
    Physical pages come from TLSF allocators over three regions of
    device memory: a boot region, an optional reserved page-table
    region and the main region. Virtual ranges are mapped to physical
    ranges by walking a multi-level page-table tree, using the largest
    page size each level permits. The page-table entry format is
    target-specific and injected as {!type-pt_ops}.

    Page-table entry words are 64-bit ({!Int64.t}); addresses and
    sizes are native [int]s and must stay below [2{^48}]. *)

(** {1:mappings Address spaces and mappings} *)

type addr_space =
  | Phys  (** Device-local physical memory. *)
  | Sys  (** System (host) memory. *)
  | Peer  (** Memory of a peer device. *)

(** The type for address spaces a page can point into. *)

type virt_mapping = {
  va_addr : int;  (** Start of the virtual range. *)
  size : int;  (** Size of the virtual range in bytes. *)
  paddrs : (int * int) list;
      (** Backing physical ranges as [(paddr, size)] pairs, in mapping
          order. Their sizes sum to [size]. *)
  aspace : addr_space;  (** Address space of the physical ranges. *)
  uncached : bool;  (** Pages are mapped uncached. *)
  snooped : bool;  (** Pages are mapped cache-coherent with the host. *)
}
(** The type for virtual mappings returned by {!map_range} and
    {!valloc}. *)

(** {1:pt Page tables} *)

type 'pt pt_ops = {
  make : paddr:int -> lv:int -> 'pt;
      (** [make ~paddr ~lv] is a view of the page table stored at
          physical address [paddr], at level [lv] of the tree. *)
  set_entry :
    'pt ->
    idx:int ->
    paddr:int ->
    ?table:bool ->
    ?uncached:bool ->
    ?aspace:addr_space ->
    ?snooped:bool ->
    ?frag:int ->
    valid:bool ->
    unit ->
    unit;
      (** [set_entry pt ~idx ~paddr ... ~valid ()] writes entry [idx] of
          [pt]. [table] marks the entry as pointing to a child page
          table rather than a page (defaults to [false]). [frag] is
          the TLB fragment size exponent: the entry is part of a
          naturally aligned run of [2{^frag}] 4KB pages (defaults to
          [0]). [aspace] defaults to {!Phys}; [uncached] and [snooped]
          default to [false]. *)
  entry : 'pt -> int -> int64;
      (** [entry pt idx] is the raw 64-bit word of entry [idx]. *)
  valid : 'pt -> int -> bool;
      (** [valid pt idx] is [true] iff entry [idx] is valid. *)
  address : 'pt -> int -> int;
      (** [address pt idx] is the physical address entry [idx] points
          to. *)
  is_page : 'pt -> int -> bool;
      (** [is_page pt idx] is [true] iff entry [idx] maps a page (it
          terminates the walk) rather than a child page table. *)
  supports_huge_page : 'pt -> paddr:int -> bool;
      (** [supports_huge_page pt ~paddr] is [true] iff a page of
          [pt]'s level can map [paddr] directly. *)
  paddr : 'pt -> int;  (** [paddr pt] is [pt]'s physical address. *)
  lv : 'pt -> int;  (** [lv pt] is [pt]'s level in the tree. *)
}
(** The type for page-table operations. ['pt] is the target-specific
    page-table view; the manager creates, reads and writes page tables
    exclusively through these operations. *)

(** {1:managers Managers} *)

type 'pt t
(** The type for memory managers. Mutable. *)

val create :
  pt_ops:'pt pt_ops ->
  vram_size:int ->
  boot_size:int ->
  va_bits:int ->
  va_shifts:int list ->
  va_base:int ->
  palloc_ranges:(int * int) list ->
  va_allocator:Tlsf.t ->
  is_booting:(unit -> bool) ->
  zero_vram:(paddr:int -> size:int -> unit) ->
  ?first_lv:int ->
  ?reserve_ptable:bool ->
  ?smi_dev:bool ->
  ?dbg_name:string ->
  ?on_range_mapped:(unit -> unit) ->
  unit ->
  'pt t
(** [create ()] is a memory manager for a device with [vram_size]
    bytes of physical memory.

    Physical memory is split into a [boot_size]-byte boot region at
    address [0], an optional page-table region (present when
    [reserve_ptable] is [true], sized to [vram_size / 512] rounded up
    to 1MB) and the main region covering the rest.

    [va_shifts] lists the page-size shifts of the page-table levels in
    increasing order (e.g. [[12; 21; 30; 39]] for a four-level tree
    with 4KB leaf pages); [va_bits] is the width of the virtual
    address space. Virtual addresses are rebased by [va_base] before
    walking the tree; [va_allocator] hands out virtual ranges above
    that base and may be shared between managers. [first_lv] is the
    level of the root page table (defaults to [0]).

    [palloc_ranges] lists [(size, align)] candidates for backing
    non-contiguous {!valloc} requests, largest first.

    [is_booting] queries the device boot state: while it returns
    [true] only boot-region memory can be allocated. [zero_vram]
    clears a physical range of device memory. [on_range_mapped] runs
    after every {!map_range} (e.g. to flush TLBs; defaults to a
    no-op). [dbg_name] prefixes debug output (see below). When
    [smi_dev] is [true] the root page table is not zeroed on creation
    (monitoring-only access to a live device).

    The root page table is allocated from the boot region, so the
    device must be booting at creation time.

    With the environment variable [MM_DEBUG] set to a non-zero
    integer, {!map_range} and {!unmap_range} print the ranges they
    touch. With [GMMU=0], {!valloc} and {!vfree} bypass per-request
    mappings and use a single identity mapping of the whole physical
    memory (a debug mode). *)

val vram_size : 'pt t -> int
(** [vram_size t] is the physical memory size managed by [t]. *)

val va_base : 'pt t -> int
(** [va_base t] is the base of [t]'s virtual address space. *)

val va_bits : 'pt t -> int
(** [va_bits t] is the width of [t]'s virtual address space. *)

val level_cnt : 'pt t -> int
(** [level_cnt t] is the number of page-table levels. *)

val pte_covers : 'pt t -> int -> int
(** [pte_covers t lv] is the number of bytes one entry at level [lv]
    covers. Levels count down from the root at [0]. *)

val pte_cnt : 'pt t -> int -> int
(** [pte_cnt t lv] is the number of entries in a level-[lv] page
    table. *)

val root_page_table : 'pt t -> 'pt
(** [root_page_table t] is [t]'s root page table. *)

(** {1:virtual Virtual memory} *)

val map_range :
  'pt t ->
  vaddr:int ->
  size:int ->
  (int * int) list ->
  addr_space ->
  ?uncached:bool ->
  ?snooped:bool ->
  ?boot:bool ->
  unit ->
  virt_mapping
(** [map_range t ~vaddr ~size paddrs aspace ()] maps [size] bytes of
    virtual address space starting at [vaddr] to the physical ranges
    [paddrs] (as [(paddr, size)] pairs whose sizes sum to [size]),
    creating intermediate page tables as needed and using the largest
    page size each level allows. [uncached], [snooped] and the TLB
    fragment hint are recorded in the entries. [boot] allocates
    intermediate page tables from the boot region.

    Raises [Invalid_argument] if the sizes don't add up or any page
    of the range is already mapped. *)

val unmap_range : 'pt t -> vaddr:int -> size:int -> unit
(** [unmap_range t ~vaddr ~size] invalidates every entry mapping the
    range and frees page tables that become empty.

    Raises [Invalid_argument] if part of the range is not mapped. *)

val page_tables : 'pt t -> vaddr:int -> size:int -> 'pt list
(** [page_tables t ~vaddr ~size] is the chain of page tables covering
    the start of the range, root first, creating them as needed. *)

val alloc_vaddr : 'pt t -> int -> ?align:int -> unit -> int
(** [alloc_vaddr t size ()] is a fresh virtual range of [size] bytes.
    The range is aligned to the largest power of two not exceeding
    [size], or to [align] if that is larger. [align] defaults to
    [0x1000].

    Raises {!Tlsf.Out_of_memory} if the virtual address space is
    exhausted. *)

val valloc :
  'pt t ->
  int ->
  ?align:int ->
  ?uncached:bool ->
  ?contiguous:bool ->
  unit ->
  virt_mapping
(** [valloc t size ()] allocates [size] bytes (rounded up to 4KB) of
    physical memory and maps them at a fresh virtual range. With
    [contiguous] the backing memory is a single zeroed physical
    range; otherwise it is assembled from the manager's allocation
    ranges, largest first, to reduce TLB pressure. [align] constrains
    the virtual range as in {!alloc_vaddr}.

    Raises {!Tlsf.Out_of_memory} if physical memory is exhausted; any
    partially allocated ranges are released. *)

val vfree : 'pt t -> virt_mapping -> unit
(** [vfree t vm] unmaps [vm] and releases its virtual range and
    backing physical memory. *)

(** {1:physical Physical memory} *)

val palloc :
  'pt t ->
  int ->
  ?align:int ->
  ?zero:bool ->
  ?boot:bool ->
  ?ptable:bool ->
  unit ->
  int
(** [palloc t size ()] is the physical address of a freshly allocated
    range of [size] bytes, rounded up to 4KB. [align] defaults to
    [0x1000]. The range is zeroed unless [zero] is [false]. [boot]
    allocates from the boot region; [ptable] from the page-table
    region when the manager reserves one.

    Raises [Invalid_argument] unless [boot] matches the device's boot
    state: while booting only boot memory can be allocated, and boot
    memory only while booting. Raises {!Tlsf.Out_of_memory} if the
    region is exhausted. *)

val pfree : 'pt t -> int -> ?ptable:bool -> unit -> unit
(** [pfree t paddr ()] releases the physical range at [paddr].
    [ptable] must match the flag the range was allocated with. *)
