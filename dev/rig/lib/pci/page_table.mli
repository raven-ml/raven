(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A GPU's physical memory and the page tables that map it.

    A driver that drives a GPU itself allocates the GPU's physical memory, takes
    virtual addresses from a {!Space}, and writes the multi-level page tables
    that map one to the other. The tables live in the GPU's memory and their
    entries are in the vendor's format, which the vendor gives as a {!format};
    this module walks and edits the tree.

    A page maps at the highest level that allows pages, whose size the range
    holds, and to whose size both its physical and virtual addresses are
    aligned. Each entry states the {e fragment} of its run, the ranges that
    follow each other in both address spaces: the largest block of the run,
    naturally aligned in both, that holds the entry.

    Physical memory comes in three pools: a boot pool at the start of memory,
    for the state that survives the driver's reopening of the GPU; an optional
    pool for page tables; and the main pool.

    Nothing here is synchronized: the GPU's owner serializes calls. *)

(** {1:format Formats} *)

(** The type for the memory an entry points into. *)
type target =
  | Gpu  (** The GPU's own memory. *)
  | System  (** System memory, which the GPU reaches over the bus. *)
  | Peer of int
      (** [Peer i] is the memory of GPU [i] of this GPU's fabric, reached over a
          direct link, at a physical address of that GPU's memory
          ({!Memory.link}). *)

type format = {
  levels : int list;
      (** The bit position of the address each level indexes, from the leaf
          level up. The leaf level indexes bit 12: it maps 4 KiB pages. Each
          table fits in a 4 KiB page. *)
  bits : int;
      (** The number of bits of a virtual address. A table has [2{^(above - l)}]
          entries, [l] the bit its level indexes and [above] the next level's,
          or [bits] for the root: with 48 bits, [[12; 21; 30; 39]] is four
          levels of 512 entries; with 49 bits, [[12; 21; 29; 38; 47]] has a root
          of 4 entries and a level of 256. *)
  pa_bits : int;
      (** The number of bits of a physical address an entry holds: {!map}
          refuses memory past [2{^pa_bits}]. *)
  first : int;  (** The number of the root level. *)
  set_table : level:int -> table:int -> int -> child:int -> unit;
      (** [set_table ~level ~table i ~child] points entry [i] of that table to
          the table at physical address [child]. *)
  set_page :
    level:int ->
    table:int ->
    int ->
    pa:int ->
    target ->
    uncached:bool ->
    snooped:bool ->
    fragment:int ->
    unit;
      (** [set_page ~level ~table i ~pa tg ~uncached ~snooped ~fragment] makes
          entry [i] of that table map the page at physical address [pa] of [tg],
          part of a naturally aligned run of [2{^fragment}] 4 KiB pages.
          [fragment] bounds the run: a format may record a smaller one, or none.
      *)
  clear : level:int -> table:int -> int -> unit;
      (** [clear ~level ~table i] makes entry [i] of that table map nothing. *)
  large : level:int -> bool;
      (** [large ~level] is [true] iff pages may map at [level]. *)
  zero : int -> int -> unit;
      (** [zero pa n] zeroes the [n] bytes of the GPU's memory at [pa]. *)
  flush : unit -> bool;
      (** [flush ()] makes the GPU walk the entries written and zeroed since the
          last flush and forget what it cached of those cleared, and is [true]
          iff the GPU confirmed it. [false] means the GPU may still reach what
          the cleared entries mapped: the vendor records it as a failure of the
          GPU. The setters and [zero] may leave their stores in flight until
          then. *)
}
(** The type for a vendor's page-table format and the access to the GPU's memory
    that holds the tables. Page_table keeps what each entry holds and never
    reads one back, so a format may write through a window {!Function.map} made
    with [~combine:true]. An entry in memory that [zero] zeroed maps nothing.

    Page_table calls [set_page] only at the leaf level or where [large] holds,
    with an address below [2{^pa_bits}], and [set_table] only with tables in the
    GPU's memory. The setters, [clear] and [zero] raise nothing. *)

(** {1:page_tables Page tables} *)

type t
(** The type for the page tables and physical memory of one GPU. *)

(** The type for where page tables come from. *)
type tables =
  | Pool
      (** A pool of their own: [memory / 512] bytes after the boot pool, rounded
          up to 1 MiB. *)
  | Main  (** The main pool, as other memory does. *)

val create :
  ?base:int ->
  format ->
  Space.t ->
  memory:int ->
  boot:int ->
  tables:tables ->
  pages:(int * int) list ->
  t
(** [create fmt s ~memory ~boot ~tables ~pages] manages [memory] bytes of a
    GPU's physical memory: a boot pool of its first [boot] bytes, the page
    tables' pool if [tables] is {!Pool}, and the main pool. [pages] lists the
    physical block sizes and their alignments that {!alloc} tries, largest
    first. The tables translate the addresses from [base] on (defaults to
    [Space.base s]), of which [s] hands out some. It allocates the root table in
    the boot pool and starts {e booting}: until {!booted}, tables and the memory
    {!palloc} and {!alloc} take by default come from the boot pool.

    Raises [Invalid_argument] if [fmt]'s levels do not rise from 12 to below
    [bits], its [pa_bits] is below 12 or past 61, [base] is not a multiple of
    the largest page a level maps, the pools do not fit in [memory] or the boot
    pool cannot hold the root table. *)

val booted : t -> unit
(** [booted t] ends booting: tables and memory come from the other pools. *)

val space : t -> Space.t
(** [space t] is the virtual address space [t] maps. *)

val base : t -> int
(** [base t] is the first virtual address the tables translate. *)

val span : t -> int
(** [span t] is the number of virtual addresses the tables reach from {!base}:
    [2{^bits}]. *)

val root : t -> int
(** [root t] is the physical address of the root table. *)

val memory : t -> int
(** [memory t] is the number of bytes of the GPU's physical memory [t] manages:
    [memory] of {!create}. *)

val pa_bits : t -> int
(** [pa_bits t] is the number of bits of a physical address [t]'s entries hold:
    [pa_bits] of its format. *)

val main_pool : t -> int
(** [main_pool t] is the number of bytes of the main pool: {!memory} less the
    boot pool and the tables' pool. It is the GPU's memory a driver hands out to
    its users. *)

(** {1:physical Physical memory} *)

val palloc : ?align:int -> ?zero:bool -> ?boot:bool -> t -> int -> int option
(** [palloc t n] is the physical address of [n] new bytes rounded up to 4 KiB,
    aligned to [align] (defaults to 4096) and zeroed if [zero] (defaults to
    [true]): from the boot pool if [boot], from the main pool otherwise. [boot]
    defaults to whether [t] is booting. It is [Some _] while the pool has a free
    block of [2 * (n' + align)] bytes, [n'] being [n] rounded up, and may be
    [None] with a smaller one that would fit.

    Raises [Invalid_argument] if [n <= 0] or [align] is not a positive power of
    two. *)

val pfree : t -> int -> unit
(** [pfree t pa] frees the block of physical memory at [pa] that {!palloc}
    returned or that a mapping of {!alloc} holds.

    Raises [Invalid_argument] if no such block starts at [pa], or if a table is
    there. *)

(** {1:mappings Mappings} *)

type mapping = {
  va : int;  (** The first virtual address. *)
  size : int;  (** The number of bytes. *)
  pages : (int * int) list;
      (** The physical ranges, as (address, bytes), in virtual order. *)
  target : target;  (** The memory they are in. *)
  uncached : bool;  (** Whether the GPU bypasses its caches for them. *)
  snooped : bool;
      (** Whether the GPU's accesses snoop the processors' caches. *)
}
(** The type for mapped virtual ranges. *)

val map :
  ?uncached:bool ->
  ?snooped:bool ->
  t ->
  va:int ->
  target ->
  (int * int) list ->
  mapping option
(** [map t ~va tg ranges] maps the physical [ranges] of [tg], in order, from
    [va] on, creating the tables it needs, and flushes, whatever the flush
    answers. [uncached] and [snooped] default to [false]. [None] if the GPU's
    memory has no room for a table it needs, having unmapped what it mapped and
    freed the tables it made.

    Raises [Invalid_argument], changing nothing, if [va] or an address or length
    of [ranges] is not a multiple of 4 KiB, a range ends past the addresses
    {!pa_bits} bits hold, the range is not within {!span} bytes from {!base}, or
    an address of it is mapped already. *)

val tables : t -> va:int -> int -> int list option
(** [tables t ~va n] is the physical addresses of the tables from the root down
    to the one whose entries would map the [n] bytes from [va], root first,
    creating those that are missing and flushing, as {!map} does. They are never
    freed: the caller may write the entries that map those bytes itself. [t]
    knows only the entries it wrote, so {!map} does not refuse the addresses the
    caller's entries map, and maps them inside these tables. [None] if the GPU's
    memory has no room for one, keeping those it made.

    Raises [Invalid_argument] if the range is not as {!map} requires, or a page
    larger than [n] bytes maps [va]. *)

val unmap : t -> va:int -> int -> bool
(** [unmap t ~va n] unmaps the [n] bytes mapped from [va], frees the tables that
    become empty, and flushes. It is [true] iff the flush was confirmed, so that
    the GPU no longer reaches the memory; on [false] it may still reach it.

    Raises [Invalid_argument], changing nothing, if the range is not as {!map}
    requires, an address of it is not mapped, or a page maps addresses on both
    sides of its bounds. *)

val alloc :
  ?uncached:bool -> ?contiguous:bool -> ?below:int -> t -> int -> mapping option
(** [alloc t n] is [n] bytes, rounded up to 4 KiB, of new physical memory from
    the pool {!palloc} takes from by default, mapped at new virtual addresses of
    [space t]:
    - if [contiguous] (defaults to [false]), one zeroed block, aligned as the
      largest block of [pages] it could hold when the pool has one, so that it
      maps with large pages;
    - otherwise the largest blocks of [pages] the pool has, not zeroed.

    With [below], the memory lies at physical addresses that end at or below it,
    such as the part of the GPU's memory its BAR reaches, chosen before anything
    is written.

    [None] if the space or the pool cannot supply them, as {!Space.alloc} and
    {!palloc} bound each request, or if a table has no room, having given back
    what it took.

    Raises [Invalid_argument] if [n <= 0]. *)

val free : t -> mapping -> bool
(** [free t m] unmaps [m] and flushes. It is [true] iff the flush was confirmed,
    having freed [m]'s virtual addresses and, if it is in the {!Gpu}'s memory,
    its physical memory; on [false] it keeps both, which the GPU may still
    reach. *)
