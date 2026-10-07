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

    A page maps at the highest level that allows pages and to whose size both
    its virtual and physical addresses are aligned, and each entry states the
    {e fragment} of its run: the largest naturally aligned block, in both
    address spaces, of contiguous pages it belongs to.

    Physical memory comes in three pools: a boot pool at the start of memory,
    for the state that survives the driver's reopening of the GPU; an optional
    pool for page tables; and the main pool.

    Nothing here is synchronized: the GPU's owner serializes calls. *)

(** {1:format Formats} *)

(** The type for the memory an entry points into. *)
type target =
  | Gpu  (** The GPU's own memory. *)
  | System  (** System memory, which the GPU reaches over the bus. *)
  | Peer  (** Another GPU's memory, reached over a direct link. *)

type format = {
  levels : int list;
      (** The bit position of the address each level indexes, from the leaf
          level up: [[12; 21; 30; 39]] for four levels of 512 entries. *)
  bits : int;  (** The number of bits of a virtual address. *)
  first : int;  (** The number of the root level. *)
  get : level:int -> table:int -> int -> int64;
      (** [get ~level ~table i] is entry [i] of the table of [level] at physical
          address [table]: the 64 bits of it that the format uses. *)
  set : level:int -> table:int -> int -> int64 -> unit;
      (** [set ~level ~table i e] writes entry [i] of that table: [e] as [get]
          reads it. *)
  encode :
    level:int ->
    table:bool ->
    target ->
    uncached:bool ->
    snooped:bool ->
    fragment:int ->
    valid:bool ->
    int ->
    int64;
      (** [encode ~level ~table t ~uncached ~snooped ~fragment ~valid pa] is the
          entry at [level] for the physical address [pa] in [t]: a child table
          if [table], a page otherwise, part of a naturally aligned run of
          [2{^fragment}] 4 KiB pages. *)
  valid : int64 -> bool;  (** [valid e] is [true] iff [e] is in use. *)
  leaf : level:int -> int64 -> bool;
      (** [leaf ~level e] is [true] iff [e] maps a page rather than a table. *)
  address : int64 -> int;
      (** [address e] is the physical address of the table [e] points to. *)
  large : level:int -> bool;
      (** [large ~level] is [true] iff pages may map at [level]. *)
  zero : int -> int -> unit;
      (** [zero pa n] zeroes the [n] bytes of the GPU's memory at [pa]. *)
  flush : unit -> unit;
      (** [flush ()] makes the GPU see the entries written so far. *)
}
(** The type for a vendor's page-table format and the access to the GPU's memory
    that holds the tables. *)

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
    the boot pool and starts {e booting}: until {!booted}, physical memory comes
    only from the boot pool.

    Raises [Invalid_argument] if the pools do not fit in [memory] or the boot
    pool cannot hold the root table. *)

val booted : t -> unit
(** [booted t] ends booting: physical memory comes from the other pools. *)

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
(** [memory t] is the size of the main pool in bytes. *)

(** {1:physical Physical memory} *)

val palloc : ?align:int -> ?zero:bool -> ?boot:bool -> t -> int -> int option
(** [palloc t n] is the physical address of [n] new bytes rounded up to 4 KiB,
    aligned to [align] (defaults to 4096) and zeroed if [zero] (defaults to
    [true]): from the boot pool if [boot], from the main pool otherwise. [boot]
    defaults to whether [t] is booting. It is [Some _] while the pool has a free
    block of [2 * (n + align)] bytes, and may be [None] with a smaller one that
    would fit. *)

val pfree : t -> int -> unit
(** [pfree t pa] frees the physical memory {!palloc} returned at [pa].

    Raises [Invalid_argument] if none was. *)

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
    [va] on, creating the tables it needs, and flushes. [uncached] and [snooped]
    default to [false]. [None] if the GPU's memory has no room for a table it
    needs, having unmapped what it mapped.

    Raises [Invalid_argument] if an address of the range is mapped already. *)

val tables : t -> va:int -> int -> int list option
(** [tables t ~va n] is the physical addresses of the tables from the root down
    to the one whose entries would map the [n] bytes from [va], root first,
    creating those that are missing as {!map} would. [None] if the GPU's memory
    has no room for one. *)

val unmap : t -> va:int -> int -> unit
(** [unmap t ~va n] unmaps the [n] bytes mapped from [va], frees the tables that
    become empty, and flushes, so the GPU no longer reaches the memory.

    Raises [Invalid_argument] if an address of the range is not mapped. *)

val alloc :
  ?align:int -> ?uncached:bool -> ?contiguous:bool -> t -> int -> mapping option
(** [alloc t n] is [n] bytes, rounded up to 4 KiB, of new physical memory of the
    main pool mapped at new virtual addresses of [space t]:
    - if [contiguous] (defaults to [false]), one zeroed block, aligned as the
      largest block of [pages] it could hold when the pool has one, so that it
      maps with large pages;
    - otherwise the largest blocks of [pages] the pool has, not zeroed.

    [None] if the pool or the space cannot supply them, as {!palloc} and
    {!Space.alloc} bound it, or if a table has no room, having freed what it
    took. *)

val free : t -> mapping -> unit
(** [free t m] unmaps [m], frees its virtual addresses and, if it is in the
    {!Gpu}'s memory, its physical memory. *)
