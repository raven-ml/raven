(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** GPU virtual memory: physical memory, address spaces and page tables.

    A runtime that drives a GPU without its kernel driver manages the GPU's
    memory itself. It allocates the GPU's physical memory, allocates virtual
    addresses, and writes the multi-level page tables that map one to the other.
    The page tables live in the GPU's memory and their entry format is the
    vendor's: the vendor gives it as an {!entry}, and this module walks and
    edits the tree. A page maps at the highest level that allows pages and to
    whose size both its virtual and physical addresses are aligned, and each
    entry states the {e fragment} of its run: the largest naturally aligned
    block, in both address spaces, of contiguous pages it belongs to.

    Physical memory comes in three pools: a boot pool at the start of memory for
    the state that survives the runtime's reopening of the GPU, an optional pool
    for page tables, and the main pool.

    Nothing here is synchronized: the device's owner serializes calls, and
    {!Space} serializes its own. *)

(** {1:spaces Address spaces} *)

(** The type for the memory an entry points into. *)
type space =
  | Phys  (** The GPU's own memory. *)
  | Sys  (** System memory, which the GPU reaches over the bus. *)
  | Peer  (** Another GPU's memory, reached over a direct link. *)

(** Virtual address spaces shared by the GPUs of one vendor. *)
module Space : sig
  type t
  (** The type for virtual address spaces. The GPUs of a vendor share one, so
      that a range means the same memory on each of them. It is synchronized. *)

  val create : base:int -> int -> t
  (** [create ~base n] is the [n] virtual addresses from [base] on. *)

  val base : t -> int
  (** [base s] is [s]'s first address. *)

  val length : t -> int
  (** [length s] is [s]'s number of addresses. *)

  val alloc : ?align:int -> t -> int -> int option
  (** [alloc s n] is the first address of [n] free addresses of [s], aligned to
      the largest power of two not above [n] and to [align] (defaults to 4096),
      so that large ranges map with large pages; [None] if [s] has no such
      range. *)

  val free : t -> int -> unit
  (** [free s a] frees the range {!alloc} returned at [a]. *)
end

(** {1:tables Page tables} *)

type entry = {
  levels : int list;
      (** The bit position of the address each level indexes, from the leaf
          level up; [[12; 21; 30; 39]] for four levels of 512 entries. *)
  bits : int;  (** The number of bits of a virtual address. *)
  first : int;  (** The number of the root level. *)
  get : level:int -> table:int -> int -> int64;
      (** [get ~level ~table i] is entry [i] of the table of [level] at physical
          address [table]. Where the format's entries are wider than 64 bits, it
          is the 64 bits of them that the entry uses. *)
  set : level:int -> table:int -> int -> int64 -> unit;
      (** [set ~level ~table i e] writes entry [i] of the table of [level] at
          [table]: [e] as {!get} reads it. *)
  encode :
    level:int ->
    table:bool ->
    space ->
    uncached:bool ->
    snooped:bool ->
    fragment:int ->
    valid:bool ->
    int ->
    int64;
      (** [encode ~level ~table sp ~uncached ~snooped ~fragment ~valid pa] is
          the entry at [level] for the physical address [pa] in [sp]: a child
          table if [table], a page otherwise, part of a naturally aligned run of
          [2{^fragment}] 4 KiB pages. *)
  valid : int64 -> bool;  (** [valid e] is [true] iff [e] is in use. *)
  leaf : level:int -> int64 -> bool;
      (** [leaf ~level e] is [true] iff [e] maps a page rather than a table. *)
  address : int64 -> int;
      (** [address e] is the physical address of the table [e] points to. *)
  large : level:int -> bool;
      (** [large ~level] is [true] iff pages may be mapped at [level]. *)
  zero : int -> int -> unit;
      (** [zero pa n] zeroes the [n] bytes of GPU memory at [pa]. *)
  flush : unit -> unit;
      (** [flush ()] makes the GPU see the entries written so far. *)
}
(** The type for a vendor's page-table format and the access to its memory. *)

type mapping = {
  va : int;  (** The first virtual address. *)
  size : int;  (** The number of bytes. *)
  pages : (int * int) list;
      (** The physical ranges, as (address, bytes), in virtual order. *)
  space : space;  (** The memory they are in. *)
  uncached : bool;  (** Whether the GPU bypasses its caches for them. *)
  snooped : bool;
      (** Whether the GPU's accesses snoop the processors' caches. *)
}
(** The type for mapped virtual ranges. *)

type t
(** The type for the page tables and physical memory of one GPU. *)

val create :
  ?base:int ->
  entry ->
  Space.t ->
  memory:int ->
  boot:int ->
  tables:bool ->
  pages:(int * int) list ->
  t
(** [create e s ~memory ~boot ~tables ~pages] manages [memory] bytes of a GPU's
    physical memory: a boot pool of its first [boot] bytes, a pool for page
    tables of [memory / 512] bytes rounded up to 1 MiB if [tables], and the main
    pool. [pages] lists the physical block sizes and alignments that {!alloc}
    tries, largest first. The tables translate the addresses from [base] on
    (defaults to [s]'s base), of which [s] allocates some. It allocates the root
    table in the boot pool, and starts {e booting}: until {!booted}, physical
    memory comes only from the boot pool.

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

val palloc : ?align:int -> ?zero:bool -> ?boot:bool -> t -> int -> int option
(** [palloc t n] is the physical address of [n] new bytes rounded up to 4 KiB,
    aligned to [align] (defaults to 4096) and zeroed if [zero] (defaults to
    [true]): from the boot pool if [boot], from the main pool otherwise. [boot]
    defaults to whether [t] is booting. [None] if the pool has no such block. *)

val pfree : t -> int -> unit
(** [pfree t pa] frees the physical memory {!palloc} returned at [pa]. *)

val map :
  ?uncached:bool ->
  ?snooped:bool ->
  t ->
  va:int ->
  space ->
  (int * int) list ->
  mapping
(** [map t ~va sp ranges] maps the physical [ranges] of [sp], in order, from
    [va] on, creating the tables it needs, and flushes. [uncached] and [snooped]
    default to [false].

    Raises [Invalid_argument] if an address of the range is mapped already, and
    [Failure] if a table cannot be allocated, having unmapped what it mapped. *)

val tables : t -> va:int -> int -> int list
(** [tables t ~va n] is the physical addresses of the tables from the root down
    to the one whose entries would map the [n] bytes from [va], creating those
    that are missing, as {!map} would, root first.

    Raises [Failure] if a table cannot be allocated. *)

val unmap : t -> va:int -> int -> unit
(** [unmap t ~va n] unmaps the [n] bytes mapped from [va], frees the tables that
    become empty, and flushes, so the GPU no longer reaches the memory.

    Raises [Invalid_argument] if an address of the range is not mapped. *)

val alloc :
  ?align:int ->
  ?uncached:bool ->
  ?contiguous:bool ->
  ?zero:bool ->
  t ->
  int ->
  mapping option
(** [alloc t n] is [n] bytes, rounded up to 4 KiB, of new physical memory mapped
    at new virtual addresses of [t]'s space:
    - if [contiguous] (defaults to [false]), one zeroed physical block, aligned
      as the largest block of [pages] it could hold when the pool has such a
      block, so that it maps with large pages;
    - otherwise the largest blocks of [pages] that the pool has, zeroed if
      [zero] (defaults to [false]).

    [None] if the pool cannot supply them, having freed what it took. Raises
    [Failure] as {!map} does, having freed what it took. *)

val free : t -> mapping -> unit
(** [free t m] unmaps [m], frees its virtual addresses and, if it is in {!Phys},
    its physical memory. *)
