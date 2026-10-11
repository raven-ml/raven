(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The memory controller's formats: its window on the GPU's memory, and the
    page-table entries its hubs walk.

    The memory controller places the GPU's own memory, and gives the GPU's
    physical addresses the addresses its hubs and page tables name. Every
    value is pure, and any domain may call it. *)

val window :
  base:int -> top:int -> fabric:int -> memory:int -> (unit, string) result
(** [window ~base ~top ~fabric ~memory] checks that the memory controller's
    window on the GPU's memory holds the [memory] bytes of the GPU's own
    memory at [fabric] bytes past its start. [base] and [top] are the FB
    location registers, the window's first and last 16 MiB unit. The result
    is [Error msg], saying where the window is, if it does not hold them:
    the GPU takes an address outside the window for system memory, which a
    write then reaches. *)

val entry :
  gc:Gpu.version ->
  level:int ->
  pa:int ->
  [ `Table | `Page of [ `Gpu | `System ] ] ->
  uncached:bool ->
  snooped:bool ->
  fragment:int ->
  int64
(** [entry ~gc ~level ~pa target ~uncached ~snooped ~fragment] is the
    page-table entry a GPU of GC [gc] reads at [level], from 0, the root, to
    3, the leaf.
    {ul
    {- [`Table] points to the table at [pa].}
    {- [`Page `Gpu] maps GPU memory, this GPU's or a peer's over the fabric,
       at the address its memory controller gives it.}
    {- [`Page `System] maps system memory at the bus address [pa]. On GC 12
       such a page is uncached whatever [uncached] says.}}
    With [uncached] the GPU bypasses its caches for the page, with [snooped]
    its accesses snoop the processors' caches, and the page is part of a
    naturally aligned run of [2{^fragment}] 4 KiB pages.

    Raises [Invalid_argument] if [pa] is not 4 KiB aligned or has bits above
    the 48 the entry holds. *)
