(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** NVIDIA's page-table entries.

    Two versions: version 2 (Pascal to Ada) translates 49-bit addresses through
    five levels, its root of 4 entries; version 3 (Hopper on) 57-bit addresses
    through six, its root of 2. The level above the leaf holds 16-byte {e dual}
    entries: a 2 MiB page in the low half, or the leaf table in the high half.
    Pages are of kind [GENERIC_MEMORY] ([0x06]).

    The encoders are pure. {!format} writes them into the GPU's memory and
    publishes them with a TLB invalidation. *)

(** The type for versions of the page-table format. *)
type version = V2 | V3

val version : Chip.family -> version
(** [version f] is [V2] for Ampere and Ada, [V3] for Blackwell. *)

val levels : version -> int list
(** [levels v] is the bit each level indexes, from the leaf up:
    [[12; 21; 29; 38; 47]] for [V2], with [56] above for [V3]. *)

val bits : version -> int
(** [bits v] is the width of a virtual address: [49] or [57]. *)

val pte :
  version ->
  pa:int ->
  Rig_pci.Page_table.target ->
  uncached:bool ->
  snooped:bool ->
  int64
(** [pte v ~pa tg ~uncached ~snooped] is the entry that maps the page at [pa] of
    [tg]: aperture [0] for the GPU's memory, [1] for a peer with its index, [2]
    for system memory reached snooped and [3] for system memory not snooped;
    [uncached] sets [VOL] ([V2]) or the uncached [PCF] ([V3]). *)

val pde : version -> child:int -> int64
(** [pde v ~child] is the directory entry that points to the table at [child] in
    the GPU's memory. *)

val dual :
  version -> [ `Table of int | `Page of int64 | `None ] -> int64 * int64
(** [dual v e] is the low and high halves of a dual entry: [`Table pa] points to
    the leaf table at [pa] (with [NO_ATS] in the low half on [V2]), [`Page pte]
    maps a 2 MiB page, [`None] maps nothing. *)

val format : Chip.t -> Rig_pci.Window.t -> Rig_pci.Page_table.format
(** [format c bar] is the format, of [c]'s {!version}, whose tables are in the
    GPU's memory, which [bar] (its memory BAR) reaches. Its [flush] flushes
    [bar] ({!Rig_pci.Window.flush}), triggers the TLB invalidation of every
    level and waits for the GPU to clear its trigger, at most 2 seconds; it
    raises {!Rig_nv.Fault} if the GPU does not. *)
