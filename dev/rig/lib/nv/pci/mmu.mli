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

    {!format} writes them into the GPU's memory and publishes them with a TLB
    invalidation. *)

(** The type for versions of the page-table format. *)
type version = V2 | V3

val version : Chip.family -> version
(** [version f] is [V2] for Ampere and Ada, [V3] for Blackwell. *)

val levels : version -> int list
(** [levels v] is the bit each level indexes, from the leaf up:
    [[12; 21; 29; 38; 47]] for [V2], with [56] above for [V3]. *)

val bits : version -> int
(** [bits v] is the width of a virtual address: [49] or [57]. *)

val pages : version -> (int * int) list
(** [pages v] is the blocks [v]'s tables map memory with, largest first, each
    aligned to its size: pages of 512 MiB, 2 MiB and 4 KiB, at the level above
    the dual level, the dual level and the leaf. *)

val format :
  Chip.t ->
  Rig_pci.Window.t ->
  failed:(string -> unit) ->
  Rig_pci.Page_table.format
(** [format c bar ~failed] is the format, of [c]'s {!version}, whose tables are
    in the GPU's memory, which [bar] (its memory BAR) reaches, and whose entries
    hold system addresses of 58 bits on [V2] and 52 on [V3]. Its [flush] flushes
    [bar] ({!Rig_pci.Window.flush}), triggers the TLB invalidation of every
    level and waits for the GPU to clear its trigger, at most 2 seconds; if the
    GPU does not, it calls [failed] with the reason and answers [false]. *)
