(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** NVIDIA's page-table entries.

    Two versions: version 2 (Pascal to Ada) translates 49-bit addresses through
    five levels, its root of 4 entries; version 3 (Hopper on) 57-bit addresses
    through six, its root of 2. The level above the leaf holds 16-byte {e dual}
    entries: a 2 MiB page in the low half, or the leaf table in the high half.
    Pages are of kind [GENERIC_MEMORY] ([0x06]). The bits are [dev_mmu.h]'s,
    [tu102] for version 2 and [gh100] for version 3. Pure functions. *)

(** The type for versions of the page-table format. *)
type version = V2 | V3

val levels : version -> int list
(** [levels v] is the bit each level indexes, from the leaf up:
    [[12; 21; 29; 38; 47]] for [V2], with [56] above for [V3]. *)

val bits : version -> int
(** [bits v] is the width of a virtual address: [49] or [57]. *)

val pages : version -> (int * int) list
(** [pages v] is the blocks [v]'s tables map memory with, largest first, each
    aligned to its size: pages of 512 MiB, 2 MiB and 4 KiB, at the level above
    the dual level, the dual level and the leaf. *)

val pa_bits : version -> int
(** [pa_bits v] is the width of a system address an entry holds: [58] for [V2],
    [52] for [V3]. *)

(** The type for what a page maps. *)
type target =
  | Gpu  (** The GPU's memory. *)
  | Peer of int  (** The memory of the peer of that index. *)
  | System of { snooped : bool }
      (** System memory, reached through the processors' caches if [snooped]. *)

val pte : version -> pa:int -> target -> uncached:bool -> int64
(** [pte v ~pa tg ~uncached] is the entry that maps the page at [pa] of [tg]:
    aperture [0] for the GPU's memory, [1] for a peer with its index, [2] for
    system memory reached snooped and [3] for system memory not snooped;
    [uncached] sets [VOL] ([V2]) or the uncached [PCF] ([V3]). *)

val pde : version -> child:int -> int64
(** [pde v ~child] is the directory entry that points to the table at [child] in
    the GPU's memory. *)

val dual :
  version -> [ `Table of int | `Page of int64 | `None ] -> int64 * int64
(** [dual v e] is the low and high halves of a dual entry: [`Table pa] points to
    the leaf table at [pa] (with [NO_ATS] in the low half on [V2]), [`Page pte]
    maps a 2 MiB page, [`None] maps nothing. *)
