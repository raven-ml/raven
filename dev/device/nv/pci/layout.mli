(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Where the GSP's firmware goes in memory (private).

    The GSP reserves the top of the GPU's memory for itself: the VGA workspace,
    the FRTS region, its boot binary, its image, its heap and its metadata, and
    below them a heap outside its protected region (WPR2). On Ampere and Ada the
    process lays that region out and writes it into the WPR metadata
    ([kgspCalculateFbLayout_TU102]); on Blackwell the FMC lays it out from the
    sizes it is given ([kgspCalculateFbLayout_GH100]). The process manages the
    memory below it.

    The GSP reads its image through a radix-3 table: three levels of 64-bit page
    addresses over the image's 4 KiB pages. Pure functions. *)

val radix3 : int -> int array
(** [radix3 n] is the number of 4 KiB pages of each level of the radix-3 table
    over an image of [n] bytes, from the root down, then of the image: each page
    of a level holds 512 addresses of the next. *)

val wpr : memory:int -> boot:int -> image:int -> ((int * int) * int) list
(** [wpr ~memory ~boot ~image] is the region the GSP reserves on an Ampere or
    Ada GPU of [memory] bytes, for a bootloader of [boot] bytes and an image of
    [image] bytes: each field of the WPR metadata ([GspFwWprMeta]) the process
    sets, as [Defs.Wpr_meta]'s (offset, bytes), with its value. *)

val fmc_sizes : ((int * int) * int) list
(** [fmc_sizes] is the fields of the WPR metadata the process sets on Blackwell,
    as {!wpr}'s, with their values: the sizes of the parts the FMC lays out. *)

val frts : memory:int -> int
(** [frts ~memory] is the offset of the 1 MiB FRTS region on an Ampere or Ada
    GPU of [memory] bytes, where {!wpr} puts it: below the 1 MiB VGA workspace
    at the end of memory. *)

val cot_frts : int * int
(** [cot_frts] is where the COT payload puts the FRTS region on Blackwell: its
    end's distance below the end of memory, 28 MiB, and its size, 1 MiB, as the
    payload's [frtsVidmemOffset] and [frtsVidmemSize]. *)

val top : Chip.family -> memory:int -> boot:int -> image:int -> int
(** [top f ~memory ~boot ~image] is the end of the memory the process manages,
    on 2 MiB: below the GSP's region and 64 MiB below the end of memory. On
    Blackwell it sums the parts the FMC places below its FRTS region, with 6 MiB
    for their alignment, since the FMC alone knows where they go. *)

val check : wpr2:int -> top:int -> (unit, string) result
(** [check ~wpr2 ~top] is [Ok ()] iff the memory below [top] lies below the heap
    the GSP put under its protected region, which starts at [wpr2]: the check of
    {!top}'s bound once the GSP runs. *)
