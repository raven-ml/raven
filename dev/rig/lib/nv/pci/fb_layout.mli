(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Where the GSP's firmware goes in memory.

    The GSP reserves the top of the GPU's memory for itself: the VGA workspace,
    the FRTS region, its boot binary, its image, its heap and its metadata, and
    below them a heap outside its protected region (WPR2). The process manages
    the memory below it. The GSP reads its image through a radix-3 table: three
    levels of 64-bit page addresses over the image's 4 KiB pages. The rules are
    those of release 570.144 ([kgspCalculateFbLayout_TU102],
    [kgspCalculateFbLayout_GH100]). Pure functions. *)

(** The type for who lays the GSP's region out. *)
type layout =
  | Process
      (** The process, which writes the region into the WPR metadata (Ampere,
          Ada). *)
  | Fmc  (** The FMC, from the sizes the metadata gives it (Blackwell). *)

val radix3 : int -> int array
(** [radix3 n] is the number of 4 KiB pages of each level of the radix-3 table
    over an image of [n] bytes, from the root down, then of the image: each page
    of a level holds 512 addresses of the next. *)

(** {1:meta The WPR metadata} *)

type image = { address : int; size : int }
(** The type for an image in system memory: its bus address and its size. *)

val wpr_meta :
  layout ->
  memory:int ->
  gsp:image ->
  signature:image ->
  bootloader:image ->
  code:int ->
  data:int ->
  manifest:int ->
  string
(** [wpr_meta l ~memory ~gsp ~signature ~bootloader ~code ~data ~manifest] is
    the WPR metadata ([GspFwWprMeta]) of a boot of a GPU of [memory] bytes: the
    GSP's image, which [gsp] gives by its radix-3 table's address and the
    image's size, its signature, and its bootloader with the offsets of its
    code, data and manifest. With [Process] it holds the region the process lays
    out, which {!frts} and {!top} also follow; with [Fmc], the sizes of the
    parts the FMC lays out. *)

(** {1:region The region} *)

val frts : memory:int -> int
(** [frts ~memory] is the offset of the 1 MiB FRTS region the process lays out
    on a GPU of [memory] bytes: below the 1 MiB VGA workspace at the end of
    memory. *)

val cot_frts : int * int
(** [cot_frts] is where the COT payload puts the FRTS region the FMC lays out:
    its end's distance below the end of memory, 28 MiB, and its size, 1 MiB, as
    the payload's [frtsVidmemOffset] and [frtsVidmemSize]. *)

val top : layout -> memory:int -> boot:int -> image:int -> int
(** [top l ~memory ~boot ~image] is the end of the memory the process manages,
    on 2 MiB, for a bootloader of [boot] bytes and an image of [image] bytes:
    below the GSP's region and 64 MiB below the end of memory. With [Fmc] it
    sums the parts the FMC places below its FRTS region, with 6 MiB for their
    alignment, since the FMC alone knows where they go. *)

val check : wpr2:int -> top:int -> (unit, string) result
(** [check ~wpr2 ~top] is [Ok ()] iff the memory below [top] lies below the heap
    the GSP put under its protected region, which starts at [wpr2]: the check of
    {!top}'s bound once the GSP runs. *)
