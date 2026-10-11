(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** NVIDIA's page tables in the GPU's memory.

    {!format} writes {!Rig_nv_abi.Page_entry}'s entries into the GPU's
    memory, and a TLB invalidation makes the GPU read them. *)

val version : Chip.family -> Rig_nv_abi.Page_entry.version
(** [version f] is [V2] for Ampere and Ada, [V3] for Blackwell. *)

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
    GPU does not, it calls [failed] with the reason and returns [false]. *)
