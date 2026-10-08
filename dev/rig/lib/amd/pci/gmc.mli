(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The GPU's memory controller: where it places its own memory, its two hubs
   (GC's, which compute work goes through, and MM's, the copy engines' and
   firmware's), their page-table walkers and TLBs, the entries they read, and
   the host data path (HDP) the host's writes through the memory BAR take.

   The GPU's owner serializes calls. *)

type t

(* [make r vram] reads the GPU's memory apertures: where its memory starts for
   the memory controller and, in a fabric of GPUs (XGMI), for its peers. *)
val make : Regs.t -> Rig_pci.Window.t -> t

(* [window ~base ~top ~fabric ~memory] checks that the memory controller's
   window on the GPU's memory, from the FB location registers' [base] and [top]
   (its first and last 16 MiB unit), holds the [memory] bytes of the GPU's own
   memory at [fabric] bytes past its start. [Error msg] says where the window
   is: an address outside it is system memory to the GPU, which a write then
   reaches. Pure. *)
val window :
  base:int -> top:int -> fabric:int -> memory:int -> (unit, string) result

(* [covers g ~memory] is {!window} of [g]'s registers. *)
val covers : t -> memory:int -> (unit, string) result

(* [mc g pa] is the address the memory controller gives the GPU's physical
   address [pa]. *)
val mc : t -> int -> int

(* [fabric g pa] is the address its page tables and hubs name the GPU's own
   memory at [pa] by: past the memory controller's offset (MC_VM_FB_OFFSET, none
   on a virtual function) and, in a fabric, its node's segment. *)
val fabric : t -> int -> int

(* [hive g] is [true] iff the GPU is one of several joined by a fabric. *)
val hive : t -> bool

(* [instances g hub] is the instances of [hub]: GC's dies, or MM's hubs (the
   live accelerator dies on a GPU of several). *)
val instances : t -> [ `Gc | `Mm ] -> int list

(* [entry ~gc ~level ~pa target ~uncached ~snooped ~fragment] is the page-table
   entry a GPU of GC [gc] reads at [level] (0, the root, to 3, the leaf) for the
   table at [pa] ([`Table]) or the page there ([`Page]): GPU memory, this GPU's
   or a peer's over the fabric, at the address the GPU's memory controller gives
   it, or system memory at its bus address; [uncached], [snooped] and [fragment]
   (the log2 of the run, in 4 KiB pages) as Rig_pci.Page_table.format states,
   but that GC 12 reaches system memory non-coherently cached (MTYPE_NC) even
   when [uncached], working around a hardware bug as the kernel does. Pure.

   Raises [Invalid_argument] if [pa] is not on 4 KiB or has bits above the 48
   the entry holds. *)
val entry :
  gc:Discovery.version ->
  level:int ->
  pa:int ->
  [ `Table | `Page of [ `Gpu | `System ] ] ->
  uncached:bool ->
  snooped:bool ->
  fragment:int ->
  int64

(* [format g ~flush] is the format of the GPU's page tables, which it writes
   through the memory BAR; a page in [Page_table.Gpu] memory is named by its
   fabric address. [flush] makes the hubs walk the entries written: an HDP
   flush, then the hubs' invalidation, which a virtual function asks its KIQ for
   (Gfx). *)
val format : t -> flush:(unit -> unit) -> Rig_pci.Page_table.format

(* [start_hub g hub tables ~scratch] programs [hub] on every instance: its
   apertures, its L2 cache, and its context 0 over [tables], with faults
   reported and redirected to the system memory page {!fault_page} names;
   [scratch] answers accesses outside the apertures. *)
val start_hub :
  t -> [ `Gc | `Mm ] -> Rig_pci.Page_table.t -> scratch:int -> unit

(* [fault_page g hub a] makes the page of system memory at bus address [a] the
   one [hub]'s faulting accesses reach on every instance, as the kernel's dummy
   page. A hub keeps it across sessions, so each session names its own page
   before the GPU masters the bus. *)
val fault_page : t -> [ `Gc | `Mm ] -> int -> unit

(* [flush_hdp g] writes the HDP flush register and reads it back, so that the
   host's writes through the memory BAR before it are in the GPU's memory. *)
val flush_hdp : t -> unit

(* [hdp g] is the byte offset in the register BAR of the HDP flush register,
   whose store the driver makes before a doorbell. *)
val hdp : t -> int

(* [invalidate g] invalidates both hubs' TLBs for context 0 through their
   invalidation engine 17, waiting for each acknowledgement. Not on a virtual
   function. *)
val invalidate : t -> unit

(* [fault g] is the report of GC's last protection fault: its address and status
   fields. It clears the status. *)
val fault : t -> string
