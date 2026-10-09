(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The GPU's memory controller: where it places its own memory, its two hubs
    (GC's, which compute work goes through, and MM's, the copy engines' and
    firmware's), their page-table walkers and TLBs, the entries they read, and
    the host data path (HDP) the host's writes through the memory BAR take.

    The GPU's owner serializes calls. *)

type t

val make : Regs.t -> Rig_pci.Window.t -> t
(** [make r vram] reads the GPU's memory apertures: where its memory starts for
    the memory controller and, in a fabric of GPUs (XGMI), for its peers. *)

val window :
  base:int -> top:int -> fabric:int -> memory:int -> (unit, string) result
(** [window ~base ~top ~fabric ~memory] checks that the memory controller's
    window on the GPU's memory, from the FB location registers' [base] and [top]
    (its first and last 16 MiB unit), holds the [memory] bytes of the GPU's own
    memory at [fabric] bytes past its start. [Error msg] says where the window
    is: an address outside it is system memory to the GPU, which a write then
    reaches. Pure. *)

val covers : t -> memory:int -> (unit, string) result
(** [covers g ~memory] is {!window} of [g]'s registers. *)

val mc : t -> int -> int
(** [mc g pa] is the address the memory controller gives the GPU's physical
    address [pa]. *)

val fabric : t -> int -> int
(** [fabric g pa] is the address its page tables and hubs name the GPU's own
    memory at [pa] by: past the memory controller's offset (MC_VM_FB_OFFSET,
    none on a virtual function) and, in a fabric, its node's segment. *)

val hive : t -> bool
(** [hive g] is [true] iff the GPU is one of several joined by a fabric. *)

val link : t -> gpus:int Lazy.t -> Rig_pci.Memory.link option
(** [link g ~gpus] is the GPU's place in its fabric, its node's number, if
    {!hive} and the machine's [gpus] AMD GPUs number the fabric's nodes; [None]
    otherwise. [gpus] is forced only if {!hive}. Halves of two fabrics that number one fabric's nodes are taken
    for one fabric. *)

val instances : t -> [ `Gc | `Mm ] -> int list
(** [instances g hub] is the instances of [hub]: GC's dies, or MM's hubs (the
    live accelerator dies on a GPU of several). *)

val entry :
  gc:Discovery.version ->
  level:int ->
  pa:int ->
  [ `Table | `Page of [ `Gpu | `System ] ] ->
  uncached:bool ->
  snooped:bool ->
  fragment:int ->
  int64
(** [entry ~gc ~level ~pa target ~uncached ~snooped ~fragment] is the page-table
    entry a GPU of GC [gc] reads at [level] (0, the root, to 3, the leaf) for
    the table at [pa] ([`Table]) or the page there ([`Page]): GPU memory, this
    GPU's or a peer's over the fabric, at the address the GPU's memory
    controller gives it, or system memory at its bus address; [uncached],
    [snooped] and [fragment] (the log2 of the run, in 4 KiB pages) as
    {!Rig_pci.Page_table.format} states, but that GC 12 reaches system memory
    uncached (MTYPE_UC) even when not [uncached], as the kernel's GART does for
    the queues of VMID 0. Pure.

    Raises [Invalid_argument] if [pa] is not on 4 KiB or has bits above the 48
    the entry holds. *)

val format : t -> flush:(unit -> bool) -> Rig_pci.Page_table.format
(** [format g ~flush] is the format of the GPU's page tables, which it writes
    through the memory BAR; a page in [Page_table.Gpu] memory is named by its
    fabric address, and one in [Page_table.Peer i] memory by the address of node
    [i]'s segment of the fabric. Its [pa_bits] is the GPU's: 48 on GC 9.4 and
    9.5, 44 on the others. [flush] makes the hubs walk the entries written: an
    HDP flush, then the hubs' invalidation, which a virtual function asks its
    KIQ for ({!Gfx}), and is [true] iff they confirmed it. *)

val start_hub :
  t -> [ `Gc | `Mm ] -> Rig_pci.Page_table.t -> scratch:int -> unit
(** [start_hub g hub tables ~scratch] programs [hub] on every instance: its
    apertures, its L2 cache, and its context 0 over [tables], with faults
    reported and redirected to the system memory page {!fault_page} names;
    [scratch] answers accesses outside the apertures. *)

val translates :
  t ->
  [ `Gc | `Mm ] ->
  Rig_pci.Page_table.t ->
  fault:int ->
  (unit, string) result
(** [translates g hub tables ~fault] is [Ok ()] iff every instance of [hub]
    holds what {!start_hub} and {!fault_page} gave it over [tables] and the page
    at bus address [fault]: its L2 cache and context 0 on, the context over the
    tables' root, its faults sent to [fault], and its system aperture and access
    mode. A block's reset clears them, as the RLC's autoload clears the GC's
    hub, which then passes the GPU's addresses through untranslated, to host
    memory if no IOMMU stands between, and sends its faults to host address 0.
    [Error msg] names the hub, the instance, the setting, and what it reads. *)

val fault_page : t -> [ `Gc | `Mm ] -> int -> unit
(** [fault_page g hub a] makes the page of system memory at bus address [a] the
    one [hub]'s faulting accesses reach on every instance, as the kernel's dummy
    page. A hub keeps it across sessions, so each session names its own page
    before the GPU masters the bus. *)

val flush_hdp : t -> unit
(** [flush_hdp g] writes the HDP flush register and reads it back, so that the
    host's writes through the memory BAR before it are in the GPU's memory. *)

val hdp : t -> int
(** [hdp g] is the byte offset in the register BAR of the HDP flush register,
    whose store the driver makes before a doorbell. *)

val invalidate : t -> unit
(** [invalidate g] invalidates both hubs' TLBs for context 0 through their
    invalidation engine 17, waiting for each acknowledgement. Not on a virtual
    function. *)

val fault : t -> string
(** [fault g] is the report of GC's last protection fault: its address and
    status fields. It clears the status. *)
