(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The GPU's memory controller and its two hubs.

    The memory controller places the GPU's own memory. GC's hub carries the
    compute work's accesses, and MM's hub those of the copy engines and the
    firmware. Each hub has page-table walkers and TLBs, and the host's
    writes through the memory BAR pass through the host data path (HDP).
    Calls for one GPU run one at a time. *)

type t
(** The type for a GPU's memory controller and hubs. *)

val make : Regs.t -> Rig_pci.Window.t -> t
(** [make r vram] reads the GPU's memory apertures: where its memory starts
    for the memory controller and, in a fabric of GPUs (XGMI), for its
    peers. *)

val covers : t -> memory:int -> (unit, string) result
(** [covers g ~memory] is {!Rig_amd_abi.Gmc.window} of [g]'s registers. *)

val mc : t -> int -> int
(** [mc g pa] is the address the memory controller gives the GPU's physical
    address [pa]. *)

val fabric : t -> int -> int
(** [fabric g pa] is the address by which the page tables and hubs name the
    GPU's own memory at [pa]. It lies past the memory controller's offset
    ([MC_VM_FB_OFFSET], [0] on a virtual function) and, in a fabric, past
    the start of the GPU's node's segment. *)

val hive : t -> bool
(** [hive g] is [true] iff the GPU is one of several joined by a fabric. *)

val link : t -> gpus:int Lazy.t -> Rig_pci.Memory.link option
(** [link g ~gpus] is the GPU's node in its fabric if {!hive} and the
    machine's [gpus] AMD GPUs number the fabric's nodes, and [None]
    otherwise. [gpus] is forced only if {!hive}. The library reads no
    fabric's identity, so halves of two fabrics that number one fabric's
    nodes are taken for one fabric. *)

val instances : t -> [ `Gc | `Mm ] -> int list
(** [instances g hub] is the instances of [hub]: GC's dies, or MM's hubs,
    which on NBIO 7.9 are the live accelerator dies. *)

val format : t -> flush:(unit -> bool) -> Rig_pci.Page_table.format
(** [format g ~flush] is the format of the GPU's page tables, which it writes
    through the memory BAR. Its [pa_bits] is 48 on GC 9.4 and 9.5 and 44 on
    the others.
    {ul
    {- A page in [Page_table.Gpu] memory is named by its {!fabric}
       address.}
    {- A page in [Page_table.Peer i] memory is named by the address of node
       [i]'s segment of the fabric.}
    {- [flush ()] makes the hubs walk the entries written and is [true] iff
       they confirmed it. It is an HDP flush then the hubs' invalidation,
       which a virtual function asks its KIQ for ({!Gfx.invalidate}).}} *)

val start_hub :
  t -> [ `Gc | `Mm ] -> Rig_pci.Page_table.t -> scratch:int -> unit
(** [start_hub g hub tables ~scratch] programs [hub] on every instance: its
    apertures, its L2 cache, and its context 0 over [tables], whose faults
    raise an interrupt and go to the page {!fault_page} names. The page at
    bus address [scratch] serves accesses outside the apertures. *)

val translates :
  t ->
  [ `Gc | `Mm ] ->
  Rig_pci.Page_table.t ->
  fault:int ->
  (unit, string) result
(** [translates g hub tables ~fault] is [Ok ()] iff every instance of [hub]
    still holds the settings {!start_hub} and {!fault_page} gave it for
    [tables] and the fault page at bus address [fault]: its L2 cache and
    context 0 on, the context over the tables' root, its faults sent to
    [fault], and its system aperture and access mode. A block's reset, or
    the RLC's autoload for GC's hub, clears them, and the hub then passes
    the GPU's addresses through untranslated and sends its faults to host
    address 0.

    The result is [Error msg], naming the hub, the instance, the setting and
    what it reads, otherwise. *)

val fault_page : t -> [ `Gc | `Mm ] -> int -> unit
(** [fault_page g hub a] makes the page of system memory at bus address [a]
    the one [hub]'s faulting accesses reach on every instance. A hub keeps
    the page across sessions, so each session names its own before the GPU
    masters the bus. *)

val flush_hdp : t -> unit
(** [flush_hdp g] writes the HDP flush register and reads it back, so that
    the host's writes through the memory BAR before it are in the GPU's
    memory. *)

val hdp : t -> int
(** [hdp g] is the byte offset in the register BAR of the HDP flush
    register, which the driver stores to before it rings a doorbell. *)

val invalidate : t -> unit
(** [invalidate g] invalidates the TLBs of both hubs for context 0 through
    their invalidation engine 17, and waits for each acknowledgement. It
    skips GC's hub until {!start_hub} has started it. A virtual function
    asks its KIQ instead ({!Gfx.invalidate}).

    Raises {!Regs.Stuck} if a hub does not acknowledge within 10 s. *)

val fault : t -> string
(** [fault g] is the report of GC's last protection fault, its address and
    status fields, and clears the status. *)
