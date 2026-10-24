(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The GSP, the GPU's system processor, which runs NVIDIA's resource
    manager. The process boots it and then plays the kernel's half of the
    RM (CPU-RM) through {!Msgq}; a lock of the GSP's serializes the calls,
    which any domain may make. *)

(** {1:boot Booting} *)

type t
(** The type for a GSP, from the system memory of its boot on. *)

type placement = {
  chip : Chip.t;
  memory : int;  (** The size of the GPU's memory ({!Chip.memory}). *)
  tables : Rig_pci.Page_table.t;
      (** The GPU's page tables, which must translate from address 0, as the
          GPU's MMU does, and leave the GSP's addresses ({!Rpc.reserved})
          unmapped: the GSP's objects' memory comes from them. *)
  tables_lock : Rig_nv.Lock.t;
      (** The lock under which every call of [tables] runs, which the GSP
          takes too after its boot. *)
  bar : Rig_pci.Window.t;  (** The GPU's memory BAR. *)
  space : Rig_pci.Space.t;
      (** Where the system memory the GSP is given lies, reserved on the GPU's
          machine. *)
}
(** The type for where a boot places the GSP's memory. *)

val boot_pool : [ `Booter of Images.booters | `Fmc of Images.fmc ] -> int
(** [boot_pool start] is the size of the boot pool
    ({!Rig_pci.Page_table.create}'s [boot]) a boot started by [start] needs, on
    2 MiB: its root table, and on Ampere and Ada the FWSEC ucode, at most
    {!Vbios.window} bytes, and the larger of the booters' images, which the
    falcons read from GPU memory that any memory BAR reaches, at the boot and
    at the stop. Every other allocation of the boot comes from the main
    pool. *)

val create : placement -> Images.t -> Vbios.fwsec option -> (t, string) result
(** [create p fw fwsec] takes the system memory a boot of [p.chip] with the
    firmware [fw] gives the GSP, and writes into it the queues, the libos
    arguments, the radix-3 table and images, the WPR metadata and, on Blackwell,
    the FMC and its arguments. It also takes a page for the GPU's flushes of
    its writes to system memory, which read it. On Ampere and Ada, [fwsec] is
    the VBIOS's FWSEC, which the boot sets up for the GPU's memory, and
    [create] takes the GPU memory of FWSEC's image and of the larger booter
    from the boot pool; Blackwell's boot runs none. It writes nothing to the
    GPU. [Error] if the machine refuses the memory, if the flush page lies past
    the bus addresses the GPU's registers hold (2{^47} on Ampere and Ada,
    2{^52} on Blackwell), if the boot pool has no room for the images, or if
    [fw] and [fwsec] are not of the GPU's family. It gives the system memory
    back when it fails or raises. *)

val boot : t -> (unit, string) result
(** [boot g] boots the GSP: it gives the GPU the flush page's bus address,
    which a reset sets to 0, turns the GPU's bus mastering on, sends the
    system's description and the registry, writes the falcons' images to GPU
    memory, starts the GSP ({!Falcon.legacy} or {!Falcon.cot}), waits for its
    [GSP_INIT_DONE], and sets up its golden context: a channel of the GSP's own
    client whose context buffers later channels' contexts copy. [Error] names
    the step that failed. Whether it fails or raises, the GPU may then run until
    {!stop}. *)

val stop : t -> [> `Clean | `Lost | `Unknown ]
(** [stop g] stops the GPU [g] booted, or began to. If the GPU responds
    ({!Rig_pci.Function.failed}):
    - it unloads the GSP if the GSP set its queue up and answered every call
      within its bound ({!check}), waiting at most 10 seconds for an answer
      still owed, to a call an exception cut or whose answer could not be
      read, 10 for room in its queue and 10 more for its reply, then up to 30
      seconds for the GSP's
      firmware to suspend ({!Falcon.suspended});
    - on Ampere and Ada, if the boot set the GPU's protected memory (WPR2)
      up, it runs FWSEC's SB command and the unloading booter, which take
      WPR2 down ({!Falcon.fwsec_sb}, {!Falcon.unload}), as NVIDIA's driver
      needs it to boot the GPU again.

    It then turns the GPU's bus mastering off and gives back the memory
    {!create} took, and the addresses of the GPU memory the GSP still holds
    to the space every GPU shares. It is [`Unknown], both kept, if the GPU
    stops responding once its bus mastering is off: it may still read that
    memory. It is [`Lost] if the boot raised WPR2 and the unload did not stop
    its channels, if the unloading booter failed or left WPR2 up, or, on
    Blackwell, if the GSP's core did not halt; and [`Clean] otherwise. *)

(** {1:rm The resource manager} *)

val rm :
  t ->
  locate:(int -> int -> ([ `Gpu | `System ] * int) option) ->
  (Rig_nv.rm, string) result
(** [rm g ~locate] is a new client of the GSP's RM, of release 570. Its
    allocations do the kernel's part: a channel's USERD is described to the GSP
    in the memory and at the address [locate h off] gives for the memory of
    handle [h] at offset [off] (a physical address in the GPU's memory, a bus
    address in system memory); a virtual address space gets the GPU's page
    directory; a compute object's context is handed over as NVIDIA's RM and
    its UVM do ({!Rpc.compute_context}). Its commands go to the GSP as NVIDIA's
    RM sends them, but for a channel's work submit token, which it makes
    without the GSP and which carries the channel's runlist, and a boost,
    which goes as the GSP's own command ({!Rpc.perf_boost}). Its [free] frees
    an object and every object under it, then gives back the GPU memory the
    client gave the GSP for them; once the GPU has left the bus or faulted
    ({!check}) it is a refusal, and calls nothing. *)

val objects : Rig_nv.rm -> (int * int * int, string) result
(** [objects rm] is the GPU's device, subdevice and virtual address space, new
    objects of [rm]'s client: the objects a path gives the driver. The address
    space spans the GPU's addresses from 4 KiB, and is externally owned: its
    page directory is the process's. *)

val gpu : t -> Rig_nv.rm -> subdevice:int -> (Rig_nv.gpu, string) result
(** [gpu g rm ~subdevice] is the GPU's classes, by its family, and its counts,
    from the RM's static graphics information. *)

(** {1:events Events} *)

val check : t -> string option
(** [check g] handles the events the GSP sent since the last call, and is the
    first fault one reported ({!Rpc.fault}). A message that cannot be read
    is a fault too, and so is a call the GSP leaves unanswered past its
    bound, whose error names the first fault and the last report of each
    error the GSP logged since the call began ({!Rpc.report}). Once it is
    [Some], it stays. An event stays in the GSP's queue until it is handled, so an
    exception that interrupts [check] or a call leaves it for the next one;
    one that interrupts a CPU sequence the GSP asked for counts as a fault,
    since a sequence runs once, front to back. It never waits for the GSP's
    lock: while a call holds it, the call's own reads handle the events, and
    [check] returns what they found. It handles none while the calling
    thread holds a path's lock ({!Rig_nv.Lock.try_protect}). *)
