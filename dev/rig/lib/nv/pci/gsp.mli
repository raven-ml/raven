(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The GSP: the GPU's system processor, which runs NVIDIA's resource manager.

    The process boots it and then plays the part of the kernel's half of the RM
    (CPU-RM): it describes the system and the GPU's memory to the GSP, runs the
    register sequences the GSP asks of the CPU, hands the GSP the memory of the
    objects it allocates (a channel's instance and method buffer, a compute
    engine's context buffers, a virtual address space's page directory), and
    reads the GSP's events. Every call goes through {!Msgq}.

    The memory the boot gives the GSP is system memory, as the RM places it, but
    for what the GPU's falcons and the GSP's objects read from the GPU's memory.
    The structures are release 570.144's.

    Calls are serialized by a lock of the GSP's; any domain may make them. *)

(** {1:boot Booting} *)

type t
(** The type for a GSP, from the system memory of its boot on. *)

type placement = {
  chip : Chip.t;
  memory : int;  (** The size of the GPU's memory ({!Chip.memory}). *)
  fn : Rig_pci.Function.t;  (** The GPU's function. *)
  tables : Rig_pci.Page_table.t;
      (** The GPU's page tables: the GSP's objects' memory comes from them. *)
  bar : Rig_pci.Window.t;  (** The GPU's memory BAR. *)
  space : Rig_pci.Space.t;
      (** Where the system memory the GSP is given lies, reserved on the GPU's
          machine. *)
}
(** The type for where a boot places the GSP's memory. *)

val boot_pool : [ `Booter of Images.booter | `Fmc of Images.fmc ] -> int
(** [boot_pool start] is the size of the boot pool
    ({!Rig_pci.Page_table.create}'s [boot]) a boot started by [start] needs, on
    2 MiB: its root table, and on Ampere and Ada the FWSEC ucode, at most
    {!Vbios.window} bytes, and the booter's image, which the falcons read from
    GPU memory that any memory BAR reaches. Every other allocation of the boot
    comes from the main pool. *)

val create : placement -> Images.t -> Vbios.fwsec option -> (t, string) result
(** [create p fw fwsec] takes the system memory a boot of [p.chip] with the
    firmware [fw] gives the GSP, and writes into it the queues, the libos
    arguments, the radix-3 table and images, the WPR metadata and, on Blackwell,
    the FMC and its arguments. On Ampere and Ada, [fwsec] is the VBIOS's FWSEC,
    which the boot sets up for the GPU's memory; Blackwell's boot runs none. It
    writes nothing to the GPU. [Error] if the machine refuses the memory, or if
    [fw] and [fwsec] are not of the GPU's family, the memory given back; so if
    it raises. *)

val boot : t -> (unit, string) result
(** [boot g] boots the GSP: it turns the GPU's bus mastering on, sends the
    system's description and the registry, writes the falcons' images to GPU
    memory, starts the GSP ({!Falcon.legacy} or {!Falcon.cot}), waits for its
    [GSP_INIT_DONE], and sets up its golden context: a channel of the GSP's own
    client whose context buffers later channels' contexts copy. [Error] names
    the step that failed. Whether it fails or raises, the GPU may then run until
    {!stop}. *)

val stop : t -> [> `Clean | `Unknown ]
(** [stop g] stops the GPU [g] booted, or began to: it unloads the GSP if the
    GPU answers ({!Rig_pci.Function.failed}) and the GSP set its queue up,
    waiting at most 10 seconds for room in its queue and 10 more for its answer,
    turns the GPU's bus mastering off, and gives back the memory {!create} took.
    It is [`Unknown], the memory kept, if the GPU does not answer once its bus
    mastering is off: it may still read that memory. The GSP runs on until the
    GPU's next reset. *)

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
    directory; a compute engine's context buffers are promoted; a channel's work
    submit token carries its runlist. Its [free] answers [Error]: the GSP's
    objects live as long as the GSP. *)

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
    first fault one reported ({!Msgq.fault}). Once it is [Some], it stays. It
    never waits for the GSP's lock: while a call holds it, the call's own reads
    handle the events, and [check] answers from what they found. *)
