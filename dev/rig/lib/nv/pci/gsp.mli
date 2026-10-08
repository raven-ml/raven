(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The GSP: the GPU's system processor, which runs NVIDIA's resource manager
    (private).

    The process boots it and then plays the part of the kernel's half of the RM:
    it describes the system and the GPU's memory to the GSP, runs the register
    sequences the GSP asks of the CPU, hands the GSP the memory of the objects
    it allocates (a channel's instance and method buffer, a compute engine's
    context buffers, a virtual address space's page directory), and reads the
    GSP's events. Every call goes through {!Msgq}.

    Calls are serialized by a lock of the GSP's; any domain may make them. *)

(** {1:boot Booting} *)

type t
(** The type for a running GSP. *)

type placement = {
  chip : Chip.t;
  tables : Rig_pci.Page_table.t;
      (** The GPU's page tables, booting: the GSP's memory comes from its boot
          pool. *)
  bar : Rig_pci.Window.t;  (** The GPU's memory BAR. *)
  space : Rig_pci.Space.t;
      (** Where the system memory the GSP is given lies, reserved on the GPU's
          machine. *)
}
(** The type for where a boot places the GSP's memory. *)

val boot : placement -> Images.t -> (t, string) result
(** [boot m fw] boots the GSP of [m.chip] with the firmware [fw]: it writes the
    images, the radix-3 table, the queues, the libos arguments, the WPR
    metadata, the system's description and the registry into memory the GPU
    reads, starts the GSP ({!Falcon.boot}), waits for its [GSP_INIT_DONE], and
    sets up its golden context: a channel of the GSP's own client whose context
    buffers later channels' contexts copy. The memory it takes stays the GSP's
    for the process. [Error] names the step that failed. *)

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

val gpu : t -> Rig_nv.rm -> subdevice:int -> (Rig_nv.gpu, string) result
(** [gpu g rm ~subdevice] is the GPU's classes, by its family, and its counts,
    from the RM's static graphics information. *)

(** {1:events Events} *)

val check : t -> string option
(** [check g] handles the events the GSP sent since the last call, and is the
    first fault one reported ({!Msgq.fault}). Once it is [Some], it stays. It
    never waits for the GSP's lock: while a call holds it, the call's own reads
    handle the events, and [check] answers from what they found. *)

val unload : t -> (unit, string) result
(** [unload g] tells the GSP the driver unloads, and waits for its answer, at
    most 10 seconds. *)
