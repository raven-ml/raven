(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The GSP: the GPU's system processor, which runs NVIDIA's resource manager
    (private).

    The process boots it and then plays the part of the kernel's half of the RM
    (CPU-RM): it describes the system and the GPU's memory to the GSP, runs the
    register sequences the GSP asks of the CPU, hands the GSP the memory of the
    objects it allocates (a channel's instance and method buffer, a compute
    engine's context buffers, a virtual address space's page directory), and
    reads the GSP's events. Every call goes through {!Msgq}.

    The encodings below are pure; the rest acts on a GPU. The memory the boot
    gives the GSP is system memory, as the RM places it, but for what the GPU's
    falcons and the GSP's objects read from the GPU's memory. The structures are
    release 570.144's. A boot runs on Ada only so far; nothing here has run on
    hardware in this library (no host gives root).

    Calls are serialized by a lock of the GSP's; any domain may make them. *)

(** {1:encodings Encodings} *)

val rm_alloc :
  client:int -> parent:int -> obj:int -> cls:int -> string -> string
(** [rm_alloc ~client ~parent ~obj ~cls p] is the body of the RPC [GSP_RM_ALLOC]
    that makes the object [obj] of class [cls] under [parent] of [client] with
    the parameters [p]. *)

val rm_control : client:int -> obj:int -> cmd:int -> string -> string
(** [rm_control ~client ~obj ~cmd p] is the body of the RPC [GSP_RM_CONTROL]
    that runs the command [cmd] on [obj] of [client] with the parameters [p]. *)

val rm_answer : [ `Alloc | `Control ] -> string -> (int * string, string) result
(** [rm_answer k body] is the RM's status and the parameters it wrote back in
    the GSP's answer [body] to an RPC of kind [k], or [Error] if [body] is
    shorter than its header says. *)

val page_directory :
  client:int -> device:int -> vaspace:int -> root:int -> entries:int -> string
(** [page_directory ~client ~device ~vaspace ~root ~entries] is the body of
    [SET_PAGE_DIRECTORY], which points the virtual address space [vaspace] to
    the root table at the physical address [root] of the GPU's memory, of
    [entries] entries. *)

val unloading : string
(** [unloading] is the body of [UNLOADING_GUEST_DRIVER], unloading to level 6:
    the GSP stops every channel and stays idle for the next boot. *)

val registry : (string * int) list -> string
(** [registry keys] is the RM's registry the GSP reads at boot
    ([PACKED_REGISTRY_TABLE]): each key with its 32-bit value. *)

val sequence : libos:int -> string -> (Falcon.op list, string) result
(** [sequence ~libos body] is the register sequence a [GSP_RUN_CPU_SEQUENCER]
    event's [body] asks of the CPU ([rmgspseq.h]): writes, modifications, polls
    and delays, and the resets, starts, halts and resumption of the GSP's falcon
    they name, resumption giving the GSP its libos arguments at the bus address
    [libos] again. [Error] names an opcode it does not know or a sequence that
    ends inside a command. *)

(** {1:boot Booting} *)

type t
(** The type for a running GSP. *)

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

val boot : placement -> Images.t -> (t, string) result
(** [boot p fw] boots the GSP of [p.chip] with the firmware [fw]: it writes the
    images, the radix-3 table, the queues, the libos arguments, the WPR
    metadata, the system's description and the registry into memory the GPU
    reads, starts the GSP ({!Falcon.legacy} or {!Falcon.cot}), waits for its
    [GSP_INIT_DONE], and sets up its golden context: a channel of the GSP's own
    client whose context buffers later channels' contexts copy. The memory it
    takes stays the GSP's for the process. [Error] names the step that failed.
*)

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

val unload : t -> (unit, string) result
(** [unload g] tells the GSP the driver unloads, and waits for its answer, at
    most 10 seconds. *)
