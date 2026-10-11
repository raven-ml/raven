(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The GPU's security processor (MP0), which loads every other block's
    firmware.

    Its bootloader starts the processor's own OS from the SOS components.
    The OS then loads the firmware the process hands it through a ring in
    the GPU's memory, and keeps the trusted memory region (TMR) the firmware
    runs from. A command is a 1024-byte buffer that a 64-byte frame of the
    ring points to. Once the processor has run a command, it writes its
    response into the same buffer and the frame's fence value. Calls for one
    GPU run one at a time. *)

type t
(** The type for security processors. *)

type memory = {
  message : int;  (** 1 MiB, 1 MiB aligned: what a command loads. *)
  command : int;  (** The command buffer. *)
  fence : int;  (** The fence the ring writes. *)
  ring : int;  (** The ring of frames, 64 KiB. *)
}
(** The type for the GPU memory the processor works in, at physical
    addresses. [Boot] lays it out in the boot pool, so that a partial boot
    finds it where the last boot left it. *)

val make :
  Regs.t -> Gmc.t -> Rig_pci.Window.t -> Rig_pci.Page_table.t -> memory -> t
(** [make r gmc vram tables m] is the processor, whose memory is [m]. *)

val bootloader : Regs.t -> bool
(** [bootloader r] is [true] iff the bootloader of the security processor of
    [r]'s GPU is ready for a command, as after a reset. *)

val running : Regs.t -> bool
(** [running r] is [true] iff the OS of the security processor of [r]'s GPU
    runs, as its sign-of-life register says. *)

val alive : t -> bool
(** [alive p] is {!running} of [p]'s GPU. *)

val os : t -> Rig_amd_abi.Firmware.t -> unit
(** [os p images] is the first part of a full {!start}, before any firmware
    is loaded. Unless the OS runs, it starts it from the SOS components of
    [images]. It then makes the ring and reads the TMR's size from the table
    of contents. Raises as {!start} does. *)

val power_firmware : t -> Rig_amd_abi.Firmware.t -> unit
(** [power_firmware p images] loads the power manager's image of [images],
    if it has one, through the running OS. Raises as {!start} does. *)

val firmware : t -> Rig_amd_abi.Firmware.t -> unit
(** [firmware p images] is the second part of a full {!start}, after {!os}.
    Through the running OS it loads the power manager's image, sets up the
    TMR where its version needs it, and loads every other piece of
    [images]. It then has the RLC load the GC's engines, or on GC 9 loads
    the register list. Raises as {!start} does. *)

val start : t -> Rig_amd_abi.Firmware.t -> partial:bool -> unit
(** [start p images ~partial] starts the processor and loads [images].
    {ul
    {- A full boot is {!os} then {!firmware}.}
    {- A partial boot reads the size of the TMR the last boot left from a
       scratch register.}}
    Where this library allocates the TMR, it is the first runtime allocation
    of the GPU's memory in both boots, so it keeps its address.

    Raises {!Regs.Stuck} if the processor does not respond, if it responds
    to a command with an error status, naming the command and the status,
    or if the GPU has no memory for the TMR. *)

val tmr : t -> int
(** [tmr p] is the TMR's size in bytes. *)

val set_partition : t -> int -> unit
(** [set_partition p mode] runs {!Rig_amd_abi.Psp.partition}[ ~mode]. *)
