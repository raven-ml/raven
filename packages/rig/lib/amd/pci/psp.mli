(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The GPU's security processor (MP0), which loads every other block's
    firmware.

    Its bootloader starts the processor's own OS from the SOS components
    ({!steps}). The OS then loads the firmware the process hands it through
    a ring in the GPU's memory, and keeps the trusted memory region (TMR)
    the firmware runs from. A command is a 1024-byte buffer that a 64-byte
    frame of the ring points to. Once the processor has run a command, it
    writes its response into the same buffer and the frame's fence value.

    The commands ({!section-commands}) are pure, and any domain may call
    them. Calls for one GPU's processor ({!section-processor}) run one at a
    time. *)

(** {1:commands Commands}

    Each value is a command buffer, its addresses as the GPU's memory
    controller gives them. *)

val load_ip_fw : at:int -> bytes:int -> fw_type:int -> string
(** [load_ip_fw ~at ~bytes ~fw_type] has the processor load the [bytes]
    bytes at [at] as firmware of type [fw_type]. *)

val load_toc : at:int -> bytes:int -> string
(** [load_toc ~at ~bytes] has the processor read the table of contents of
    the [bytes] bytes at [at] and return the TMR's size. *)

val setup_tmr : at:int -> fabric:int -> bytes:int -> string
(** [setup_tmr ~at ~fabric ~bytes] gives the processor the TMR of [bytes]
    bytes at [at], which is [fabric] in the fabric's addresses. With [bytes]
    of [0] it uses the TMR it set up at boot. *)

val autoload_rlc : string
(** [autoload_rlc] tells the processor every firmware is loaded, so that the
    RLC loads the GC's engines. *)

val partition : mode:int -> string
(** [partition ~mode] sets the compute partition mode of a GPU of several
    dies. *)

val frame : command:int -> fence:int -> value:int -> string
(** [frame ~command ~fence ~value] is the ring's frame for the command buffer
    at [command]. The processor writes [value] to [fence] once it has run
    the command. *)

val status : string -> int
(** [status b] is the status the processor returned in command buffer [b],
    [0] for success. *)

val tmr_bytes : string -> int
(** [tmr_bytes b] is the TMR size the processor returned for a
    {!load_toc}. *)

val steps : (int * int) list
(** [steps] is the bootloader's steps, in the order it takes them: the type
    of the SOS component each loads ([PSP_FW_TYPE_*]) and the command that
    loads it ([PSP_BL__LOAD_*]). The last step loads the SOS itself. *)

(** {1:processor The processor} *)

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

val os : t -> Firmware.t -> unit
(** [os p images] is the first part of a full {!start}, before any firmware
    is loaded. Unless the OS runs, it starts it from the SOS components of
    [images]. It then makes the ring and reads the TMR's size from the table
    of contents. Raises as {!start} does. *)

val power_firmware : t -> Firmware.t -> unit
(** [power_firmware p images] loads the power manager's image of [images],
    if it has one, through the running OS. Raises as {!start} does. *)

val firmware : t -> Firmware.t -> unit
(** [firmware p images] is the second part of a full {!start}, after {!os}.
    Through the running OS it loads the power manager's image, sets up the
    TMR where its version needs it, and loads every other piece of
    [images]. It then has the RLC load the GC's engines, or on GC 9 loads
    the register list. Raises as {!start} does. *)

val start : t -> Firmware.t -> partial:bool -> unit
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
(** [set_partition p mode] runs {!partition}[ ~mode]. *)
